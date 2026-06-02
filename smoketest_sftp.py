#!/usr/bin/env python3
"""
smoketest_sftp.py — Test de fumée SFTP multi-angle pour vm-storage.

Diagnostique le problème "Accepted password … Connection reset by peer"
en testant chaque couche séparément : TCP → SSH banner → auth → subsystème
SFTP → permissions → opérations fichier → concurrence.

Usage:
    python3 smoketest_sftp.py [--host HOST] [--port PORT] [--user USER] [--pass PASS]

Défauts lus depuis vm-storage/.env (SFTP_USER / SFTP_PASS) ou variables d'env.
"""

import argparse
import concurrent.futures
import io
import os
import socket
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Couleurs terminal
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


# ---------------------------------------------------------------------------
# Lecture .env
# ---------------------------------------------------------------------------
def _load_env_file():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_load_env_file()


# ---------------------------------------------------------------------------
# Résultats agrégés
# ---------------------------------------------------------------------------
_results: list[tuple[str, bool, str]] = []

def record(name: str, passed: bool, detail: str = ""):
    _results.append((name, passed, detail))
    if passed:
        ok(f"{name}" + (f"  —  {detail}" if detail else ""))
    else:
        fail(f"{name}" + (f"  —  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    """1. Connectivité TCP brute."""
    print(f"\n{BOLD}[1] Connectivité TCP{RESET}")
    try:
        t0 = time.monotonic()
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        ms = (time.monotonic() - t0) * 1000
        record("TCP connect", True, f"{ms:.0f} ms")
        return True
    except Exception as e:
        record("TCP connect", False, str(e))
        print(f"\n  {RED}BLOQUANT : aucune connectivité TCP sur {host}:{port}{RESET}")
        print("  Vérifiez que le container sftp tourne : docker ps | grep sftp")
        return False


def test_ssh_banner(host: str, port: int) -> bool:
    """2. Bannière SSH — confirme que c'est bien sshd qui répond."""
    print(f"\n{BOLD}[2] Bannière SSH{RESET}")
    try:
        s = socket.create_connection((host, port), timeout=5)
        banner = s.recv(256).decode(errors="replace").strip()
        s.close()
        if banner.startswith("SSH-"):
            record("SSH banner", True, banner[:60])
            return True
        else:
            record("SSH banner", False, f"Réponse inattendue : {banner[:60]!r}")
            return False
    except Exception as e:
        record("SSH banner", False, str(e))
        return False


def _make_transport(host: str, port: int, user: str, password: str):
    """Ouvre un transport SSH authentifié (lève en cas d'échec)."""
    import paramiko
    t = paramiko.Transport((host, port))
    t.connect(username=user, password=password)
    return t


def test_auth(host: str, port: int, user: str, password: str) -> bool:
    """3. Authentification SSH par mot de passe."""
    print(f"\n{BOLD}[3] Authentification SSH{RESET}")
    try:
        import paramiko
    except ImportError:
        warn("paramiko non installé — pip install paramiko")
        return False

    try:
        t = _make_transport(host, port, user, password)
        t.close()
        record("SSH auth (password)", True, f"user={user}")
        return True
    except Exception as e:
        record("SSH auth (password)", False, str(e))
        if "Authentication failed" in str(e):
            info("Mauvais mot de passe ou utilisateur introuvable dans le container")
        elif "Connection reset" in str(e) or "EOF" in str(e):
            info("La connexion se ferme après auth → problème de shell/subsystème (causa prima)")
        return False


def test_sftp_open(host: str, port: int, user: str, password: str):
    """4. Ouverture du sous-système SFTP — c'est là que 'Connection reset' se produit."""
    print(f"\n{BOLD}[4] Ouverture sous-système SFTP{RESET}")
    import paramiko

    try:
        t = _make_transport(host, port, user, password)
        try:
            sftp = paramiko.SFTPClient.from_transport(t)
            record("SFTP subsystem open", True)
            return sftp, t
        except Exception as e:
            record("SFTP subsystem open", False, str(e))
            info("Causes fréquentes :")
            info("  • sshd_config manque 'Subsystem sftp /usr/lib/ssh/sftp-server'")
            info("  • chroot directory introuvable ou permissions incorrectes")
            info("  • /home/<user>/sessions n'existe pas dans le volume monté")
            t.close()
            return None, None
    except Exception as e:
        record("SFTP subsystem open", False, f"transport: {e}")
        return None, None


def test_listdir(sftp, remote_path: str = ".") -> bool:
    """5. Listage du répertoire racine et du dossier sessions."""
    print(f"\n{BOLD}[5] Listage répertoires{RESET}")
    try:
        entries = sftp.listdir(remote_path)
        record(f"listdir({remote_path!r})", True, f"{len(entries)} entrées: {entries[:8]}")
    except Exception as e:
        record(f"listdir({remote_path!r})", False, str(e))
        return False

    for sub in ("sessions", "."):
        try:
            sub_entries = sftp.listdir(sub)
            record(f"listdir({sub!r})", True, f"{len(sub_entries)} entrées")
        except Exception as e:
            record(f"listdir({sub!r})", False, str(e))

    return True


def test_stat(sftp) -> bool:
    """6. Stat sur les chemins critiques."""
    print(f"\n{BOLD}[6] Stat répertoires{RESET}")
    for path in (".", "sessions"):
        try:
            st = sftp.stat(path)
            import stat as stat_mod
            mode = oct(stat_mod.S_IMODE(st.st_mode))
            record(f"stat({path!r})", True, f"mode={mode} uid={st.st_uid} gid={st.st_gid}")
        except Exception as e:
            record(f"stat({path!r})", False, str(e))
    return True


def test_write_read(sftp) -> bool:
    """7. Écriture puis relecture d'un fichier test dans sessions/."""
    print(f"\n{BOLD}[7] Écriture / Lecture{RESET}")
    remote_file = f"sessions/.smoketest_{int(time.time())}.tmp"
    payload = b"smoketest-ok-" + os.urandom(16).hex().encode()

    # Upload
    try:
        sftp.putfo(io.BytesIO(payload), remote_file)
        record("put (upload)", True, remote_file)
    except Exception as e:
        record("put (upload)", False, str(e))
        if "Permission denied" in str(e):
            info("Le user SFTP n'a pas les droits d'écriture dans sessions/")
            info("Vérifiez sftp-setup.sh : chown 1000:1000 et chmod 755/775")
        return False

    # Download
    try:
        buf = io.BytesIO()
        sftp.getfo(remote_file, buf)
        downloaded = buf.getvalue()
        if downloaded == payload:
            record("get (download)", True, f"{len(downloaded)} bytes, contenu correct")
        else:
            record("get (download)", False, "contenu corrompu")
    except Exception as e:
        record("get (download)", False, str(e))

    # Nettoyage
    try:
        sftp.remove(remote_file)
        record("remove (delete)", True)
    except Exception as e:
        record("remove (delete)", False, str(e))

    return True


def test_large_file(sftp, size_mb: int = 5) -> bool:
    """8. Upload / download d'un fichier de taille réelle (test stabilité)."""
    print(f"\n{BOLD}[8] Fichier volumineux ({size_mb} MB){RESET}")
    remote_file = f"sessions/.smoketest_large_{int(time.time())}.tmp"
    data = os.urandom(size_mb * 1024 * 1024)

    try:
        t0 = time.monotonic()
        sftp.putfo(io.BytesIO(data), remote_file)
        upload_s = time.monotonic() - t0
        record(f"upload {size_mb} MB", True, f"{upload_s:.2f}s — {size_mb/upload_s:.1f} MB/s")
    except Exception as e:
        record(f"upload {size_mb} MB", False, str(e))
        return False

    try:
        buf = io.BytesIO()
        t0 = time.monotonic()
        sftp.getfo(remote_file, buf)
        dl_s = time.monotonic() - t0
        match = buf.getvalue() == data
        record(f"download {size_mb} MB", match, f"{dl_s:.2f}s — {size_mb/dl_s:.1f} MB/s")
    except Exception as e:
        record(f"download {size_mb} MB", False, str(e))

    try:
        sftp.remove(remote_file)
    except Exception:
        pass

    return True


def test_mkdir_rmdir(sftp) -> bool:
    """9. Création d'un sous-dossier session (simule le comportement du vrai client)."""
    print(f"\n{BOLD}[9] Création / suppression dossier session{RESET}")
    session_dir = f"sessions/session_smoketest_{int(time.time())}"

    try:
        sftp.mkdir(session_dir)
        record("mkdir (session dir)", True, session_dir)
    except Exception as e:
        record("mkdir (session dir)", False, str(e))
        return False

    # Écriture d'un fichier dans ce dossier
    test_file = f"{session_dir}/tracker_positions.csv"
    try:
        sftp.putfo(io.BytesIO(b"timestamp,x,y,z\n0.0,1,2,3\n"), test_file)
        record("put dans session dir", True)
    except Exception as e:
        record("put dans session dir", False, str(e))

    # Nettoyage
    try:
        sftp.remove(test_file)
        sftp.rmdir(session_dir)
        record("rmdir (session dir)", True)
    except Exception as e:
        record("rmdir (session dir)", False, str(e))

    return True


def test_chroot_escape(sftp) -> bool:
    """10. Sécurité : le client ne doit pas pouvoir sortir du chroot."""
    print(f"\n{BOLD}[10] Sécurité chroot (ne doit PAS pouvoir lister /etc){RESET}")
    escape_paths = ["../etc", "../../etc", "/etc", "/../etc"]
    escaped = False
    for path in escape_paths:
        try:
            entries = sftp.listdir(path)
            if "passwd" in entries or "hostname" in entries:
                record(f"chroot escape via {path!r}", False, "FUITE CHROOT DÉTECTÉE — /etc accessible !")
                escaped = True
            else:
                warn(f"listdir({path!r}) a répondu mais sans /etc/passwd ({len(entries)} entrées)")
        except Exception:
            pass  # Normal : accès refusé

    if not escaped:
        record("chroot confinement", True, "impossible de lister /etc depuis l'extérieur")
    return not escaped


def test_concurrent(host: str, port: int, user: str, password: str, n: int = 5) -> bool:
    """11. Connexions simultanées — reproduit le comportement des VMs."""
    print(f"\n{BOLD}[11] Connexions simultanées ({n} connexions){RESET}")
    import paramiko

    errors: list[str] = []
    lock = threading.Lock()

    def connect_and_list(idx: int):
        try:
            t = _make_transport(host, port, user, password)
            sftp = paramiko.SFTPClient.from_transport(t)
            sftp.listdir("sessions")
            sftp.close()
            t.close()
        except Exception as e:
            with lock:
                errors.append(f"conn#{idx}: {e}")

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(connect_and_list, i) for i in range(n)]
        concurrent.futures.wait(futs)
    elapsed = time.monotonic() - t0

    ok_count = n - len(errors)
    passed = len(errors) == 0
    record(f"concurrence {n} connexions", passed,
           f"{ok_count}/{n} réussies en {elapsed:.2f}s")
    for e in errors[:3]:
        info(f"  Erreur: {e}")
    return passed


def test_reconnect_stability(host: str, port: int, user: str, password: str,
                              attempts: int = 10) -> bool:
    """12. Reconnexions répétées — reproduit le pattern des logs (connect/disconnect rapide)."""
    print(f"\n{BOLD}[12] Stabilité reconnexions ({attempts} fois){RESET}")
    import paramiko

    ok_count = 0
    first_error = None
    for i in range(attempts):
        try:
            t = _make_transport(host, port, user, password)
            sftp = paramiko.SFTPClient.from_transport(t)
            sftp.listdir("sessions")
            sftp.close()
            t.close()
            ok_count += 1
        except Exception as e:
            if first_error is None:
                first_error = str(e)

    passed = ok_count == attempts
    record(f"reconnexions {attempts}x", passed,
           f"{ok_count}/{attempts} réussies" + (f" — 1ère erreur: {first_error}" if first_error else ""))
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("VM_STORAGE_IP", "192.168.1.20"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SFTP_PORT", "2222")))
    p.add_argument("--user", default=os.environ.get("SFTP_USER", "exoria"))
    p.add_argument("--pass", dest="password", default=os.environ.get("SFTP_PASS", ""))
    p.add_argument("--large-mb", type=int, default=5, help="Taille du fichier volumineux en MB (défaut: 5)")
    p.add_argument("--concurrent", type=int, default=5, help="Nombre de connexions simultanées (défaut: 5)")
    p.add_argument("--reconnects", type=int, default=10, help="Nombre de reconnexions stabilité (défaut: 10)")
    p.add_argument("--skip-large", action="store_true", help="Passer le test de fichier volumineux")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  SFTP SMOKETEST — {args.host}:{args.port}{RESET}")
    print(f"  user={args.user}  port={args.port}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Vérification de paramiko
    try:
        import paramiko
    except ImportError:
        print(f"\n{RED}ERREUR : paramiko non installé.{RESET}")
        print("  pip install paramiko")
        sys.exit(1)

    # Supprime les logs verbeux de paramiko
    import logging
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)

    # ---- Tests séquentiels ----
    if not test_tcp(args.host, args.port):
        sys.exit(2)

    if not test_ssh_banner(args.host, args.port):
        sys.exit(2)

    if not test_auth(args.host, args.port, args.user, args.password):
        sys.exit(2)

    sftp, transport = test_sftp_open(args.host, args.port, args.user, args.password)
    if sftp is None:
        _print_summary()
        sys.exit(2)

    try:
        test_listdir(sftp)
        test_stat(sftp)
        test_write_read(sftp)
        if not args.skip_large:
            test_large_file(sftp, size_mb=args.large_mb)
        test_mkdir_rmdir(sftp)
        test_chroot_escape(sftp)
    finally:
        try:
            sftp.close()
            transport.close()
        except Exception:
            pass

    # ---- Tests concurrence (nouvelles connexions) ----
    test_concurrent(args.host, args.port, args.user, args.password, n=args.concurrent)
    test_reconnect_stability(args.host, args.port, args.user, args.password, attempts=args.reconnects)

    _print_summary()


def _print_summary():
    total  = len(_results)
    passed = sum(1 for _, p, _ in _results if p)
    failed = total - passed

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RÉSUMÉ : {passed}/{total} tests réussis{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if failed:
        print(f"\n{RED}Tests échoués :{RESET}")
        for name, p, detail in _results:
            if not p:
                print(f"  {RED}✗{RESET} {name}  —  {detail}")
        print()
        _print_diagnosis()
    else:
        print(f"\n{GREEN}Tous les tests sont passés.{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


def _print_diagnosis():
    """Affiche un diagnostic ciblé selon les tests échoués."""
    failed_names = {name for name, p, _ in _results if not p}

    print(f"{BOLD}Diagnostic :{RESET}")

    if "SFTP subsystem open" in failed_names:
        print(f"""
  {YELLOW}→ Le sous-système SFTP ne s'ouvre pas après auth.{RESET}
    C'est la cause directe des "Connection reset by peer" dans vos logs.
    Vérifications :

    1. Le dossier sessions/ existe dans le volume monté ?
       Sur la VM storage :  ls -la $SESSIONS_PATH
       Dans le container :  docker exec sftp ls -la /home/exoria/

    2. Permissions du dossier sessions/ (doit être 755, owned 1000:1000) :
       docker exec sftp ls -la /home/exoria/sessions

    3. Permissions du chroot parent (doit être owned root:root, mode 755) :
       docker exec sftp ls -la /home/exoria

    4. sftp-setup.sh est-il exécuté au démarrage ?
       docker exec sftp cat /etc/sftp.d/fix-perms.sh
       docker logs sftp | head -30
""")

    if any("listdir" in n for n in failed_names):
        print(f"""
  {YELLOW}→ Impossible de lister le répertoire.{RESET}
    Le sous-système SFTP s'ouvre mais le chroot échoue.
    Commande de diagnostic :
      docker exec sftp ls -la /home/{os.environ.get('SFTP_USER','exoria')}/
      docker exec sftp id {os.environ.get('SFTP_USER','exoria')}
""")

    if any("put" in n.lower() or "upload" in n.lower() for n in failed_names):
        print(f"""
  {YELLOW}→ Permission refusée en écriture.{RESET}
    Le dossier sessions/ n'est pas writable pour l'uid 1000.
    Fix : dans sftp-setup.sh → chmod 775 /home/*/sessions
    Puis redémarrer : docker compose restart sftp
""")

    if "chroot confinement" in failed_names:
        print(f"""
  {RED}→ FUITE CHROOT DÉTECTÉE ! Le client peut lire /etc.{RESET}
    atmoz/sftp doit avoir ChrootDirectory configuré.
    Vérifier : docker exec sftp grep -i chroot /etc/ssh/sshd_config
""")

    if any("concurren" in n.lower() or "reconnex" in n.lower() for n in failed_names):
        print(f"""
  {YELLOW}→ Instabilité sous charge / reconnexions rapides.{RESET}
    Vérifiez MaxSessions et MaxStartups dans sshd_config :
      docker exec sftp grep -E 'MaxSessions|MaxStartups' /etc/ssh/sshd_config
    Valeurs recommandées : MaxSessions 100, MaxStartups 100:30:200
""")


if __name__ == "__main__":
    main()
