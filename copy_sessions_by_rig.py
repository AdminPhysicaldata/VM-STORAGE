#!/usr/bin/env python3
"""
copy_sessions_by_rig.py — Copie N sessions d'un rig donné vers un dossier
de destination, pour envoi à la demande (debug, contrôle qualité, partage...).

Parcourt directement SESSIONS_DIR (/data/sessions), lit le config.json de
chaque session pour en extraire le rig (config["rig"]["code"]), garde celles
qui correspondent au rig_id demandé, puis copie les dossiers vers --dest.
Optionnellement, envoie ensuite ce dossier vers une machine distante via
rsync (SSH).

Usage :
    python3 copy_sessions_by_rig.py rig_06
    python3 copy_sessions_by_rig.py rig_06 --count 10 --dest /data/exports/rig_06
    python3 copy_sessions_by_rig.py rig_06 --order recent --dry-run
    python3 copy_sessions_by_rig.py rig_06 --remote user@host:/srv/incoming/

Variables d'environnement :
    SESSIONS_DIR
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/data/sessions")


def _session_mtime(session_dir: Path) -> float:
    """Dernière date de modification connue d'une session (fallback de tri sans BDD)."""
    try:
        return session_dir.stat().st_mtime
    except OSError:
        return 0.0


def _read_config(session_dir: Path) -> dict | None:
    config_path = session_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("config.json illisible : %s", config_path)
        return None


def find_rig_sessions(sessions_dir: Path, rig_id: str, count: int, order: str) -> list[Path]:
    """Parcourt sessions_dir et garde les dossiers dont config.json a rig.code == rig_id."""
    matches: list[Path] = []
    for entry in sorted(sessions_dir.iterdir()):
        if not entry.is_dir():
            continue
        cfg = _read_config(entry)
        if not cfg:
            continue
        rig = cfg.get("rig", {})
        if rig.get("code") != rig_id:
            continue
        matches.append(entry)

    if order == "recent":
        matches.sort(key=_session_mtime, reverse=True)
    elif order == "oldest":
        matches.sort(key=_session_mtime)
    elif order == "random":
        import random
        random.shuffle(matches)

    return matches[:count]


def folder_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total / 1_048_576


def copy_session(src: Path, dest_dir: Path, dry_run: bool) -> tuple[bool, str]:
    dst = dest_dir / src.name
    if dst.exists():
        return True, f"déjà présent dans {dest_dir}"
    if dry_run:
        return True, f"[DRY-RUN] serait copié — {folder_size_mb(src):.1f} MB"
    try:
        shutil.copytree(str(src), str(dst))
        return True, f"copié ({folder_size_mb(dst):.1f} MB)"
    except Exception as exc:
        return False, f"erreur copie : {exc}"


def send_remote(dest_dir: Path, remote: str, dry_run: bool) -> bool:
    """Envoie dest_dir vers `remote` (format user@host:/chemin/) via rsync over SSH."""
    cmd = ["rsync", "-avz", "--progress", f"{dest_dir}/", remote]
    if dry_run:
        cmd.insert(1, "--dry-run")
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Copie N sessions d'un rig vers un dossier de destination (envoi à la demande)"
    )
    parser.add_argument("rig_id", help="Identifiant du rig, ex: rig_06")
    parser.add_argument("--count", type=int, default=10, help="Nombre de sessions à copier (défaut: 10)")
    parser.add_argument("--order", choices=["recent", "oldest", "random"], default="recent",
                         help="Ordre de sélection des sessions (défaut: recent)")
    parser.add_argument("--dest", default=None,
                         help="Dossier de destination (défaut: ./<rig_id>_export_<timestamp>)")
    parser.add_argument("--remote", default=None,
                         help="Envoie ensuite le dossier vers user@host:/chemin/ via rsync (SSH)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans copier ni envoyer")
    parser.add_argument("--yes", action="store_true", help="Pas de confirmation interactive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = Path(args.dest) if args.dest else Path(f"./{args.rig_id}_export_{timestamp}")

    sessions_dir = Path(SESSIONS_DIR)
    if not sessions_dir.is_dir():
        print(f"[ERREUR] Dossier de sessions introuvable : {sessions_dir}")
        sys.exit(1)

    sessions = find_rig_sessions(sessions_dir, args.rig_id, args.count, args.order)

    if not sessions:
        print(f"Aucune session trouvée pour le rig '{args.rig_id}' dans {sessions_dir}.")
        sys.exit(1)

    print(f"\n{len(sessions)} session(s) trouvée(s) pour {args.rig_id} (ordre: {args.order}) :\n")

    resolved = []
    for src in sessions:
        print(f"  • {src.name} — {src}")
        resolved.append((src.name, src))

    if not resolved:
        print("Rien à copier.")
        sys.exit(1)

    print(f"\nDestination : {dest_dir.resolve()}")
    if args.remote:
        print(f"Envoi distant après copie : {args.remote}")

    if not args.dry_run and not args.yes:
        answer = input(f"\nCopier ces {len(resolved)} session(s) ? [oui/non] : ").strip().lower()
        if answer not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            sys.exit(0)

    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    print()
    ok = 0
    fail = 0
    for name, src in resolved:
        success, msg = copy_session(src, dest_dir, args.dry_run)
        print(f"  {name} : {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n=== Copie terminée === {ok} réussie(s), {fail} échouée(s)")

    if args.remote and ok > 0:
        print(f"\nEnvoi vers {args.remote}...")
        sent = send_remote(dest_dir, args.remote, args.dry_run)
        print("Envoi réussi." if sent else "[ERREUR] Envoi échoué.")
        sys.exit(0 if sent else 2)

    sys.exit(0 if fail == 0 else 2)


if __name__ == "__main__":
    main()
