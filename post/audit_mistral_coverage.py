#!/usr/bin/env python3
"""
Audite la couverture d'envoi Mistral d'un disque entier.

Parcourt récursivement un ou plusieurs dossiers racine (n'importe quelle
profondeur — contrairement à SessionsToMistral/run_pipeline qui ne regardent
qu'un seul niveau) et trouve TOUTES les sessions (dossiers `session_*`), où
qu'elles soient rangées sur le disque. Pour chacune, la source de vérité est
la BDD backend (GET delivered-folders, client Mistral) — pas la simple
présence dans un dossier "envoyé(e)s", qui peut mentir (cf. mode
--mistral-offline, voir backfill_db_sent.py).

Classement de chaque session trouvée :
  - envoyée (confirmée BDD)        : compte dans "sent"
  - envoyée (fichier dodge local,
    PAS confirmée en BDD)          : compte dans "sent" aussi (le zip est
                                      bien parti chez Mistral), mais signalée
                                      à part — à rattraper avec
                                      backfill_db_sent.py
  - exclue (dodge "skipped")       : jamais envoyable (invalide/rejetée/vide)
                                      — ne compte ni pour ni contre les 100%
  - en attente                     : ni envoyée ni exclue — c'est ce qui
                                      empêche les 100% ; le script liste un
                                      échantillon avec la raison probable
                                      (validate_session / jamais traitée)

Affiche la progression dossier par dossier (chaque enfant direct de la
racine = un "gros dossier"), avec le % de données (en octets) déjà envoyées
à Mistral pour ce dossier, puis un résumé global à la fin.

Lecture seule : aucune écriture sur disque, aucun appel de mutation BDD.

Usage :
  python3 audit_mistral_coverage.py /data
  python3 audit_mistral_coverage.py /mnt/hdd1 /mnt/hdd2 /mnt/hdd3
  python3 audit_mistral_coverage.py /data --show-pending 30 --json rapport.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sessions-uploader"))
import SessionsToMistral as mistral_uploader  # noqa: E402

# Un vrai dossier de session est nommé session_YYYYMMDD_HHMMSS(...). Un simple
# startswith("session_") mord aussi sur des dossiers CONTENEURS qui partagent
# le préfixe par coïncidence (ex. "session_envoye", le dossier singulier créé
# par move_session_to_sent — à ne pas confondre avec "sessions_envoyes",
# pluriel, qui lui ne matche pas le préfixe) : un tel dossier serait alors pris
# pour UNE SEULE session géante au lieu d'être parcouru, et son contenu réel
# jamais vu. D'où l'ancrage sur le format de date, identique à
# SessionsToMistral._SESSION_DATE_RE.
SESSION_NAME_RE = re.compile(r"^session_\d{8}_\d{6}", re.IGNORECASE)
DODGE_FILE = mistral_uploader.DODGE_FILE  # "uploaded_sessions.json"


# ---------------------------------------------------------------------------
# Parcours disque — un seul passage : chaque fichier est visité une fois,
# soit à l'intérieur d'une session (comptée dans sa taille), soit hors
# session (ignoré). On ne redescend jamais À L'INTÉRIEUR d'un dossier
# session_* déjà identifié (cameras/, sensors/... comptés, jamais visités
# comme sous-dossiers indépendants).
# ---------------------------------------------------------------------------

def _session_size(session_dir: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(session_dir):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def scan_bucket(bucket_path: Path, dodge_paths: list) -> list[tuple[str, str, int]]:
    """
    Retourne [(chemin_complet, nom_session, taille_octets), ...] pour toutes
    les sessions trouvées récursivement sous bucket_path (bucket_path lui
    même inclus s'il s'agit directement d'une session). Empile aussi tout
    fichier uploaded_sessions.json croisé en route dans dodge_paths.
    """
    if SESSION_NAME_RE.match(bucket_path.name):
        return [(str(bucket_path), bucket_path.name, _session_size(str(bucket_path)))]

    results: list[tuple[str, str, int]] = []
    for dirpath, dirnames, filenames in os.walk(bucket_path):
        keep = []
        for d in dirnames:
            if SESSION_NAME_RE.match(d):
                sdir = os.path.join(dirpath, d)
                results.append((sdir, d, _session_size(sdir)))
            else:
                keep.append(d)
        dirnames[:] = keep
        if DODGE_FILE in filenames:
            dodge_paths.append(os.path.join(dirpath, DODGE_FILE))
    return results


def load_dodge_records(dodge_paths: list[str]) -> tuple[dict[str, int], dict[str, tuple[str, object]]]:
    """
    Agrège tous les fichiers uploaded_sessions.json trouvés sur le disque.
    Retourne (local_sent: name -> size_bytes, skipped: name -> (kind, detail)).
    """
    local_sent: dict[str, int] = {}
    skipped: dict[str, tuple[str, object]] = {}
    for path in dodge_paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Avertissement : dodge illisible '{path}' : {exc}", file=sys.stderr)
            continue
        for e in data.get("sessions", []):
            name = e.get("name")
            if name:
                local_sent[name] = e.get("size_bytes", 0)
        for e in data.get("skipped", []):
            name = e.get("name")
            if name:
                skipped[name] = (e.get("kind", "?"), e.get("detail"))
    return local_sent, skipped


# ---------------------------------------------------------------------------
# Diagnostic d'une session en attente — pourquoi n'est-elle pas partie ?
# ---------------------------------------------------------------------------

def _pending_reason(session_dir: str) -> str:
    issues = mistral_uploader.validate_session(Path(session_dir))
    if issues:
        return "structure incomplète : " + "; ".join(issues[:2])
    errors = mistral_uploader.read_analysis_errors(Path(session_dir))
    if errors:
        return "erreurs analysis.json : " + "; ".join(str(e) for e in errors[:2])
    return "structurellement valide — jamais traitée par l'uploader"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Un ou plusieurs dossiers racine à auditer (ex: /data /mnt/hdd2)")
    parser.add_argument("--client-id", default=mistral_uploader.MISTRAL_CLIENT_ID,
                        help=f"Client de livraison en BDD (défaut : {mistral_uploader.MISTRAL_CLIENT_ID})")
    parser.add_argument("--show-pending", type=int, default=15, metavar="N",
                        help="Sessions en attente détaillées par dossier (défaut : 15, 0 = aucun détail)")
    parser.add_argument("--json", type=Path, metavar="OUT",
                        help="Écrit aussi un rapport JSON complet (toutes les sessions) dans ce fichier")
    args = parser.parse_args()

    roots = [r.resolve() for r in args.roots]
    for root in roots:
        if not root.is_dir():
            print(f"ERREUR : '{root}' n'est pas un répertoire", file=sys.stderr)
            return 2

    print(f"Vérité BDD : sessions déjà livrées au client '{args.client_id}'...", flush=True)
    delivered = mistral_uploader._fetch_delivered_folders(args.client_id)
    if delivered is None:
        print("ERREUR : backend BDD injoignable — abandon "
              "(impossible de garantir un taux de couverture sans la BDD)", file=sys.stderr)
        return 1
    print(f"{len(delivered)} session(s) confirmée(s) livrées en BDD\n", flush=True)

    multi_root = len(roots) > 1

    # Totaux globaux (en octets), et détail complet pour --json
    g_sent = g_sent_unconfirmed = g_excluded = g_pending = 0
    g_n_sent = g_n_unconfirmed = g_n_excluded = g_n_pending = 0
    all_pending_detail: list[tuple[str, str, int]] = []  # (bucket_label, path, size)
    unconfirmed_examples: list[str] = []
    json_report: list[dict] = []

    for root in roots:
        root_prefix = f"{root.name}/" if multi_root else ""
        print(f"### Racine : {root} ###", flush=True)

        # "Gros dossiers" = enfants directs de root ; une session trouvée
        # juste sous root tombe dans le seau virtuel "(racine)".
        try:
            children = sorted(root.iterdir())
        except OSError as exc:
            print(f"  ERREUR de lecture : {exc}", file=sys.stderr)
            continue

        buckets: dict[str, list[Path]] = defaultdict(list)
        for c in children:
            if not c.is_dir():
                continue
            if SESSION_NAME_RE.match(c.name):
                buckets["(racine)"].append(c)
            else:
                buckets[c.name].append(c)

        if not buckets:
            print("  (vide)\n", flush=True)
            continue

        for bucket_name in sorted(buckets):
            t0 = time.monotonic()
            dodge_paths: list[str] = []
            found: list[tuple[str, str, int]] = []
            for path in buckets[bucket_name]:
                found.extend(scan_bucket(path, dodge_paths))

            local_sent, skipped = load_dodge_records(dodge_paths)

            b_sent = b_sent_unconfirmed = b_excluded = b_pending = 0
            b_n_sent = b_n_unconfirmed = b_n_excluded = b_n_pending = 0
            pending_here: list[tuple[str, int]] = []
            looks_like_sent_folder = any(k in bucket_name.lower() for k in ("envoy", "sent"))

            for path, name, size in found:
                if name in delivered:
                    status = "sent"
                    b_sent += size
                    b_n_sent += 1
                elif name in local_sent:
                    status = "sent_unconfirmed"
                    b_sent_unconfirmed += size
                    b_n_unconfirmed += 1
                    unconfirmed_examples.append(name)
                elif name in skipped:
                    status = "excluded"
                    b_excluded += size
                    b_n_excluded += 1
                else:
                    status = "pending"
                    b_pending += size
                    b_n_pending += 1
                    pending_here.append((path, size))

                if args.json:
                    json_report.append({
                        "root": str(root), "bucket": bucket_name, "path": path,
                        "name": name, "size_bytes": size, "status": status,
                    })

            label = f"{root_prefix}{bucket_name}"
            total = b_sent + b_sent_unconfirmed + b_excluded + b_pending
            sent_pct = ((b_sent + b_sent_unconfirmed) / total * 100) if total else 100.0
            elapsed = time.monotonic() - t0

            print(f"  [{label}] {len(found)} session(s) — {mistral_uploader.format_size(total)}", flush=True)
            print(f"    Envoyées à Mistral : {b_n_sent + b_n_unconfirmed} "
                  f"({mistral_uploader.format_size(b_sent + b_sent_unconfirmed)}) — {sent_pct:.1f}%", flush=True)
            if b_n_unconfirmed:
                print(f"      dont {b_n_unconfirmed} envoyée(s) localement mais PAS confirmée(s) en BDD "
                      f"({mistral_uploader.format_size(b_sent_unconfirmed)}) "
                      f"→ à rattraper avec backfill_db_sent.py", flush=True)
            if b_n_excluded:
                print(f"    Exclues (jamais envoyables) : {b_n_excluded} "
                      f"({mistral_uploader.format_size(b_excluded)})", flush=True)
            if b_n_pending:
                print(f"    EN ATTENTE : {b_n_pending} "
                      f"({mistral_uploader.format_size(b_pending)}) ← bloque les 100%", flush=True)
                if looks_like_sent_folder:
                    print(f"      ⚠ dossier '{bucket_name}' ressemble à un dossier de sessions déjà "
                          f"envoyées, mais ces sessions n'ont aucune trace (ni BDD, ni dodge) — à vérifier", flush=True)
                for path, size in sorted(pending_here, key=lambda t: -t[1])[: args.show_pending]:
                    reason = _pending_reason(path)
                    print(f"      - {Path(path).name} ({mistral_uploader.format_size(size)}) : {reason}", flush=True)
                if len(pending_here) > args.show_pending:
                    print(f"      … et {len(pending_here) - args.show_pending} autre(s)", flush=True)
                all_pending_detail.extend((label, p, s) for p, s in pending_here)
            print(f"    (scanné en {elapsed:.1f}s)\n", flush=True)

            g_sent += b_sent; g_sent_unconfirmed += b_sent_unconfirmed
            g_excluded += b_excluded; g_pending += b_pending
            g_n_sent += b_n_sent; g_n_unconfirmed += b_n_unconfirmed
            g_n_excluded += b_n_excluded; g_n_pending += b_n_pending

    if args.json:
        args.json.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Rapport JSON complet écrit dans {args.json}\n", flush=True)

    g_total = g_sent + g_sent_unconfirmed + g_excluded + g_pending
    g_pct = ((g_sent + g_sent_unconfirmed) / g_total * 100) if g_total else 100.0

    print("=== Résumé global ===")
    print(f"  Total scanné       : {g_n_sent + g_n_unconfirmed + g_n_excluded + g_n_pending} session(s), "
          f"{mistral_uploader.format_size(g_total)}")
    print(f"  Envoyées à Mistral : {g_n_sent + g_n_unconfirmed} session(s), "
          f"{mistral_uploader.format_size(g_sent + g_sent_unconfirmed)} ({g_pct:.1f}%)")
    if g_n_unconfirmed:
        print(f"    dont {g_n_unconfirmed} non confirmée(s) en BDD "
              f"({mistral_uploader.format_size(g_sent_unconfirmed)}) → backfill_db_sent.py")
    if g_n_excluded:
        print(f"  Exclues (jamais envoyables) : {g_n_excluded} session(s), "
              f"{mistral_uploader.format_size(g_excluded)}")
    if g_n_pending:
        print(f"  EN ATTENTE : {g_n_pending} session(s), {mistral_uploader.format_size(g_pending)}")

    if g_n_pending == 0:
        print("\n100% du disque est envoyé à Mistral (ou exclu à raison).")
        return 0
    print(f"\nPAS ENCORE 100% — {g_n_pending} session(s) / "
          f"{mistral_uploader.format_size(g_pending)} restent à traiter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
