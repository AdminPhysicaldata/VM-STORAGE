#!/usr/bin/env python3
"""
Rattrapage BDD des sessions envoyées à Mistral en mode --mistral-offline.

Un run `run_pipeline --send-mistral --mistral-offline` uploade les zips et
déplace les sessions vers sessions_envoyes/, mais n'enregistre RIEN en base
(ni register-bulk, ni mark-sent-bulk) : la BDD ignore que ces sessions sont
livrées — l'anti-doublon cross-disque et les stats de livraison sont faux.

Ce script rejoue l'enregistrement BDD du chemin d'envoi normal
(run_pipeline.send_session_to_mistral) pour chaque session d'un dossier
sessions_envoyes, sans rien toucher sur disque :

  1. GET delivered-folders → les sessions déjà connues de la BDD sont
     ignorées (idempotent : relançable sans risque de doublon) ;
  2. par lots : POST register-bulk (analysis/config/mission + taille)
     puis POST mark-sent-bulk (session_id BDD ou nom de dossier, taille,
     durée lue dans analysis.json).

Différence assumée avec l'envoi normal : size_bytes est la taille du dossier
(somme des fichiers), pas celle du zip disparu depuis longtemps — écart
négligeable, les vidéos ne se compressent pas.

Sans --apply : dry-run, affiche seulement ce qui serait enregistré.
Si le backend est injoignable, le script s'arrête AVANT toute écriture
(pas de fail-open : impossible de re-marquer en masse par accident).

Usage :
  python3 backfill_db_sent.py /data/sessions_envoyes            # aperçu
  python3 backfill_db_sent.py /data/sessions_envoyes --apply    # écrit en BDD
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sessions-uploader"))
import SessionsToMistral as mistral_uploader  # noqa: E402


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("sent_dir", type=Path,
                        help="Dossier des sessions envoyées (ex. /data/sessions_envoyes)")
    parser.add_argument("--apply", action="store_true",
                        help="Écrit réellement en BDD (défaut : dry-run, aucune écriture)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Sessions par POST bulk (défaut : 100)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ne traite que les N premières sessions manquantes (pour tester)")
    args = parser.parse_args()

    root = args.sent_dir.resolve()
    if not root.is_dir():
        print(f"ERREUR : {root} n'est pas un répertoire", file=sys.stderr)
        return 2

    sessions = sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name.startswith("session_"))
    ignored = sum(1 for d in root.iterdir() if d.is_dir() and not d.name.startswith("session_"))
    print(f"{len(sessions)} session(s) dans {root}"
          + (f" ({ignored} dossier(s) non-session ignoré(s))" if ignored else ""))

    # Pas de fail-open ici : si la BDD est injoignable on refuse de continuer,
    # sinon un backend en panne ferait re-marquer TOUTES les sessions.
    client_id = mistral_uploader.MISTRAL_CLIENT_ID
    delivered = mistral_uploader._fetch_delivered_folders(client_id)
    if delivered is None:
        print("ERREUR : backend BDD injoignable — abandon avant toute écriture", file=sys.stderr)
        return 1
    print(f"{len(delivered)} session(s) déjà marquées livrées en BDD (client: {client_id})")

    todo = [d for d in sessions if d.name not in delivered]
    print(f"→ {len(todo)} session(s) à enregistrer en BDD")
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"  (--limit : on n'en traite que {len(todo)})")

    if not todo:
        print("Rien à faire — la BDD est déjà à jour.")
        return 0

    if not args.apply:
        for d in todo[:10]:
            print(f"    {d.name}")
        if len(todo) > 10:
            print(f"    … et {len(todo) - 10} autres")
        print("\nDry-run : aucune écriture faite. Relancez avec --apply pour enregistrer.")
        return 0

    t0 = time.monotonic()
    n_registered = n_marked = n_no_analysis = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]

        register_items, meta_by_name = [], {}
        for d in batch:
            # dry_run=True : ne réécrit pas les config.json sur disque — la
            # correction rig a déjà eu lieu (ou pas) au moment de l'envoi réel.
            analysis, config, mission = mistral_uploader.read_session_metadata(d, dry_run=True)
            duration = mistral_uploader.read_duration(d)
            size = _dir_size(d)
            meta_by_name[d.name] = (size, duration)
            if analysis is not None:
                register_items.append({
                    "folder_name": d.name, "analysis": analysis,
                    "config": config, "mission": mission, "size_bytes": size,
                })
            else:
                n_no_analysis += 1

        session_ids = mistral_uploader.db_register_sessions_bulk(register_items)
        n_registered += sum(1 for v in session_ids.values() if v)

        deliveries = []
        for d in batch:
            size, duration = meta_by_name[d.name]
            ref = session_ids.get(d.name) or d.name
            deliveries.append({"session_ref": ref, "size_bytes": size,
                               "duration_seconds": duration})
        mistral_uploader.api_mark_sent_bulk(deliveries)
        n_marked += len(deliveries)

        done = min(i + args.batch_size, len(todo))
        rate = done / max(time.monotonic() - t0, 1e-6)
        print(f"  {done}/{len(todo)}  ({rate:.1f} sessions/s)")

    print(f"\nTerminé en {time.monotonic() - t0:.1f} s : "
          f"{n_marked} marquées livrées, {n_registered} enregistrées via register-bulk, "
          f"{n_no_analysis} sans analysis.json (marquées par nom de dossier). "
          f"Détail des éventuels refus du backend : voir les warnings ci-dessus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
