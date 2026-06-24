#!/usr/bin/env python3
"""
audit_quality.py — Audit en lecture seule de la qualité des sessions.

Lance tous les checks de checks.py sur un répertoire de sessions SANS
écrire treatment.json (contrairement à run_treatment.py). Produit un
rapport trié par score croissant pour repérer les sessions les plus
problématiques, et un CSV optionnel pour analyse externe.

Le traitement d'une session est CPU-bound (numpy, flux optique OpenCV) :
les sessions sont réparties sur plusieurs processus (multiprocessing,
contourne le GIL) pour paralléliser sur tous les cœurs disponibles.

Usage :
  python audit_quality.py --dir /data/sessions
  python audit_quality.py --dir /data/sessions --pattern "session_2026*"
  python audit_quality.py --dir /data/sessions --csv audit.csv
  python audit_quality.py --dir /data/sessions --workers 16
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NAS_SESSIONS_DIR", "")

import checks  # noqa: E402


def _read_json_safe(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_duration(session_path: str) -> Optional[float]:
    analysis = _read_json_safe(os.path.join(session_path, "analysis.json"))
    if analysis:
        cams = analysis.get("fps_check", {}).get("cameras", {})
        durs = [c["duration_sec"] for c in cams.values() if c.get("duration_sec")]
        if durs:
            durs.sort()
            return durs[len(durs) // 2]
    for fname in ("metadata.json", "config.json"):
        meta = _read_json_safe(os.path.join(session_path, fname))
        if isinstance(meta.get("duration_seconds"), (int, float)):
            return float(meta["duration_seconds"])
    return None


def _get_scenario(session_path: str) -> Optional[str]:
    mission = _read_json_safe(os.path.join(session_path, "mission.json"))
    if mission.get("scenario_id"):
        return str(mission["scenario_id"])
    if mission.get("name"):
        return str(mission["name"])
    meta = _read_json_safe(os.path.join(session_path, "metadata.json"))
    sid = meta.get("scenario_id") or meta.get("scenario")
    return str(sid) if sid else None


def build_session_dict(session_path: str) -> dict:
    session_id = os.path.basename(session_path.rstrip("/"))
    return {
        "session_id":       session_id,
        "session_folder":   session_path,
        "duration_seconds": _get_duration(session_path),
        "scenario_id":      _get_scenario(session_path),
    }


def find_sessions(search_dir: str, pattern: str = None) -> list[str]:
    if not os.path.isdir(search_dir):
        return []
    out = []
    for entry in sorted(os.listdir(search_dir)):
        full_path = os.path.join(search_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if pattern and not fnmatch.fnmatch(entry, pattern):
            continue
        out.append(full_path)
    return out


def _init_worker():
    """Limite chaque worker à 1 thread OpenCV interne pour éviter la
    sur-souscription CPU (N processus x M threads OpenCV chacun)."""
    if checks.HAS_CV2:
        checks.cv2.setNumThreads(1)


def audit_session(session_path: str) -> dict:
    """Lance les checks et retourne une ligne de rapport. N'écrit jamais sur disque."""
    session_path = str(Path(session_path).resolve())
    session = build_session_dict(session_path)
    sid = session["session_id"]

    t0 = time.monotonic()
    try:
        result = checks.run_checks(session, session_path_override=session_path)
    except Exception as e:
        logger.exception("Erreur sur %s", sid)
        return {
            "session_id": sid, "path": session_path, "score": 0.0, "grade": "F",
            "passed": False, "duration_s": round(time.monotonic() - t0, 2),
            "failed_checks": "EXCEPTION", "error_detail": str(e),
        }

    elapsed = time.monotonic() - t0
    quality = result.get("quality", {})
    failed = [k for k, v in result["checks"].items() if not v.get("ok", True)]
    details = "; ".join(
        f"{k}: {result['checks'][k].get('detail', '')}" for k in failed
    )

    return {
        "session_id":    sid,
        "path":          session_path,
        "score":         quality.get("score", 0.0),
        "grade":         quality.get("grade", "F"),
        "passed":        result["passed"],
        "duration_s":    round(elapsed, 2),
        "failed_checks": ", ".join(failed),
        "error_detail":  details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audite la qualité des sessions sans écrire de fichier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir", "-d", required=True, help="Répertoire contenant les sessions")
    parser.add_argument("--pattern", "-p", default=None, help="Filtrer par nom (ex: session_2026*)")
    parser.add_argument("--session", "-s", default=None, help="Auditer une seule session")
    parser.add_argument("--csv", default=None, help="Chemin du CSV de sortie (optionnel)")
    parser.add_argument("--top", type=int, default=20, help="Nombre de pires sessions à afficher (def: 20)")
    parser.add_argument(
        "--workers", "-w", type=int, default=0,
        help="Nombre de processus parallèles (def: tous les cœurs CPU). 1 = séquentiel",
    )
    args = parser.parse_args()

    if args.session:
        session_dirs = [args.session]
    else:
        session_dirs = find_sessions(args.dir, args.pattern)

    if not session_dirs:
        logger.warning("Aucune session trouvée dans %s (pattern=%s)", args.dir, args.pattern)
        sys.exit(0)

    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    n_workers = max(1, min(n_workers, len(session_dirs)))
    logger.info("%d session(s) à auditer avec %d worker(s)", len(session_dirs), n_workers)

    rows = []
    t_start = time.monotonic()
    done = 0

    if n_workers == 1:
        for sd in session_dirs:
            done += 1
            logger.info("[%d/%d] %s", done, len(session_dirs), os.path.basename(sd))
            rows.append(audit_session(sd))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
            futures = {pool.submit(audit_session, sd): sd for sd in session_dirs}
            for future in as_completed(futures):
                done += 1
                row = future.result()
                logger.info("[%d/%d] %s → score=%.1f", done, len(session_dirs),
                            row["session_id"], row["score"])
                rows.append(row)

    elapsed = time.monotonic() - t_start
    rows.sort(key=lambda r: r["score"])

    n_passed = sum(1 for r in rows if r["passed"])
    n_failed = len(rows) - n_passed
    avg_score = sum(r["score"] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 100)
    print(f"AUDIT QUALITÉ — {len(rows)} session(s) en {elapsed:.1f}s")
    print(f"  Passées : {n_passed}  |  Échouées : {n_failed}  |  Score moyen : {avg_score:.1f}")
    print("=" * 100)

    print(f"\n{'PIRES SESSIONS (top ' + str(args.top) + ')':^100}")
    print(f"{'session_id':<35} {'score':>6} {'grade':>5} {'passed':>7}  failed_checks")
    print("-" * 100)
    for r in rows[: args.top]:
        print(f"{r['session_id']:<35} {r['score']:>6.1f} {r['grade']:>5} {str(r['passed']):>7}  {r['failed_checks']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("CSV écrit : %s", args.csv)

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
