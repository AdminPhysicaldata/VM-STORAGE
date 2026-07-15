#!/usr/bin/env python3
"""
build_daily_batch.py — Constitue le lot journalier d'un rig (par défaut le
poste 6) à soumettre à validation humaine avant envoi à Mistral.

Chaque matin :
  1. Parcourt SESSIONS_DIR, garde les sessions du rig demandé ET du jour ciblé.
  2. Valide l'intégrité (réutilise validate_session de SessionsToMistral) —
     les sessions invalides sont ignorées.
  3. Accumule des sessions jusqu'à ~MAX_GB Go (granularité session entière :
     la session qui ferait dépasser le plafond est laissée pour un autre jour).
  4. Dépose une COPIE du lot dans COPY_DIR/batch_<date>_<rig> (consultable en
     SFTP par l'utilisateur pour contrôler la qualité).
  5. Enregistre le lot côté backend (POST /api/pipeline/batches) au statut
     'pending_validation'. La notification 10h et la validation se font
     ensuite sur le site ; l'envoi réel est fait par send_validated_batches.py.

Identification du rig : robuste au format (rig_06 / RIG-6 / rig_6) — on
compare la partie numérique de config["rig"]["rig_id"] (ou ["code"]) au numéro
demandé.

Usage :
    python3 build_daily_batch.py
    python3 build_daily_batch.py --rig-num 6 --date yesterday --max-gb 5
    python3 build_daily_batch.py --date 2026-06-28 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# Réutilise la validation/analyse éprouvée de l'uploader (même dossier).
from SessionsToMistral import (
    validate_session, analyze_session, read_duration, read_analysis_errors,
    session_date, format_size, _expected_rig_num,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/data/sessions")
COPY_DIR     = os.environ.get("BATCH_COPY_DIR", "/data/session_envoye/batches")
BACKEND_URL  = os.environ.get("BACKEND_URL", "http://localhost:5000/api").rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")
DEFAULT_MAX_GB = float(os.environ.get("BATCH_MAX_GB", "5"))
DEFAULT_RIG_NUM = int(os.environ.get("BATCH_RIG_NUM", "6"))


def _rig_num_of(config: dict | None) -> int | None:
    """
    Numéro de rig d'une session. Lit config['rig']['code'] (puis ['rig_id']),
    robuste au format (rig_06 / RIG-6 / rig_6). Si le rig est absent ou vaut le
    placeholder du logiciel de capture (rig_1), on déduit le numéro réel depuis
    l'operator_id (règle ±30, identique à fix_rig_config de l'uploader).
    """
    if not config:
        return None
    rig = config.get("rig") or {}
    num = None
    for key in ("code", "rig_id"):
        m = re.search(r"(\d+)", str(rig.get(key) or ""))
        if m:
            num = int(m.group(1))
            break
    if num is None or num == 1:
        expected = _expected_rig_num((config.get("operator") or {}).get("operator_id"))
        if expected is not None:
            num = expected
    return num


def _read_config(session_dir: Path) -> dict | None:
    p = session_dir / "config.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_target_date(value: str) -> date:
    value = (value or "").strip().lower()
    if value in ("today", "auj", "aujourdhui", ""):
        return datetime.now().date()
    if value in ("yesterday", "hier"):
        return datetime.now().date() - timedelta(days=1)
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"--date invalide : '{value}' (today|yesterday|YYYY-MM-DD)")


def _backend_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if INTERNAL_API_TOKEN:
        h["X-Internal-Token"] = INTERNAL_API_TOKEN
    return h


# Heure (locale du site) encodée dans le nom du dossier : session_YYYYMMDD_HHMMSS
_SESSION_HOUR_RE = re.compile(r"^session_\d{8}_(\d{2})(\d{2})(\d{2})", re.IGNORECASE)


def _session_hour(name: str) -> int | None:
    """Heure de capture (0-23) extraite du nom du dossier. None si illisible."""
    m = _SESSION_HOUR_RE.match(name)
    if not m:
        return None
    h = int(m.group(1))
    return h if 0 <= h <= 23 else None


def select_sessions(sessions_dir: Path, rig_num: int, target: date,
                    max_bytes: int, do_validate: bool,
                    start_hour: int = 7, end_hour: int = 24) -> list[dict]:
    """
    Retourne la liste des sessions retenues (dict folder_name/size_bytes/duration_seconds).
    Ne garde que les sessions du rig demandé, du jour 'target', et dont l'heure de
    capture est dans [start_hour, end_hour[ (heure locale du site) — exclut donc
    les sessions de nuit. Une heure illisible est écartée (strict).
    """
    selected: list[dict] = []
    cumul = 0

    for entry in sorted(sessions_dir.iterdir()):
        if not entry.is_dir() or not entry.name.lower().startswith("session"):
            continue
        if session_date(entry.name) != target:
            continue
        hour = _session_hour(entry.name)
        if hour is None or hour < start_hour or hour >= end_hour:
            continue
        cfg = _read_config(entry)
        if _rig_num_of(cfg) != rig_num:
            continue

        if do_validate:
            issues = validate_session(entry)
            if issues:
                logger.info("  [%s] ignorée (invalide) : %s", entry.name, issues[0])
                continue
            if read_analysis_errors(entry):
                logger.info("  [%s] ignorée (erreurs analysis.json)", entry.name)
                continue

        is_empty, reason, size = analyze_session(entry)
        if is_empty:
            logger.info("  [%s] ignorée (vide : %s)", entry.name, reason)
            continue

        if cumul + size > max_bytes:
            logger.info("  [%s] plafond atteint (+%s) — laissée pour un autre lot",
                        entry.name, format_size(size))
            break

        cumul += size
        selected.append({
            "folder_name": entry.name,
            "size_bytes": size,
            "duration_seconds": read_duration(entry),
        })
        logger.info("  [%s] retenue (%s) — cumul %s",
                    entry.name, format_size(size), format_size(cumul))

    return selected


def copy_batch(sessions_dir: Path, selected: list[dict], copy_root: Path) -> Path:
    """Copie les sessions du lot dans copy_root et y écrit un manifeste."""
    copy_root.mkdir(parents=True, exist_ok=True)
    for s in selected:
        src = sessions_dir / s["folder_name"]
        dst = copy_root / s["folder_name"]
        if dst.exists():
            continue
        shutil.copytree(str(src), str(dst))
    manifest = copy_root / "manifest.txt"
    manifest.write_text("\n".join(s["folder_name"] for s in selected) + "\n", encoding="utf-8")
    return copy_root


def register_batch(batch_date: date, rig_id: str, copy_path: str | None,
                   selected: list[dict]) -> dict:
    payload = {
        "batch_date": batch_date.isoformat(),
        "rig_id": rig_id,
        "copy_path": copy_path,
        "sessions": selected,
    }
    r = requests.post(f"{BACKEND_URL}/pipeline/batches", json=payload,
                      headers=_backend_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le lot journalier d'un rig pour validation")
    parser.add_argument("--sessions-dir", default=SESSIONS_DIR)
    parser.add_argument("--rig-num", type=int, default=DEFAULT_RIG_NUM, help="Numéro du poste/rig (défaut 6)")
    parser.add_argument("--date", default=os.environ.get("BATCH_DATE", "today"),
                        help="today | yesterday | YYYY-MM-DD (défaut today — données du jour même)")
    parser.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB, help="Plafond du lot en Go (défaut 5)")
    parser.add_argument("--start-hour", type=int, default=int(os.environ.get("BATCH_START_HOUR", "7")),
                        help="N'inclut que les sessions à partir de cette heure locale (défaut 7 — exclut la nuit)")
    parser.add_argument("--end-hour", type=int, default=int(os.environ.get("BATCH_END_HOUR", "24")),
                        help="N'inclut que les sessions avant cette heure locale (défaut 24)")
    parser.add_argument("--copy-dir", default=COPY_DIR, help="Racine de dépôt de la copie SFTP")
    parser.add_argument("--no-copy", action="store_true", help="Ne pas déposer la copie SFTP")
    parser.add_argument("--no-validate", action="store_true", help="Ne pas valider l'intégrité (plus rapide)")
    parser.add_argument("--no-register", action="store_true", help="Ne pas enregistrer côté backend")
    parser.add_argument("--dry-run", action="store_true", help="N'écrit/copie/enregistre rien")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.is_dir():
        raise SystemExit(f"Dossier de sessions introuvable : {sessions_dir}")

    target = _parse_target_date(args.date)
    batch_date = target
    rig_id = f"rig_{args.rig_num:02d}"
    max_bytes = int(args.max_gb * 1024 ** 3)

    logger.info("Construction du lot %s pour %s (plafond %.1f Go, heures %dh–%dh) dans %s",
                target.isoformat(), rig_id, args.max_gb, args.start_hour, args.end_hour, sessions_dir)

    selected = select_sessions(sessions_dir, args.rig_num, target, max_bytes,
                               do_validate=not args.no_validate,
                               start_hour=args.start_hour, end_hour=args.end_hour)

    if not selected:
        logger.warning("Aucune session retenue pour %s le %s — aucun lot créé.",
                       rig_id, target.isoformat())
        sys.exit(0)

    total = sum(s["size_bytes"] for s in selected)
    logger.info("=== Lot %s : %d session(s), %s ===", rig_id, len(selected), format_size(total))

    if args.dry_run:
        for s in selected:
            logger.info("  [DRY-RUN] %s (%s)", s["folder_name"], format_size(s["size_bytes"]))
        logger.info("*** DRY-RUN — rien copié ni enregistré ***")
        sys.exit(0)

    copy_path = None
    if not args.no_copy:
        copy_root = Path(args.copy_dir) / f"batch_{batch_date.isoformat()}_{rig_id}"
        logger.info("Copie de la session vers %s ...", copy_root)
        copy_batch(sessions_dir, selected, copy_root)
        copy_path = str(copy_root)

    if not args.no_register:
        try:
            res = register_batch(batch_date, rig_id, copy_path, selected)
            logger.info("Lot enregistré : %s", res.get("batch_id"))
        except Exception as exc:
            logger.error("Échec d'enregistrement backend : %s", exc)
            sys.exit(1)

    logger.info("Terminé — lot en attente de validation sur la page /batch.")


if __name__ == "__main__":
    main()
