#!/usr/bin/env python3
"""
send_validated_batches.py — Envoie à Mistral les lots journaliers VALIDÉS.

Tourne périodiquement (cron). À chaque passage :
  1. Récupère les lots validés non envoyés (GET /pipeline/batches/approved-unsent).
  2. Pour chacun : écrit un manifeste des dossiers de session, marque le lot
     'sending', puis lance SessionsToMistral.py --only <manifeste> (toute la
     mécanique d'analyse/zip/upload/suivi de run est réutilisée telle quelle).
  3. Marque le lot 'sent' si l'envoi réussit (code retour 0), sinon 'send_failed'.

Le filtre "jour courant" de SessionsToMistral est désactivé par --only : le lot
a déjà été curé (rig + jour + intégrité) par build_daily_batch.py et validé
manuellement, on l'envoie tel quel.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/data/sessions")
BACKEND_URL  = os.environ.get("BACKEND_URL", "http://localhost:5000/api").rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")
UPLOADER = os.environ.get("UPLOADER_SCRIPT", str(Path(__file__).with_name("SessionsToMistral.py")))


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if INTERNAL_API_TOKEN:
        h["X-Internal-Token"] = INTERNAL_API_TOKEN
    return h


def fetch_approved_unsent() -> list[dict]:
    r = requests.get(f"{BACKEND_URL}/pipeline/batches/approved-unsent",
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("batches", [])


def _post(path: str, body: dict) -> None:
    try:
        requests.post(f"{BACKEND_URL}{path}", json=body, headers=_headers(), timeout=30)
    except requests.RequestException as exc:
        logger.warning("Backend %s en échec : %s", path, exc)


def _start_run(total: int) -> str | None:
    """Démarre un upload_run côté backend et retourne son run_id (suivi % + page Clients)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/uploads/runs/start",
            json={"client_id": os.environ.get("DELIVERY_CLIENT_ID", "mistral"),
                  "total_sessions": total},
            headers=_headers(), timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("run_id")
    except requests.RequestException as exc:
        logger.warning("Démarrage du run échoué : %s", exc)
    return None


def send_batch(batch: dict) -> bool:
    batch_id = batch["batch_id"]
    sessions = batch.get("sessions") or []
    if not sessions:
        logger.warning("Lot %s sans session — marqué échoué", batch_id)
        _post(f"/pipeline/batches/{batch_id}/mark-failed", {"error_msg": "aucune session dans le lot"})
        return False

    logger.info("Envoi du lot %s — %d session(s)", batch_id, len(sessions))
    # Crée le run AVANT l'envoi pour relier lot↔run (suivi % sur /batch + page Clients)
    run_id = _start_run(len(sessions))
    _post(f"/pipeline/batches/{batch_id}/mark-sending", {"run_id": run_id} if run_id else {})

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for s in sessions:
            f.write(s["folder_name"] + "\n")
        manifest = f.name

    try:
        cmd = [sys.executable, UPLOADER, SESSIONS_DIR, "--only", manifest, "--all"]
        logger.info("  $ %s", " ".join(cmd))
        env = dict(os.environ)
        if run_id:
            env["UPLOAD_RUN_ID"] = run_id  # SessionsToMistral reporte sa progression sur CE run
        result = subprocess.run(cmd, env=env)
        ok = result.returncode == 0
    finally:
        Path(manifest).unlink(missing_ok=True)

    if ok:
        _post(f"/pipeline/batches/{batch_id}/mark-sent", {})
        logger.info("Lot %s envoyé ✅", batch_id)
    else:
        _post(f"/pipeline/batches/{batch_id}/mark-failed",
              {"error_msg": f"SessionsToMistral code retour {result.returncode}"})
        logger.error("Lot %s en échec (code %s)", batch_id, result.returncode)
    return ok


def main() -> None:
    try:
        batches = fetch_approved_unsent()
    except Exception as exc:
        logger.error("Impossible de récupérer les lots validés : %s", exc)
        sys.exit(1)

    if not batches:
        logger.info("Aucun lot validé en attente d'envoi.")
        return

    logger.info("%d lot(s) validé(s) à envoyer.", len(batches))
    failed = 0
    for batch in batches:
        if not send_batch(batch):
            failed += 1

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
