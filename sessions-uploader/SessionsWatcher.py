"""
SessionsWatcher.py — Surveille les sessions SFTP et met à jour la BDD sur la notation.

Détecte l'apparition de analysis.json dans chaque dossier session_*, calcule un
score de qualité de capture, et enregistre quality_score + quality_grade en BDD.

Score calculé depuis analysis.json uniquement (léger, instantané) :
  sync_inter_cameras (40%) — dérive relative inter-caméras (drift_check.pairs)
  fps_stability      (35%) — sequence gaps V4L2 kernel (fps_check)
  cam_sensor_sync    (25%) — delta temporel caméra/capteur (sync_check)

Ce score initial est écrasé par le treatment-worker dès qu'il effectue
les vérifications complètes (tracking gripper ArUco, dérive JSONL, etc.).

Résolution session_id depuis le nom de dossier (3 stratégies) :
  1. session_folder = folder_name (colonne déjà renseignée)
  2. session_id     = folder_name (sessions importées via reimport_sessions.py)
  3. Timestamp      : extrait YYYYMMDD_HHMMSS du nom, cherche sess_YYYYMMDD_HHMMSS*
     → fonctionne pour les dossiers anciens format YYYY-MM-DD_HH-MM-SS_NNNNNN
       ET les dossiers nouveaux format session_YYYYMMDD_HHMMSS
     → met à jour session_folder pour que le backend trouve le dossier NAS
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SessionsWatcher: %(message)s",
)
logger = logging.getLogger(__name__)

SESSIONS_DIR  = os.environ.get("SESSIONS_DIR",          "/data/sessions")
POLL_INTERVAL = int(os.environ.get("WATCHER_POLL_INTERVAL", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# BDD
# ─────────────────────────────────────────────────────────────────────────────

def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST",     "192.168.1.18"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB",     "robotics"),
        user=os.environ.get("POSTGRES_USER",     "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


def _load_already_scored() -> set[str]:
    """
    Charge depuis la BDD les noms de dossiers déjà notés (metadata contient
    'capture_quality') pour éviter de les retraiter au démarrage.
    """
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(session_folder, session_id)
                    FROM sessions
                    WHERE metadata ? 'capture_quality'
                """)
                result = {row[0] for row in cur.fetchall() if row[0]}
        conn.close()
        return result
    except Exception as exc:
        logger.warning("Impossible de charger les sessions déjà notées : %s", exc)
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Score depuis analysis.json
# ─────────────────────────────────────────────────────────────────────────────

def _score_from_analysis(analysis: dict) -> tuple[float, str]:
    """
    Score 0-100 et grade depuis analysis.json.

    Retourne (0.0, 'F') immédiatement si errors[] est non vide.
    Critères et pondérations :
      sync_inter_cameras (40%) : drift relatif inter-caméras (drift_check.pairs)
      fps_stability      (35%) : sequence gaps kernel V4L2  (fps_check.cameras)
      cam_sensor_sync    (25%) : delta caméra/capteur        (sync_check)
    """
    if analysis.get("errors"):
        return 0.0, "F"

    scores = {}

    # ── Sync inter-caméras ────────────────────────────────────────────────────
    pairs = analysis.get("drift_check", {}).get("pairs", {})
    if pairs:
        max_rel = max(abs(v.get("relative_drift_ms_per_min", 0)) for v in pairs.values())
        scores["sync"] = max(0.0, 100.0 - max_rel * 5.0)
    else:
        scores["sync"] = 50.0

    # ── Stabilité FPS (gaps kernel) ───────────────────────────────────────────
    cams = analysis.get("fps_check", {}).get("cameras", {})
    if cams:
        total_est   = sum(
            (c.get("measured_fps", 0) or c.get("expected_fps", 0)) * c.get("duration_sec", 0)
            for c in cams.values()
        )
        total_gaps  = sum(c.get("sequence_gaps", 0) for c in cams.values())
        total_qdrop = sum(c.get("queue_drops",   0) for c in cams.values())
        gap_pct     = total_gaps / total_est * 100.0 if total_est > 0 else 0.0
        scores["fps"] = max(0.0, 100.0 - gap_pct * 2.0 - total_qdrop * 5.0)
    else:
        scores["fps"] = 50.0

    # ── Sync caméra/capteur ───────────────────────────────────────────────────
    sync  = analysis.get("sync_check", {})
    delta = sync.get("delta_sec", 0.0)
    if sync.get("ok", True):
        scores["cam_sensor"] = max(0.0, 100.0 - delta * 300.0)
    else:
        scores["cam_sensor"] = max(0.0, 30.0 - delta * 100.0)

    total = round(
        scores["sync"]       * 0.40
        + scores["fps"]      * 0.35
        + scores["cam_sensor"] * 0.25,
        1,
    )
    grade = ("A" if total >= 85 else
             "B" if total >= 70 else
             "C" if total >= 55 else
             "D" if total >= 40 else "F")
    return total, grade


# ─────────────────────────────────────────────────────────────────────────────
# Résolution session_id ← nom de dossier SFTP
# ─────────────────────────────────────────────────────────────────────────────

def _folder_to_timestamp(folder_name: str) -> str | None:
    """
    Extrait YYYYMMDD_HHMMSS depuis un nom de dossier, quelle que soit la forme :
      session_20260604_214054          → 20260604_214054
      2026-06-04_21-40-54_000000       → 20260604_214054
    Retourne None si le format n'est pas reconnu.
    """
    # Nouveau format : session_YYYYMMDD_HHMMSS[_...]
    m = re.match(r"^session_(\d{8})_(\d{6})", folder_name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Ancien format robot : YYYY-MM-DD_HH-MM-SS_NNNNNN
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", folder_name)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return None


def _resolve_session_id(cur, folder_name: str) -> str | None:
    """
    Cherche le session_id dans la BDD via 3 stratégies :
      1. session_folder = folder_name  (déjà renseigné)
      2. session_id     = folder_name  (sessions importées manuellement)
      3. session_id LIKE 'sess_YYYYMMDD_HHMMSS%'  (sessions Kafka, session_folder null)
    """
    # 1 & 2 — correspondance directe
    for query, params in [
        ("SELECT session_id FROM sessions WHERE session_folder = %s LIMIT 1", (folder_name,)),
        ("SELECT session_id FROM sessions WHERE session_id     = %s LIMIT 1", (folder_name,)),
    ]:
        cur.execute(query, params)
        row = cur.fetchone()
        if row:
            return row[0]

    # 3 — correspondance par timestamp (sessions enregistrées via Kafka sans session_folder)
    ts = _folder_to_timestamp(folder_name)
    if ts:
        cur.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE %s LIMIT 1",
            (f"sess_{ts}%",),
        )
        row = cur.fetchone()
        if row:
            return row[0]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Mise à jour BDD
# ─────────────────────────────────────────────────────────────────────────────

def _update_db(folder_name: str, analysis: dict) -> bool:
    """
    Résout le session_id depuis folder_name, puis :
    - Écrit session_folder si null → le backend retrouvera le dossier NAS.
    - Met quality_score et quality_grade si encore NULL.
    - Stocke capture_quality dans metadata (additive).

    Retourne True si la BDD a été mise à jour, False si session introuvable.
    """
    score, grade = _score_from_analysis(analysis)
    errors       = [str(e) for e in analysis.get("errors",   [])]
    warnings     = [str(w) for w in analysis.get("warnings", [])]

    capture_meta = json.dumps({
        "score":          score,
        "grade":          grade,
        "errors":         errors,
        "warnings_count": len(warnings),
        "scored_at":      datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False)

    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                session_id = _resolve_session_id(cur, folder_name)

                if not session_id:
                    logger.debug("'%s' pas encore en BDD — sera retenté", folder_name)
                    return False

                cur.execute("""
                    UPDATE sessions
                    SET quality_score  = COALESCE(quality_score, %s),
                        quality_grade  = COALESCE(quality_grade, %s),
                        session_folder = COALESCE(session_folder, %s),
                        metadata       = metadata
                                         || jsonb_build_object('capture_quality', %s::jsonb)
                    WHERE session_id = %s
                """, (score, grade, folder_name, capture_meta, session_id))

                updated = cur.rowcount > 0

        conn.close()

        if updated:
            flag = "ERREURS" if errors else f"{len(warnings)} warning(s)" if warnings else "OK"
            logger.info(
                "%s (%s) → score=%.1f grade=%s [%s]",
                folder_name, session_id, score, grade, flag,
            )
        return updated

    except Exception as exc:
        logger.error("BDD erreur pour '%s' : %s", folder_name, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

def watch_loop():
    logger.info("Démarré — répertoire: %s  intervalle: %ds", SESSIONS_DIR, POLL_INTERVAL)

    # Charge les sessions déjà notées pour éviter de les retraiter au restart
    processed: set[str] = _load_already_scored()
    logger.info("%d session(s) déjà notée(s) chargées depuis la BDD", len(processed))

    # Sessions avec analysis.json présent mais pas encore trouvées en BDD
    # (session enregistrée quelques secondes après le début de l'upload)
    pending_retry: set[str] = set()

    while True:
        if not os.path.isdir(SESSIONS_DIR):
            time.sleep(POLL_INTERVAL)
            continue

        # Réessayer les sessions en attente d'enregistrement BDD
        still_pending: set[str] = set()
        for folder_name in list(pending_retry):
            session_dir   = os.path.join(SESSIONS_DIR, folder_name)
            analysis_path = os.path.join(session_dir, "analysis.json")
            if not os.path.isfile(analysis_path):
                continue
            try:
                with open(analysis_path, encoding="utf-8") as f:
                    analysis = json.load(f)
            except Exception:
                processed.add(folder_name)     # fichier illisible → abandon
                continue
            if _update_db(folder_name, analysis):
                processed.add(folder_name)
            else:
                still_pending.add(folder_name)
        pending_retry = still_pending

        # Scan du répertoire pour les nouvelles sessions
        try:
            entries = sorted(os.listdir(SESSIONS_DIR))
        except OSError:
            time.sleep(POLL_INTERVAL)
            continue

        for entry in entries:
            if not entry.lower().startswith("session"):
                continue
            if entry in processed or entry in pending_retry:
                continue

            session_dir   = os.path.join(SESSIONS_DIR, entry)
            analysis_path = os.path.join(session_dir, "analysis.json")

            if not os.path.isfile(analysis_path):
                continue

            try:
                with open(analysis_path, encoding="utf-8") as f:
                    analysis = json.load(f)
            except Exception as exc:
                logger.warning("Impossible de lire %s : %s", analysis_path, exc)
                processed.add(entry)
                continue

            if _update_db(entry, analysis):
                processed.add(entry)
            else:
                # Session pas encore en BDD — réessayer lors du prochain cycle
                pending_retry.add(entry)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    watch_loop()
