"""
StorageHeartbeat.py — Heartbeat Kafka périodique pour vm-storage (SFTP).

Publie sur le topic Kafka "monitoring" un message source="server_heartbeat"
contenant :
  - taux de remplissage du disque /data/sessions
  - nombre de sessions présentes sur le disque
  - nombre de sessions enregistrées en BDD + nombre d'heures de capture stockées

Consommé par kafka_consumer.py (vm-backend), exposé dans get_snapshot()["servers"].
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import psycopg2
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] StorageHeartbeat: %(message)s",
)
logger = logging.getLogger(__name__)

SESSIONS_DIR       = os.environ.get("SESSIONS_DIR", "/data/sessions")
SERVER_ID          = os.environ.get("SERVER_ID", "vm-storage")
SERVER_ROLE        = os.environ.get("SERVER_ROLE", "storage")
KAFKA_BROKER       = os.environ.get("KAFKA_BROKER", f"{os.environ.get('VM_DATA_IP', '192.168.1.18')}:9092")
KAFKA_TOPIC        = os.environ.get("KAFKA_TOPIC", "monitoring")
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))


def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST",     "192.168.1.18"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB",     "robotics"),
        user=os.environ.get("POSTGRES_USER",     "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


def _disk_usage(path: str) -> dict:
    st    = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    free  = st.f_bavail * st.f_frsize
    used  = total - free
    return {
        "disk_total_gb": round(total / 1e9, 2),
        "disk_free_gb":  round(free / 1e9, 2),
        "disk_used_gb":  round(used / 1e9, 2),
        "disk_used_pct": round(used / total * 100.0, 1) if total else 0.0,
    }


def _sessions_on_disk(path: str) -> int:
    """Compte les dossiers session_* sur /data/sessions (os.scandir, rapide)."""
    try:
        with os.scandir(path) as it:
            return sum(
                1 for e in it
                if e.is_dir(follow_symlinks=False) and e.name.lower().startswith("session")
            )
    except OSError as exc:
        logger.warning("Impossible de scanner %s : %s", path, exc)
        return 0


def _captured_hours() -> tuple:
    """Retourne (nb sessions en BDD, total heures capturées) pour ce serveur."""
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*), COALESCE(SUM(duration_seconds), 0)
                    FROM sessions
                    WHERE session_folder IS NOT NULL
                """)
                count, total_sec = cur.fetchone()
        conn.close()
        return int(count), round(float(total_sec) / 3600.0, 2)
    except Exception as exc:
        logger.warning("Impossible de lire les stats BDD : %s", exc)
        return 0, 0.0


def _quality_stats() -> dict:
    """
    Statistiques de qualité des sessions notées par fs_scanner :
      - distribution des notes A/B/C/D/F (count + %)
      - % de sessions exploitables (grade A/B/C/D) vs inutilisables (F)
      - heures totales capturées / heures "propres" (sessions sans erreur)
    """
    empty = {
        "quality_scored_count":  0,
        "quality_total_count":   0,
        "quality_avg_score":     None,
        "quality_pct_usable":    None,
        "quality_pct_unusable":  None,
        "quality_grade_pct":     {"A": None, "B": None, "C": None, "D": None, "F": None},
        "quality_grade_count":   {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
        "total_hours_clean":     0.0,
    }
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                                                   AS total,
                        COUNT(quality_score)                                       AS scored,
                        ROUND(AVG(quality_score)::numeric, 1)                      AS avg_score,
                        COUNT(*) FILTER (WHERE quality_grade = 'A')                AS a,
                        COUNT(*) FILTER (WHERE quality_grade = 'B')                AS b,
                        COUNT(*) FILTER (WHERE quality_grade = 'C')                AS c,
                        COUNT(*) FILTER (WHERE quality_grade = 'D')                AS d,
                        COUNT(*) FILTER (WHERE quality_grade = 'F')                AS f,
                        COALESCE(SUM(duration_seconds)
                            FILTER (WHERE quality_grade IS NOT NULL
                                      AND quality_grade != 'F'), 0)                AS clean_sec
                    FROM sessions
                    WHERE session_folder IS NOT NULL
                """)
                total, scored, avg_score, a, b, c, d, f, clean_sec = cur.fetchone()
        conn.close()

        scored = int(scored or 0)
        if scored == 0:
            return empty

        usable = a + b + c + d
        pct = lambda n: round(n / scored * 100.0, 1)

        return {
            "quality_scored_count":  scored,
            "quality_total_count":   int(total or 0),
            "quality_avg_score":     float(avg_score) if avg_score is not None else None,
            "quality_pct_usable":    pct(usable),
            "quality_pct_unusable":  pct(f),
            "quality_grade_pct":     {"A": pct(a), "B": pct(b), "C": pct(c), "D": pct(d), "F": pct(f)},
            "quality_grade_count":   {"A": int(a), "B": int(b), "C": int(c), "D": int(d), "F": int(f)},
            "total_hours_clean":     round(float(clean_sec) / 3600.0, 2),
        }
    except Exception as exc:
        logger.warning("Impossible de lire les stats qualité BDD : %s", exc)
        return empty


def main():
    logger.info("Démarré — broker=%s topic=%s intervalle=%ds dir=%s",
                 KAFKA_BROKER, KAFKA_TOPIC, HEARTBEAT_INTERVAL, SESSIONS_DIR)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        api_version=(2, 8, 1),
    )

    start_ts = time.monotonic()

    while True:
        disk             = _disk_usage(SESSIONS_DIR)
        sessions_on_disk = _sessions_on_disk(SESSIONS_DIR)
        sessions_in_db, total_hours = _captured_hours()
        quality          = _quality_stats()

        msg = {
            "source":               "server_heartbeat",
            "server_id":            SERVER_ID,
            "role":                 SERVER_ROLE,
            "ts":                   time.time(),
            "ts_iso":               datetime.now(timezone.utc).isoformat(),
            "uptime_s":             int(time.monotonic() - start_ts),
            "sessions_count":       sessions_on_disk,
            "sessions_count_db":    sessions_in_db,
            "total_hours_captured": total_hours,
            **disk,
            **quality,
        }

        try:
            producer.send(KAFKA_TOPIC, msg)
            producer.flush()
            logger.info(
                "heartbeat envoyé — disque %.1f%% (%.1f/%.1f Go libres : %.1f Go) "
                "| %d sessions disque | %d sessions BDD | %.1fh capturées "
                "| qualité : %d/%d notées, %s%% exploitables, %s%% inutilisables",
                disk["disk_used_pct"], disk["disk_used_gb"], disk["disk_total_gb"],
                disk["disk_free_gb"], sessions_on_disk, sessions_in_db, total_hours,
                quality["quality_scored_count"], quality["quality_total_count"],
                quality["quality_pct_usable"], quality["quality_pct_unusable"],
            )
        except Exception as exc:
            logger.warning("Échec envoi Kafka : %s", exc)

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
