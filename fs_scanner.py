#!/usr/bin/env python3
"""
fs_scanner.py — Scanne /data/sessions directement sur vm-storage (pas de SFTP),
lie les sessions à la BDD, les note via analysis.json et met à jour les KPIs.

Vérification d'intégrité (_check_files_integrity) :
  Pour chaque caméra/capteur déclaré dans config.json, vérifie la présence
  ET la validité des fichiers de capture :
    - cameras/<name>.mp4   doit exister, taille > 0, atomes 'ftyp'+'moov' présents
    - cameras/<name>.jsonl doit exister, taille > 0, première/dernière ligne JSON valides
    - cameras/resample_report.json + resampled_30hz.jsonl (si caméras déclarées)
    - sensors/<name>.jsonl doit exister, taille > 0, première/dernière ligne JSON valides
  Toute session avec un fichier manquant ou corrompu est notée F / score 0,
  et le détail est ajouté à la liste d'erreurs stockée en BDD.

Modes :
  --once   Scan complet ultra-rapide (multiprocessing + batch DB) puis quitte.
           Idéal pour rattraper un backlog de dizaines de milliers de sessions.
  --watch  Surveillance continue avec polling du FS (défaut).

Architecture --once :
  1. os.scandir() → liste des dossiers non encore traités
  2. Chargement du session_map complet depuis la BDD (1 seule SELECT)
  3. ProcessPoolExecutor (SCAN_WORKERS processus) → chaque processus traite
     un chunk de sessions, et utilise lui-même un ThreadPoolExecutor
     (SCAN_THREADS_PER_WORKER threads) pour paralléliser l'I/O par session
     (lecture JSON, parcours des atomes mp4, lecture jsonl)
  4. Écriture DB en batch (une seule connexion, commit toutes les DB_BATCH sessions)
  5. Recalcul KPIs en une seule requête SQL d'agrégation
  6. Bilan final : sessions scannées, heures propres, % valides / invalides

Variables d'environnement :
  SESSIONS_DIR            Répertoire sessions           (défaut: /data/sessions)
  SCAN_WORKERS            Processus parallèles          (défaut: nb CPUs)
  SCAN_THREADS_PER_WORKER Threads I/O par processus      (défaut: 4)
  DB_BATCH                Sessions par commit DB         (défaut: 500)
  STABILITY_SECONDS       Stabilité avant traitement     (défaut: 60)
  SCAN_INTERVAL           Intervalle watch en s          (défaut: 15)
  DISK_MOUNTS             Disques SSD/HDD à suivre explicitement, format
                          "mount_path:type:label:disk_uuid" séparés par ';'
                          (défaut: "<SESSIONS_DIR>:ssd:SSD principal:local-ssd-data")
  DISK_AUTODISCOVER_GLOB  Pattern glob de points de montage HDD à découvrir
                          automatiquement (défaut: "/mnt/*"). Chaque répertoire
                          qui est un vrai point de montage (os.path.ismount) et
                          absent de DISK_MOUNTS est ajouté comme disque HDD —
                          permet d'ajouter/retirer des disques physiques sans
                          modifier la config.
  DISK_SCAN_INTERVAL      Intervalle scan disques en s   (défaut: 300)
  POSTGRES_HOST/PORT/DB/USER/PASSWORD
"""

import glob
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path

import psycopg2

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

SESSIONS_DIR      = os.environ.get("SESSIONS_DIR",      "/data/sessions")
SCAN_WORKERS      = int(os.environ.get("SCAN_WORKERS",  str(max(1, (cpu_count() or 4)))))
DB_BATCH          = int(os.environ.get("DB_BATCH",      "500"))
STABILITY_SECONDS = int(os.environ.get("STABILITY_SECONDS", "60"))
SCAN_INTERVAL     = int(os.environ.get("SCAN_INTERVAL", "15"))

# Disques de stockage (SSD à expédier / HDD de sauvegarde) — voir _scan_disks().
# Format : "mount_path:type:label:disk_uuid" (type = ssd|hdd), séparés par ';'.
DISK_MOUNTS = os.environ.get(
    "DISK_MOUNTS",
    f"{SESSIONS_DIR}:ssd:SSD principal:local-ssd-data",
)
# Pattern glob de points de montage à découvrir automatiquement (nouveaux
# disques HDD branchés/montés sans avoir à toucher la config) — voir
# _discover_disks().
DISK_AUTODISCOVER_GLOB = os.environ.get("DISK_AUTODISCOVER_GLOB", "/mnt/*")
DISK_SCAN_INTERVAL = int(os.environ.get("DISK_SCAN_INTERVAL", "300"))


# ── Connexion PostgreSQL ───────────────────────────────────────────────────────

def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST",     "192.168.1.18"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB",     "robotics"),
        user=os.environ.get("POSTGRES_USER",     "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", "YsLuB46NKoF6WlS3NwUm97vhEtLkjLRQ"),
        connect_timeout=10,
    )


def _ensure_kpi_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kpi_quality_snapshots (
                project_id      TEXT        NOT NULL,
                site_id         TEXT        NOT NULL DEFAULT 'default',
                snapshot_date   DATE        NOT NULL,
                session_count   INTEGER     NOT NULL DEFAULT 0,
                scored_count    INTEGER     NOT NULL DEFAULT 0,
                avg_score       FLOAT,
                min_score       FLOAT,
                max_score       FLOAT,
                grade_a_count   INTEGER     NOT NULL DEFAULT 0,
                grade_b_count   INTEGER     NOT NULL DEFAULT 0,
                grade_c_count   INTEGER     NOT NULL DEFAULT 0,
                grade_d_count   INTEGER     NOT NULL DEFAULT 0,
                grade_f_count   INTEGER     NOT NULL DEFAULT 0,
                errors_count    INTEGER     NOT NULL DEFAULT 0,
                warnings_count  INTEGER     NOT NULL DEFAULT 0,
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (project_id, site_id, snapshot_date)
            )
        """)
    conn.commit()


# ── Score depuis analysis.json ─────────────────────────────────────────────────
# Doit rester une fonction top-level (picklable pour multiprocessing).

def _score_from_analysis(analysis: dict) -> tuple:
    """Score 0-100 et grade depuis analysis.json. Retourne (0.0, 'F') si errors."""
    if analysis.get("errors"):
        return 0.0, "F"

    scores = {}

    pairs = analysis.get("drift_check", {}).get("pairs", {})
    valid_pairs = [v for v in (pairs or {}).values() if isinstance(v, dict)]
    if valid_pairs:
        max_rel = max(abs(v.get("relative_drift_ms_per_min") or 0) for v in valid_pairs)
        scores["sync"] = max(0.0, 100.0 - max_rel * 5.0)
    else:
        scores["sync"] = 50.0

    cams = analysis.get("fps_check", {}).get("cameras", {})
    valid_cams = [c for c in (cams or {}).values() if isinstance(c, dict)]
    if valid_cams:
        total_est   = sum(
            (c.get("measured_fps", 0) or c.get("expected_fps", 0)) * c.get("duration_sec", 0)
            for c in valid_cams
        )
        total_gaps  = sum(c.get("sequence_gaps", 0) for c in valid_cams)
        total_qdrop = sum(c.get("queue_drops",   0) for c in valid_cams)
        if total_est > 0:
            gap_pct   = total_gaps  / total_est * 100.0
            qdrop_pct = total_qdrop / total_est * 100.0
        else:
            gap_pct = qdrop_pct = 0.0
        scores["fps"] = max(0.0, 100.0 - gap_pct * 1.5 - min(qdrop_pct * 0.2, 10.0))
    else:
        scores["fps"] = 50.0

    sync  = analysis.get("sync_check", {})
    delta = sync.get("delta_sec") or 0.0
    if sync.get("ok", True):
        scores["cam_sensor"] = max(0.0, 100.0 - delta * 200.0)
    else:
        scores["cam_sensor"] = max(0.0, 20.0 - delta * 100.0)

    total = round(
        scores["sync"]         * 0.40
        + scores["fps"]        * 0.35
        + scores["cam_sensor"] * 0.25,
        1,
    )
    grade = ("A" if total >= 85 else
             "B" if total >= 70 else
             "C" if total >= 55 else
             "D" if total >= 40 else "F")
    return total, grade


# ── Vérification d'intégrité des fichiers ──────────────────────────────────────

def _is_valid_mp4(path: Path) -> bool:
    """
    Vérifie qu'un fichier .mp4 a une structure de boîtes (atoms) valide et
    qu'il a été finalisé (présence de 'ftyp' et 'moov'). Un fichier tronqué
    pendant l'écriture (crash, coupure) n'aura pas de 'moov' lisible.
    """
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        found_ftyp = found_moov = False
        with open(path, "rb") as f:
            offset = 0
            while offset < size:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    break
                box_size = int.from_bytes(header[0:4], "big")
                box_type = header[4:8]
                header_len = 8
                if box_size == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    box_size = int.from_bytes(ext, "big")
                    header_len = 16
                elif box_size == 0:
                    box_size = size - offset
                if box_size < header_len:
                    return False
                if box_type == b"ftyp":
                    found_ftyp = True
                elif box_type == b"moov":
                    found_moov = True
                offset += box_size
        return found_ftyp and found_moov
    except OSError:
        return False


def _is_valid_jsonl(path: Path) -> bool:
    """Vérifie qu'un fichier .jsonl est non vide et que sa première et sa
    dernière ligne sont du JSON valide (sans lire le fichier entier)."""
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "rb") as f:
            first = f.readline()
            if not first.strip():
                return False
            json.loads(first)

            f.seek(0, os.SEEK_END)
            fsize = f.tell()
            chunk = min(fsize, 8192)
            f.seek(fsize - chunk)
            tail_lines = [l for l in f.read(chunk).split(b"\n") if l.strip()]
            if not tail_lines:
                return False
            json.loads(tail_lines[-1])
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _is_valid_json(path: Path) -> bool:
    """Vérifie qu'un fichier .json est non vide et parsable."""
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _check_files_integrity(base: Path, config: dict | None) -> tuple:
    """
    Vérifie la présence et la validité des fichiers de capture attendus
    (caméras + capteurs déclarés dans config.json).

    Retourne (missing_files, corrupted_files) : listes de chemins relatifs
    au dossier de session.
    """
    missing:   list = []
    corrupted: list = []

    cam_names    = [c.get("name") for c in (config or {}).get("cameras", []) if c.get("name")]
    sensor_names = [s.get("name") for s in (config or {}).get("sensors", []) if s.get("name")]

    cameras_dir = base / "cameras"
    sensors_dir = base / "sensors"

    for name in cam_names:
        mp4   = cameras_dir / f"{name}.mp4"
        jsonl = cameras_dir / f"{name}.jsonl"

        if not mp4.exists() or mp4.stat().st_size == 0:
            missing.append(f"cameras/{name}.mp4")
        elif not _is_valid_mp4(mp4):
            corrupted.append(f"cameras/{name}.mp4")

        if not jsonl.exists() or jsonl.stat().st_size == 0:
            missing.append(f"cameras/{name}.jsonl")
        elif not _is_valid_jsonl(jsonl):
            corrupted.append(f"cameras/{name}.jsonl")

    if cam_names:
        resample  = cameras_dir / "resample_report.json"
        resampled = cameras_dir / "resampled_30hz.jsonl"

        if not resample.exists() or resample.stat().st_size == 0:
            missing.append("cameras/resample_report.json")
        elif not _is_valid_json(resample):
            corrupted.append("cameras/resample_report.json")

        if not resampled.exists() or resampled.stat().st_size == 0:
            missing.append("cameras/resampled_30hz.jsonl")
        elif not _is_valid_jsonl(resampled):
            corrupted.append("cameras/resampled_30hz.jsonl")

    for name in sensor_names:
        jsonl = sensors_dir / f"{name}.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            missing.append(f"sensors/{name}.jsonl")
        elif not _is_valid_jsonl(jsonl):
            corrupted.append(f"sensors/{name}.jsonl")

    return missing, corrupted


def _evaluate_session(base: Path, analysis: dict, config: dict | None) -> tuple:
    """
    Calcule score, grade, erreurs et warnings d'une session en combinant :
      - le score qualité issu de analysis.json (_score_from_analysis)
      - la vérification d'intégrité des fichiers de capture (_check_files_integrity)

    Toute session avec un fichier manquant ou corrompu est notée F / 0,
    quel que soit le score calculé depuis analysis.json.
    """
    score, grade = _score_from_analysis(analysis)
    errors   = [str(e) for e in analysis.get("errors",   [])]
    warnings = [str(w) for w in analysis.get("warnings", [])]

    missing_files, corrupted_files = _check_files_integrity(base, config)
    for fpath in missing_files:
        errors.append(f"fichier manquant: {fpath}")
    for fpath in corrupted_files:
        errors.append(f"fichier corrompu: {fpath}")

    if missing_files or corrupted_files:
        score, grade = 0.0, "F"

    return score, grade, errors, warnings


# ── Worker multiprocessing (top-level = picklable) ─────────────────────────────

def _session_duration_sec(analysis: dict) -> float:
    """
    Durée totale d'une session, calculée depuis analysis.json.

    Prend le maximum des duration_sec de toutes les caméras (fps_check) et
    de tous les capteurs (sensor_check) — c'est le flux le plus long qui
    détermine la durée réelle de la session, même si un autre flux s'est
    arrêté plus tôt (capteur déconnecté, etc.).
    """
    durations = []

    cams = analysis.get("fps_check", {}).get("cameras", {})
    for c in (cams or {}).values():
        if isinstance(c, dict):
            durations.append(c.get("duration_sec") or 0)

    sensors = analysis.get("sensor_check", {}).get("sensors", {})
    for s in (sensors or {}).values():
        if isinstance(s, dict):
            durations.append(s.get("duration_sec") or 0)

    return max(durations, default=0.0)


SCAN_THREADS_PER_WORKER = int(os.environ.get("SCAN_THREADS_PER_WORKER", "4"))


def _scan_one_session(sessions_dir: str, folder_name: str) -> tuple:
    """
    Lit analysis.json + config.json pour une session, vérifie l'intégrité
    des fichiers et calcule le score.

    Retourne : (folder_name, score, grade, errors, warnings_count, config,
                 duration_sec, err)
    err vaut "no_analysis" si analysis.json est absent, ou un message
    d'erreur en cas d'exception, sinon None.
    """
    base = Path(sessions_dir) / folder_name
    try:
        analysis_path = base / "analysis.json"
        if not analysis_path.exists():
            return (folder_name, None, None, None, 0, None, 0, "no_analysis")

        with open(analysis_path, "r", encoding="utf-8", errors="replace") as f:
            analysis = json.load(f)

        config = None
        config_path = base / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                    config = json.load(f)
            except Exception:
                pass

        score, grade, errors, warnings = _evaluate_session(base, analysis, config)

        duration_sec = _session_duration_sec(analysis)

        return (folder_name, score, grade, errors, len(warnings),
                config, duration_sec, None)

    except Exception as exc:
        return (folder_name, None, None, None, 0, None, 0, str(exc))


def _scan_chunk(args: tuple) -> list:
    """
    Worker process : traite un lot de dossiers en parallèle via un pool de
    threads (les opérations sont dominées par l'I/O — lecture JSON, parcours
    des atomes mp4, lecture jsonl — donc le GIL n'est pas un goulot).

    args: (sessions_dir, [folder_name, ...])
    Retourne: [(folder_name, score, grade, errors, warnings_count, config,
                 duration_sec, err), ...]
    """
    sessions_dir, folder_names = args

    if len(folder_names) == 1:
        return [_scan_one_session(sessions_dir, folder_names[0])]

    workers = min(SCAN_THREADS_PER_WORKER, len(folder_names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(
            lambda fname: _scan_one_session(sessions_dir, fname),
            folder_names,
        ))


# ── Utilitaires FS ─────────────────────────────────────────────────────────────

def _list_sessions(sessions_dir: str) -> list:
    """Liste les dossiers session_* via os.scandir (bien plus rapide que listdir)."""
    try:
        with os.scandir(sessions_dir) as it:
            return sorted(
                e.name for e in it
                if e.is_dir(follow_symlinks=False) and e.name.lower().startswith("session")
            )
    except Exception as exc:
        logger.error("Impossible de lister %s : %s", sessions_dir, exc)
        return []


def _analysis_stat(folder_name: str) -> tuple | None:
    """Retourne (size, mtime) d'analysis.json ou None si absent."""
    path = Path(SESSIONS_DIR) / folder_name / "analysis.json"
    try:
        st = path.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None


def _get_folder_size(folder_name: str) -> int:
    """Taille récursive d'un dossier session via os.scandir."""
    total = 0
    stack = [str(Path(SESSIONS_DIR) / folder_name)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    else:
                        try:
                            total += entry.stat().st_size
                        except OSError:
                            pass
        except OSError:
            pass
    return total


# ── Résolution session_id ──────────────────────────────────────────────────────

def _folder_to_timestamp(folder_name: str) -> str | None:
    m = re.match(r"^session_(\d{8})_(\d{6})", folder_name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", folder_name)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return None


def _resolve_from_map(session_map: dict, folder_name: str, analysis: dict | None) -> str | None:
    """
    Résout session_id depuis le dictionnaire pré-chargé (pas de SELECT).
    Stratégies identiques à _resolve_session_id.
    """
    if folder_name in session_map:
        return session_map[folder_name]

    ts = _folder_to_timestamp(folder_name)
    if ts:
        prefix = f"sess_{ts}"
        for key, sid in session_map.items():
            if key.startswith(prefix):
                return sid

    if analysis:
        robot_path = analysis.get("session", "")
        if robot_path:
            basename = robot_path.split("/")[-1]
            for candidate in (robot_path, f"./data/{basename}", f"/data/{basename}", basename):
                if candidate in session_map:
                    return session_map[candidate]

    return None


def _resolve_session_id(cur, folder_name: str, analysis: dict | None = None) -> str | None:
    """Résolution par SELECT DB — utilisé en mode --watch uniquement."""
    for query, params in [
        ("SELECT session_id FROM sessions WHERE session_folder = %s LIMIT 1", (folder_name,)),
        ("SELECT session_id FROM sessions WHERE session_id     = %s LIMIT 1", (folder_name,)),
    ]:
        cur.execute(query, params)
        row = cur.fetchone()
        if row:
            return row[0]

    ts = _folder_to_timestamp(folder_name)
    if ts:
        cur.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE %s LIMIT 1",
            (f"sess_{ts}%",),
        )
        row = cur.fetchone()
        if row:
            return row[0]

    if analysis:
        robot_path = analysis.get("session", "")
        if robot_path:
            basename = robot_path.split("/")[-1]
            for candidate in (robot_path, f"./data/{basename}", f"/data/{basename}", basename):
                cur.execute(
                    "SELECT session_id FROM sessions WHERE session_folder = %s LIMIT 1",
                    (candidate,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]

    return None


def _create_session_from_config(cur, folder_name: str, config: dict) -> str:
    ts = _folder_to_timestamp(folder_name)
    session_id = f"sess_{ts}_fs" if ts else f"sess_{folder_name}"

    operator = config.get("operator") or {}
    rig      = config.get("rig")      or {}
    project  = config.get("project")  or {}

    started_at = None
    if ts:
        try:
            started_at = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    cur.execute("""
        INSERT INTO sessions (
            session_id, operator_id, operator_name, site_id, project_id,
            rig_id, pipeline_status, session_folder, started_at, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, 'captured', %s, %s, now())
        ON CONFLICT (session_id) DO NOTHING
    """, (
        session_id,
        operator.get("operator_id"),
        operator.get("full_name"),
        rig.get("site_id", "default"),
        project.get("project_id"),
        rig.get("rig_id"),
        folder_name,
        started_at,
    ))
    return session_id


# ── Chargement DB ──────────────────────────────────────────────────────────────

def _load_already_scored() -> set:
    """Dossiers déjà scorés depuis la BDD."""
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_folder
                    FROM sessions
                    WHERE session_folder ~ '^session_\\d{8}_\\d{6}'
                      AND quality_score IS NOT NULL
                    UNION
                    SELECT COALESCE(session_folder, session_id)
                    FROM sessions
                    WHERE metadata ? 'capture_quality'
                """)
                result = {row[0] for row in cur.fetchall() if row[0]}
        conn.close()
        return result
    except Exception as exc:
        logger.warning("Impossible de charger les sessions déjà scorées : %s", exc)
        return set()


def _load_session_map(conn) -> dict:
    """
    Charge TOUS les mappings folder→session_id en une seule requête.
    Évite 40k SELECT individuels dans _resolve_session_id.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(session_folder, session_id), session_id
            FROM sessions
            WHERE session_folder IS NOT NULL OR session_id IS NOT NULL
        """)
        return {row[0]: row[1] for row in cur.fetchall() if row[0]}


# ── Écriture DB (session + KPI) ────────────────────────────────────────────────

def _write_session(cur, folder_name: str, session_id: str,
                   score: float, grade: str,
                   errors: list, warnings_count: int,
                   size_bytes: int = 0) -> bool:
    """UPDATE sessions + UPSERT KPI pour une session. Pas de commit (fait par l'appelant)."""
    capture_meta = json.dumps({
        "score":          score,
        "grade":          grade,
        "errors":         errors,
        "warnings_count": warnings_count,
        "size_bytes":     size_bytes,
        "scored_at":      datetime.now(timezone.utc).isoformat(),
        "source":         "fs_scanner",
    }, ensure_ascii=False)

    cur.execute("""
        UPDATE sessions
        SET session_folder  = %s,
            size_bytes      = CASE WHEN %s > 0 THEN %s ELSE size_bytes END,
            quality_score   = %s,
            quality_grade   = %s,
            pipeline_status = CASE
                WHEN pipeline_status IN ('queued', 'captured') THEN 'captured'
                ELSE pipeline_status
            END,
            metadata = metadata || jsonb_build_object('capture_quality', %s::jsonb)
        WHERE session_id = %s
    """, (folder_name, size_bytes, size_bytes, score, grade, capture_meta, session_id))

    return cur.rowcount > 0


def _recalculate_kpis(conn) -> None:
    """
    Recalcule TOUS les KPI depuis la table sessions en une seule requête.
    Remplace les N UPSERT incrémentaux — résultat correct garanti.
    """
    logger.info("Recalcul KPI en cours...")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO kpi_quality_snapshots (
                project_id, site_id, snapshot_date,
                session_count, scored_count,
                avg_score, min_score, max_score,
                grade_a_count, grade_b_count, grade_c_count,
                grade_d_count, grade_f_count,
                errors_count, warnings_count, updated_at
            )
            SELECT
                COALESCE(project_id,  'unknown'),
                COALESCE(site_id,     'default'),
                COALESCE(started_at::date, CURRENT_DATE),
                COUNT(*),
                COUNT(quality_score),
                ROUND(AVG(quality_score)::numeric, 1),
                MIN(quality_score),
                MAX(quality_score),
                COUNT(*) FILTER (WHERE quality_grade = 'A'),
                COUNT(*) FILTER (WHERE quality_grade = 'B'),
                COUNT(*) FILTER (WHERE quality_grade = 'C'),
                COUNT(*) FILTER (WHERE quality_grade = 'D'),
                COUNT(*) FILTER (WHERE quality_grade = 'F'),
                COALESCE(SUM(
                    jsonb_array_length(
                        COALESCE(metadata->'capture_quality'->'errors', '[]'::jsonb)
                    )
                ), 0),
                COALESCE(SUM(
                    (metadata->'capture_quality'->>'warnings_count')::int
                ), 0),
                now()
            FROM sessions
            WHERE quality_score IS NOT NULL
            GROUP BY
                COALESCE(project_id,  'unknown'),
                COALESCE(site_id,     'default'),
                COALESCE(started_at::date, CURRENT_DATE)
            ON CONFLICT (project_id, site_id, snapshot_date) DO UPDATE SET
                session_count  = EXCLUDED.session_count,
                scored_count   = EXCLUDED.scored_count,
                avg_score      = EXCLUDED.avg_score,
                min_score      = EXCLUDED.min_score,
                max_score      = EXCLUDED.max_score,
                grade_a_count  = EXCLUDED.grade_a_count,
                grade_b_count  = EXCLUDED.grade_b_count,
                grade_c_count  = EXCLUDED.grade_c_count,
                grade_d_count  = EXCLUDED.grade_d_count,
                grade_f_count  = EXCLUDED.grade_f_count,
                errors_count   = EXCLUDED.errors_count,
                warnings_count = EXCLUDED.warnings_count,
                updated_at     = now()
        """)
    conn.commit()
    logger.info("Recalcul KPI terminé.")


# ── Disques de stockage (SSD/HDD) ───────────────────────────────────────────────
# Répartit les sessions par disque physique (storage_disks, table créée par
# la migration 008). Code additif : n'interfère jamais avec le scoring /
# les KPI ci-dessus, et n'interrompt jamais la boucle principale en cas
# d'erreur (ex : migration pas encore appliquée, disque pas encore monté).

def _parse_disk_mounts() -> list:
    """Parse DISK_MOUNTS = 'mount_path:type:label:disk_uuid;...'."""
    disks = []
    for entry in DISK_MOUNTS.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 4:
            logger.warning("DISK_MOUNTS : entrée invalide ignorée : %s", entry)
            continue
        mount_path, disk_type, label = parts[0].strip(), parts[1].strip().lower(), parts[2].strip()
        disk_uuid = ":".join(parts[3:]).strip()
        if disk_type not in ("ssd", "hdd"):
            logger.warning("DISK_MOUNTS : type invalide (%s) pour %s, ignoré", disk_type, entry)
            continue
        if not disk_uuid:
            logger.warning("DISK_MOUNTS : disk_uuid manquant pour %s, ignoré", entry)
            continue
        disks.append({
            "mount_path": mount_path,
            "disk_type":  disk_type,
            "label":      label,
            "disk_uuid":  disk_uuid,
        })
    return disks


def _autodiscover_disks(known_paths: set) -> list:
    """Découvre les points de montage HDD non déclarés explicitement dans
    DISK_MOUNTS, via le pattern DISK_AUTODISCOVER_GLOB (défaut /mnt/*).
    Permet d'ajouter/retirer des disques physiques sans toucher la config :
    seuls les vrais points de montage (os.path.ismount) sont retenus, label
    et disk_uuid sont dérivés du nom du dossier."""
    disks = []
    for path in sorted(glob.glob(DISK_AUTODISCOVER_GLOB)):
        if path in known_paths or not os.path.ismount(path):
            continue
        name = os.path.basename(path.rstrip("/"))
        disks.append({
            "mount_path": path,
            "disk_type":  "hdd",
            "label":      name,
            "disk_uuid":  f"local-{name}",
        })
    return disks


def _discover_disks() -> list:
    """Calcule total/used/free (os.statvfs) pour chaque disque configuré
    (explicite + auto-découvert) dont le mount_path existe déjà. Les disques
    pas encore montés sont ignorés."""
    explicit = _parse_disk_mounts()
    autodiscovered = _autodiscover_disks({d["mount_path"] for d in explicit})

    disks = []
    for disk in explicit + autodiscovered:
        path = disk["mount_path"]
        if not os.path.isdir(path):
            logger.debug("Disque %s (%s) : chemin %s absent, ignoré",
                         disk["disk_uuid"], disk["disk_type"], path)
            continue
        try:
            st = os.statvfs(path)
            total = st.f_frsize * st.f_blocks
            free  = st.f_frsize * st.f_bavail
            used  = total - (st.f_frsize * st.f_bfree)
        except OSError as exc:
            logger.warning("Disque %s : statvfs(%s) a échoué : %s", disk["disk_uuid"], path, exc)
            continue
        disks.append({**disk, "total_bytes": total, "used_bytes": used, "free_bytes": free})
    return disks


def _upsert_disks(conn, disks: list) -> None:
    with conn.cursor() as cur:
        for d in disks:
            cur.execute("""
                INSERT INTO storage_disks (
                    disk_uuid, server_id, disk_type, label, mount_path,
                    total_bytes, used_bytes, free_bytes, last_scanned_at, updated_at
                ) VALUES (%s, 'vm-storage', %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (disk_uuid) DO UPDATE SET
                    disk_type       = EXCLUDED.disk_type,
                    label           = COALESCE(storage_disks.label, EXCLUDED.label),
                    mount_path      = EXCLUDED.mount_path,
                    total_bytes     = EXCLUDED.total_bytes,
                    used_bytes      = EXCLUDED.used_bytes,
                    free_bytes      = EXCLUDED.free_bytes,
                    last_scanned_at = now(),
                    updated_at      = now()
            """, (
                d["disk_uuid"], d["disk_type"], d["label"], d["mount_path"],
                d["total_bytes"], d["used_bytes"], d["free_bytes"],
            ))
    conn.commit()


def _assign_sessions_to_disk(conn, disk: dict) -> None:
    """Met à jour ssd_disk_uuid / hdd_disk_uuid : assigne ce disque aux sessions
    dont le dossier est présent dans son mount_path, et désassigne les sessions
    qui y pointaient mais dont le dossier a disparu (ex : SSD vidé après envoi)."""
    folders = _list_sessions(disk["mount_path"])
    col = "ssd_disk_uuid" if disk["disk_type"] == "ssd" else "hdd_disk_uuid"
    with conn.cursor() as cur:
        if folders:
            cur.execute(f"""
                UPDATE sessions SET {col} = %s
                WHERE session_folder = ANY(%s) AND {col} IS DISTINCT FROM %s
            """, (disk["disk_uuid"], folders, disk["disk_uuid"]))
        cur.execute(f"""
            UPDATE sessions SET {col} = NULL
            WHERE {col} = %s
              AND session_folder IS NOT NULL
              AND NOT (session_folder = ANY(%s))
        """, (disk["disk_uuid"], folders))
    conn.commit()


def _recalculate_disk_stats(conn) -> None:
    """Recalcule session_count / sessions_delivered_count pour chaque disque
    (une session est "livrée" si pipeline_status='delivered' OU si elle a une
    ligne client_deliveries au statut 'delivered')."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE storage_disks sd SET
                session_count = (
                    SELECT COUNT(*) FROM sessions s
                    WHERE s.ssd_disk_uuid = sd.disk_uuid OR s.hdd_disk_uuid = sd.disk_uuid
                ),
                sessions_delivered_count = (
                    SELECT COUNT(*) FROM sessions s
                    WHERE (s.ssd_disk_uuid = sd.disk_uuid OR s.hdd_disk_uuid = sd.disk_uuid)
                      AND (
                          s.pipeline_status = 'delivered'
                          OR EXISTS (
                              SELECT 1 FROM client_deliveries cd
                              WHERE cd.session_id = s.session_id AND cd.status = 'delivered'
                          )
                      )
                ),
                updated_at = now()
        """)
    conn.commit()


def _scan_disks(conn) -> None:
    """Point d'entrée disques : découverte + assignation + agrégats.
    Ne lève jamais (table éventuellement absente si migration 008 pas
    encore appliquée, disques pas encore montés, etc.)."""
    try:
        disks = _discover_disks()
        if not disks:
            return
        _upsert_disks(conn, disks)
        for disk in disks:
            _assign_sessions_to_disk(conn, disk)
        _recalculate_disk_stats(conn)
        logger.info("Disques : %d disque(s) scanné(s)", len(disks))
    except Exception:
        conn.rollback()
        logger.exception("Erreur lors du scan des disques (migration 008 appliquée ?)")


# ── Bilan final ───────────────────────────────────────────────────────────────

def _print_summary(conn, total_on_disk: int, valid_count: int,
                    total_duration_sec: float, clean_duration_sec: float,
                    elapsed: float) -> None:
    """Affiche la distribution des notes, le total sessions, le total
    d'heures et le détail heures propres / % sessions valides-invalides."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                          AS scored,
                    COUNT(*) FILTER (WHERE quality_grade = 'A')      AS a,
                    COUNT(*) FILTER (WHERE quality_grade = 'B')      AS b,
                    COUNT(*) FILTER (WHERE quality_grade = 'C')      AS c,
                    COUNT(*) FILTER (WHERE quality_grade = 'D')      AS d,
                    COUNT(*) FILTER (WHERE quality_grade = 'F')      AS f,
                    ROUND(AVG(quality_score)::numeric, 1)            AS avg
                FROM sessions
                WHERE quality_score IS NOT NULL
            """)
            row = cur.fetchone()
    except Exception as exc:
        logger.warning("Impossible de récupérer les stats : %s", exc)
        return

    scored, a, b, c, d, f, avg = row
    scored = scored or 0

    def pct(n):
        return f"{n / scored * 100:.1f}%" if scored else "—"

    h  = int(total_duration_sec // 3600)
    m  = int((total_duration_sec % 3600) // 60)
    ch = int(clean_duration_sec // 3600)
    cm = int((clean_duration_sec % 3600) // 60)

    invalid_count = max(0, total_on_disk - valid_count)
    valid_pct   = f"{valid_count   / total_on_disk * 100:.1f}%" if total_on_disk else "—"
    invalid_pct = f"{invalid_count / total_on_disk * 100:.1f}%" if total_on_disk else "—"

    sep = "─" * 44
    lines = [
        "",
        f"┌{sep}┐",
        f"│{'BILAN FINAL':^44}│",
        f"├{sep}┤",
        f"│  Sessions sur disque        : {total_on_disk:>10}  │",
        f"│  Sessions notées en BDD     : {scored:>10}  │",
        f"│  Score moyen                : {str(avg or 0) + '/100':>10}  │",
        f"│  Durée totale capturée      : {f'{h}h {m:02d}min':>10}  │",
        f"│  Heures propres (sans erreur): {f'{ch}h {cm:02d}min':>9}  │",
        f"│  Sessions valides           : {valid_pct:>10}  │",
        f"│  Sessions invalides         : {invalid_pct:>10}  │",
        f"├{sep}┤",
        f"│  A (≥85)  {a:>7}   {pct(a):>7}              │",
        f"│  B (≥70)  {b:>7}   {pct(b):>7}              │",
        f"│  C (≥55)  {c:>7}   {pct(c):>7}              │",
        f"│  D (≥40)  {d:>7}   {pct(d):>7}              │",
        f"│  F (<40)  {f:>7}   {pct(f):>7}              │",
        f"├{sep}┤",
        f"│  Temps de traitement        : {elapsed:>9.1f}s  │",
        f"└{sep}┘",
        "",
    ]
    for line in lines:
        logger.info(line)


# ── Mode --once : scan complet parallèle ──────────────────────────────────────

def run_once():
    """
    Scan complet ultra-rapide :
      1. Liste tous les dossiers → filtre déjà traités
      2. Charge session_map depuis DB (1 SELECT)
      3. ProcessPoolExecutor : lecture JSON + scoring en parallèle
      4. Batch DB writes (DB_BATCH par commit)
      5. Recalcul KPI final en 1 requête SQL
    """
    logger.info("=== fs_scanner --once | workers=%d batch=%d dir=%s ===",
                SCAN_WORKERS, DB_BATCH, SESSIONS_DIR)

    t0 = time.monotonic()

    # ── Étape 1 : liste FS (tout retraiter sans exception) ────────────────────
    to_scan = _list_sessions(SESSIONS_DIR)

    logger.info("Total dossiers : %d | mode full-rescan (aucun filtrage)", len(to_scan))

    if not to_scan:
        logger.info("Rien à faire.")
        return

    # ── Étape 2 : connexion DB + session_map ───────────────────────────────────
    conn = _pg_connect()
    _ensure_kpi_table(conn)
    session_map = _load_session_map(conn)
    logger.info("session_map chargé : %d entrées", len(session_map))

    # ── Étape 3 : scan parallèle par chunks ────────────────────────────────────
    chunk_size = max(1, len(to_scan) // (SCAN_WORKERS * 4))
    chunks = [
        (SESSIONS_DIR, to_scan[i:i + chunk_size])
        for i in range(0, len(to_scan), chunk_size)
    ]
    logger.info("Chunks : %d × ~%d sessions | %d processus", len(chunks), chunk_size, SCAN_WORKERS)

    ok = skipped = failed = no_analysis = 0
    valid_count = 0
    total_duration_sec = 0.0
    clean_duration_sec = 0.0
    pending_new_sessions: list = []  # créées en cours de scan, ajoutées au map

    t_scan = time.monotonic()

    with ProcessPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        for chunk_results in pool.map(_scan_chunk, chunks, chunksize=1):
            # ── Écriture DB pour ce chunk ───────────────────────────────────────
            for folder_name, score, grade, errors, warnings_count, config, duration_sec, err in chunk_results:

                if err == "no_analysis":
                    no_analysis += 1
                    continue
                if err:
                    logger.debug("%s : erreur lecture — %s", folder_name, err)
                    skipped += 1
                    continue

                try:
                    with conn.cursor() as cur:
                        # Résolution depuis le map pré-chargé (pas de SELECT)
                        session_id = _resolve_from_map(session_map, folder_name, None)

                        if not session_id:
                            if config:
                                session_id = _create_session_from_config(cur, folder_name, config)
                                session_map[folder_name] = session_id  # mise à jour locale
                                logger.info("Nouvelle session créée : %s → %s",
                                            folder_name, session_id)
                            else:
                                logger.debug("'%s' introuvable en BDD (pas de config.json)",
                                             folder_name)
                                failed += 1
                                continue

                        updated = _write_session(
                            cur, folder_name, session_id,
                            score, grade, errors, warnings_count,
                        )

                    if updated:
                        ok += 1
                        total_duration_sec += duration_sec or 0
                        if not errors:
                            valid_count += 1
                            clean_duration_sec += duration_sec or 0
                    else:
                        failed += 1

                except Exception as exc:
                    conn.rollback()
                    logger.warning("%s : erreur DB — %s", folder_name, exc)
                    skipped += 1
                    continue

                # Commit par lot
                if (ok + failed) % DB_BATCH == 0:
                    conn.commit()
                    logger.info("  ... %d traités (%.1f/s)",
                                ok + failed,
                                (ok + failed) / max(1, time.monotonic() - t_scan))

    conn.commit()

    t_db = time.monotonic()
    logger.info("Scan FS terminé en %.1fs | %d traités %d sans session_id "
                "%d sans analysis.json %d erreurs",
                t_db - t_scan, ok, failed, no_analysis, skipped)

    # ── Étape 5 : recalcul KPI final ───────────────────────────────────────────
    _recalculate_kpis(conn)

    # ── Étape 6 : répartition par disque (SSD/HDD) ─────────────────────────────
    _scan_disks(conn)

    # ── Bilan final ────────────────────────────────────────────────────────────
    _print_summary(conn, len(to_scan), valid_count, total_duration_sec,
                    clean_duration_sec, time.monotonic() - t0)
    conn.close()


# ── Mode --watch : surveillance continue ──────────────────────────────────────

def watch_loop():
    """
    Surveillance continue du dossier sessions.
    Détecte les nouveaux dossiers, attend la stabilité d'analysis.json,
    puis traite avec un ThreadPoolExecutor.
    """
    logger.info("=== fs_scanner --watch | intervalle=%ds stabilité=%ds dir=%s ===",
                SCAN_INTERVAL, STABILITY_SECONDS, SESSIONS_DIR)

    processed: set   = _load_already_scored()
    # {folder_name: (size, mtime, first_stable_ts)}
    pending_stable: dict = {}
    pending_db:     set  = set()

    logger.info("%d session(s) déjà traitées", len(processed))

    last_disk_scan = 0.0

    while True:
        now = time.monotonic()

        # ── 0. Répartition par disque (SSD/HDD), throttlée ─────────────────────
        if now - last_disk_scan >= DISK_SCAN_INTERVAL:
            try:
                disk_conn = _pg_connect()
                _scan_disks(disk_conn)
                disk_conn.close()
            except Exception:
                logger.exception("Connexion DB échouée pour le scan des disques")
            last_disk_scan = now

        # ── 1. Vérifier stabilité en parallèle ─────────────────────────────────
        newly_stable = []
        still_watching = {}

        if pending_stable:
            def _check_one(item):
                fname, (ps, pm, fs) = item
                stat = _analysis_stat(fname)
                return fname, (ps, pm, fs), stat

            with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, len(pending_stable))) as pool:
                for fname, (ps, pm, fs), stat in pool.map(_check_one, pending_stable.items()):
                    if stat is None:
                        logger.debug("%s : analysis.json disparu, abandon", fname)
                        continue
                    cs, cm = stat
                    if cs != ps or cm != pm:
                        still_watching[fname] = (cs, cm, now)
                    elif now - fs >= STABILITY_SECONDS:
                        newly_stable.append(fname)
                    else:
                        still_watching[fname] = (cs, cm, fs)

        pending_stable = still_watching

        # ── 2. Traiter les sessions stables ────────────────────────────────────
        if newly_stable:
            def _fetch_and_process(fname):
                base = Path(SESSIONS_DIR) / fname
                try:
                    with open(base / "analysis.json", "r", encoding="utf-8") as f:
                        analysis = json.load(f)
                except Exception:
                    return fname, False, "analysis illisible"

                config = None
                cp = base / "config.json"
                if cp.exists():
                    try:
                        with open(cp, "r", encoding="utf-8") as f:
                            config = json.load(f)
                    except Exception:
                        pass

                size_bytes = _get_folder_size(fname)
                score, grade, errors, warnings = _evaluate_session(base, analysis, config)

                try:
                    conn = _pg_connect()
                    _ensure_kpi_table(conn)
                    with conn:
                        with conn.cursor() as cur:
                            session_id = _resolve_session_id(cur, fname, analysis)
                            if not session_id:
                                if config:
                                    session_id = _create_session_from_config(cur, fname, config)
                                else:
                                    return fname, False, "session introuvable"

                            _write_session(cur, fname, session_id, score, grade,
                                           errors, len(warnings), size_bytes)

                            # KPI incrémental (une seule session à la fois)
                            cur.execute(
                                "SELECT project_id, site_id, started_at::date "
                                "FROM sessions WHERE session_id = %s", (session_id,)
                            )
                            row = cur.fetchone()
                            if row:
                                pid, sid, sdate = row
                                pid   = pid   or "unknown"
                                sid   = sid   or "default"
                                sdate = sdate or datetime.now(timezone.utc).date()
                                gcol  = f"grade_{grade.lower()}_count"
                                cur.execute(f"""
                                    INSERT INTO kpi_quality_snapshots AS kq
                                        (project_id, site_id, snapshot_date,
                                         session_count, scored_count,
                                         avg_score, min_score, max_score,
                                         grade_a_count, grade_b_count, grade_c_count,
                                         grade_d_count, grade_f_count,
                                         errors_count, warnings_count, updated_at)
                                    VALUES (%s,%s,%s, 1,1, %s,%s,%s, 0,0,0,0,0, %s,%s, now())
                                    ON CONFLICT (project_id, site_id, snapshot_date) DO UPDATE SET
                                        session_count  = kq.session_count + 1,
                                        scored_count   = kq.scored_count  + 1,
                                        avg_score      = (kq.avg_score * kq.scored_count
                                                          + EXCLUDED.avg_score)
                                                         / (kq.scored_count + 1),
                                        min_score      = LEAST(kq.min_score, EXCLUDED.min_score),
                                        max_score      = GREATEST(kq.max_score, EXCLUDED.max_score),
                                        {gcol}         = kq.{gcol} + 1,
                                        errors_count   = kq.errors_count   + EXCLUDED.errors_count,
                                        warnings_count = kq.warnings_count + EXCLUDED.warnings_count,
                                        updated_at     = now()
                                """, (pid, sid, sdate, score, score, score,
                                      len(errors), len(warnings)))
                    conn.close()
                    flag = "ERREURS" if errors else f"{len(warnings)}w" if warnings else "OK"
                    return fname, True, f"score={score} grade={grade} [{flag}]"

                except Exception as exc:
                    return fname, False, str(exc)

            with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, len(newly_stable))) as pool:
                for fname, success, msg in pool.map(_fetch_and_process, newly_stable):
                    if success:
                        processed.add(fname)
                        logger.info("%s : %s", fname, msg)
                    else:
                        pending_db.add(fname)
                        logger.debug("%s : échec — %s, réessai prochain cycle", fname, msg)

        # ── 3. Retry pending_db ─────────────────────────────────────────────────
        if pending_db:
            still_pending = set()
            for fname in list(pending_db):
                base = Path(SESSIONS_DIR) / fname
                try:
                    with open(base / "analysis.json", "r", encoding="utf-8") as f:
                        analysis = json.load(f)
                    cp = base / "config.json"
                    config = None
                    if cp.exists():
                        with open(cp, "r", encoding="utf-8") as f:
                            config = json.load(f)
                    score, grade, errors, warnings = _evaluate_session(base, analysis, config)
                    conn = _pg_connect()
                    with conn:
                        with conn.cursor() as cur:
                            session_id = _resolve_session_id(cur, fname, analysis)
                            if not session_id:
                                if config:
                                    session_id = _create_session_from_config(cur, fname, config)
                                else:
                                    still_pending.add(fname)
                                    conn.close()
                                    continue
                            _write_session(cur, fname, session_id, score, grade,
                                           errors, len(warnings))
                    conn.close()
                    processed.add(fname)
                except Exception as exc:
                    logger.debug("%s : retry DB échoué — %s", fname, exc)
                    still_pending.add(fname)
            pending_db = still_pending

        # ── 4. Nouvelles sessions ───────────────────────────────────────────────
        sessions  = _list_sessions(SESSIONS_DIR)
        new_found = 0
        for fname in sessions:
            if fname in processed or fname in pending_stable or fname in pending_db:
                continue
            stat = _analysis_stat(fname)
            if stat is None:
                continue
            pending_stable[fname] = (*stat, now)
            new_found += 1

        if new_found or newly_stable:
            logger.info("Cycle : %d nouvelle(s) | %d traitée(s) | "
                        "en attente : %d stable, %d db",
                        new_found, len(newly_stable),
                        len(pending_stable), len(pending_db))

        time.sleep(SCAN_INTERVAL)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] fs_scanner: %(message)s",
    )

    if "--once" in sys.argv:
        run_once()
    else:
        watch_loop()
