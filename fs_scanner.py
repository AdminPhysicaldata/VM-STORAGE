#!/usr/bin/env python3
"""
fs_scanner.py — Scanne /data/sessions directement sur vm-storage (pas de SFTP),
lie les sessions à la BDD via le backend HTTP (vm-backend), et met à jour les
KPIs. Aucun accès direct à PostgreSQL : tout passe par BACKEND_URL (comme
sessions-uploader/SessionsToMistral.py).

Vérification d'intégrité (_check_files_integrity) :
  Pour chaque caméra/capteur déclaré dans config.json, vérifie la présence
  ET la validité des fichiers de capture :
    - cameras/<name>.mp4   doit exister, taille > 0, atomes 'ftyp'+'moov' présents
    - cameras/<name>.jsonl doit exister, taille > 0, première/dernière ligne JSON valides
    - cameras/resample_report.json + resampled_30hz.jsonl (si caméras déclarées)
    - sensors/<name>.jsonl doit exister, taille > 0, première/dernière ligne JSON valides
  Toute session avec un fichier manquant ou corrompu est notée F / score 0,
  et le détail est ajouté à la liste d'erreurs envoyée au backend.

Modes :
  --once   Scan complet ultra-rapide (multiprocessing + écritures HTTP en
           parallèle) puis quitte. Idéal pour rattraper un backlog.
  --watch  Surveillance continue avec polling du FS (défaut).

Architecture --once :
  1. os.scandir() → liste des dossiers
  2. ProcessPoolExecutor (SCAN_WORKERS processus) → chaque processus traite
     un chunk de sessions, et utilise lui-même un ThreadPoolExecutor
     (SCAN_THREADS_PER_WORKER threads) pour paralléliser l'I/O par session
     (lecture JSON, parcours des atomes mp4, lecture jsonl) — tout en local,
     aucun accès réseau dans cette étape
  3. ThreadPoolExecutor (HTTP_WORKERS threads) → POST du résultat de chaque
     session vers /api/pipeline/sessions/scan-result (le backend résout ou
     crée la session et écrit le score)
  4. Recalcul KPIs via /api/pipeline/kpis/recalculate
  5. Bilan final : sessions scannées, heures propres, % valides / invalides

Variables d'environnement :
  SESSIONS_DIR            Répertoire sessions           (défaut: /data/sessions)
  SCAN_WORKERS            Processus parallèles (scan)   (défaut: nb CPUs)
  SCAN_THREADS_PER_WORKER Threads I/O par processus      (défaut: 4)
  HTTP_WORKERS            Threads d'écriture backend     (défaut: 8)
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
  KPI_RECALC_INTERVAL     Intervalle recalcul KPI en s (mode --watch, défaut: 60)
  BACKEND_URL             URL de base de l'API backend   (défaut: http://vm-backend:5000/api)
  INTERNAL_API_TOKEN      Jeton d'authentification interne (header X-Internal-Token)
"""

import glob
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

SESSIONS_DIR      = os.environ.get("SESSIONS_DIR",      "/data/sessions")
SCAN_WORKERS      = int(os.environ.get("SCAN_WORKERS",  str(max(1, (cpu_count() or 4)))))
HTTP_WORKERS      = int(os.environ.get("HTTP_WORKERS",  "8"))
HTTP_BATCH_SIZE   = int(os.environ.get("HTTP_BATCH_SIZE", "200"))
STABILITY_SECONDS = int(os.environ.get("STABILITY_SECONDS", "60"))
SCAN_INTERVAL     = int(os.environ.get("SCAN_INTERVAL", "15"))
KPI_RECALC_INTERVAL = int(os.environ.get("KPI_RECALC_INTERVAL", "60"))

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

# ── Backend HTTP (remplace l'accès direct PostgreSQL) ──────────────────────────

BACKEND_URL         = os.environ.get("BACKEND_URL", "http://vm-backend:5000/api")
INTERNAL_API_TOKEN  = os.environ.get("INTERNAL_API_TOKEN", "")


def _backend_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN
    return headers


def _post_scan_result(folder_name: str, score: float, grade: str,
                       errors: list, warnings_count: int,
                       size_bytes: int = 0, config: dict | None = None) -> tuple:
    """
    POST /api/pipeline/sessions/scan-result — le backend résout (ou crée
    depuis 'config') la session et écrit le score en écrasant la valeur
    précédente. Retourne (ok, session_id, created, err).
    """
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/scan-result",
            json={
                "folder_name":     folder_name,
                "score":           score,
                "grade":           grade,
                "errors":          errors,
                "warnings_count":  warnings_count,
                "size_bytes":      size_bytes,
                "config":          config,
            },
            headers=_backend_headers(),
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            return True, data.get("session_id"), bool(data.get("created")), None
        return False, None, False, f"HTTP {r.status_code}: {r.text}"
    except requests.RequestException as exc:
        return False, None, False, str(exc)


def _post_scan_results_batch(items: list) -> tuple:
    """
    POST /api/pipeline/sessions/scan-result/batch — un seul aller-retour
    réseau pour tout un lot (mode --once, où une requête par session serait
    trop coûteuse sur un backlog de dizaines de milliers de sessions).

    'items' : liste de dicts {folder_name, score, grade, errors,
    warnings_count, size_bytes, config}.

    Retourne (ok_global, results, err) :
      - ok_global=False  → la requête HTTP elle-même a échoué (timeout, 5xx...)
      - results          → dict {folder_name: {ok, session_id?, created?, error?}}
        renvoyé par le backend, valable même si ok_global=True (chaque item
        peut individuellement réussir ou échouer).
    """
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/scan-result/batch",
            json={"results": items},
            headers=_backend_headers(),
            timeout=120,
        )
        if r.status_code == 200:
            return True, r.json().get("results", {}), None
        return False, {}, f"HTTP {r.status_code}: {r.text}"
    except requests.RequestException as exc:
        return False, {}, str(exc)


def _load_already_scored() -> set:
    """Dossiers déjà scorés, via le backend (mode --watch)."""
    try:
        r = requests.get(
            f"{BACKEND_URL}/pipeline/sessions/scanned-folders",
            headers=_backend_headers(), timeout=30,
        )
        if r.status_code == 200:
            return set(r.json().get("folders") or [])
        logger.warning("Impossible de charger les sessions déjà scorées [%s]: %s",
                        r.status_code, r.text)
    except requests.RequestException as exc:
        logger.warning("Impossible de charger les sessions déjà scorées : %s", exc)
    return set()


def _recalculate_kpis() -> None:
    """Déclenche le recalcul KPI côté backend (une seule requête d'agrégation)."""
    logger.info("Recalcul KPI en cours...")
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/kpis/recalculate",
            headers=_backend_headers(), timeout=60,
        )
        if r.status_code == 200:
            logger.info("Recalcul KPI terminé.")
        else:
            logger.warning("Recalcul KPI échoué [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logger.warning("Recalcul KPI : erreur réseau — %s", exc)


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


# ── Disques de stockage (SSD/HDD) ───────────────────────────────────────────────
# Répartit les sessions par disque physique (storage_disks). Code additif :
# n'interfère jamais avec le scoring / les KPI ci-dessus, et n'interrompt
# jamais la boucle principale en cas d'erreur (ex : disque pas encore monté).

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


def _scan_disks() -> None:
    """Point d'entrée disques : découverte locale (FS) + synchronisation via
    le backend (POST /api/storage-disks/sync). Ne lève jamais."""
    try:
        disks = _discover_disks()
        if not disks:
            return
        payload = [{**d, "folders": _list_sessions(d["mount_path"])} for d in disks]
        r = requests.post(
            f"{BACKEND_URL}/storage-disks/sync",
            json={"disks": payload},
            headers=_backend_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            logger.info("Disques : %d disque(s) synchronisé(s)", len(disks))
        else:
            logger.warning("Sync disques échouée [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logger.warning("Sync disques : erreur réseau — %s", exc)
    except Exception:
        logger.exception("Erreur lors du scan des disques")


# ── Bilan final ───────────────────────────────────────────────────────────────

def _print_summary(total_on_disk: int, valid_count: int,
                    total_duration_sec: float, clean_duration_sec: float,
                    elapsed: float) -> None:
    """Affiche la distribution des notes, le total sessions, le total
    d'heures et le détail heures propres / % sessions valides-invalides."""
    scored = a = b = c = d = f = 0
    avg = None
    try:
        r = requests.get(f"{BACKEND_URL}/quality-kpis", headers=_backend_headers(), timeout=30)
        if r.status_code == 200:
            data = r.json()
            grades = {g["grade"]: g["count"] for g in data.get("grade_distribution", [])}
            a, b, c, d, f = (grades.get(k, 0) for k in ("A", "B", "C", "D", "F"))
            scored = data.get("global", {}).get("scored_count") or 0
            avg    = data.get("global", {}).get("avg_score")
        else:
            logger.warning("Impossible de récupérer les stats [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logger.warning("Impossible de récupérer les stats : %s", exc)

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
      1. Liste tous les dossiers (tout retraiter, sans filtrage)
      2. ProcessPoolExecutor : lecture JSON + scoring en parallèle (local)
      3. ThreadPoolExecutor : POST de chaque résultat vers le backend
      4. Recalcul KPI final via le backend
    """
    logger.info("=== fs_scanner --once | workers=%d http_workers=%d dir=%s backend=%s ===",
                SCAN_WORKERS, HTTP_WORKERS, SESSIONS_DIR, BACKEND_URL)

    t0 = time.monotonic()

    # ── Étape 1 : liste FS (tout retraiter sans exception) ────────────────────
    to_scan = _list_sessions(SESSIONS_DIR)

    logger.info("Total dossiers : %d | mode full-rescan (aucun filtrage)", len(to_scan))

    if not to_scan:
        logger.info("Rien à faire.")
        return

    # ── Étape 2 : scan parallèle par chunks (local, sans réseau) ───────────────
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

    t_scan = time.monotonic()

    with ProcessPoolExecutor(max_workers=SCAN_WORKERS) as pool, \
            ThreadPoolExecutor(max_workers=HTTP_WORKERS) as http_pool:

        pending_http: dict = {}
        buffer: list = []  # items prêts à être envoyés, en attente d'un lot complet

        def flush_buffer():
            if not buffer:
                return
            batch = buffer.copy()
            buffer.clear()
            payload = [{k: v for k, v in item.items() if k != "_duration_sec"} for item in batch]
            fut = http_pool.submit(_post_scan_results_batch, payload)
            pending_http[fut] = batch

        def process_batch_result(fut):
            nonlocal ok, failed, valid_count, total_duration_sec, clean_duration_sec
            batch = pending_http.pop(fut)
            success, results, http_err = fut.result()
            for item in batch:
                folder_name  = item["folder_name"]
                duration_sec = item["_duration_sec"]
                item_result  = results.get(folder_name) if success else None
                if not item_result or not item_result.get("ok"):
                    logger.warning("%s : échec backend — %s", folder_name,
                                    http_err or (item_result or {}).get("error", "?"))
                    failed += 1
                    continue
                ok += 1
                total_duration_sec += duration_sec or 0
                if not item["errors"]:
                    valid_count += 1
                    clean_duration_sec += duration_sec or 0
            if (ok + failed) % 500 == 0:
                logger.info("  ... %d traités (%.1f/s)",
                            ok + failed,
                            (ok + failed) / max(1, time.monotonic() - t_scan))

        def drain_completed():
            for fut in [f for f in pending_http if f.done()]:
                process_batch_result(fut)

        for chunk_results in pool.map(_scan_chunk, chunks, chunksize=1):
            for folder_name, score, grade, errors, warnings_count, config, duration_sec, err in chunk_results:
                if err == "no_analysis":
                    no_analysis += 1
                    continue
                if err:
                    logger.debug("%s : erreur lecture — %s", folder_name, err)
                    skipped += 1
                    continue
                buffer.append({
                    "folder_name":     folder_name,
                    "score":           score,
                    "grade":           grade,
                    "errors":          errors,
                    "warnings_count":  warnings_count,
                    "size_bytes":      0,
                    "config":          config,
                    "_duration_sec":   duration_sec,
                })
                if len(buffer) >= HTTP_BATCH_SIZE:
                    flush_buffer()

            # Vide la fenêtre de lots HTTP terminés pour ne pas accumuler des
            # centaines de futures en mémoire sur un gros backlog.
            drain_completed()

        flush_buffer()
        for fut in as_completed(list(pending_http.keys())):
            process_batch_result(fut)

    t_db = time.monotonic()
    logger.info("Scan FS terminé en %.1fs | %d traités %d échecs backend "
                "%d sans analysis.json %d erreurs lecture",
                t_db - t_scan, ok, failed, no_analysis, skipped)

    # ── Étape 4 : recalcul KPI final ───────────────────────────────────────────
    _recalculate_kpis()

    # ── Étape 5 : répartition par disque (SSD/HDD) ─────────────────────────────
    _scan_disks()

    # ── Bilan final ────────────────────────────────────────────────────────────
    _print_summary(len(to_scan), valid_count, total_duration_sec,
                    clean_duration_sec, time.monotonic() - t0)


# ── Mode --watch : surveillance continue ──────────────────────────────────────

def _fetch_and_process(fname: str) -> tuple:
    """Lit analysis.json/config.json localement, évalue le score, puis
    POST le résultat au backend. Retourne (fname, success, msg)."""
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

    success, session_id, created, err = _post_scan_result(
        fname, score, grade, errors, len(warnings), size_bytes, config,
    )
    if not success:
        return fname, False, err

    flag = "ERREURS" if errors else f"{len(warnings)}w" if warnings else "OK"
    return fname, True, f"score={score} grade={grade} [{flag}]"


def watch_loop():
    """
    Surveillance continue du dossier sessions.
    Détecte les nouveaux dossiers, attend la stabilité d'analysis.json,
    puis traite avec un ThreadPoolExecutor.
    """
    logger.info("=== fs_scanner --watch | intervalle=%ds stabilité=%ds dir=%s backend=%s ===",
                SCAN_INTERVAL, STABILITY_SECONDS, SESSIONS_DIR, BACKEND_URL)

    processed: set   = _load_already_scored()
    # {folder_name: (size, mtime, first_stable_ts)}
    pending_stable: dict = {}
    pending_db:     set  = set()

    logger.info("%d session(s) déjà traitées", len(processed))

    last_disk_scan = 0.0
    last_kpi_recalc = 0.0

    while True:
        now = time.monotonic()

        # ── 0. Répartition par disque (SSD/HDD), throttlée ─────────────────────
        if now - last_disk_scan >= DISK_SCAN_INTERVAL:
            _scan_disks()
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
            with ThreadPoolExecutor(max_workers=min(HTTP_WORKERS, len(newly_stable))) as pool:
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
            with ThreadPoolExecutor(max_workers=min(HTTP_WORKERS, len(pending_db))) as pool:
                results = pool.map(_fetch_and_process, list(pending_db))
                for fname, success, msg in results:
                    if success:
                        processed.add(fname)
                    else:
                        logger.debug("%s : retry échoué — %s", fname, msg)
                        still_pending.add(fname)
            pending_db = still_pending

        # ── 4. Recalcul KPI, throttlé ────────────────────────────────────────────
        if newly_stable and now - last_kpi_recalc >= KPI_RECALC_INTERVAL:
            _recalculate_kpis()
            last_kpi_recalc = now

        # ── 5. Nouvelles sessions ───────────────────────────────────────────────
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
