"""
SessionsToMistral.py — Envoie les sessions complètes/valides vers Mistral.

Pipeline en continu : un pool de processus (ANALYZE_PROCESSES) analyse les
candidats au fil de l'eau (pas de passe d'analyse complète préalable), et un
pool de threads (UPLOAD_WORKERS) envoie les sessions valides dès qu'elles
sont prêtes. Les sessions invalides/rejetées/vides sont persistées dans le
fichier de dodge dès leur analyse, pour ne jamais être ré-analysées.

Une session dont la date (encodée dans le nom du dossier) n'est pas celle du
jour courant est écartée définitivement, AVANT toute analyse — jamais envoyée,
même si elle est par ailleurs valide (cf. session_date()/filtre "jour
courant" dans main()).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import date, datetime
from multiprocessing import cpu_count
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)

BASE_URL = "http://13.62.206.125:5001"
USERNAME = "pd_umi"
PASSWORD = "sqiu763hQP1"

# Backend interne (API pipeline) — pour marquer les sessions comme envoyées
BACKEND_URL        = os.environ.get("BACKEND_URL", "https://slouchy-trombone-bats.ngrok-free.dev/api")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "b1445bb1f44ca2ae6dab11a11cbd62b41805e02726b49c4f9e87199011176970")

DODGE_FILE = "uploaded_sessions.json"

# Client Mistral dans la BDD — configurable via env
MISTRAL_CLIENT_ID   = os.environ.get("DELIVERY_CLIENT_ID",   "mistral")
MISTRAL_CLIENT_NAME = os.environ.get("DELIVERY_CLIENT_NAME", "Mistral AI")

# Limite de volume par exécution (défaut 5 Go, surchargeable via MAX_RUN_GB)
MAX_RUN_BYTES = int(os.environ.get("MAX_RUN_GB", "5")) * 1024 ** 3

# Files that alone do not constitute a meaningful session
METADATA_ONLY_FILES = {"metadata.json"}

# Parallélisme — analyse (validation/intégrité) : multiprocessing, l'I/O
# (rglob, lecture JSON) domine et est répartie sur plusieurs processus
ANALYZE_PROCESSES = int(os.environ.get("ANALYZE_PROCESSES", str(min(4, max(1, cpu_count() or 1)))))

# Parallélisme — envoi (zip + upload réseau) : threads, l'I/O réseau libère le GIL
UPLOAD_WORKERS = int(os.environ.get("UPLOAD_WORKERS", "12"))

# Taille des lots envoyés au backend (register/mark-sent/mark-send-failed) :
# un seul appel HTTP toutes les BACKEND_BATCH_SIZE sessions plutôt qu'un
# appel par session, pour limiter le nombre de requêtes vers BACKEND_URL.
BACKEND_BATCH_SIZE = int(os.environ.get("BACKEND_BATCH_SIZE", "500"))


# ---------------------------------------------------------------------------
# Backend API
# ---------------------------------------------------------------------------

OFFLINE = False  # quand True : aucun appel réseau vers BACKEND_URL (register/mark-sent/mark-failed)


def _backend_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN
    return headers


# ---------------------------------------------------------------------------
# Validation/correction du config.json — cohérence rig <-> opérateur
# ---------------------------------------------------------------------------

def _expected_rig_num(operator_id: str | None) -> int | None:
    """
    Règle métier : rig_id = operator_id si 0-30, sinon operator_id - 30.
    'operator_id' est de la forme 'op_45'. Retourne None si non parsable.
    """
    m = re.search(r"(\d+)", operator_id or "")
    if not m:
        return None
    opnum = int(m.group(1))
    return opnum if opnum <= 30 else opnum - 30


def fix_rig_config(config: dict, session_name: str) -> bool:
    """
    Corrige config["rig"]["rig_id"]/["code"] uniquement dans les cas où le
    rig est absent ou vaut le placeholder par défaut du logiciel de capture
    ('rig_1') — un rig déjà renseigné avec une autre valeur est considéré
    comme volontaire et n'est jamais modifié, même incohérent avec la règle
    operator_id±30.
    Retourne True si une correction a été appliquée.
    """
    operator = config.get("operator") or {}
    rig = config.get("rig")

    current_rig_id = (rig or {}).get("rig_id") or ""
    m = re.search(r"(\d+)", current_rig_id)
    current_num = int(m.group(1)) if m else None

    if current_num is not None and current_num != 1:
        return False  # rig déjà renseigné avec une valeur volontaire — on ne touche pas

    expected = _expected_rig_num(operator.get("operator_id"))
    if expected is None:
        return False

    if current_num == expected:
        return False

    new_rig_id = f"rig_{expected}"
    new_code = f"RIG-{expected}"
    logging.warning(
        "  [%s] rig %s avec operator_id='%s' : '%s' → corrigé en '%s'",
        session_name,
        "absent" if rig is None else "manquant" if current_num is None else "== 1 (placeholder)",
        operator.get("operator_id"), current_rig_id or "(absent)", new_rig_id,
    )
    if rig is None:
        rig = {}
        config["rig"] = rig
    rig["rig_id"] = new_rig_id
    rig["code"] = new_code
    return True


def read_session_metadata(session_dir: Path, dry_run: bool = False) -> tuple[dict | None, dict | None, dict | None]:
    """
    Lit config.json, mission.json et analysis.json (requis pour le register).
    Si config.json est présent, vérifie/corrige la cohérence rig <-> opérateur
    (fix_rig_config) et réécrit le fichier sur disque si une correction a été
    appliquée et que dry_run est False — la correction doit être faite avant
    le zip pour se propager dans l'archive envoyée.

    Retourne (analysis, config, mission). 'analysis' est None si
    analysis.json est manquant/illisible (la session ne sera alors pas
    enregistrée via register-bulk, mais config est tout de même corrigé).
    """
    config = None
    config_path = session_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("  Impossible de lire config.json pour '%s': %s", session_dir.name, exc)
            config = None

    if config is not None and fix_rig_config(config, session_dir.name):
        if dry_run:
            logging.info("  [%s] dry-run — correction rig non écrite sur disque", session_dir.name)
        else:
            try:
                config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logging.warning("  Impossible d'écrire config.json corrigé pour '%s': %s", session_dir.name, exc)

    mission = None
    mission_path = session_dir / "mission.json"
    if mission_path.exists():
        try:
            mission = json.loads(mission_path.read_text(encoding="utf-8"))
        except Exception:
            mission = None

    analysis_path = session_dir / "analysis.json"
    if not analysis_path.exists():
        return None, config, mission
    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("  Impossible de lire analysis.json pour '%s': %s", session_dir.name, exc)
        return None, config, mission

    return analysis, config, mission


def db_register_sessions_bulk(items: list[dict]) -> dict[str, str | None]:
    """
    Enregistre/relie un lot de sessions en BDD via /register-bulk.
    items : [{"folder_name", "analysis", "config", "mission", "size_bytes"}, ...]
    Retourne folder_name -> session_id (None en cas d'échec pour cet item).
    """
    if OFFLINE or not items:
        return {}
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/register-bulk",
            json={"sessions": items},
            headers=_backend_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            out: dict[str, str | None] = {}
            for res in r.json().get("results", []):
                fname = res.get("folder_name")
                if res.get("error"):
                    logging.warning("  Backend erreur (register-bulk) '%s': %s", fname, res["error"])
                    out[fname] = None
                else:
                    out[fname] = res.get("session_id")
            return out
        logging.warning("  Backend erreur (register-bulk) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.warning("  Backend: échec register-bulk (%d session(s)) : %s", len(items), exc)
    return {item["folder_name"]: None for item in items}


def api_mark_sent_bulk(deliveries: list[dict]) -> None:
    """
    Marque un lot de sessions comme envoyées au client Mistral via /mark-sent-bulk.
    deliveries : [{"session_ref", "size_bytes", "duration_seconds"}, ...]
    """
    if OFFLINE or not deliveries:
        return
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/mark-sent-bulk",
            json={
                "client_id": MISTRAL_CLIENT_ID,
                "client_name": MISTRAL_CLIENT_NAME,
                "deliveries": deliveries,
            },
            headers=_backend_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            for res in r.json().get("results", []):
                if res.get("error"):
                    logging.warning("  Backend erreur (mark-sent-bulk) '%s': %s",
                                     res.get("session_ref"), res["error"])
            logging.info("  Backend: %d session(s) → delivered (client: %s)",
                         len(deliveries), MISTRAL_CLIENT_ID)
        else:
            logging.warning("  Backend erreur (mark-sent-bulk) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.error("  Backend erreur (mark-sent-bulk) : %s", exc)


def api_mark_send_failed_bulk(session_refs: list[str]) -> None:
    """Marque un lot d'envois comme échoués via /mark-send-failed-bulk."""
    if OFFLINE or not session_refs:
        return
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/mark-send-failed-bulk",
            json={"client_id": MISTRAL_CLIENT_ID, "session_refs": session_refs},
            headers=_backend_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            for res in r.json().get("results", []):
                if res.get("error"):
                    logging.warning("  Backend erreur (mark-send-failed-bulk) '%s': %s",
                                     res.get("session_ref"), res["error"])
            logging.info("  Backend: %d session(s) → delivery_failed", len(session_refs))
        else:
            logging.warning("  Backend erreur (mark-send-failed-bulk) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.error("  Backend erreur (mark-send-failed-bulk) : %s", exc)


def _fetch_delivered_folders(client_id: str) -> set[str] | None:
    """
    Ensemble des session_folder déjà livrés au client `client_id`, lu depuis
    la BDD (source de vérité cross-disque : une session livrée depuis un autre
    disque ou un run précédent y figure, même si son dossier est encore
    physiquement présent ici). Permet de ne jamais ré-envoyer une session déjà
    délivrée, indépendamment du fichier de dodge (local à un seul dossier
    racine) et du déplacement vers session_envoye/ (local à un seul arbre).

    Retourne None (et non set()) si OFFLINE ou si le backend est indisponible,
    pour que l'appelant distingue « aucune livraison » de « statut inconnu »
    (le cache TTL garde alors son dernier set connu au lieu de tout ré-ouvrir
    à l'envoi).
    """
    if OFFLINE:
        return None
    try:
        r = requests.get(
            f"{BACKEND_URL}/pipeline/sessions/delivered-folders",
            params={"client_id": client_id},
            headers=_backend_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            return set(r.json().get("folders") or [])
        logging.warning("  Backend erreur (delivered-folders) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.warning("  Backend: échec delivered-folders : %s", exc)
    return None


def fetch_delivered_folders(client_id: str) -> set[str]:
    """Version fail-open (set vide si OFFLINE/backend indisponible), pour les
    exécutions one-shot (SessionsToMistral.main, run_pipeline) : un seul appel
    au démarrage, on préfère risquer un ré-envoi plutôt que bloquer un envoi
    légitime faute d'avoir pu joindre la BDD."""
    return _fetch_delivered_folders(client_id) or set()


# Cache TTL du set des dossiers livrés, pour les services CONTINUS (ex.
# mission-pipeline/pipeline_service.py) qui tournent en permanence : un snapshot
# pris une seule fois deviendrait périmé (une session livrée par un AUTRE disque
# après le démarrage ne serait pas vue). On rafraîchit donc le set toutes les
# _DELIVERED_TTL secondes, en gardant un lookup O(1) par session. En cas
# d'échec de rafraîchissement, on conserve le dernier set connu.
_delivered_lock = threading.Lock()
_delivered_cache: dict = {"folders": None, "ts": 0.0}
_DELIVERED_TTL = float(os.environ.get("DELIVERED_FOLDERS_TTL", "120"))


def delivered_folders_cached(client_id: str) -> set[str]:
    """fetch_delivered_folders() mis en cache _DELIVERED_TTL secondes,
    thread-safe — pour les services continus qui vérifient session par session."""
    now = time.monotonic()
    with _delivered_lock:
        cached = _delivered_cache["folders"]
        if cached is None or now - _delivered_cache["ts"] > _DELIVERED_TTL:
            fresh = _fetch_delivered_folders(client_id)
            if fresh is not None:
                _delivered_cache["folders"] = fresh
                _delivered_cache["ts"] = now
            elif cached is None:
                _delivered_cache["folders"] = set()  # rien de connu encore
        return _delivered_cache["folders"] or set()


# ---------------------------------------------------------------------------
# Upload runs — suivi de cette exécution comme entité groupée côté backend
# (affichée dans la page /clients : statut validé/en cours/avorté + barre
# de progression, alimentée par des heartbeats périodiques).
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_SECONDS = 5


def start_upload_run(total_sessions: int) -> str | None:
    """Démarre le suivi du run. Retourne le run_id, ou None si indisponible."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/uploads/runs/start",
            json={"client_id": MISTRAL_CLIENT_ID, "total_sessions": total_sessions},
            headers=_backend_headers(),
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("run_id")
        logging.warning("  Backend erreur (run start) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.warning("  Backend: impossible de démarrer le suivi du run : %s", exc)
    return None


def send_run_progress(run_id: str, processed_count: int, sent_count: int, failed_count: int) -> None:
    """Heartbeat de progression — fire-and-forget, ne doit jamais bloquer le run."""
    try:
        requests.post(
            f"{BACKEND_URL}/pipeline/uploads/runs/{run_id}/progress",
            json={
                "processed_count": processed_count,
                "sent_count": sent_count,
                "failed_count": failed_count,
            },
            headers=_backend_headers(),
            timeout=10,
        )
    except requests.RequestException as exc:
        logging.warning("  Backend: échec de la mise à jour de progression (run %s) : %s", run_id, exc)


def finish_upload_run(run_id: str, status: str, processed_count: int, sent_count: int,
                       failed_count: int, error_msg: str | None = None) -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/pipeline/uploads/runs/{run_id}/finish",
            json={
                "status": status,
                "processed_count": processed_count,
                "sent_count": sent_count,
                "failed_count": failed_count,
                "error_msg": error_msg,
            },
            headers=_backend_headers(),
            timeout=10,
        )
    except requests.RequestException as exc:
        logging.warning("  Backend: échec de la clôture du run %s : %s", run_id, exc)


# ---------------------------------------------------------------------------
# Validation d'intégrité — structure complète de la session
# ---------------------------------------------------------------------------

# Taille minimale d'un MP4 valide (encodage non-corrompu)
MP4_MIN_BYTES = 100_000  # 100 KB


def validate_session(session_dir: Path) -> list[str]:
    """
    Vérifie que la session est structurellement complète et sans problème connu.
    Retourne une liste d'issues (vide = session valide).

    Checks :
      1. result.json existe et indique SUCCESS
      2. config.json lisible, caméras sans erreur
      3. mission.json présent
      4. analysis.json : sync_check.ok doit être True
      5. Pour chaque caméra (config) : <name>.mp4 ≥ 100 KB et <name>.jsonl non-vide
      6. cameras/resampled_30hz.jsonl non-vide
      7. Pour chaque capteur (config) : sensors/<name>.jsonl non-vide
    """
    issues: list[str] = []

    # 1. result.json
    result_path = session_dir / "result.json"
    if not result_path.exists():
        issues.append("result.json manquant")
    else:
        try:
            res = json.loads(result_path.read_text(encoding="utf-8"))
            result_val = str(res.get("result", "")).upper()
            if result_val != "SUCCESS":
                issues.append(f"result.json non-SUCCESS (valeur : '{res.get('result')}')")
        except Exception as exc:
            issues.append(f"result.json illisible : {exc}")

    # 2. config.json — lit les caméras et capteurs attendus
    config_path = session_dir / "config.json"
    expected_cameras: list[str] = []
    expected_sensors: list[str] = []
    if not config_path.exists():
        issues.append("config.json manquant")
    else:
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            for cam in cfg.get("cameras", []):
                name = cam.get("name")
                if not name:
                    continue
                expected_cameras.append(name)
                if cam.get("error"):
                    issues.append(f"caméra '{name}' : erreur hardware ({cam['error']})")
            for sen in cfg.get("sensors", []):
                name = sen.get("name")
                if name:
                    expected_sensors.append(name)
        except Exception as exc:
            issues.append(f"config.json illisible : {exc}")

    # 3. mission.json
    if not (session_dir / "mission.json").exists():
        issues.append("mission.json manquant")

    # 4. analysis.json — sync_check
    analysis_path = session_dir / "analysis.json"
    if analysis_path.exists():
        try:
            data = json.loads(analysis_path.read_text(encoding="utf-8"))
            sync = data.get("sync_check", {})
            if isinstance(sync.get("ok"), bool) and not sync["ok"]:
                delta = sync.get("delta_sec", "?")
                issues.append(f"sync_check échoué — delta={delta}s (caméras/capteurs désynchronisés)")
        except Exception:
            pass  # read_analysis_errors lèvera l'erreur si le fichier est corrompu

    # 5. Fichiers caméra
    cam_dir = session_dir / "cameras"
    for name in expected_cameras:
        mp4  = cam_dir / f"{name}.mp4"
        jsonl = cam_dir / f"{name}.jsonl"

        if not mp4.exists():
            issues.append(f"cameras/{name}.mp4 manquant")
        else:
            size = mp4.stat().st_size
            if size < MP4_MIN_BYTES:
                issues.append(
                    f"cameras/{name}.mp4 trop petit ({size} octets < {MP4_MIN_BYTES}) — "
                    "encodage probablement corrompu"
                )

        if not jsonl.exists():
            issues.append(f"cameras/{name}.jsonl manquant")
        elif jsonl.stat().st_size == 0:
            issues.append(f"cameras/{name}.jsonl vide — aucune frame enregistrée")

    # resampled_30hz.jsonl — produit par le post-processing
    resampled = cam_dir / "resampled_30hz.jsonl"
    if expected_cameras:  # seulement si des caméras sont attendues
        if not resampled.exists():
            issues.append("cameras/resampled_30hz.jsonl manquant — post-processing non terminé")
        elif resampled.stat().st_size == 0:
            issues.append("cameras/resampled_30hz.jsonl vide — resample échoué")

    # 6. Fichiers capteurs
    sen_dir = session_dir / "sensors"
    for name in expected_sensors:
        jsonl = sen_dir / f"{name}.jsonl"
        if not jsonl.exists():
            issues.append(f"sensors/{name}.jsonl manquant")
        elif jsonl.stat().st_size == 0:
            issues.append(f"sensors/{name}.jsonl vide — aucune donnée capteur")

    return issues


# ---------------------------------------------------------------------------
# Analysis.json — vérification des erreurs de capture
# ---------------------------------------------------------------------------

def read_analysis_errors(session_dir: Path) -> list[str]:
    """
    Lit analysis.json et retourne la liste des erreurs.
    Retourne [] si le fichier est absent, illisible, ou sans erreurs.
    """
    analysis_path = session_dir / "analysis.json"
    if not analysis_path.exists():
        return []
    try:
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
        errors = data.get("errors", [])
        return [str(e) for e in errors] if errors else []
    except Exception as exc:
        logging.warning("  Impossible de lire analysis.json pour '%s': %s", session_dir.name, exc)
        return []


# ---------------------------------------------------------------------------
# Session analysis — single rglob pass (size + empty check)
# ---------------------------------------------------------------------------

def analyze_session(session_dir: Path) -> tuple[bool, str, int]:
    """
    Single directory traversal.
    Returns (is_empty, reason, total_bytes).
    """
    try:
        all_files = [f for f in session_dir.rglob("*") if f.is_file()]
    except Exception as exc:
        return True, f"impossible de lister les fichiers : {exc}", 0

    if not all_files:
        return True, "dossier vide", 0

    total_bytes = 0
    data_bytes = 0
    data_count = 0
    for f in all_files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        if f.name.lower() not in METADATA_ONLY_FILES:
            data_bytes += size
            data_count += 1

    if data_count == 0:
        return True, "uniquement metadata.json, pas de données", 0
    if data_bytes == 0:
        return True, f"{data_count} fichier(s) de données mais tous vides (0 octet)", 0

    return False, "", total_bytes


# ---------------------------------------------------------------------------
# Filtre "jour courant" — n'envoie jamais une session capturée un jour
# précédent, même si elle vient seulement d'être détectée (rig en retard,
# resync SFTP...). Volontairement strict : une date illisible est traitée
# comme "pas aujourd'hui", jamais comme "OK par défaut".
# ---------------------------------------------------------------------------

_SESSION_DATE_RE = re.compile(r"^session_(\d{4})(\d{2})(\d{2})_\d{6}", re.IGNORECASE)


def session_date(name: str) -> date | None:
    """Date encodée dans le nom du dossier (session_YYYYMMDD_HHMMSS...).
    None si le format n'est pas reconnu ou la date invalide."""
    m = _SESSION_DATE_RE.match(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Analyse d'un candidat
# ---------------------------------------------------------------------------

def _analyze_one(session_dir_str: str) -> tuple:
    """
    Classifie une session candidate (lecture seule) :
      ("invalid",  name, issues)   — structure incomplète/corrompue
      ("rejected", name, errors)   — erreurs dans analysis.json
      ("empty",    name, reason)   — pas de données
      ("valid",    name, size)     — prête à être envoyée
    """
    s = Path(session_dir_str)

    issues = validate_session(s)
    if issues:
        return (s.name, "invalid", issues)

    errors = read_analysis_errors(s)
    if errors:
        return (s.name, "rejected", errors)

    is_empty, reason, size = analyze_session(s)
    if is_empty:
        return (s.name, "empty", reason)

    return (s.name, "valid", size)


# ---------------------------------------------------------------------------
# Dodge file helpers
# ---------------------------------------------------------------------------

def load_dodge(root: Path) -> dict:
    dodge_path = root / DODGE_FILE
    if dodge_path.exists():
        try:
            dodge = json.loads(dodge_path.read_text(encoding="utf-8"))
            dodge.setdefault("sessions", [])
            dodge.setdefault("skipped", [])
            return dodge
        except Exception:
            pass
    return {"sessions": [], "skipped": []}


def save_dodge(root: Path, dodge: dict) -> None:
    dodge_path = root / DODGE_FILE
    dodge_path.write_text(json.dumps(dodge, indent=2, ensure_ascii=False), encoding="utf-8")


def mark_uploaded(root: Path, dodge: dict, session_name: str, size_bytes: int, duration_seconds: float) -> None:
    dodge["sessions"].append({
        "name": session_name,
        "size_bytes": size_bytes,
        "duration_seconds": duration_seconds,
    })
    save_dodge(root, dodge)


def mark_skipped(root: Path, dodge: dict, session_name: str, kind: str, detail) -> None:
    dodge["skipped"].append({
        "name": session_name,
        "kind": kind,
        "detail": detail,
    })
    save_dodge(root, dodge)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def read_duration(session_dir: Path) -> float:
    """
    Durée totale d'une session, calculée depuis analysis.json : on prend le
    max des duration_sec de toutes les caméras (fps_check) et de tous les
    capteurs (sensor_check), c'est-à-dire le flux le plus long.
    """
    analysis_path = session_dir / "analysis.json"
    if not analysis_path.exists():
        return 0.0
    try:
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0

    durations = []

    cams = data.get("fps_check", {}).get("cameras", {})
    for c in (cams or {}).values():
        if isinstance(c, dict):
            durations.append(c.get("duration_sec") or 0)

    sensors = data.get("sensor_check", {}).get("sensors", {})
    for s in (sensors or {}).values():
        if isinstance(s, dict):
            durations.append(s.get("duration_sec") or 0)

    return max(durations, default=0.0)


def format_duration(total_seconds: float) -> str:
    h = int(total_seconds) // 3600
    m = (int(total_seconds) % 3600) // 60
    s = int(total_seconds) % 60
    return f"{h}h {m}m {s}s"


def format_size(total_bytes: int) -> str:
    gb = total_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f} Go"
    mb = total_bytes / (1024 ** 2)
    return f"{mb:.1f} Mo"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_zip_to_mistral(zip_path: str) -> bool:
    path = Path(zip_path)
    if not path.exists() or path.suffix.lower() != ".zip":
        print(f"  Erreur : '{zip_path}' n'est pas un fichier .zip valide")
        return False

    file_name = path.stem
    session = requests.Session()

    payload = {
        "username": USERNAME,
        "password": PASSWORD,
        "repo_id": file_name,
        "filename": path.name,
    }

    print(f"  Demande d'URL signée pour '{path.name}'...")
    try:
        r = session.post(
            url=f"{BASE_URL}/pd/upload",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"  Erreur de connexion : {e}")
        return False

    print(f"  STATUS: {r.status_code}")

    if r.status_code != 200:
        try:
            err = r.json().get("error", "Unknown error")
        except Exception:
            err = r.text
        print(f"  Erreur serveur : {err}")
        return False

    signed_url = r.json().get("url")
    if not signed_url:
        print("  Pas d'URL d'upload reçue.")
        return False

    print(f"  Upload de '{path.name}' ({format_size(path.stat().st_size)})...")
    try:
        with open(path, "rb") as f:
            response = session.put(
                signed_url,
                data=f,
                headers={"Content-Type": "application/zip"},
                timeout=300,
            )
    except requests.RequestException as e:
        print(f"  Erreur upload : {e}")
        return False

    if response.status_code in (200, 201, 204):
        print("  Upload réussi !")
        return True
    else:
        print(f"  Upload échoué {response.status_code}: {response.text}")
        return False


# ---------------------------------------------------------------------------
# Move after upload
# ---------------------------------------------------------------------------

def move_session_to_sent(session_dir: Path, sent_dir: Path) -> bool:
    """Moves session_dir into sent_dir. Returns True on success."""
    try:
        sent_dir.mkdir(parents=True, exist_ok=True)
        dest = sent_dir / session_dir.name
        if dest.exists():
            # Already present (previous partial move?) — remove and replace
            shutil.rmtree(dest)
        shutil.move(str(session_dir), str(dest))
        print(f"  Déplacé vers {dest}")
        return True
    except Exception as exc:
        print(f"  Avertissement : impossible de déplacer '{session_dir.name}' : {exc}")
        return False


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------

def zip_session(session_dir: Path, tmp_dir: Path) -> Path:
    """
    Crée tmp_dir/<session_dir.name>.zip, contenu sous '<session_dir.name>/...'
    (équivalent à shutil.make_archive). Implémentation manuelle car
    make_archive() fait os.chdir() — non thread-safe avec UPLOAD_WORKERS > 1
    (plusieurs threads se marchent sur le cwd du process, d'où
    'FileNotFoundError: ... sessions').

    ZIP_STORED (pas de compression) : le contenu est presque exclusivement
    des .mp4 déjà compressés, donc ZIP_DEFLATED ne réduisait quasiment pas
    la taille tout en consommant du CPU pour rien — le serveur destinataire
    attend un .zip, peu importe sa méthode de compression interne.
    """
    zip_path = tmp_dir / f"{session_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for f in session_dir.rglob("*"):
            if f.is_file():
                zf.write(f, Path(session_dir.name) / f.relative_to(session_dir))
    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Upload sessions to Mistral")
    parser.add_argument("dossier", help="Dossier racine contenant les sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse et affiche ce qui serait envoyé, sans rien uploader ni déplacer")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Nombre maximum de sessions à envoyer par exécution (0 = illimité)")
    parser.add_argument("--all", action="store_true",
                        help="Envoie toutes les sessions valides, sans plafond de volume "
                             "(ignore MAX_RUN_GB). Combinable avec --max-sessions.")
    parser.add_argument("--workers", type=int, default=UPLOAD_WORKERS,
                        help=f"Nombre de threads d'upload en parallèle (défaut : {UPLOAD_WORKERS}, "
                             "surchargeable aussi via UPLOAD_WORKERS)")
    parser.add_argument("--analyze-processes", type=int, default=ANALYZE_PROCESSES,
                        help=f"Nombre de processus d'analyse en parallèle (défaut : {ANALYZE_PROCESSES})")
    parser.add_argument("--offline", action="store_true",
                        help="Désactive tous les appels réseau vers BACKEND_URL "
                             "(register/mark-sent/mark-send-failed). Le suivi des sessions "
                             "envoyées reste fait localement via le fichier de dodge. "
                             "À utiliser quand BACKEND_URL est lent/indisponible (ex. ngrok).")
    parser.add_argument("--only", default=None, metavar="MANIFEST",
                        help="Fichier manifeste (un nom de dossier de session par ligne) : "
                             "n'envoie QUE ces sessions. Utilisé pour les lots validés "
                             "(build_daily_batch.py) — désactive aussi le filtre 'jour courant', "
                             "le lot ayant déjà été curé.")
    args = parser.parse_args()

    global OFFLINE
    OFFLINE = args.offline

    dry_run = args.dry_run
    max_sessions = args.max_sessions
    send_all = args.all
    upload_workers = max(1, args.workers)
    analyze_processes = max(1, args.analyze_processes)
    max_run_bytes = float("inf") if send_all else MAX_RUN_BYTES
    root = Path(args.dossier)

    if not root.is_dir():
        print(f"Erreur : '{root}' n'est pas un dossier valide")
        sys.exit(1)

    if dry_run:
        print("*** MODE DRY-RUN — aucun upload, aucun déplacement, aucune écriture DB/persistance ***\n")
    if OFFLINE:
        print("*** MODE OFFLINE — aucun appel à BACKEND_URL (register/mark-sent/mark-send-failed) ***\n")

    sent_dir = root.parent / "session_envoye"

    dodge = load_dodge(root)
    sent_names    = {e["name"] for e in dodge["sessions"]}
    skipped_names = {e["name"] for e in dodge["skipped"]}
    already_done  = sent_names | skipped_names
    # Source de vérité cross-disque : dossiers déjà livrés à Mistral en BDD,
    # quel que soit le disque. Traités juste après la sélection des candidats
    # (déplacés hors du disque sans ré-upload — cf. plus bas).
    delivered_names = fetch_delivered_folders(MISTRAL_CLIENT_ID)  # {} si OFFLINE

    # Mode lot validé (--only) : on restreint aux sessions listées dans le
    # manifeste et on saute le filtre "jour courant" (le lot a déjà été curé
    # par build_daily_batch.py, qui a appliqué le filtre rig + jour en amont).
    only_names: set[str] | None = None
    if args.only:
        only_path = Path(args.only)
        if not only_path.is_file():
            print(f"Erreur : manifeste '{only_path}' introuvable", flush=True)
            sys.exit(1)
        only_names = {
            line.strip() for line in only_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        if not only_names:
            print(f"Manifeste '{only_path}' vide — rien à envoyer", flush=True)
            sys.exit(0)
        print(f"Mode --only : {len(only_names)} session(s) ciblée(s) par le manifeste", flush=True)

    print(f"Lecture de '{root}'...", flush=True)
    all_sessions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("session"))
    if only_names is not None:
        all_sessions = [s for s in all_sessions if s.name in only_names]
        missing = only_names - {s.name for s in all_sessions}
        for name in sorted(missing):
            print(f"  [{name}] ABSENTE du dossier — ignorée", flush=True)

    if not all_sessions:
        print(f"Aucun dossier 'session*' trouvé dans '{root}'", flush=True)
        sys.exit(0)

    candidates = [s for s in all_sessions if s.name not in already_done]

    # Sessions déjà livrées à Mistral depuis un autre disque (BDD) mais encore
    # physiquement présentes ici : jamais ré-uploadées, sorties du disque sans
    # ré-envoi. (En dry-run : signalées, ni déplacées ni marquées.)
    if delivered_names:
        delivered_here = [s for s in candidates if s.name in delivered_names]
        for s in delivered_here:
            print(f"  [{s.name}] DÉJÀ LIVRÉE à Mistral (BDD, autre disque) — déplacée sans ré-upload", flush=True)
            if not dry_run:
                mark_skipped(root, dodge, s.name, "already_sent_elsewhere",
                             "delivered_at déjà renseigné en BDD pour le client mistral")
                move_session_to_sent(s, sent_dir)
        if delivered_here:
            print(flush=True)
        candidates = [s for s in candidates if s.name not in delivered_names]

    if only_names is None:
        # Filtre "jour courant" — appliqué avant tout le reste (analyse, upload) :
        # une session d'un jour précédent est écartée définitivement, jamais
        # envoyée, même si elle est par ailleurs structurellement valide.
        today = datetime.now().date()
        same_day, other_day = [], []
        for s in candidates:
            (same_day if session_date(s.name) == today else other_day).append(s)

        for s in other_day:
            d = session_date(s.name)
            reason = f"session du {d.isoformat()}, pas du jour courant ({today.isoformat()})" if d \
                else "date illisible dans le nom du dossier"
            print(f"  [{s.name}] ECARTEE (jour) — {reason}", flush=True)
            if not dry_run:
                mark_skipped(root, dodge, s.name, "wrong_day", reason)
        if other_day:
            print(flush=True)

        candidates = same_day
    else:
        other_day = []

    print(f"{len(all_sessions)} dossier(s) — {len(sent_names)} déjà envoyé(s), "
          f"{len(skipped_names)} déjà écarté(s) (problème connu), "
          f"{len(other_day)} écarté(s) (jour précédent). "
          f"{len(candidates)} candidat(s) à traiter.", flush=True)
    print(flush=True)

    sent_count   = 0   # réservation pour le quota (sessions soumises à l'envoi)
    capped_count = 0
    cumul_bytes  = 0   # réservation pour le quota
    processed_count = 0
    sent_this_run: list[tuple[str, int, float]] = []
    failed_names: list[str] = []

    # Sessions uploadées en attente de synchronisation backend (register +
    # mark-sent/mark-send-failed), regroupées par lots de BACKEND_BATCH_SIZE
    # pour limiter le nombre d'appels HTTP vers BACKEND_URL.
    pending_backend: list[dict] = []

    def flush_backend_batch() -> None:
        if not pending_backend or OFFLINE:
            pending_backend.clear()
            return

        batch = list(pending_backend)
        pending_backend.clear()

        register_items = [
            {
                "folder_name": e["name"],
                "analysis": e["analysis"],
                "config": e["config"],
                "mission": e["mission"],
                "size_bytes": e["size"],
            }
            for e in batch if e["analysis"] is not None
        ]
        session_ids = db_register_sessions_bulk(register_items)

        delivered = []
        failed_refs = []
        for e in batch:
            ref = session_ids.get(e["name"]) or e["name"]
            if e["status"] == "ok":
                delivered.append({
                    "session_ref": ref,
                    "size_bytes": e["zip_size"],
                    "duration_seconds": e["duration"],
                })
            else:
                failed_refs.append(ref)

        api_mark_sent_bulk(delivered)
        api_mark_send_failed_bulk(failed_refs)

        print(f"  [batch backend] {len(batch)} session(s) synchronisée(s) "
              f"({len(register_items)} register, {len(delivered)} mark-sent, "
              f"{len(failed_refs)} mark-send-failed)", flush=True)

    total = len(candidates)
    candidates_iter = iter(candidates)

    # Si UPLOAD_RUN_ID est fourni (par send_validated_batches), on l'utilise
    # tel quel pour relier ce run au lot validé ; sinon on en crée un nouveau.
    if dry_run or OFFLINE:
        run_id = None
    else:
        run_id = os.environ.get("UPLOAD_RUN_ID") or start_upload_run(total)
    last_heartbeat_ts = time.monotonic()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir, \
                ProcessPoolExecutor(max_workers=analyze_processes) as analysis_pool, \
                ThreadPoolExecutor(max_workers=upload_workers) as upload_pool:

            def _upload_one(session_dir: Path, size: int) -> tuple:
                name = session_dir.name
                # Lu/corrigé même en OFFLINE : la correction du rig doit se propager
                # dans l'archive envoyée à Mistral, indépendamment du suivi backend.
                analysis, config, mission = read_session_metadata(session_dir, dry_run=dry_run)
                duration = read_duration(session_dir)

                if dry_run:
                    return (session_dir, size, duration, analysis, config, mission, "dry-run", 0)

                if not OFFLINE and analysis is None:
                    print(f"  [{name}] Avertissement : analysis.json manquant/illisible — upload sans mise à jour DB", flush=True)

                zip_path = zip_session(session_dir, Path(tmp_dir))
                zip_size = zip_path.stat().st_size
                print(f"  [{name}] Archive : {zip_path.name}  ({format_size(zip_size)})", flush=True)

                success = upload_zip_to_mistral(str(zip_path))
                zip_path.unlink(missing_ok=True)

                return (session_dir, size, duration, analysis, config, mission,
                         "ok" if success else "failed", zip_size)

            # pending : future -> ("analyze", session_dir) | ("upload",)
            pending: dict = {}

            def submit_next_analysis() -> None:
                try:
                    s = next(candidates_iter)
                except StopIteration:
                    return
                fut = analysis_pool.submit(_analyze_one, str(s))
                pending[fut] = ("analyze", s)

            # Amorce le pipeline d'analyse (fenêtre glissante, pas d'attente globale)
            for _ in range(max(analyze_processes * 2, 1)):
                submit_next_analysis()

            while pending:
                # timeout = intervalle de heartbeat : la boucle se réveille
                # régulièrement même si aucun upload n'a fini, pour continuer à
                # envoyer des heartbeats pendant les gros uploads (sinon le run
                # est marqué "avorté" à tort côté backend).
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED,
                               timeout=HEARTBEAT_INTERVAL_SECONDS)

                if run_id and time.monotonic() - last_heartbeat_ts >= HEARTBEAT_INTERVAL_SECONDS:
                    send_run_progress(run_id, processed_count, len(sent_this_run), len(failed_names))
                    last_heartbeat_ts = time.monotonic()

                for fut in done:
                    tag, *rest = pending.pop(fut)

                    if tag == "analyze":
                        session_dir, = rest
                        name = session_dir.name
                        _, kind, detail = fut.result()
                        processed_count += 1
                        print(f"[{processed_count}/{total}] {name}", flush=True)

                        if kind == "invalid":
                            print("  INVALIDE — structure incomplète/corrompue :")
                            for issue in detail:
                                print(f"    - {issue}")
                            if not dry_run:
                                mark_skipped(root, dodge, name, kind, detail)
                            print(flush=True)
                            submit_next_analysis()
                            continue

                        if kind == "rejected":
                            print("  REJET — erreurs dans analysis.json :")
                            for err in detail:
                                print(f"    - {err}")
                            if not dry_run:
                                mark_skipped(root, dodge, name, kind, detail)
                            print(flush=True)
                            submit_next_analysis()
                            continue

                        if kind == "empty":
                            print(f"  VIDE — {detail}")
                            if not dry_run:
                                mark_skipped(root, dodge, name, kind, detail)
                            print(flush=True)
                            submit_next_analysis()
                            continue

                        # kind == "valid" → detail = taille en octets
                        size = detail
                        if (max_sessions > 0 and sent_count >= max_sessions) or cumul_bytes + size > max_run_bytes:
                            capped_count += 1
                            print(f"  Plafond atteint — reporté au prochain run ({format_size(size)})", flush=True)
                            print(flush=True)
                            submit_next_analysis()
                            continue

                        sent_count += 1
                        cumul_bytes += size
                        ufut = upload_pool.submit(_upload_one, session_dir, size)
                        pending[ufut] = ("upload",)
                        submit_next_analysis()
                        continue

                    # tag == "upload"
                    session_dir, size, duration, analysis, config, mission, status, zip_size = fut.result()
                    name = session_dir.name

                    if status == "dry-run":
                        print(f"  [{name}] VALIDE — serait envoyée ({format_size(size)}, {format_duration(duration)})")
                        if analysis is not None:
                            print("           DB → serait enregistrée (analysis.json présent)")
                        else:
                            print("           DB → introuvable (upload sans mise à jour DB)")
                        print(flush=True)
                        continue

                    if status == "ok":
                        mark_uploaded(root, dodge, name, zip_size, duration)
                        move_session_to_sent(session_dir, sent_dir)
                        sent_this_run.append((name, zip_size, duration))
                    else:
                        failed_names.append(name)

                    pending_backend.append({
                        "name": name,
                        "analysis": analysis,
                        "config": config,
                        "mission": mission,
                        "size": size,
                        "duration": duration,
                        "status": status,
                        "zip_size": zip_size,
                    })
                    if len(pending_backend) >= BACKEND_BATCH_SIZE:
                        flush_backend_batch()

                    print(flush=True)

            flush_backend_batch()
    except BaseException as exc:
        if run_id:
            finish_upload_run(run_id, "aborted", processed_count,
                               len(sent_this_run), len(failed_names), error_msg=str(exc))
        raise

    if run_id:
        finish_upload_run(run_id, "completed", processed_count,
                           len(sent_this_run), len(failed_names))

    if dry_run:
        print("*** DRY-RUN terminé — rien n'a été modifié ***")
        sys.exit(0)

    sent_bytes = sum(sz for _, sz, _ in sent_this_run)
    print("=== Résumé de cette exécution ===")
    print(f"  Sessions envoyées : {len(sent_this_run)}  ({format_size(sent_bytes)})")
    for name, sz, dur in sent_this_run:
        print(f"  OK     {name}  ({format_size(sz)}, {format_duration(dur)})")
    for name in failed_names:
        print(f"  ECHEC  {name}")
    if capped_count:
        print(f"  REPORT {capped_count} session(s) — plafond atteint, prochains runs")

    total_bytes   = sum(e["size_bytes"] for e in dodge["sessions"])
    total_seconds = sum(e["duration_seconds"] for e in dodge["sessions"])

    print()
    print("=== Cumul total envoyé (toutes exécutions) ===")
    print(f"  Sessions envoyées : {len(dodge['sessions'])}")
    print(f"  Volume total      : {format_size(total_bytes)}")
    print(f"  Durée totale      : {format_duration(total_seconds)}")

    sys.exit(0 if not failed_names else 1)


if __name__ == "__main__":
    main()
