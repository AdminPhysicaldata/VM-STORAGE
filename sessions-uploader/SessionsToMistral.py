"""
SessionsToMistral.py — Envoie les sessions complètes/valides vers Mistral.

Pipeline en continu : un pool de processus (ANALYZE_PROCESSES) analyse les
candidats au fil de l'eau (pas de passe d'analyse complète préalable), et un
pool de threads (UPLOAD_WORKERS) envoie les sessions valides dès qu'elles
sont prêtes. Les sessions invalides/rejetées/vides sont persistées dans le
fichier de dodge dès leur analyse, pour ne jamais être ré-analysées.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
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
UPLOAD_WORKERS = int(os.environ.get("UPLOAD_WORKERS", "3"))


# ---------------------------------------------------------------------------
# Backend API
# ---------------------------------------------------------------------------

def db_register_session(session_dir: Path) -> str | None:
    """
    Enregistre/relie la session en BDD via le backend, comme le fait
    sftp_scanner.py (résolution ou création, scoring depuis analysis.json,
    mise à jour session_folder/size_bytes/quality_score/pipeline_status).

    Lit analysis.json (requis) et config.json (optionnel, pour la création
    si la session n'existe pas encore en BDD). Retourne le session_id DB,
    ou None en cas d'échec / analysis.json manquant.
    """
    analysis_path = session_dir / "analysis.json"
    if not analysis_path.exists():
        return None

    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("  Impossible de lire analysis.json pour '%s': %s", session_dir.name, exc)
        return None

    config = None
    config_path = session_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = None

    mission = None
    mission_path = session_dir / "mission.json"
    if mission_path.exists():
        try:
            mission = json.loads(mission_path.read_text(encoding="utf-8"))
        except Exception:
            mission = None

    size_bytes = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())

    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/register",
            json={
                "folder_name": session_dir.name,
                "analysis": analysis,
                "config": config,
                "mission": mission,
                "size_bytes": size_bytes,
            },
            headers=_backend_headers(),
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("session_id")
        logging.warning("  Backend erreur (register) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.warning("  Backend: impossible d'enregistrer la session '%s': %s", session_dir.name, exc)
    return None


def _backend_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN
    return headers


def api_mark_sent(session_ref: str, size_bytes: int, duration_seconds: float) -> bool:
    """
    Appelle le backend pour marquer la session comme envoyée au client Mistral.
    'session_ref' : session_id DB ou nom de dossier NAS (session_folder).
    """
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/{session_ref}/mark-sent",
            json={
                "client_id": MISTRAL_CLIENT_ID,
                "client_name": MISTRAL_CLIENT_NAME,
                "size_bytes": size_bytes,
                "duration_seconds": duration_seconds,
            },
            headers=_backend_headers(),
            timeout=10,
        )
        if r.status_code == 200:
            logging.info("  Backend: %s → delivered (client: %s)", session_ref, MISTRAL_CLIENT_ID)
            return True
        logging.warning("  Backend erreur (mark-sent) [%s]: %s", r.status_code, r.text)
        return False
    except requests.RequestException as exc:
        logging.error("  Backend erreur (mark-sent) : %s", exc)
        return False


def api_mark_send_failed(session_ref: str) -> None:
    """Appelle le backend pour marquer l'envoi de la session comme échoué."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/pipeline/sessions/{session_ref}/mark-send-failed",
            json={"client_id": MISTRAL_CLIENT_ID},
            headers=_backend_headers(),
            timeout=10,
        )
        if r.status_code == 200:
            logging.info("  Backend: %s → delivery_failed", session_ref)
        else:
            logging.warning("  Backend erreur (mark-send-failed) [%s]: %s", r.status_code, r.text)
    except requests.RequestException as exc:
        logging.error("  Backend erreur (mark-send-failed) : %s", exc)


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
    """
    zip_path = tmp_dir / f"{session_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
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
    args = parser.parse_args()

    dry_run = args.dry_run
    max_sessions = args.max_sessions
    send_all = args.all
    max_run_bytes = float("inf") if send_all else MAX_RUN_BYTES
    root = Path(args.dossier)

    if not root.is_dir():
        print(f"Erreur : '{root}' n'est pas un dossier valide")
        sys.exit(1)

    if dry_run:
        print("*** MODE DRY-RUN — aucun upload, aucun déplacement, aucune écriture DB/persistance ***\n")

    sent_dir = root.parent / "session_envoye"

    dodge = load_dodge(root)
    sent_names    = {e["name"] for e in dodge["sessions"]}
    skipped_names = {e["name"] for e in dodge["skipped"]}
    already_done  = sent_names | skipped_names

    print(f"Lecture de '{root}'...", flush=True)
    all_sessions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("session"))

    if not all_sessions:
        print(f"Aucun dossier 'session*' trouvé dans '{root}'", flush=True)
        sys.exit(0)

    candidates = [s for s in all_sessions if s.name not in already_done]

    print(f"{len(all_sessions)} dossier(s) — {len(sent_names)} déjà envoyé(s), "
          f"{len(skipped_names)} déjà écarté(s) (problème connu). "
          f"{len(candidates)} candidat(s) à traiter.", flush=True)
    print(flush=True)

    sent_count   = 0   # réservation pour le quota (sessions soumises à l'envoi)
    capped_count = 0
    cumul_bytes  = 0   # réservation pour le quota
    processed_count = 0
    sent_this_run: list[tuple[str, int, float]] = []
    failed_names: list[str] = []

    total = len(candidates)
    candidates_iter = iter(candidates)

    with tempfile.TemporaryDirectory() as tmp_dir, \
            ProcessPoolExecutor(max_workers=ANALYZE_PROCESSES) as analysis_pool, \
            ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as upload_pool:

        def _upload_one(session_dir: Path, size: int) -> tuple:
            name = session_dir.name
            db_session_id = db_register_session(session_dir)
            duration = read_duration(session_dir)

            if dry_run:
                return (session_dir, size, duration, db_session_id, "dry-run", 0)

            if db_session_id:
                print(f"  [{name}] DB session_id : {db_session_id}", flush=True)
            else:
                print(f"  [{name}] Avertissement : session introuvable/non enregistrée en DB — upload sans mise à jour DB", flush=True)

            zip_path = zip_session(session_dir, Path(tmp_dir))
            zip_size = zip_path.stat().st_size
            print(f"  [{name}] Archive : {zip_path.name}  ({format_size(zip_size)})", flush=True)

            success = upload_zip_to_mistral(str(zip_path))
            zip_path.unlink(missing_ok=True)

            session_ref = db_session_id or name
            if success:
                api_mark_sent(session_ref, zip_size, duration)
            else:
                api_mark_send_failed(session_ref)

            return (session_dir, size, duration, db_session_id, "ok" if success else "failed", zip_size)

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
        for _ in range(max(ANALYZE_PROCESSES * 2, 1)):
            submit_next_analysis()

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
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
                session_dir, size, duration, db_session_id, status, zip_size = fut.result()
                name = session_dir.name

                if status == "dry-run":
                    print(f"  [{name}] VALIDE — serait envoyée ({format_size(size)}, {format_duration(duration)})")
                    if db_session_id:
                        print(f"           DB → {db_session_id}")
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

                print(flush=True)

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
