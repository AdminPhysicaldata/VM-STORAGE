import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import requests

logging.basicConfig(level=logging.INFO)

BASE_URL = "http://13.62.206.125:5001"
USERNAME = "pd_umi"
PASSWORD = "sqiu763hQP1"

DODGE_FILE = "uploaded_sessions.json"

# Files that alone do not constitute a meaningful session
METADATA_ONLY_FILES = {"metadata.json"}


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgresql"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "robotics"),
        user=os.environ.get("POSTGRES_USER", "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", "robotics123"),
        connect_timeout=10,
    )


def _now():
    return datetime.now(timezone.utc)


def db_mark_delivering(session_id: str, size_bytes: int) -> bool:
    """Transition captured/treatment_done/qa_accepted/delivery_pending → delivering."""
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions
                    SET pipeline_status    = 'delivering',
                        delivering_at      = COALESCE(delivering_at, %s),
                        delivery_pending_at= COALESCE(delivery_pending_at, %s),
                        size_bytes         = COALESCE(size_bytes, %s)
                    WHERE session_id = %s
                    RETURNING session_id
                """, (_now(), _now(), size_bytes, session_id))
                updated = cur.fetchone() is not None
        conn.close()
        if updated:
            logging.info("  DB: %s → delivering", session_id)
        else:
            logging.warning("  DB: session '%s' introuvable, statut non mis à jour", session_id)
        return updated
    except Exception as exc:
        logging.error("  DB erreur (mark_delivering) : %s", exc)
        return False


def db_mark_delivered(session_id: str, size_bytes: int, duration_seconds: float) -> bool:
    """Marque la session comme livrée."""
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions
                    SET pipeline_status  = 'delivered',
                        delivered_at     = %s,
                        size_bytes       = COALESCE(size_bytes, %s),
                        duration_seconds = COALESCE(duration_seconds, %s)
                    WHERE session_id = %s
                    RETURNING session_id
                """, (_now(), size_bytes, duration_seconds or None, session_id))
                updated = cur.fetchone() is not None
        conn.close()
        if updated:
            logging.info("  DB: %s → delivered", session_id)
        else:
            logging.warning("  DB: session '%s' introuvable", session_id)
        return updated
    except Exception as exc:
        logging.error("  DB erreur (mark_delivered) : %s", exc)
        return False


def db_mark_delivery_failed(session_id: str) -> None:
    """Marque la session comme échouée."""
    try:
        conn = _pg_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions
                    SET pipeline_status  = 'delivery_failed',
                        delivery_failed_at = %s
                    WHERE session_id = %s
                """, (_now(), session_id))
        conn.close()
        logging.info("  DB: %s → delivery_failed", session_id)
    except Exception as exc:
        logging.error("  DB erreur (mark_delivery_failed) : %s", exc)


# ---------------------------------------------------------------------------
# Empty session detection
# ---------------------------------------------------------------------------

def is_session_empty(session_dir: Path) -> tuple[bool, str]:
    """
    Returns (is_empty, reason).
    A session is considered empty when it has no data files beyond metadata.json,
    or when all data files add up to 0 bytes.
    """
    try:
        all_files = [f for f in session_dir.rglob("*") if f.is_file()]
    except Exception as exc:
        return True, f"impossible de lister les fichiers : {exc}"

    if not all_files:
        return True, "dossier vide"

    data_files = [f for f in all_files if f.name.lower() not in METADATA_ONLY_FILES]
    if not data_files:
        return True, "uniquement metadata.json, pas de données"

    total_data_bytes = sum(f.stat().st_size for f in data_files)
    if total_data_bytes == 0:
        return True, f"{len(data_files)} fichier(s) de données mais tous vides (0 octet)"

    return False, ""


# ---------------------------------------------------------------------------
# Dodge file helpers
# ---------------------------------------------------------------------------

def load_dodge(root: Path) -> dict:
    dodge_path = root / DODGE_FILE
    if dodge_path.exists():
        try:
            return json.loads(dodge_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sessions": []}


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


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def read_duration(session_dir: Path) -> float:
    meta = session_dir / "metadata.json"
    if not meta.exists():
        return 0.0
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return float(data.get("duration_seconds", 0))
    except Exception:
        return 0.0


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
    zip_base = tmp_dir / session_dir.name
    archive = shutil.make_archive(
        str(zip_base), "zip",
        root_dir=session_dir.parent,
        base_dir=session_dir.name,
    )
    return Path(archive)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python SessionsToMistral.py <dossier_racine>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Erreur : '{root}' n'est pas un dossier valide")
        sys.exit(1)

    sent_dir = root.parent / "session_envoye"

    dodge = load_dodge(root)
    already_done = {e["name"] for e in dodge["sessions"]}

    all_sessions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("session"))

    if not all_sessions:
        print(f"Aucun dossier 'session*' trouvé dans '{root}'")
        sys.exit(0)

    # --- Filtrage : sessions vides et déjà envoyées ---
    empty_sessions = []
    valid_sessions = []
    for s in all_sessions:
        empty, reason = is_session_empty(s)
        if empty:
            empty_sessions.append((s.name, reason))
        else:
            valid_sessions.append(s)

    to_send = [s for s in valid_sessions if s.name not in already_done]
    skipped = [s.name for s in valid_sessions if s.name in already_done]

    print(f"{len(all_sessions)} session(s) trouvée(s) au total :")
    print(f"  {len(empty_sessions)} vide(s) — ignorées")
    print(f"  {len(skipped)} déjà envoyée(s) — ignorées")
    print(f"  {len(to_send)} à envoyer")

    if empty_sessions:
        print("\nSessions vides :")
        for name, reason in empty_sessions:
            print(f"  VIDE  {name}  ({reason})")

    if skipped:
        print(f"\nIgnorées (dodge) : {skipped}")
    print()

    if not to_send:
        print("Aucune session à envoyer.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, session_dir in enumerate(to_send, 1):
                session_id = session_dir.name
                print(f"[{i}/{len(to_send)}] '{session_id}'")

                zip_path = zip_session(session_dir, Path(tmp_dir))
                zip_size = zip_path.stat().st_size
                print(f"  Archive : {zip_path.name}  ({format_size(zip_size)})")

                db_mark_delivering(session_id, zip_size)

                success = upload_zip_to_mistral(str(zip_path))

                if success:
                    duration = read_duration(session_dir)
                    db_mark_delivered(session_id, zip_size, duration)
                    mark_uploaded(root, dodge, session_id, zip_size, duration)
                    move_session_to_sent(session_dir, sent_dir)
                else:
                    db_mark_delivery_failed(session_id)

                print()

    # --- Résumé global depuis le dodge file ---
    total_bytes = sum(e["size_bytes"] for e in dodge["sessions"])
    total_seconds = sum(e["duration_seconds"] for e in dodge["sessions"])
    sent_this_run = [e for e in dodge["sessions"] if e["name"] not in already_done]
    failed_this_run = [s.name for s in to_send if s.name not in {e["name"] for e in dodge["sessions"]}]

    print("=== Résumé de cette exécution ===")
    for e in sent_this_run:
        print(f"  OK     {e['name']}  ({format_size(e['size_bytes'])}, {format_duration(e['duration_seconds'])})")
    for name in failed_this_run:
        print(f"  ECHEC  {name}")
    for name, reason in empty_sessions:
        print(f"  VIDE   {name}  ({reason})")

    print()
    print("=== Cumul total envoyé (toutes exécutions) ===")
    print(f"  Sessions envoyées : {len(dodge['sessions'])}")
    print(f"  Volume total      : {format_size(total_bytes)}")
    print(f"  Durée totale      : {format_duration(total_seconds)}")

    sys.exit(0 if not failed_this_run else 1)


if __name__ == "__main__":
    main()
