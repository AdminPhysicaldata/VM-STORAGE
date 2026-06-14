"""
SessionsToMail.py — Envoie un échantillon de sessions robotiques par email.

À chaque exécution (toutes les heures via cron) :
  - sélectionne au plus 1 session non-encore-envoyée par opérateur
  - valide chaque session (mêmes checks que SessionsToMistral.py)
  - regroupe les sessions sélectionnées dans une unique archive ZIP
  - envoie UN email (avec ZIP joint) à une liste de destinataires codée en dur

Utilise un fichier dodge séparé (uploaded_sessions_mail.json) — n'interfère
pas avec le pipeline Mistral. Les sessions ne sont pas déplacées.
"""

import argparse
import json
import logging
import os
import smtplib
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from SessionsToMistral import (
    MAX_RUN_BYTES,
    analyze_session,
    format_duration,
    format_size,
    read_analysis_errors,
    read_duration,
    validate_session,
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration — modifier selon l'environnement
# ---------------------------------------------------------------------------

RECIPIENTS = [
    "chris.loisel94@gmail.com",
    "data@mistral.ai",
]

SMTP_HOST   = os.environ.get("SMTP_HOST",   "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER   = os.environ.get("SMTP_USER",   "")
SMTP_PASS   = os.environ.get("SMTP_PASS",   "")
SMTP_SENDER = os.environ.get("SMTP_SENDER", SMTP_USER)

# Si l'archive groupée dépasse cette limite, l'email est envoyé sans pièce
# jointe (résumé seul) — ajuster selon les limites du serveur SMTP.
MAX_ATTACH_BYTES = int(os.environ.get("MAX_ATTACH_MB", "200")) * 1024 * 1024

DODGE_FILE_MAIL = "uploaded_sessions_mail.json"

# ---------------------------------------------------------------------------
# Dodge file (variante mail — fichier séparé du pipeline Mistral)
# ---------------------------------------------------------------------------

def load_dodge_mail(root: Path) -> dict:
    path = root / DODGE_FILE_MAIL
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sessions": []}


def save_dodge_mail(root: Path, dodge: dict) -> None:
    path = root / DODGE_FILE_MAIL
    path.write_text(json.dumps(dodge, indent=2, ensure_ascii=False), encoding="utf-8")


def mark_mailed(dodge: dict, session_name: str, size_bytes: int, duration_seconds: float) -> None:
    dodge["sessions"].append({
        "name":             session_name,
        "size_bytes":       size_bytes,
        "duration_seconds": duration_seconds,
        "sent_at":          datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------------------
# Lecture des métadonnées d'une session
# ---------------------------------------------------------------------------

def read_session_meta(session_dir: Path) -> dict:
    """Lit result.json, mission.json, config.json et analysis.json."""
    meta: dict = {}
    for fname, key in [
        ("result.json",   "result"),
        ("mission.json",  "mission"),
        ("config.json",   "config"),
        ("analysis.json", "analysis"),
    ]:
        path = session_dir / fname
        if path.exists():
            try:
                meta[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return meta


def read_operator_id(session_dir: Path) -> str:
    """Lit l'identifiant opérateur depuis config.json (fallback 'unknown')."""
    config_path = session_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            op = cfg.get("operator", {})
            return op.get("operator_id") or op.get("username") or op.get("full_name") or "unknown"
        except Exception:
            pass
    return "unknown"


# ---------------------------------------------------------------------------
# Archive groupée
# ---------------------------------------------------------------------------

def zip_sessions_batch(sessions: list[Path], tmp_dir: Path) -> Path:
    """Zippe plusieurs sessions dans une seule archive (1 dossier par session)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = tmp_dir / f"sessions_batch_{timestamp}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for session_dir in sessions:
            for f in session_dir.rglob("*"):
                if f.is_file():
                    arcname = Path(session_dir.name) / f.relative_to(session_dir)
                    zf.write(f, arcname)
    return archive_path


# ---------------------------------------------------------------------------
# Construction de l'email
# ---------------------------------------------------------------------------

def _build_body(entries: list[dict], zip_size: int, attached: bool) -> str:
    """Corps en texte brut — résumé de toutes les sessions du batch."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "═" * 60,
        f"  ENVOI HORAIRE — {len(entries)} session(s), 1 par opérateur",
        f"  Envoyé le : {now_str}",
        "═" * 60,
        "",
    ]

    for e in entries:
        mission = e["meta"].get("mission", {})
        config  = e["meta"].get("config", {})
        result  = e["meta"].get("result", {})
        op      = config.get("operator", {})
        rig     = config.get("rig", {})

        lines += [
            f"── {e['session_name']} ────────────────────────",
            f"  Opérateur : {op.get('full_name', '—')} ({e['operator_id']})",
            f"  Rig       : {rig.get('code', '—')} — {rig.get('site_id', '—')}",
            f"  Mission   : {mission.get('name', '—')}",
            f"  Résultat  : {result.get('result', '—')}",
            f"  Durée     : {format_duration(e['duration'])}",
            f"  Taille    : {format_size(e['size'])}",
            "",
        ]

    lines += ["── Livraison ────────────────────────────────────"]
    if attached:
        lines.append(f"  ✓ Archive ZIP jointe à cet email ({format_size(zip_size)}).")
    else:
        lines.append(
            f"  ⚠ Archive trop volumineuse pour être jointe "
            f"({format_size(zip_size)} > limite {format_size(MAX_ATTACH_BYTES)})."
        )
        lines.append("    Les données sont disponibles sur le serveur NAS.")

    lines += ["", "═" * 60]
    return "\n".join(lines)


def send_batch_email(entries: list[dict], zip_path: Path) -> bool:
    """
    Envoie un email unique pour le batch de sessions.
    Joint le ZIP si sa taille est <= MAX_ATTACH_BYTES, sinon résumé seul.
    Retourne True si l'envoi a réussi.
    """
    if not SMTP_USER or not SMTP_PASS:
        logging.error("  SMTP_USER / SMTP_PASS non configurés — envoi impossible")
        return False

    zip_size = zip_path.stat().st_size
    attached = zip_size <= MAX_ATTACH_BYTES

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    subject = f"[Sessions] {len(entries)} session(s) — {now_str} UTC"

    msg = MIMEMultipart()
    msg["From"]    = SMTP_SENDER
    msg["To"]      = ", ".join(RECIPIENTS)
    msg["Subject"] = subject

    body = _build_body(entries, zip_size, attached)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if attached:
        with open(zip_path, "rb") as fh:
            part = MIMEBase("application", "zip")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=zip_path.name)
        msg.attach(part)
        logging.info("  ZIP joint : %s (%s)", zip_path.name, format_size(zip_size))
    else:
        logging.info("  Trop volumineux (%s) — résumé sans pièce jointe", format_size(zip_size))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_SENDER, RECIPIENTS, msg.as_bytes())
        logging.info("  Email envoyé à : %s", ", ".join(RECIPIENTS))
        return True
    except Exception as exc:
        logging.error("  Erreur SMTP : %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Envoie 1 session par opérateur par email")
    parser.add_argument("dossier", help="Dossier racine contenant les sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse sans envoyer ni écrire de dodge file")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Nombre maximum d'opérateurs/sessions par run (0 = illimité)")
    args = parser.parse_args()

    dry_run      = args.dry_run
    max_sessions = args.max_sessions
    root         = Path(args.dossier)

    if not root.is_dir():
        print(f"Erreur : '{root}' n'est pas un dossier valide")
        sys.exit(1)

    if dry_run:
        print("*** MODE DRY-RUN — aucun envoi, aucune écriture ***\n")

    dodge        = load_dodge_mail(root)
    already_done = {e["name"] for e in dodge["sessions"]}

    print(f"Lecture de '{root}'...", flush=True)
    all_sessions = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.lower().startswith("session")
    )

    if not all_sessions:
        print(f"Aucun dossier 'session*' trouvé dans '{root}'", flush=True)
        sys.exit(0)

    skipped    = [s.name for s in all_sessions if s.name in already_done]
    candidates = [s for s in all_sessions if s.name not in already_done]

    print(f"{len(all_sessions)} dossier(s) dont {len(skipped)} déjà envoyé(s). "
          f"Analyse de {len(candidates)} candidat(s)...", flush=True)

    empty_sessions:    list[tuple[str, str]]       = []
    invalid_sessions:  list[tuple[str, list[str]]] = []
    rejected_sessions: list[tuple[str, list[str]]] = []
    to_send:           list[dict]                  = []
    selected_operators: set[str] = set()
    cumul_bytes  = 0

    for s in candidates:
        if max_sessions > 0 and len(to_send) >= max_sessions:
            break
        if cumul_bytes >= MAX_RUN_BYTES:
            break

        operator_id = read_operator_id(s)
        if operator_id in selected_operators:
            continue  # déjà une session sélectionnée pour cet opérateur

        print(f"  Analyse {s.name} (op={operator_id}) ...", end="\r", flush=True)

        # 1. Vérification d'intégrité
        issues = validate_session(s)
        if issues:
            invalid_sessions.append((s.name, issues))
            continue

        # 2. analysis.json — erreurs de capture
        errors = read_analysis_errors(s)
        if errors:
            rejected_sessions.append((s.name, errors))
            continue

        # 3. Session non-vide
        is_empty, reason, size = analyze_session(s)
        if is_empty:
            empty_sessions.append((s.name, reason))
            continue

        to_send.append({
            "session_dir":  s,
            "session_name": s.name,
            "operator_id":  operator_id,
            "size":         size,
            "duration":     read_duration(s),
            "meta":         read_session_meta(s),
        })
        selected_operators.add(operator_id)
        cumul_bytes += size

    print(" " * 60, end="\r", flush=True)

    print(f"{len(all_sessions)} session(s) trouvée(s) au total :", flush=True)
    print(f"  {len(invalid_sessions)} incomplète(s)/corrompue(s) — bloquées")
    print(f"  {len(rejected_sessions)} rejetée(s) — erreurs analysis.json")
    print(f"  {len(empty_sessions)} vide(s) — ignorées")
    print(f"  {len(skipped)} déjà envoyée(s) — ignorées")
    print(f"  {len(to_send)} sélectionnée(s) — {len(selected_operators)} opérateur(s) "
          f"({format_size(cumul_bytes)})")

    if invalid_sessions:
        print("\nSessions invalides (bloquées) :")
        for name, issues in invalid_sessions:
            print(f"  INVALIDE  {name}")
            for issue in issues:
                print(f"            - {issue}")

    if rejected_sessions:
        print("\nSessions rejetées (erreurs analysis.json) :")
        for name, errs in rejected_sessions:
            print(f"  REJET  {name}")
            for e in errs:
                print(f"         - {e}")

    if empty_sessions:
        print("\nSessions vides :")
        for name, reason in empty_sessions:
            print(f"  VIDE  {name}  ({reason})")

    if not to_send:
        print("\nAucune session à envoyer.", flush=True)
        if dry_run:
            sys.exit(0)
        sys.exit(0)

    if dry_run:
        print("\nSessions qui seraient regroupées et envoyées :")
        for i, e in enumerate(to_send, 1):
            print(f"  [{i}/{len(to_send)}] {e['session_name']}  (op={e['operator_id']}, "
                  f"{format_size(e['size'])}, {format_duration(e['duration'])})")
        print(f"\nDestinataires : {', '.join(RECIPIENTS)}")
        print("\n*** DRY-RUN terminé — rien n'a été modifié ***")
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nCréation de l'archive groupée ({len(to_send)} session(s))...", flush=True)
        zip_path = zip_sessions_batch([e["session_dir"] for e in to_send], Path(tmp_dir))
        zip_size = zip_path.stat().st_size
        print(f"Archive : {zip_path.name}  ({format_size(zip_size)})", flush=True)

        success = send_batch_email(to_send, zip_path)

        if success:
            for e in to_send:
                mark_mailed(dodge, e["session_name"], e["size"], e["duration"])
            save_dodge_mail(root, dodge)
            print(f"\nOK — email envoyé à {', '.join(RECIPIENTS)} "
                  f"({len(to_send)} session(s))", flush=True)
        else:
            print("\nECHEC — envoi email échoué, rien n'a été marqué comme envoyé", flush=True)
            sys.exit(1)

    # --- Cumul global ---
    total_bytes   = sum(e["size_bytes"]       for e in dodge["sessions"])
    total_seconds = sum(e["duration_seconds"] for e in dodge["sessions"])

    print()
    print("=== Cumul total envoyé par email (toutes exécutions) ===")
    print(f"  Sessions envoyées : {len(dodge['sessions'])}")
    print(f"  Volume total      : {format_size(total_bytes)}")
    print(f"  Durée totale      : {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
