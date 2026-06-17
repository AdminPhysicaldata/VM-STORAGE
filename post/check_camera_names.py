#!/usr/bin/env python3
"""Détecte les sessions dont les caméras (ou capteurs) ont des noms inattendus.

Par défaut les noms attendus sont : left, right, head (sous-dossier "cameras").
Toute entrée avec un nom différent est signalée.

Usage :
    python3 check_camera_names.py /media/qbee/T9/sessions/
    python3 check_camera_names.py /media/qbee/T9/sessions/ --expected left right head front
    python3 check_camera_names.py /media/qbee/T9/sessions/ -m /media/qbee/T9/bad/
    python3 check_camera_names.py --sftp
    python3 check_camera_names.py /media/qbee/T9/sessions/ -j 16

    # Vérifier le sous-dossier sensors/ (noms attendus différents : pas de "head")
    python3 check_camera_names.py /media/qbee/T9/sessions/ --subdir sensors --expected left right
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_DEFAULT_EXPECTED = {"left", "right", "head"}
_IGNORE_FILES = {"resample_report.json", "resampled_30hz.jsonl"}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov"}
_DEFAULT_WORKERS = min(32, (os.cpu_count() or 4) * 4)
_PROGRESS_EVERY = 1000


# ─── Helpers ────────────────────────────────────────────────────────────────

def _camera_names_local(session_dir: Path, subdir: str = "cameras") -> set[str]:
    cameras_dir = session_dir / subdir
    if not cameras_dir.is_dir():
        return set()
    names: set[str] = set()
    try:
        with os.scandir(cameras_dir) as it:
            for entry in it:
                if not entry.is_file() or entry.name in _IGNORE_FILES:
                    continue
                p = Path(entry.name)
                if p.suffix.lower() in _VIDEO_EXT or p.suffix.lower() == ".jsonl":
                    names.add(p.stem)
    except OSError:
        pass
    return names


def _camera_names_sftp(sftp, remote_session: str, subdir: str = "cameras") -> set[str]:
    cameras_path = f"{remote_session}/{subdir}"
    names: set[str] = set()
    try:
        for attr in sftp.listdir_attr(cameras_path):
            fname = attr.filename
            if fname in _IGNORE_FILES:
                continue
            p = Path(fname)
            if p.suffix.lower() in _VIDEO_EXT or p.suffix.lower() == ".jsonl":
                names.add(p.stem)
    except Exception:
        pass
    return names


# ─── Worker local ────────────────────────────────────────────────────────────

def _check_session_local(
    session_dir: Path, expected: frozenset[str], subdir: str = "cameras"
) -> tuple[Path, set[str], set[str]] | None:
    """Retourne (session_dir, unexpected, missing) ou None si la session est clean."""
    if not (session_dir / "config.json").is_file():
        return None
    names = _camera_names_local(session_dir, subdir)
    unexpected = names - expected
    missing    = expected - names
    if unexpected or missing:
        return (session_dir, unexpected, missing)
    return None


# ─── Scan local ──────────────────────────────────────────────────────────────

def scan_local(
    root: Path,
    expected: set[str],
    move_to: Path | None = None,
    workers: int = _DEFAULT_WORKERS,
    subdir: str = "cameras",
) -> None:
    sessions = sorted(
        Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False)
    )
    if not sessions:
        print(f"Aucun sous-dossier trouvé dans {root}")
        return

    total = len(sessions)
    print(f"{total} sous-dossiers trouvés, scan avec {workers} workers…\n")

    if move_to is not None:
        move_to.mkdir(parents=True, exist_ok=True)

    fset = frozenset(expected)
    anomalies: list[tuple[Path, set[str], set[str]]] = []
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check_session_local, s, fset, subdir): s for s in sessions}
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                done += 1
                if result is not None:
                    anomalies.append(result)
                if done % _PROGRESS_EVERY == 0 or done == total:
                    print(f"  {done}/{total} analysées, {len(anomalies)} anomalies…", end="\r")

    print()  # newline après le \r

    anomalies.sort(key=lambda x: x[0].name)
    moved = 0
    for session_dir, unexpected, missing in anomalies:
        parts = []
        if unexpected:
            parts.append(f"inattendu={sorted(unexpected)}")
        if missing:
            parts.append(f"manquant={sorted(missing)}")
        line = f"{session_dir.name}  →  {',  '.join(parts)}"

        if move_to is not None:
            dest = move_to / session_dir.name
            if dest.exists():
                print(f"{line}  [SKIP — déjà dans {move_to.name}/]")
            else:
                shutil.move(str(session_dir), str(dest))
                moved += 1
                print(f"{line}  [déplacé → {dest}]")
        else:
            print(line)

    print(f"\n{'─' * 50}")
    print(f"Sessions analysées : {total}")
    print(f"Anomalies          : {len(anomalies)}")
    if move_to is not None:
        print(f"Déplacées          : {moved}")


# ─── Worker SFTP (thread-local) ──────────────────────────────────────────────

_tls = threading.local()


def _sftp_worker(sname: str, remote_base: str, transport, expected: frozenset[str], subdir: str = "cameras"):
    if not hasattr(_tls, "sftp"):
        import paramiko
        _tls.sftp = paramiko.SFTPClient.from_transport(transport)
    sftp = _tls.sftp
    names = _camera_names_sftp(sftp, f"{remote_base}/{sname}", subdir)
    unexpected = names - expected
    missing    = expected - names
    if unexpected or missing:
        return (sname, unexpected, missing)
    return None


# ─── Scan SFTP ───────────────────────────────────────────────────────────────

def scan_sftp(expected: set[str], workers: int = _DEFAULT_WORKERS, subdir: str = "cameras") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    config_path = Path(__file__).resolve().parent / "config.json"
    try:
        sftp_cfg = json.loads(config_path.read_text(encoding="utf-8")).get("sftp", {})
    except (json.JSONDecodeError, OSError):
        sftp_cfg = {}

    host        = (os.environ.get("SFTP_HOST")        or sftp_cfg.get("host", "")).strip()
    port        = int(os.environ.get("SFTP_PORT")     or sftp_cfg.get("port", 22))
    username    = (os.environ.get("SFTP_USERNAME")    or sftp_cfg.get("username", "")).strip()
    password    = os.environ.get("SFTP_PASSWORD")     or sftp_cfg.get("password", "")
    remote_base = (os.environ.get("SFTP_REMOTE_PATH") or sftp_cfg.get("remote_path", "")).strip().rstrip("/")

    if not all([host, username, remote_base]):
        print("[sftp] configuration incomplète (host/username/remote_path)", file=sys.stderr)
        sys.exit(1)

    try:
        import paramiko
    except ImportError:
        print("[sftp] paramiko non installé — uv add paramiko", file=sys.stderr)
        sys.exit(1)

    try:
        from postprocess_runner import _sftp_connect
    except ImportError as exc:
        print(f"[sftp] impossible d'importer postprocess_runner : {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Connexion SFTP : {username}@{host}:{port}{remote_base}")
    transport = _sftp_connect(host, port, username, password)
    try:
        sftp_main = paramiko.SFTPClient.from_transport(transport)
        try:
            session_names = sorted(
                a.filename for a in sftp_main.listdir_attr(remote_base)
                if a.filename.startswith("session_")
            )
        except Exception as exc:
            print(f"[sftp] impossible de lister {remote_base} : {exc}", file=sys.stderr)
            return

        total = len(session_names)
        print(f"{total} sessions trouvées, scan avec {workers} workers…\n")

        fset = frozenset(expected)
        anomalies: list[tuple[str, set, set]] = []
        done = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_sftp_worker, sname, remote_base, transport, fset, subdir): sname
                for sname in session_names
            }
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    done += 1
                    if result is not None:
                        anomalies.append(result)
                    if done % _PROGRESS_EVERY == 0 or done == total:
                        print(f"  {done}/{total} analysées, {len(anomalies)} anomalies…", end="\r")

        print()
        anomalies.sort(key=lambda x: x[0])
        for sname, unexpected, missing in anomalies:
            parts = []
            if unexpected:
                parts.append(f"inattendu={sorted(unexpected)}")
            if missing:
                parts.append(f"manquant={sorted(missing)}")
            print(f"{sname}  →  {',  '.join(parts)}")

        print(f"\n{'─' * 50}")
        print(f"Sessions analysées : {total}")
        print(f"Anomalies          : {len(anomalies)}")
    finally:
        transport.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path,
                   help="Répertoire local contenant les sessions")
    p.add_argument("--sftp", action="store_true",
                   help="Scanner directement le serveur SFTP")
    p.add_argument("--expected", nargs="+", default=sorted(_DEFAULT_EXPECTED),
                   metavar="NOM",
                   help=f"Noms de caméras attendus (défaut : {sorted(_DEFAULT_EXPECTED)})")
    p.add_argument("-m", "--move", type=Path, metavar="DEST",
                   help="Déplacer les sessions anormales dans ce répertoire")
    p.add_argument("-j", "--jobs", type=int, default=_DEFAULT_WORKERS, metavar="N",
                   help=f"Nombre de workers parallèles (défaut : {_DEFAULT_WORKERS})")
    p.add_argument("--subdir", default="cameras", metavar="NOM",
                   help="Sous-dossier de session à inspecter (défaut : cameras ; ex : sensors)")
    args = p.parse_args()

    expected = set(args.expected)
    print(f"Sous-dossier : {args.subdir}  —  Noms attendus : {sorted(expected)}\n")

    if args.sftp:
        scan_sftp(expected, workers=args.jobs, subdir=args.subdir)
        return 0

    if args.directory:
        scan_local(args.directory.resolve(), expected, move_to=args.move, workers=args.jobs, subdir=args.subdir)
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
