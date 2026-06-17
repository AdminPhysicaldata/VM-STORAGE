#!/usr/bin/env python3
"""Diagnostic des sessions contaminées par un upload multi-device.

Le bug : deux devices qui enregistrent dans la même seconde uploadent dans
le même dossier SFTP (suffixe compteur strippé) → leurs fichiers caméra
se mélangent ou s'écrasent.

Tous les devices ont les mêmes noms de caméra, donc la détection utilise
des critères temporels et statistiques extraits des fichiers .jsonl :

  1. Alignement du début (t_start) — toutes les caméras du même device
     démarrent dans un window de < 1 s. Écart > 3 s → caméra étrangère.
  2. Alignement de la fin (t_end) — même logique, tolérance 3 s.
  3. Durée relative — une caméra < 70 % ou > 130 % de la médiane est
     suspecte (autre device a enregistré moins/plus longtemps).
  4. Nombre de frames — idem, seuil 70 % / 130 %.

Deux signaux simultanés → caméra étrangère (HIGH confidence).
Un seul signal → signalée en WARNING (LOW confidence).

Usage :
    # Scan local (ex : dossier téléchargé depuis le serveur)
    python3 diagnose_shuffle.py /chemin/vers/sessions/

    # Analyser une session unique
    python3 diagnose_shuffle.py --session session_20260605_190710

    # Scan SFTP direct
    python3 diagnose_shuffle.py --sftp
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov"}
_IGNORE_FILES = {"resample_report.json", "resampled_30hz.jsonl"}

# Seuils de détection
_START_END_TOLERANCE_S = 3.0   # secondes
_RATIO_LOW = 0.70               # 70 % de la médiane
_RATIO_HIGH = 1.30              # 130 % de la médiane
_SIGNALS_HIGH = 2               # nb signaux → HIGH confidence


# ─── ffprobe (durée réelle du conteneur vidéo) ───────────────────────────────

def _ffprobe_duration(path: Path) -> float | None:
    """Durée du flux vidéo en secondes via ffprobe, ou None si indisponible."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


# ─── Structures de données ───────────────────────────────────────────────────

@dataclass
class CameraStats:
    name: str
    t_start: float
    t_end: float
    duration: float
    frames: int
    files: list  # Path (local) ou str (remote)
    video_duration: float | None = None  # durée réelle du .mp4 via ffprobe
    video_size: int | None = None        # taille du .mp4 en octets


@dataclass
class Finding:
    camera: CameraStats
    reasons: list[str]

    @property
    def confidence(self) -> str:
        return "HIGH" if len(self.reasons) >= _SIGNALS_HIGH else "LOW"


@dataclass
class SessionReport:
    session_name: str
    cameras: list[CameraStats]
    findings: list[Finding]
    missing_config: bool = False

    @property
    def is_contaminated(self) -> bool:
        return any(f.confidence == "HIGH" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return bool(self.findings)


# ─── Lecture des stats depuis un .jsonl ─────────────────────────────────────

def _parse_jsonl_stats(lines: list[str]) -> tuple[float, float, int] | None:
    """Retourne (t_start, t_end, frame_count) ou None si JSONL invalide."""
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 2:
        return None
    try:
        first = json.loads(non_empty[0])
        last  = json.loads(non_empty[-1])
        return (
            float(first["capture_timestamp_sec"]),
            float(last["capture_timestamp_sec"]),
            len(non_empty),
        )
    except (KeyError, json.JSONDecodeError, ValueError):
        return None


# ─── Détection des caméras étrangères ───────────────────────────────────────

def detect_foreign_cameras(cameras: list[CameraStats]) -> list[Finding]:
    """
    Compare chaque caméra aux médianes du groupe.
    Retourne une liste de Finding (caméras suspectes).
    """
    if len(cameras) < 2:
        return []

    med_tstart   = median(c.t_start   for c in cameras)
    med_tend     = median(c.t_end     for c in cameras)
    med_duration = median(c.duration  for c in cameras)
    med_frames   = median(c.frames    for c in cameras)

    findings: list[Finding] = []
    for cam in cameras:
        reasons: list[str] = []

        # 1. Alignement du début
        d_start = abs(cam.t_start - med_tstart)
        if d_start > _START_END_TOLERANCE_S:
            med_fmt = datetime.fromtimestamp(med_tstart).strftime("%H:%M:%S.%f")[:12]
            cam_fmt = datetime.fromtimestamp(cam.t_start).strftime("%H:%M:%S.%f")[:12]
            reasons.append(
                f"début décalé de {d_start:.1f}s "
                f"({cam_fmt} vs médiane {med_fmt})"
            )

        # 2. Alignement de la fin
        d_end = abs(cam.t_end - med_tend)
        if d_end > _START_END_TOLERANCE_S:
            med_fmt = datetime.fromtimestamp(med_tend).strftime("%H:%M:%S.%f")[:12]
            cam_fmt = datetime.fromtimestamp(cam.t_end).strftime("%H:%M:%S.%f")[:12]
            reasons.append(
                f"fin décalée de {d_end:.1f}s "
                f"({cam_fmt} vs médiane {med_fmt})"
            )

        # 3. Durée relative
        if med_duration > 0:
            ratio = cam.duration / med_duration
            if ratio < _RATIO_LOW or ratio > _RATIO_HIGH:
                reasons.append(
                    f"durée {cam.duration:.1f}s vs médiane {med_duration:.1f}s "
                    f"({ratio:.0%})"
                )

        # 4. Nombre de frames relatif
        if med_frames > 0:
            fratio = cam.frames / med_frames
            if fratio < _RATIO_LOW or fratio > _RATIO_HIGH:
                reasons.append(
                    f"frames {cam.frames} vs médiane {med_frames:.0f} "
                    f"({fratio:.0%})"
                )

        # 5. Durée réelle du fichier vidéo (ffprobe) — indépendant du .jsonl,
        #    détecte aussi les mp4 tronqués/écrasés par un autre device.
        video_durations = [c.video_duration for c in cameras if c.video_duration is not None]
        if cam.video_duration is not None and len(video_durations) >= 2:
            med_video_duration = median(video_durations)
            if med_video_duration > 0:
                vratio = cam.video_duration / med_video_duration
                if vratio < _RATIO_LOW or vratio > _RATIO_HIGH:
                    reasons.append(
                        f"durée vidéo (ffprobe) {cam.video_duration:.1f}s vs "
                        f"médiane {med_video_duration:.1f}s ({vratio:.0%})"
                    )

        if reasons:
            findings.append(Finding(camera=cam, reasons=reasons))

    return findings


# ─── Analyse locale ──────────────────────────────────────────────────────────

def _load_cameras_local(session_dir: Path) -> list[CameraStats]:
    cameras_dir = session_dir / "cameras"
    if not cameras_dir.is_dir():
        return []

    by_name: dict[str, dict] = {}
    for f in cameras_dir.iterdir():
        if not f.is_file() or f.name in _IGNORE_FILES:
            continue
        stem = f.stem
        entry = by_name.setdefault(stem, {"files": []})
        entry["files"].append(f)
        if f.suffix.lower() == ".jsonl":
            entry["jsonl"] = f

    result: list[CameraStats] = []
    for name, entry in sorted(by_name.items()):
        jsonl_path: Path | None = entry.get("jsonl")
        if jsonl_path is None:
            continue
        try:
            raw = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        parsed = _parse_jsonl_stats(raw)
        if parsed is None:
            continue
        t0, t1, nframes = parsed

        video_duration = video_size = None
        video_path = next((f for f in entry["files"] if f.suffix.lower() in _VIDEO_EXT), None)
        if video_path is not None:
            try:
                video_size = video_path.stat().st_size
            except OSError:
                video_size = None
            video_duration = _ffprobe_duration(video_path)

        result.append(CameraStats(
            name=name,
            t_start=t0,
            t_end=t1,
            duration=t1 - t0,
            frames=nframes,
            files=entry["files"],
            video_duration=video_duration,
            video_size=video_size,
        ))
    return result


def analyze_session(session_dir: Path) -> SessionReport:
    has_config = (session_dir / "config.json").is_file()
    cameras = _load_cameras_local(session_dir)
    findings = detect_foreign_cameras(cameras)
    return SessionReport(
        session_name=session_dir.name,
        cameras=cameras,
        findings=findings,
        missing_config=not has_config,
    )


# ─── Scan local ──────────────────────────────────────────────────────────────

def scan_local(root: Path, verbose: bool = False) -> None:
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        print(f"Aucun sous-dossier trouvé dans {root}")
        return

    contaminated = warnings = analyzed = 0

    for session_dir in sessions:
        if not (session_dir / "config.json").is_file():
            continue
        analyzed += 1
        report = analyze_session(session_dir)
        if report.is_contaminated:
            contaminated += 1
            if verbose:
                _print_report(report)
            else:
                print(f"[HIGH] {report.session_name}")
        elif report.has_warnings:
            warnings += 1
            if verbose:
                _print_report(report)
            else:
                print(f"[LOW]  {report.session_name}")

    print(f"\n{'─' * 60}")
    print(f"Sessions analysées   : {analyzed}")
    print(f"Contaminées (HIGH)   : {contaminated}")
    print(f"Avertissements (LOW) : {warnings}")


# ─── Scan SFTP ───────────────────────────────────────────────────────────────

def _sftp_load_cameras(sftp, remote_session: str) -> list[CameraStats]:
    cameras_path = f"{remote_session}/cameras"
    by_name: dict[str, dict] = {}

    try:
        for attr in sftp.listdir_attr(cameras_path):
            fname = attr.filename
            if fname in _IGNORE_FILES:
                continue
            p = Path(fname)
            stem = p.stem
            fpath = f"{cameras_path}/{fname}"
            entry = by_name.setdefault(stem, {"files": []})
            entry["files"].append(fpath)
            if p.suffix.lower() == ".jsonl":
                entry["jsonl_remote"] = fpath
                entry["size"] = attr.st_size
    except Exception:
        return []

    result: list[CameraStats] = []
    for name, entry in sorted(by_name.items()):
        if "jsonl_remote" not in entry:
            continue
        try:
            buf = io.BytesIO()
            sftp.getfo(entry["jsonl_remote"], buf)
            raw = buf.getvalue().decode("utf-8", errors="replace").splitlines()
        except Exception:
            continue
        parsed = _parse_jsonl_stats(raw)
        if parsed is None:
            continue
        t0, t1, nframes = parsed
        result.append(CameraStats(
            name=name,
            t_start=t0,
            t_end=t1,
            duration=t1 - t0,
            frames=nframes,
            files=entry["files"],
        ))
    return result


def scan_sftp(verbose: bool = False) -> None:
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
        sftp = paramiko.SFTPClient.from_transport(transport)
        _run_sftp_scan(sftp, remote_base, verbose=verbose)
    finally:
        transport.close()


def _run_sftp_scan(sftp, remote_base: str, verbose: bool = False) -> None:
    try:
        session_names = sorted(
            a.filename for a in sftp.listdir_attr(remote_base)
            if a.filename.startswith("session_")
        )
    except Exception as exc:
        print(f"[sftp] impossible de lister {remote_base} : {exc}", file=sys.stderr)
        return

    contaminated = warnings = 0

    for sname in session_names:
        remote_session = f"{remote_base}/{sname}"
        cameras = _sftp_load_cameras(sftp, remote_session)
        if not cameras:
            continue
        findings = detect_foreign_cameras(cameras)
        if not findings:
            continue

        report = SessionReport(session_name=sname, cameras=cameras, findings=findings)
        if report.is_contaminated:
            contaminated += 1
            if verbose:
                _print_report(report)
            else:
                print(f"[HIGH] {sname}")
        else:
            warnings += 1
            if verbose:
                _print_report(report)
            else:
                print(f"[LOW]  {sname}")

    print(f"\n{'─' * 60}")
    print(f"Sessions analysées   : {len(session_names)}")
    print(f"Contaminées (HIGH)   : {contaminated}")
    print(f"Avertissements (LOW) : {warnings}")


# ─── Affichage ───────────────────────────────────────────────────────────────

def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:12]


def _print_report(report: SessionReport) -> None:
    print(f"\n{'═' * 60}")
    print(f"  SESSION : {report.session_name}")

    # Tableau des caméras
    print(f"\n  {'Caméra':<10} {'Frames':>8} {'Durée':>8} {'t_start':>13} {'t_end':>13}")
    print(f"  {'─' * 56}")
    flagged_names = {f.camera.name for f in report.findings}
    for cam in sorted(report.cameras, key=lambda c: c.name):
        marker = " ⚠" if cam.name in flagged_names else "  "
        print(
            f"  {cam.name:<10} {cam.frames:>8} {cam.duration:>7.1f}s "
            f"{_fmt_ts(cam.t_start):>13} {_fmt_ts(cam.t_end):>13}{marker}"
        )

    # Détail des findings
    for finding in sorted(report.findings, key=lambda f: f.camera.name):
        label = f"[{finding.confidence}]"
        files_str = ", ".join(
            (str(p) if isinstance(p, Path) else p).split("/")[-1]
            for p in finding.camera.files
        )
        print(f"\n  {label} {finding.camera.name}  ({files_str})")
        for reason in finding.reasons:
            print(f"      • {reason}")


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
    p.add_argument("--session", type=Path,
                   help="Analyser un seul dossier de session")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Afficher le détail complet (tableau + signaux)")
    args = p.parse_args()

    if args.session:
        report = analyze_session(args.session.resolve())
        _print_report(report)
        return 0

    if args.sftp:
        scan_sftp(verbose=args.verbose)
        return 0

    if args.directory:
        scan_local(args.directory.resolve(), verbose=args.verbose)
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
