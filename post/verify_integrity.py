#!/usr/bin/env python3
"""Vérifie l'existence EXACTE des fichiers attendus dans chaque session.

Une session "propre" doit contenir, ni plus ni moins :

  racine/        config.json, mission.json, analysis.json, result.json,
                 postprocess.log
  cameras/       {left,right,head}.mp4, {left,right,head}.jsonl,
                 resample_report.json, resampled_30hz.jsonl
  sensors/       left.jsonl, right.jsonl

Tout fichier manquant est signalé comme MANQUANT.
Tout fichier présent qui n'est pas dans cette liste (doublon, fichier de
debug oublié, ancien nom non nettoyé, etc.) est signalé comme EN TROP.
Tout fichier présent mais vide/tronqué/invalide (upload coupé en cours de
route, mp4 sans atome moov, jsonl dont la 1ʳᵉ/dernière ligne n'est pas du
JSON valide) est signalé comme CORROMPU — sinon une session avec un fichier
de 0 octet passerait à tort pour "complète".

Usage :
    python3 verify_integrity.py /media/qbee/T9/sessions/
    python3 verify_integrity.py --session ../session_20260605_190710
    python3 verify_integrity.py /media/qbee/T9/sessions/ -m /media/qbee/T9/bad/
    python3 verify_integrity.py /media/qbee/T9/sessions/ --extra-only   # ignore les manquants
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

_CAM_NAMES = ("left", "right", "head")
_SENSOR_NAMES = ("left", "right")

# Fichiers générés par l'outillage lui-même (cache de run_pipeline.py, scoring
# qualité treatment-worker, etc.) — jamais signalés comme "en trop", sinon une
# session validée comme "propre" ne le resterait jamais après son premier
# passage dans le pipeline.
_IGNORE_ANYWHERE = {".postcheck.json", "gripper_tracking.csv", "gripper_correlation.json"}

_ROOT_REQUIRED = {"config.json", "mission.json", "analysis.json", "result.json", "postprocess.log"}
_CAMERA_REQUIRED = (
    {f"{n}.mp4" for n in _CAM_NAMES}
    | {f"{n}.jsonl" for n in _CAM_NAMES}
    | {"resample_report.json", "resampled_30hz.jsonl"}
)
_SENSOR_REQUIRED = {f"{n}.jsonl" for n in _SENSOR_NAMES}

_DEFAULT_WORKERS = min(32, (os.cpu_count() or 4) * 4)
_PROGRESS_EVERY = 1000


@dataclass
class IntegrityReport:
    session_name: str
    missing: dict[str, set[str]] = field(default_factory=dict)   # subdir -> {filenames}
    extra: dict[str, set[str]] = field(default_factory=dict)     # subdir -> {filenames}
    corrupt: dict[str, set[str]] = field(default_factory=dict)   # subdir -> {filenames}

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.extra and not self.corrupt


def _is_valid_jsonl(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            lines = [l for l in fh.read().splitlines() if l.strip()]
        if not lines:
            return False
        json.loads(lines[0])
        json.loads(lines[-1])
        return True
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _is_valid_mp4(path: Path) -> bool:
    """Vérifie la présence des atomes 'ftyp' et 'moov' (sans décoder la vidéo)."""
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("rb") as fh:
            head = fh.read(1 << 20)
            fh.seek(max(0, path.stat().st_size - (1 << 20)))
            tail = fh.read()
        return b"ftyp" in head and (b"moov" in head or b"moov" in tail)
    except OSError:
        return False


def _check_subdir(
    dir_path: Path, required: set[str], aliases: dict[str, str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Retourne (missing, extra, corrupt) pour un sous-dossier (ou la racine).

    `aliases` (ex : {"wall": "head"}) tolère un nom de fichier alternatif :
    "wall.mp4" sur disque est traité comme s'il s'appelait "head.mp4" pour
    le calcul missing/extra/corrupt, sans jamais renommer le fichier réel.
    """
    if not dir_path.is_dir():
        return set(required), set(), set()
    aliases = aliases or {}
    raw_present = {e.name for e in os.scandir(dir_path) if e.is_file() and e.name not in _IGNORE_ANYWHERE}

    # nom canonique (après alias) -> nom réel sur disque
    name_map: dict[str, str] = {}
    for name in raw_present:
        stem, dot, ext = name.partition(".")
        canon = f"{aliases.get(stem, stem)}{dot}{ext}"
        name_map[canon] = name

    present = set(name_map)
    missing = required - present
    extra = present - required

    corrupt: set[str] = set()
    for canon_name in required & present:
        path = dir_path / name_map[canon_name]
        if canon_name.endswith(".jsonl") and not _is_valid_jsonl(path):
            corrupt.add(canon_name)
        elif canon_name.endswith(".mp4") and not _is_valid_mp4(path):
            corrupt.add(canon_name)
        elif canon_name.endswith(".json") and path.stat().st_size == 0:
            corrupt.add(canon_name)

    return missing, extra, corrupt


# "wall" est un nom alternatif toléré pour la caméra fixe "head".
_CAMERA_ALIASES = {"wall": "head"}


def check_session(session_dir: Path) -> IntegrityReport:
    report = IntegrityReport(session_name=session_dir.name)

    root_missing, root_extra, root_corrupt = _check_subdir(session_dir, _ROOT_REQUIRED)
    cam_missing, cam_extra, cam_corrupt = _check_subdir(
        session_dir / "cameras", _CAMERA_REQUIRED, aliases=_CAMERA_ALIASES
    )
    sens_missing, sens_extra, sens_corrupt = _check_subdir(session_dir / "sensors", _SENSOR_REQUIRED)

    if root_missing:
        report.missing["."] = root_missing
    if cam_missing:
        report.missing["cameras"] = cam_missing
    if sens_missing:
        report.missing["sensors"] = sens_missing

    if root_extra:
        report.extra["."] = root_extra
    if cam_extra:
        report.extra["cameras"] = cam_extra
    if sens_extra:
        report.extra["sensors"] = sens_extra

    if root_corrupt:
        report.corrupt["."] = root_corrupt
    if cam_corrupt:
        report.corrupt["cameras"] = cam_corrupt
    if sens_corrupt:
        report.corrupt["sensors"] = sens_corrupt

    return report


def _fmt_report(report: IntegrityReport, extra_only: bool, missing_only: bool) -> str:
    lines = [report.session_name]
    if not missing_only:
        for subdir, names in sorted(report.extra.items()):
            lines.append(f"  [EN TROP]   {subdir}/  {sorted(names)}")
    if not extra_only:
        for subdir, names in sorted(report.missing.items()):
            lines.append(f"  [MANQUANT]  {subdir}/  {sorted(names)}")
    if not extra_only and not missing_only:
        for subdir, names in sorted(report.corrupt.items()):
            lines.append(f"  [CORROMPU]  {subdir}/  {sorted(names)}")
    return "\n".join(lines)


def scan_local(
    root: Path,
    move_to: Path | None,
    workers: int,
    extra_only: bool,
    missing_only: bool,
) -> None:
    sessions = sorted(
        Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False)
        and e.name.startswith("session_")
    )
    if not sessions:
        print(f"Aucune session trouvée dans {root}")
        return

    total = len(sessions)
    print(f"{total} sessions trouvées, scan avec {workers} workers…\n")

    if move_to is not None:
        move_to.mkdir(parents=True, exist_ok=True)

    reports: list[IntegrityReport] = []
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_session, s): s for s in sessions}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                done += 1
                if not r.is_clean:
                    reports.append(r)
                if done % _PROGRESS_EVERY == 0 or done == total:
                    print(f"  {done}/{total} analysées, {len(reports)} anomalies…", end="\r")

    print()
    reports.sort(key=lambda r: r.session_name)

    moved = 0
    for report in reports:
        if extra_only and not report.extra:
            continue
        if missing_only and not report.missing:
            continue
        print(_fmt_report(report, extra_only, missing_only))
        if move_to is not None:
            session_dir = root / report.session_name
            dest = move_to / report.session_name
            if dest.exists():
                print(f"  [SKIP — déjà dans {move_to.name}/]")
            else:
                shutil.move(str(session_dir), str(dest))
                moved += 1
                print(f"  → déplacé dans {move_to.name}/")

    print(f"\n{'─' * 50}")
    print(f"Sessions analysées : {total}")
    print(f"Anomalies          : {len(reports)}")
    if move_to is not None:
        print(f"Déplacées          : {moved}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path,
                   help="Répertoire local contenant les sessions")
    p.add_argument("--session", type=Path,
                   help="Vérifier une seule session")
    p.add_argument("-m", "--move", type=Path, metavar="DEST",
                   help="Déplacer les sessions anormales dans ce répertoire")
    p.add_argument("-j", "--jobs", type=int, default=_DEFAULT_WORKERS, metavar="N",
                   help=f"Workers parallèles (défaut : {_DEFAULT_WORKERS})")
    p.add_argument("--extra-only", action="store_true",
                   help="N'afficher/traiter que les fichiers en trop")
    p.add_argument("--missing-only", action="store_true",
                   help="N'afficher/traiter que les fichiers manquants")
    args = p.parse_args()

    if args.session:
        report = check_session(args.session.resolve())
        if report.is_clean:
            print(f"{report.session_name} : OK")
        else:
            print(_fmt_report(report, args.extra_only, args.missing_only))
        return 0

    if args.directory:
        scan_local(
            args.directory.resolve(), args.move, args.jobs,
            args.extra_only, args.missing_only,
        )
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
