#!/usr/bin/env python3
"""Vérifie que cameras/resampled_30hz.jsonl est bien synchronisé avec les
noms de caméra ACTUELS (cameras/{name}.jsonl) — c'est-à-dire qu'un éventuel
renommage (fix_camera_names.py, detect_charuco_lr.py --apply) a bien été
propagé jusqu'à ce fichier.

Pourquoi ce check existe : resampled_30hz.jsonl encode le nom de caméra à
deux endroits — les clés "frames"."{nom}" et les chemins "file" ("/cameras/
{nom}/frame_xxx.jpg"). C'est du JSONL, pas un JSON simple, donc les
correcteurs génériques ne le touchent pas automatiquement ; un bug a
longtemps laissé ce fichier de côté lors des renommages (cf. fix_camera_names
.fix_resampled_jsonl, ajouté pour corriger ça). Une session déjà "corrigée"
avec l'ancien comportement reste désynchronisée tant qu'elle n'est pas
repassée par le correcteur à jour — ce script sert à détecter ces survivants.

Méthode : pour chaque caméra (left/right/head), chaque timestamp
"frames"."{nom}".capture_timestamp_sec de resampled_30hz.jsonl doit
correspondre (à _TOLERANCE_MS près) à un timestamp réellement présent dans
cameras/{nom}.jsonl. Si une fraction significative ne correspond pas, le
fichier référence encore l'ancien contenu → désynchronisation détectée.
Vérifie aussi que le segment de chemin dans "file" correspond à la clé.

Usage :
    python3 verify_camera_sync.py --session ../session_xxx
    python3 verify_camera_sync.py /media/qbee/T9/sessions/ --report report.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOLERANCE_MS = 2.0        # resampled_30hz.jsonl copie le timestamp de la frame source choisie ; en
                            # pratique on observe ~0ms d'écart pour une caméra correctement référencée
                            # (parfois jusqu'à ~1ms de jitter de précision), et plusieurs ms pour la
                            # mauvaise caméra sur un rig synchronisé matériellement (left/right à
                            # quelques ms l'une de l'autre). Une tolérance plus large masquerait un
                            # vrai swap sur ce type de rig — voir le test de reproduction dans le repo.
_MAX_BAD_RATIO = 0.02      # tolère un peu de bruit (frames bordure, arrondis) avant de signaler une anomalie
_FILE_PATH_RE = re.compile(r"/cameras/([^/]+)/")


def _load_raw_timestamps(jsonl_path: Path) -> np.ndarray:
    ts = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("capture_timestamp_sec")
                if t is not None:
                    ts.append(float(t))
    except (OSError, json.JSONDecodeError):
        return np.array([])
    return np.array(sorted(ts))


def _nearest_gap_ms(arr: np.ndarray, target: float) -> float:
    if len(arr) == 0:
        return float("inf")
    idx = np.searchsorted(arr, target)
    cands = [i for i in (idx - 1, idx) if 0 <= i < len(arr)]
    return min(abs(arr[i] - target) for i in cands) * 1000.0 if cands else float("inf")


@dataclass
class SyncIssue:
    camera: str
    kind: str           # "timestamp" | "path"
    bad: int
    total: int
    median_gap_ms: float = 0.0   # diagnostic : écart typique des points en désaccord (timestamp uniquement)

    def __str__(self) -> str:
        pct = (self.bad / self.total * 100.0) if self.total else 0.0
        if self.kind == "timestamp":
            return (f"{self.camera}: {self.bad}/{self.total} ({pct:.1f}%) timestamps de "
                     f"resampled_30hz.jsonl introuvables (>±{_TOLERANCE_MS}ms, écart médian "
                     f"{self.median_gap_ms:.2f}ms) dans {self.camera}.jsonl "
                     f"— probable renommage non propagé à resampled_30hz.jsonl")
        return (f"{self.camera}: {self.bad}/{self.total} ({pct:.1f}%) chemins 'file' de "
                 f"resampled_30hz.jsonl pointent vers un autre dossier que la clé '{self.camera}'")


def check_session(session_dir: Path) -> list[SyncIssue]:
    cam_dir = session_dir / "cameras"
    resampled_path = cam_dir / "resampled_30hz.jsonl"
    if not resampled_path.is_file():
        return []  # pas de grille resample → rien à vérifier

    # "wall" est un nom alternatif toléré pour la caméra fixe "head".
    raw_ts: dict[str, np.ndarray] = {}
    for cam in ("left", "right", "head"):
        names = (cam, "wall") if cam == "head" else (cam,)
        p = next((cam_dir / f"{n}.jsonl" for n in names if (cam_dir / f"{n}.jsonl").is_file()), None)
        if p is not None:
            raw_ts[cam] = _load_raw_timestamps(p)

    bad_ts = {cam: 0 for cam in raw_ts}
    bad_gaps_ms: dict[str, list[float]] = {cam: [] for cam in raw_ts}
    bad_path = {cam: 0 for cam in raw_ts}
    total = {cam: 0 for cam in raw_ts}

    try:
        with open(resampled_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frames = d.get("frames", {})
                for key, info in frames.items():
                    cam_key = "right" if key == "rigth" else ("head" if key == "wall" else key)
                    if cam_key not in raw_ts:
                        continue
                    total[cam_key] += 1
                    t = info.get("capture_timestamp_sec")
                    gap = _nearest_gap_ms(raw_ts[cam_key], t) if t is not None else float("inf")
                    if gap > _TOLERANCE_MS:
                        bad_ts[cam_key] += 1
                        bad_gaps_ms[cam_key].append(gap)
                    file_path = info.get("file", "")
                    m = _FILE_PATH_RE.search(file_path)
                    if m and m.group(1) != key:
                        bad_path[cam_key] += 1
    except (OSError, json.JSONDecodeError):
        return []

    issues: list[SyncIssue] = []
    for cam in raw_ts:
        if total[cam] == 0:
            continue
        if bad_ts[cam] / total[cam] > _MAX_BAD_RATIO:
            median_gap = float(np.median(bad_gaps_ms[cam])) if bad_gaps_ms[cam] else 0.0
            issues.append(SyncIssue(cam, "timestamp", bad_ts[cam], total[cam], median_gap))
        if bad_path[cam] / total[cam] > _MAX_BAD_RATIO:
            issues.append(SyncIssue(cam, "path", bad_path[cam], total[cam]))
    return issues


def _check_one(session_dir_str: str) -> dict:
    session_dir = Path(session_dir_str)
    try:
        issues = check_session(session_dir)
    except Exception as exc:  # noqa: BLE001
        return {"name": session_dir.name, "error": repr(exc)}
    return {
        "name": session_dir.name,
        "ok": not issues,
        "issues": [str(i) for i in issues],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", type=Path)
    p.add_argument("--session", type=Path)
    p.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--report", type=Path, metavar="JSONL")
    args = p.parse_args()

    if args.session:
        result = _check_one(str(args.session.resolve()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", False) else 1

    if not args.directory:
        p.print_help()
        return 1

    root = args.directory.resolve()
    sessions = sorted(
        Path(e.path) for e in os.scandir(root)
        if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
    )
    if not sessions:
        print(f"Aucune session trouvée dans {root}")
        return 0

    print(f"{len(sessions)} sessions, {args.workers} workers…\n")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_check_one, str(s)): s for s in sessions}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            rows.append(result)
            if not result.get("ok", True):
                print(f"\n{result['name']} :")
                for issue in result.get("issues", []):
                    print(f"  ⚠ {issue}")
                if "error" in result:
                    print(f"  [ERREUR] {result['error']}")
            if done % 20 == 0 or done == len(sessions):
                print(f"  … {done}/{len(sessions)}", end="\r")
    print()

    if args.report:
        with args.report.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows if r.get("ok"))
    n_bad = len(rows) - n_ok
    print(f"\n{'─' * 60}")
    print(f"Sessions vérifiées  : {len(rows)}")
    print(f"Synchronisées       : {n_ok}")
    print(f"Désynchronisées     : {n_bad}")
    if n_bad:
        print("\nRelancer fix_camera_names.py sur ces sessions pour les corriger "
              "(le correcteur propage maintenant aussi resampled_30hz.jsonl).")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
