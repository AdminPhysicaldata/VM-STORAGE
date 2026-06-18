#!/usr/bin/env python3
"""Vérifie que les vidéos sont cohérentes avec leur nom via les marqueurs
ArUco des pinces (id 244 et 255) : "head" ne doit jamais les voir, "left"/
"right" doivent toujours les voir.

Ce script NE tente PAS de déterminer laquelle des deux vidéos gripper est
"left" et laquelle est "right". Une investigation approfondie (corrélation
pleine courbe, focus sur les frames divergentes, concordance de signe,
balayage de décalage temporel, raffinement sous-pixel des coins) a montré
que la distance pixel entre les deux marqueurs est trop bruitée pour servir
de proxy fiable à l'écartement physique — alors que la détection des IDs
eux-mêmes est, elle, fiable à 77-100% des frames. Ce script se limite donc
à ce qui est réellement vérifiable : EST-CE un gripper ou non.

Ce que ça détecte et corrige :
  - "head.mp4" qui montre les marqueurs → c'est en fait un gripper mal placé.
  - "left.mp4" ou "right.mp4" qui ne les montre jamais → c'est en fait la
    vidéo head mal placée.
  - Dans ce cas précis (exactement 1 vidéo gripper-nommée-head et 1 vidéo
    head-nommée-gripper), le swap est corrigeable sans ambiguïté : on
    échange juste leurs deux noms, peu importe si le gripper en question
    est "vraiment" left ou right (cette question reste sans réponse fiable).
  - Tout autre cas (plusieurs incohérences, vidéo illisible) → signalé
    pour vérification manuelle, jamais de correction automatique.

Usage :
    python3 detect_charuco_lr.py --session ../session_20260605_190710
    python3 detect_charuco_lr.py --session ../session_20260605_190710 --apply
    python3 detect_charuco_lr.py /media/qbee/T9/sessions/   # scan d'un répertoire

Dépendances : opencv-contrib-python
    uv add opencv-contrib-python
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    print(f"Dépendance manquante : {exc}\n  uv add opencv-contrib-python", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_camera_names import _replace_in_json  # réutilisé pour les corrections JSON

_MARKER_IDS = (244, 255)
_DEFAULT_DICT = "DICT_4X4_1000"
_GRIPPER_MIN_RATIO = 0.05   # >= 5% des frames échantillonnées voient les 2 marqueurs → "gripper"
_HEAD_MAX_RATIO = 0.02      # <= 2% → "head" (bruit/faux positifs tolérés)
_SAMPLE_FPS = 2.0           # quelques frames/seconde suffisent : détection fiable à 77-100%


# ─── Détecteur ArUco (compat anciennes/nouvelles API OpenCV) ────────────────

def _make_detector(dict_name: str):
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda gray: detector.detectMarkers(gray)[1]  # ids uniquement
    else:
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
        params = cv2.aruco.DetectorParameters_create()
        return lambda gray: cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)[1]


# ─── Échantillonnage : ratio de frames où les 2 marqueurs sont visibles ─────

def _gripper_marker_ratio(path: Path, detector, sample_fps: float) -> float | None:
    """Retourne la fraction de frames échantillonnées où id244 ET id255 sont
    détectés ensemble, ou None si la vidéo est illisible/corrompue."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))

    total = both = 0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ids = detector(gray)
                total += 1
                if ids is not None:
                    flat = ids.flatten().tolist()
                    if _MARKER_IDS[0] in flat and _MARKER_IDS[1] in flat:
                        both += 1
        idx += 1
    cap.release()
    return (both / total) if total else None


def _classify(ratio: float | None) -> str:
    if ratio is None:
        return "unreadable"
    if ratio >= _GRIPPER_MIN_RATIO:
        return "gripper"
    if ratio <= _HEAD_MAX_RATIO:
        return "head"
    return "ambiguous"


def expected_role(name: str) -> str:
    return "head" if name == "head" else "gripper"


# ─── Analyse d'une session ───────────────────────────────────────────────────

@dataclass
class VideoFinding:
    current_name: str
    role: str       # "gripper" / "head" / "ambiguous" / "unreadable"
    ratio: float     # 0.0 si "unreadable"

    @property
    def mismatch(self) -> bool:
        return self.role in ("gripper", "head") and self.role != expected_role(self.current_name)


def analyze_session(
    session_dir: Path,
    dict_name: str = _DEFAULT_DICT,
    sample_fps: float = _SAMPLE_FPS,
) -> list[VideoFinding]:
    cameras_dir = session_dir / "cameras"
    detector = _make_detector(dict_name)

    findings: list[VideoFinding] = []
    for video_path in sorted(cameras_dir.glob("*.mp4")):
        name = video_path.stem
        ratio = _gripper_marker_ratio(video_path, detector, sample_fps)
        role = _classify(ratio)
        findings.append(VideoFinding(current_name=name, role=role, ratio=ratio or 0.0))
    return findings


# ─── Rapport texte ────────────────────────────────────────────────────────

def print_report(session_name: str, findings: list[VideoFinding]) -> bool:
    """Retourne True si une anomalie est détectée."""
    print(f"\n{session_name}")
    anomaly = False
    for f in findings:
        if f.role == "unreadable":
            anomaly = True
            print(f"  {f.current_name:<8} [unreadable]  ⚠ vidéo illisible/corrompue")
            continue
        marker = f"ratio_2_marqueurs={f.ratio:.1%}"
        if f.role == "ambiguous":
            anomaly = True
            print(f"  {f.current_name:<8} [ambigu    ]  {marker}  ⚠ ni clairement gripper ni clairement head")
        elif f.mismatch:
            anomaly = True
            print(f"  {f.current_name:<8} [{f.role:<9}]  {marker}  "
                  f"⚠ INCOHÉRENT : nommée '{f.current_name}' mais le contenu correspond à '{f.role}'")
        else:
            print(f"  {f.current_name:<8} [{f.role:<9}]  {marker}")
    return anomaly


# ─── Correction (renommage) ──────────────────────────────────────────────────

def apply_fix(session_dir: Path, findings: list[VideoFinding]) -> None:
    """Ne corrige QUE le cas non-ambigu : exactement 1 vidéo nommée "head" qui
    montre des marqueurs (donc en fait un gripper), et exactement 1 vidéo
    gripper-nommée (left/right) qui n'en montre aucun (donc en fait head).
    On échange juste leurs deux noms — sans jamais essayer de deviner si le
    gripper en question est "vraiment" left ou right."""
    mismatched = [f for f in findings if f.mismatch]
    head_wrong = [f for f in mismatched if f.current_name == "head" and f.role == "gripper"]
    other_wrong = [f for f in mismatched if f.current_name != "head" and f.role == "head"]

    if len(mismatched) != 2 or len(head_wrong) != 1 or len(other_wrong) != 1:
        print(f"  pas de correction automatique sûre ({len(mismatched)} vidéo(s) incohérente(s)) "
              f"— vérification manuelle requise.")
        return

    a, b = head_wrong[0], other_wrong[0]
    renames = {a.current_name: b.current_name, b.current_name: a.current_name}

    cameras_dir = session_dir / "cameras"
    tmp_suffix = ".tmp_charuco_swap"
    # Renommage en 2 phases (passe par un nom temporaire) pour gérer le swap
    # sans jamais écraser un fichier existant pendant l'opération.
    for old in renames:
        for ext in (".mp4", ".jsonl"):
            src = cameras_dir / f"{old}{ext}"
            if src.exists():
                src.rename(cameras_dir / f"{old}{ext}{tmp_suffix}")
    for old, new in renames.items():
        for ext in (".mp4", ".jsonl"):
            src = cameras_dir / f"{old}{ext}{tmp_suffix}"
            if src.exists():
                src.rename(cameras_dir / f"{new}{ext}")
        print(f"  rename {old} → {new}")

    for json_name in ("config.json", "analysis.json"):
        path = session_dir / json_name
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        new_obj = _replace_in_json(obj, renames)
        if new_obj != obj:
            path.write_text(json.dumps(new_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  update {json_name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path,
                   help="Répertoire contenant plusieurs sessions à scanner")
    p.add_argument("--session", type=Path,
                   help="Analyser une seule session")
    p.add_argument("--dict", default=_DEFAULT_DICT, dest="dict_name",
                   help=f"Dictionnaire ArUco (défaut : {_DEFAULT_DICT})")
    p.add_argument("--sample-fps", type=float, default=_SAMPLE_FPS,
                   help=f"Fréquence d'échantillonnage vidéo en Hz (défaut : {_SAMPLE_FPS})")
    p.add_argument("--apply", action="store_true",
                   help="Appliquer le swap head<->gripper si détecté sans ambiguïté")
    args = p.parse_args()

    if args.session:
        session_dir = args.session.resolve()
        findings = analyze_session(session_dir, args.dict_name, args.sample_fps)
        anomaly = print_report(session_dir.name, findings)
        if anomaly and args.apply:
            apply_fix(session_dir, findings)
        return 0

    if args.directory:
        root = args.directory.resolve()
        sessions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_"))
        anomalies = 0
        for session_dir in sessions:
            if not (session_dir / "cameras").is_dir():
                continue
            findings = analyze_session(session_dir, args.dict_name, args.sample_fps)
            if print_report(session_dir.name, findings):
                anomalies += 1
                if args.apply:
                    apply_fix(session_dir, findings)
        print(f"\n{'─' * 50}")
        print(f"Sessions analysées : {len(sessions)}")
        print(f"Anomalies          : {anomalies}")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
