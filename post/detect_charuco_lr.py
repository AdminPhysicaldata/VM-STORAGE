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
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    print(f"Dépendance manquante : {exc}\n  uv add opencv-contrib-python", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_camera_names import _replace_in_json, fix_resampled_jsonl  # réutilisés pour les corrections

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


# ─── Repli multi-passe (vidéo basse résolution / floue) ─────────────────────
#
# Constat (sessions 640x480 à fort flou de mouvement, cf. incident pipeline) :
# la détection simple échoue à décoder le motif des marqueurs même quand ils
# sont visibles à l'œil, faisant passer un gripper réel pour "head" (0% de
# détection) — exactement le cas dangereux puisqu'un mismatch peut déclencher
# un renommage. Un repli multi-prétraitements (CLAHE doux/fort, flou+CLAHE,
# accentuation — repris de gripper_tracking.py) fait passer la détection de
# 0% à 41% sur le cas observé. Coûteux (5 passes), donc utilisé seulement
# quand la détection simple ne trouve pas déjà les 2 marqueurs.

_CLAHE_SOFT = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_HARD = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
_SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


@functools.lru_cache(maxsize=4)
def _make_raw_detector(dict_name: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _detect_multipass_ids(gray: np.ndarray, dict_name: str) -> set[int]:
    detector = _make_raw_detector(dict_name)
    passes = [
        gray,
        _CLAHE_SOFT.apply(gray),
        _CLAHE_HARD.apply(gray),
        _CLAHE_SOFT.apply(cv2.GaussianBlur(gray, (3, 3), 0)),
        cv2.filter2D(gray, -1, _SHARPEN_KERNEL).clip(0, 255).astype(np.uint8),
    ]
    found: set[int] = set()
    for img in passes:
        _, ids, _ = detector.detectMarkers(img)
        if ids is not None:
            found.update(int(i) for i in ids.flatten())
    return found


# ─── Échantillonnage : ratio de frames où les 2 marqueurs sont visibles ─────

def _gripper_marker_ratio(
    path: Path, detector, sample_fps: float, dict_name: str = _DEFAULT_DICT,
    use_multipass: bool = False,
) -> float | None:
    """Retourne la fraction de frames échantillonnées où id244 ET id255 sont
    détectés ensemble, ou None si la vidéo est illisible/corrompue.

    use_multipass=False (passe rapide, par défaut) : détection simple
    uniquement — suffisant et bien plus rapide dès que le résultat n'est pas
    ambigu. use_multipass=True (repli) : ajoute le prétraitement multi-passe
    sur les frames où la détection simple échoue — coûteux, à réserver aux
    vidéos dont le premier résultat est ambigu ou suspect (cf. analyze_session)."""
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
                flat = set(ids.flatten().tolist()) if ids is not None else set()
                if use_multipass and not (_MARKER_IDS[0] in flat and _MARKER_IDS[1] in flat):
                    flat |= _detect_multipass_ids(gray, dict_name)
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
    """Passe rapide (détection simple) sur toutes les vidéos, puis repli
    multi-passe — bien plus coûteux — UNIQUEMENT sur celles dont le résultat
    rapide est ambigu ou contredit le nom (mismatch). Sur une vidéo dont le
    résultat est déjà net (vrai gripper bien détecté, vraie head jamais
    détectée), refaire l'analyse en multi-passe n'aurait rien changé — ne pas
    payer ce coût systématiquement est l'optimisation qui fait la différence
    en volume (cf. incident pipeline : multi-passe partout = beaucoup plus
    lent pour un gain nul sur les sessions déjà sans ambiguïté)."""
    cameras_dir = session_dir / "cameras"
    detector = _make_detector(dict_name)

    findings: list[VideoFinding] = []
    for video_path in sorted(cameras_dir.glob("*.mp4")):
        name = video_path.stem
        ratio = _gripper_marker_ratio(video_path, detector, sample_fps, dict_name, use_multipass=False)
        role = _classify(ratio)

        needs_recheck = role == "ambiguous" or (role in ("gripper", "head") and role != expected_role(name))
        if needs_recheck:
            ratio = _gripper_marker_ratio(video_path, detector, sample_fps, dict_name, use_multipass=True)
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

    # cameras/resampled_30hz.jsonl référence aussi les noms de caméra (clés
    # "frames" + chemins "file") — JSONL, donc pas couvert par _replace_in_json
    # ci-dessus (qui suppose un JSON simple). Sans ce correcteur dédié, ce
    # fichier resterait désynchronisé après le swap et rendrait la session
    # inutilisable pour toute corrélation caméra↔capteur basée sur ses timestamps.
    if fix_resampled_jsonl(cameras_dir / "resampled_30hz.jsonl", renames, dry_run=False):
        print("  update content resampled_30hz.jsonl")


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
