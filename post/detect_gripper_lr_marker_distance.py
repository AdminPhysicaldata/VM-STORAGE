#!/usr/bin/env python3
"""Vérifie, pour une session, que cameras/left.mp4 et cameras/right.mp4 sont
bien nommées dans le bon sens — en comparant directement la distance pixel
entre les coins haut-droit des 2 marqueurs ArUco de la pince (IDs 244 et 255)
à l'ouverture mesurée par le capteur (sensors/{side}.jsonl, champ
Opening_width).

Contrairement à detect_charuco_lr.py (qui utilise les mêmes marqueurs pour
juste distinguer "gripper" de "head") et à detect_gripper_lr_sensor_corr.py
(flux optique global, signal faible par session), ici la distance entre les
2 marqueurs EST directement l'écartement des mors de la pince : sur les
sessions vérifiées (vm-storage/session_*), la corrélation directe avec
Opening_width atteint r≈0.93-0.998 à décalage quasi nul, contre r<0.5 sur le
mauvais appariement. C'est un signal net, pas un signal moyenné sur plein de
sessions bruitées.

Point d'attention qui a fait échouer les tentatives précédentes : les
timestamps par frame de cameras/{cam}.jsonl ne correspondent PAS 1:1 aux
frames de la vidéo encodée (le brut est capturé ~60fps, la vidéo encodée
30fps) — il faut utiliser cameras/resampled_30hz.jsonl (grille 1 entrée =
1 frame vidéo) pour aligner correctement chaque frame à son timestamp réel.

Usage :
    python3 detect_gripper_lr_marker_distance.py --session ../session_xxx --plot out.png
    python3 detect_gripper_lr_marker_distance.py /media/qbee/T9/sessions/ --report report.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    print(f"Dépendance manquante : {exc}\n  uv add opencv-contrib-python", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_camera_names import _replace_in_json, fix_resampled_jsonl  # réutilisés pour appliquer un swap

_MARKER_A, _MARKER_B = 244, 255
_DICT_NAME = "DICT_4X4_1000"
_MAX_LAG_SEC = 1.0
_LAG_STEP_SEC = 0.02
_GRID_DT_SEC = 0.02
_MIN_POINTS = 30          # sous ce seuil, la corrélation est trop instable pour trancher (cf. cas dégénéré observé)
_MIN_SPAN_SEC = 3.0        # les points détectés doivent couvrir au moins ça (évite une simple rampe isolée)
_MIN_CORR_SAME = 0.80      # corrélation minimale attendue sur le bon appariement
_MIN_MARGIN = 0.30         # écart minimal entre bon et mauvais appariement


def _make_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _DICT_NAME))
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _top_right(corners: np.ndarray) -> np.ndarray:
    """corners: (1,4,2) — ordre standard OpenCV ArUco TL,TR,BR,BL."""
    return corners.reshape(4, 2)[1]


# ─── Repli multi-passe (vidéo basse résolution / floue) ─────────────────────
# Même constat que detect_charuco_lr.py : sur certaines sessions (640x480,
# fort flou de mouvement), la détection simple échoue à décoder les
# marqueurs même visibles à l'œil (0% de détection mesuré), ce qui rendait le
# check left/right systématiquement "inconclusive". Un repli multi-passe
# (CLAHE doux/fort, flou+CLAHE, accentuation) ramène la détection à ~41% sur
# le cas observé. Coûteux (jusqu'à 5 passes), donc utilisé seulement quand la
# détection simple ne trouve pas déjà les 2 marqueurs ciblés.
_CLAHE_SOFT = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_HARD = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
_SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def _detect_pair(gray: np.ndarray, detector, use_multipass: bool = True) -> dict:
    """Détecte les marqueurs ArUco visibles, avec repli multi-prétraitements
    si _MARKER_A et _MARKER_B ne sont pas trouvés ensemble du premier coup
    (sauf si use_multipass=False — passe rapide pure, cf. analyze_session).
    Retourne {id: corners (1,4,2)}."""
    corners, ids, _ = detector.detectMarkers(gray)
    found: dict = {}
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            found[int(mid)] = c
    if not use_multipass or (_MARKER_A in found and _MARKER_B in found):
        return found

    for img in (
        _CLAHE_SOFT.apply(gray),
        _CLAHE_HARD.apply(gray),
        _CLAHE_SOFT.apply(cv2.GaussianBlur(gray, (3, 3), 0)),
        cv2.filter2D(gray, -1, _SHARPEN_KERNEL).clip(0, 255).astype(np.uint8),
    ):
        c2, ids2, _ = detector.detectMarkers(img)
        if ids2 is not None:
            for c, mid in zip(c2, ids2.flatten()):
                found.setdefault(int(mid), c)
        if _MARKER_A in found and _MARKER_B in found:
            break
    return found


def _load_grid_timestamps(resampled_path: Path, cam_key: str) -> Optional[list[float]]:
    """Lit cameras/resampled_30hz.jsonl : une ligne == une frame de la vidéo
    encodée (contrairement à cameras/{cam}.jsonl qui capture à un fps brut
    différent). cam_key peut être 'left'/'right', avec repli sur la variante
    'rigth' observée dans certaines sessions."""
    if not resampled_path.is_file():
        return None
    ts: list[float] = []
    keys_tried = [cam_key] + (["rigth"] if cam_key == "right" else [])
    try:
        with open(resampled_path) as f:
            for line in f:
                d = json.loads(line)
                frames = d.get("frames", {})
                val = None
                for k in keys_tried:
                    if k in frames:
                        val = frames[k]["capture_timestamp_sec"]
                        break
                ts.append(val if val is not None else d["grid_t"])
    except Exception:
        return None
    return ts


def _fallback_camera_timestamps(jsonl_path: Path) -> list[float]:
    """Repli si resampled_30hz.jsonl est absent : utilise cameras/{cam}.jsonl
    directement. Suppose alors une correspondance 1:1 frame vidéo ↔ ligne
    jsonl, ce qui n'est vrai que si la vidéo n'a pas été ré-échantillonnée."""
    ts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("capture_timestamp_sec")
            if t is not None:
                ts.append(float(t))
    return ts


def _track_marker_distance(
    video_path: Path, ts_list: list[float], detector, use_multipass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance pixel entre les coins haut-droit des marqueurs 244 et 255,
    pour chaque frame où les deux sont détectés simultanément.

    use_multipass=False (par défaut) : détection simple uniquement — passe
    rapide sur toute la vidéo. analyze_session() ne réessaie en multi-passe
    (use_multipass=True, bien plus lent) que si cette première passe ne
    suffit pas à trancher — pas question de payer ce coût sur chaque vidéo
    alors que la plupart se résolvent déjà avec la détection simple."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])
    times, dists = [], []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = _detect_pair(gray, detector, use_multipass=use_multipass)
        if _MARKER_A in found and _MARKER_B in found and n < len(ts_list):
            tr_a = _top_right(found[_MARKER_A])
            tr_b = _top_right(found[_MARKER_B])
            times.append(ts_list[n])
            dists.append(float(np.linalg.norm(tr_a - tr_b)))
        n += 1
    cap.release()
    return np.array(times), np.array(dists)


def _sensor_opening(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, ow = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t, w = d.get("host_time_sec"), d.get("Opening_width")
            if t is not None and w is not None:
                ts.append(float(t))
                ow.append(float(w))
    return np.array(ts), np.array(ow)


def _best_lag_corr(
    cam_t: np.ndarray, cam_v: np.ndarray, sen_t: np.ndarray, sen_v: np.ndarray,
    max_lag: float = _MAX_LAG_SEC, lag_step: float = _LAG_STEP_SEC, grid_dt: float = _GRID_DT_SEC,
) -> tuple[float, float]:
    if len(cam_t) < 5 or len(sen_t) < 5:
        return 0.0, 0.0
    lo, hi = max(cam_t.min(), sen_t.min()), min(cam_t.max(), sen_t.max())
    if hi <= lo:
        return 0.0, 0.0
    grid = np.arange(lo, hi, grid_dt)
    a = np.interp(grid, cam_t, cam_v)
    if a.std() < 1e-9:
        return 0.0, 0.0
    a_z = (a - a.mean()) / a.std()
    best_r, best_lag = 0.0, 0.0
    for lag in np.arange(-max_lag, max_lag + lag_step, lag_step):
        b = np.interp(grid - lag, sen_t, sen_v)
        if b.std() < 1e-9:
            continue
        b_z = (b - b.mean()) / b.std()
        r = float(np.corrcoef(a_z, b_z)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, float(lag)
    return best_r, best_lag


def _data_sufficient(t: np.ndarray) -> bool:
    return len(t) >= _MIN_POINTS and (t.max() - t.min()) >= _MIN_SPAN_SEC


# ═════════════════════════════════════════════════════════════════════════
# MODE RAPIDE — ~10 frames ciblées par seek, au lieu de décoder toute la
# vidéo. On choisit des instants où le capteur a des valeurs d'ouverture
# bien différentes (étalées sur toute la plage observée), on saute
# directement aux frames vidéo correspondantes, et on vérifie que la
# distance des marqueurs suit le même ordre que ces valeurs.
# ═════════════════════════════════════════════════════════════════════════

_QUICK_SEEK_LOOKBACK = 8     # frames de marge avant la cible, pour un seek fiable même sans keyframe exact
_QUICK_RETRY_RADIUS = 5      # si les 2 marqueurs ne sont pas détectés à la frame visée, essaie ±1..±5


def _select_diverse_indices(values: np.ndarray, n: int) -> list[int]:
    """Choisit n indices dont les valeurs sont étalées le plus uniformément
    possible sur la plage observée (et non n valeurs proches du hasard)."""
    if len(values) == 0:
        return []
    if len(values) <= n:
        return list(range(len(values)))
    order = np.argsort(values)
    sorted_vals = values[order]
    targets = np.linspace(sorted_vals[0], sorted_vals[-1], n)
    chosen: list[int] = []
    used: set[int] = set()
    for t in targets:
        candidates = np.argsort(np.abs(sorted_vals - t))
        for c in candidates:
            orig_idx = int(order[c])
            if orig_idx not in used:
                used.add(orig_idx)
                chosen.append(orig_idx)
                break
    return chosen


def _nearest_index(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(arr - target)))


def _read_frame_at(cap: "cv2.VideoCapture", frame_idx: int, lookback: int = _QUICK_SEEK_LOOKBACK):
    """Saute près de frame_idx puis avance image par image jusqu'à l'index
    exact — évite les imprécisions de seek direct (B-frames/keyframes) tout
    en restant largement plus rapide qu'un décodage séquentiel complet."""
    start = max(0, frame_idx - lookback)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame = None
    cur = start
    while cur <= frame_idx:
        ok, f = cap.read()
        if not ok:
            return None
        frame = f
        cur += 1
    return frame


def _detect_pair_distance(frame, detector) -> Optional[float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _detect_pair(gray, detector)
    if _MARKER_A in found and _MARKER_B in found:
        tr_a = _top_right(found[_MARKER_A])
        tr_b = _top_right(found[_MARKER_B])
        return float(np.linalg.norm(tr_a - tr_b))
    return None


def _quick_track_side(
    video_path: Path, grid_ts: list[float],
    sensor_t: np.ndarray, sensor_v: np.ndarray,
    detector, n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sélectionne n_samples instants à ouverture capteur bien différente,
    saute aux frames vidéo correspondantes (avec petit retry si les 2
    marqueurs ne sont pas vus exactement à cette frame), retourne
    (timestamps_retenus, valeurs_capteur_retenues, distances_pixel_mesurées)
    — un point par échantillon réussi (les échecs sont simplement omis,
    donc les 3 tableaux retournés restent alignés entre eux)."""
    if len(sensor_v) == 0 or not grid_ts:
        return np.array([]), np.array([]), np.array([])

    idxs = _select_diverse_indices(sensor_v, n_samples)
    grid_arr = np.asarray(grid_ts)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([]), np.array([])

    chosen_t, chosen_v, chosen_d = [], [], []
    try:
        for i in idxs:
            target_t = sensor_t[i]
            frame_idx = _nearest_index(grid_arr, target_t)
            dist = None
            for delta in range(0, _QUICK_RETRY_RADIUS + 1):
                for fi in ({frame_idx} if delta == 0 else {frame_idx - delta, frame_idx + delta}):
                    if fi < 0:
                        continue
                    frame = _read_frame_at(cap, fi)
                    if frame is None:
                        continue
                    dist = _detect_pair_distance(frame, detector)
                    if dist is not None:
                        break
                if dist is not None:
                    break
            if dist is not None:
                chosen_t.append(target_t)
                chosen_v.append(sensor_v[i])
                chosen_d.append(dist)
    finally:
        cap.release()

    return np.array(chosen_t), np.array(chosen_v), np.array(chosen_d)


@dataclass
class QuickResult:
    name: str
    n_left_found: int
    n_left_requested: int
    n_right_found: int
    n_right_requested: int
    r_same_left: float
    r_same_right: float
    r_cross_left: float
    r_cross_right: float

    @property
    def verdict(self) -> str:
        min_found = min(self.n_left_found, self.n_right_found)
        min_requested = min(self.n_left_requested, self.n_right_requested)
        if min_found < max(4, 0.5 * min_requested):  # moins de 50% des échantillons demandés exploitables
            return "inconclusive (trop peu de frames exploitables)"
        score_same = abs(self.r_same_left) + abs(self.r_same_right)
        score_swap = abs(self.r_cross_left) + abs(self.r_cross_right)
        if max(abs(self.r_same_left), abs(self.r_same_right)) < _MIN_CORR_SAME and \
           max(abs(self.r_cross_left), abs(self.r_cross_right)) < _MIN_CORR_SAME:
            return "inconclusive (corrélation trop faible des deux côtés)"
        if abs(score_same - score_swap) < _MIN_MARGIN:
            return "inconclusive (marge insuffisante)"
        return "same" if score_same > score_swap else "swap"


def quick_analyze_session(session_dir: Path, n_samples: int = 10) -> Optional[QuickResult]:
    cam_dir = session_dir / "cameras"
    sens_dir = session_dir / "sensors"
    left_video, right_video = cam_dir / "left.mp4", cam_dir / "right.mp4"
    left_sensor, right_sensor = sens_dir / "left.jsonl", sens_dir / "right.jsonl"
    if not (left_video.is_file() and right_video.is_file()
            and left_sensor.is_file() and right_sensor.is_file()):
        return None

    resampled = cam_dir / "resampled_30hz.jsonl"
    ts_left = _load_grid_timestamps(resampled, "left") or _fallback_camera_timestamps(cam_dir / "left.jsonl")
    ts_right = _load_grid_timestamps(resampled, "right") or _fallback_camera_timestamps(cam_dir / "right.jsonl")

    sl_t, sl_v = _sensor_opening(left_sensor)
    sr_t, sr_v = _sensor_opening(right_sensor)
    detector = _make_detector()

    # Échantillons choisis sur la diversité du capteur LEFT, mesurés dans left.mp4
    t_left, v_left, d_left = _quick_track_side(left_video, ts_left, sl_t, sl_v, detector, n_samples)
    # Échantillons choisis sur la diversité du capteur RIGHT, mesurés dans right.mp4
    t_right, v_right, d_right = _quick_track_side(right_video, ts_right, sr_t, sr_v, detector, n_samples)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 4 or a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # même appariement : distance mesurée vs capteur qui a servi à choisir les instants
    r_same_left = _corr(d_left, v_left)
    r_same_right = _corr(d_right, v_right)

    # appariement croisé : on réutilise les MÊMES distances déjà mesurées (pas de frame
    # supplémentaire à lire) et on les compare à l'autre capteur, interpolé aux instants
    # réellement réussis (t_left / t_right, alignés par construction avec d_left / d_right)
    cross_v_for_left = np.interp(t_left, sr_t, sr_v) if len(t_left) else np.array([])
    cross_v_for_right = np.interp(t_right, sl_t, sl_v) if len(t_right) else np.array([])
    r_cross_left = _corr(d_left, cross_v_for_left)
    r_cross_right = _corr(d_right, cross_v_for_right)

    return QuickResult(
        name=session_dir.name,
        n_left_found=len(d_left), n_left_requested=n_samples,
        n_right_found=len(d_right), n_right_requested=n_samples,
        r_same_left=r_same_left, r_same_right=r_same_right,
        r_cross_left=r_cross_left, r_cross_right=r_cross_right,
    )


def apply_swap_fix(session_dir: Path) -> list[str]:
    """Échange cameras/left.mp4↔right.mp4 (+ .jsonl) quand quick_analyze_session
    a tranché "swap" avec confiance. Ne touche JAMAIS sensors/ : c'est la
    vérité-terrain (capteur identifié par le firmware) qui a servi à détecter
    le swap, donc c'est cameras/ qu'on aligne dessus, pas l'inverse.

    Propage aussi le renommage à cameras/resampled_30hz.jsonl (sinon la
    session redevient désynchronisée — cf. verify_camera_sync.py) et à
    config.json/analysis.json. Retourne le journal des actions effectuées."""
    cam_dir = session_dir / "cameras"
    renames = {"left": "right", "right": "left"}
    tmp_suffix = ".tmp_lr_swap"
    log: list[str] = []

    for old in renames:
        for ext in (".mp4", ".jsonl"):
            src = cam_dir / f"{old}{ext}"
            if src.exists():
                src.rename(cam_dir / f"{old}{ext}{tmp_suffix}")
    for old, new in renames.items():
        for ext in (".mp4", ".jsonl"):
            src = cam_dir / f"{old}{ext}{tmp_suffix}"
            if src.exists():
                src.rename(cam_dir / f"{new}{ext}")
        log.append(f"rename {old} → {new}")

    if fix_resampled_jsonl(cam_dir / "resampled_30hz.jsonl", renames, dry_run=False):
        log.append("update content resampled_30hz.jsonl")

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
            log.append(f"update {json_name}")

    return log


@dataclass
class SessionResult:
    name: str
    n_left: int
    n_right: int
    r_LL: float
    r_RR: float
    r_LR: float
    r_RL: float
    lag_LL: float
    lag_RR: float
    sufficient_left: bool
    sufficient_right: bool

    @property
    def verdict(self) -> str:
        if not (self.sufficient_left and self.sufficient_right):
            return "inconclusive (pas assez de détections simultanées 244+255)"
        score_same = abs(self.r_LL) + abs(self.r_RR)
        score_swap = abs(self.r_LR) + abs(self.r_RL)
        if max(abs(self.r_LL), abs(self.r_RR)) < _MIN_CORR_SAME and max(abs(self.r_LR), abs(self.r_RL)) < _MIN_CORR_SAME:
            return "inconclusive (corrélation trop faible des deux côtés)"
        if abs(score_same - score_swap) < _MIN_MARGIN:
            return "inconclusive (marge insuffisante entre les deux hypothèses)"
        return "same" if score_same > score_swap else "swap"


def _full_scan_result(
    session_dir: Path, left_video: Path, right_video: Path,
    ts_left: list, ts_right: list, sl_t, sl_v, sr_t, sr_v,
    detector, use_multipass: bool,
) -> SessionResult:
    t_left, d_left = _track_marker_distance(left_video, ts_left, detector, use_multipass=use_multipass)
    t_right, d_right = _track_marker_distance(right_video, ts_right, detector, use_multipass=use_multipass)

    r_LL, lag_LL = _best_lag_corr(t_left, d_left, sl_t, sl_v)
    r_LR, _ = _best_lag_corr(t_left, d_left, sr_t, sr_v)
    r_RL, _ = _best_lag_corr(t_right, d_right, sl_t, sl_v)
    r_RR, lag_RR = _best_lag_corr(t_right, d_right, sr_t, sr_v)

    return SessionResult(
        name=session_dir.name,
        n_left=len(t_left), n_right=len(t_right),
        r_LL=r_LL, r_RR=r_RR, r_LR=r_LR, r_RL=r_RL,
        lag_LL=lag_LL, lag_RR=lag_RR,
        sufficient_left=_data_sufficient(t_left),
        sufficient_right=_data_sufficient(t_right),
    )


def analyze_session(session_dir: Path) -> Optional[SessionResult]:
    """Scan complet (toute la vidéo, contrairement à quick_analyze_session) —
    déjà la passe coûteuse de secours quand l'échantillonnage rapide ne
    suffit pas. Optimisation : une PREMIÈRE passe en détection simple
    (use_multipass=False, rapide) ; le repli multi-passe — bien plus lent —
    n'est tenté que si cette première passe reste inconclusive. La plupart
    des vidéos n'ont pas besoin du repli, pas question de le payer partout."""
    cam_dir = session_dir / "cameras"
    sens_dir = session_dir / "sensors"
    left_video, right_video = cam_dir / "left.mp4", cam_dir / "right.mp4"
    left_sensor, right_sensor = sens_dir / "left.jsonl", sens_dir / "right.jsonl"
    if not (left_video.is_file() and right_video.is_file()
            and left_sensor.is_file() and right_sensor.is_file()):
        return None

    resampled = cam_dir / "resampled_30hz.jsonl"
    ts_left = _load_grid_timestamps(resampled, "left")
    ts_right = _load_grid_timestamps(resampled, "right")
    if ts_left is None:
        ts_left = _fallback_camera_timestamps(cam_dir / "left.jsonl")
    if ts_right is None:
        ts_right = _fallback_camera_timestamps(cam_dir / "right.jsonl")

    detector = _make_detector()
    sl_t, sl_v = _sensor_opening(left_sensor)
    sr_t, sr_v = _sensor_opening(right_sensor)

    result = _full_scan_result(
        session_dir, left_video, right_video, ts_left, ts_right, sl_t, sl_v, sr_t, sr_v,
        detector, use_multipass=False,
    )
    if result.verdict.startswith("inconclusive"):
        result = _full_scan_result(
            session_dir, left_video, right_video, ts_left, ts_right, sl_t, sl_v, sr_t, sr_v,
            detector, use_multipass=True,
        )
    return result


def plot_session(session_dir: Path, out_path: Path) -> None:
    """Génère le graphique demandé : distance(244,255) par pince + Opening_width capteur correspondant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam_dir = session_dir / "cameras"
    sens_dir = session_dir / "sensors"
    resampled = cam_dir / "resampled_30hz.jsonl"
    ts_left = _load_grid_timestamps(resampled, "left") or _fallback_camera_timestamps(cam_dir / "left.jsonl")
    ts_right = _load_grid_timestamps(resampled, "right") or _fallback_camera_timestamps(cam_dir / "right.jsonl")

    detector = _make_detector()
    t_left, d_left = _track_marker_distance(cam_dir / "left.mp4", ts_left, detector)
    t_right, d_right = _track_marker_distance(cam_dir / "right.mp4", ts_right, detector)
    sl_t, sl_v = _sensor_opening(sens_dir / "left.jsonl")
    sr_t, sr_v = _sensor_opening(sens_dir / "right.jsonl")

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    axes[0].plot(t_left, d_left, ".-", ms=2, label=f"distance coin TR (id{_MARKER_A}-id{_MARKER_B}) — left.mp4")
    axes[0].set_ylabel("px"); axes[0].legend()
    axes[1].plot(sl_t, sl_v, ".-", ms=2, color="orange", label="Opening_width — sensors/left.jsonl")
    axes[1].set_ylabel("mm"); axes[1].legend()
    axes[2].plot(t_right, d_right, ".-", ms=2, color="green", label=f"distance coin TR (id{_MARKER_A}-id{_MARKER_B}) — right.mp4")
    axes[2].set_ylabel("px"); axes[2].legend()
    axes[3].plot(sr_t, sr_v, ".-", ms=2, color="red", label="Opening_width — sensors/right.jsonl")
    axes[3].set_ylabel("mm"); axes[3].set_xlabel("temps (s, host_time)"); axes[3].legend()
    fig.suptitle(session_dir.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _analyze_one(session_dir_str: str) -> Optional[dict]:
    try:
        r = analyze_session(Path(session_dir_str))
    except Exception as exc:  # noqa: BLE001
        return {"name": Path(session_dir_str).name, "error": repr(exc)}
    if r is None:
        return None
    return {
        "name": r.name, "n_left": r.n_left, "n_right": r.n_right,
        "r_LL": round(r.r_LL, 4), "r_RR": round(r.r_RR, 4),
        "r_LR": round(r.r_LR, 4), "r_RL": round(r.r_RL, 4),
        "lag_LL": round(r.lag_LL, 3), "lag_RR": round(r.lag_RR, 3),
        "verdict": r.verdict,
    }


def _quick_analyze_one(session_dir_str: str, n_samples: int) -> Optional[dict]:
    try:
        r = quick_analyze_session(Path(session_dir_str), n_samples=n_samples)
    except Exception as exc:  # noqa: BLE001
        return {"name": Path(session_dir_str).name, "error": repr(exc)}
    if r is None:
        return None
    return {
        "name": r.name,
        "n_left_found": r.n_left_found, "n_left_requested": r.n_left_requested,
        "n_right_found": r.n_right_found, "n_right_requested": r.n_right_requested,
        "r_same_left": round(r.r_same_left, 4), "r_same_right": round(r.r_same_right, 4),
        "r_cross_left": round(r.r_cross_left, 4), "r_cross_right": round(r.r_cross_right, 4),
        "verdict": r.verdict,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", type=Path)
    p.add_argument("--session", type=Path)
    p.add_argument("--plot", type=Path, metavar="PNG", help="Avec --session : écrit le graphique demandé")
    p.add_argument("--quick", action="store_true",
                   help="Mode rapide : ~10 frames ciblées par seek (valeurs d'ouverture diverses) "
                        "au lieu de décoder toute la vidéo")
    p.add_argument("--n-samples", type=int, default=10, metavar="N",
                   help="Nombre de frames échantillonnées en mode --quick (défaut : 10)")
    p.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--report", type=Path, metavar="JSONL")
    args = p.parse_args()

    if args.session:
        if args.quick:
            result = _quick_analyze_one(str(args.session.resolve()), args.n_samples)
        else:
            result = _analyze_one(str(args.session.resolve()))
        if result is None:
            print(f"{args.session.name} : données insuffisantes (vidéos/capteurs manquants)")
            return 0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.plot:
            plot_session(args.session.resolve(), args.plot)
            print(f"Graphique écrit : {args.plot}")
        return 0

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

    mode = "rapide (--quick)" if args.quick else "complet"
    print(f"{len(sessions)} sessions, {args.workers} workers, mode {mode}…\n")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.quick:
            futures = {pool.submit(_quick_analyze_one, str(s), args.n_samples): s for s in sessions}
        else:
            futures = {pool.submit(_analyze_one, str(s)): s for s in sessions}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                rows.append(result)
            if done % 20 == 0 or done == len(sessions):
                print(f"  … {done}/{len(sessions)}", end="\r")
    print()

    if args.report:
        with args.report.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = [r for r in rows if "error" not in r]
    n_same = sum(1 for r in valid if r["verdict"] == "same")
    n_swap = sum(1 for r in valid if r["verdict"] == "swap")
    n_inconclusive = len(valid) - n_same - n_swap
    print(f"\n{'─' * 60}")
    print(f"Sessions analysées : {len(rows)} (exploitables : {len(valid)})")
    print(f"  nommage correct  : {n_same}")
    print(f"  nommage inversé  : {n_swap}")
    print(f"  inconclusive     : {n_inconclusive}")
    if n_swap:
        print("\nSessions à corriger :")
        for r in valid:
            if r["verdict"] == "swap":
                if args.quick:
                    print(f"  {r['name']}  r_same_left={r['r_same_left']:+.3f} r_same_right={r['r_same_right']:+.3f} "
                          f"r_cross_left={r['r_cross_left']:+.3f} r_cross_right={r['r_cross_right']:+.3f}")
                else:
                    print(f"  {r['name']}  r_LL={r['r_LL']:+.3f} r_RR={r['r_RR']:+.3f} "
                          f"r_LR={r['r_LR']:+.3f} r_RL={r['r_RL']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
