"""
gripper_tracking.py — Tracking multi-sources des positions de grippers.

Pipeline (ordre d'exécution) :
  1. Capteurs pinces (sensors/left.jsonl + sensors/right.jsonl) :
       • q [w,x,y,z] → orientation directe par quaternion (pas de filtre complémentaire)
       • host_time_sec → timestamp Unix absolu (× 1000 = ms)
       • Filtrage : q=[0,0,0,0] ignoré (capteur non initialisé)
  2. Stéréo (cameras/left → pince gauche, cameras/right → pince droite) :
       • Multi-marqueurs ArUco : solvePnP global si board layout connu,
         sinon médiane pondérée + distances inter-marqueurs pour qualité
  3. Caméra head → les deux pinces simultanément
  4. Corrélation croisée stéréo ↔ head (Pearson/axe, lag optimal, RMSE)

Format capteur (sensors/{side}.jsonl) :
  {
    "host_time_sec": 1779552002.977,           <- timestamp Unix (s)
    "sensor": "left",
    "q": [0.6314, -0.7089, 0.3048, 0.0769],   <- quaternion [w,x,y,z]
    "a": [-0.348, 0.141, -0.176],              <- accélération (m/s²)
    "Opening_width": 32.4,
    "Angle": 50.92,
    "SW": "ON"
  }

Format caméra (cameras/{side}.jsonl) :
  { "frame_index": 0, "capture_timestamp_sec": 1779552002.972, ... }

Sorties :
  {session}/gripper_tracking.csv      — trajectoires + orientation IMU
  {session}/gripper_correlation.json  — corrélations stéréo vs head
"""
import csv
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_FPS      = float(os.environ.get("TRACKING_SAMPLE_FPS",     "5.0"))
MARKER_SIZE_MM  = float(os.environ.get("TRACKING_MARKER_SIZE_MM", "50.0"))
ARUCO_DICT_NAME = os.environ.get("TRACKING_ARUCO_DICT",           "DICT_4X4_50")
MIN_CROSS_CORR  = float(os.environ.get("TRACKING_MIN_CROSS_CORR", "0.70"))
OUTPUT_CSV_NAME = os.environ.get("TRACKING_CSV_NAME",             "gripper_tracking.csv")
# Désactivé par post/run_pipeline.py (audit en lecture seule) : écrire ces
# fichiers dans la session ferait échouer verify_integrity.py (fichiers "en
# trop" non attendus), même quand la session est par ailleurs parfaitement
# propre. En production (treatment-worker), ces fichiers sont le livrable
# attendu du check #12 → on les laisse s'écrire par défaut (True).
WRITE_OUTPUTS   = os.environ.get("GRIPPER_TRACKING_WRITE_OUTPUT", "true").lower() == "true"

# Configuration ChArUco (board 3×3 = 5 marqueurs, 4 coins intérieurs)
CHARUCO_SQUARES_X   = int(os.environ.get("CHARUCO_SQUARES_X",          "3"))
CHARUCO_SQUARES_Y   = int(os.environ.get("CHARUCO_SQUARES_Y",          "3"))
CHARUCO_SQUARE_MM   = float(os.environ.get("CHARUCO_SQUARE_MM",        "60.0"))
CHARUCO_MARKER_MM   = float(os.environ.get("CHARUCO_MARKER_MM",        "45.0"))
# Décalage d'ID : les IDs du board de base commencent à 0 ;
# on soustrait cet offset des IDs détectés avant l'interpolation ChArUco.
CHARUCO_FIRST_RIGHT = int(os.environ.get("CHARUCO_FIRST_MARKER_RIGHT", "0"))
CHARUCO_FIRST_LEFT  = int(os.environ.get("CHARUCO_FIRST_MARKER_LEFT",  "5"))

# Seuil de validation croisée (distance 3D max entre estimation stéréo et head)
CROSSVAL_MAX_DIST_MM = float(os.environ.get("CROSSVAL_MAX_DIST_MM", "50.0"))

# Board ChArUco environnemental (planches fixes dans la scène) ─────────────────
# Utilisé pour localiser chaque caméra dans le repère monde et exprimer les
# positions des pinces en coordonnées absolues (indépendantes de la caméra).
ENV_CHARUCO_SQUARES_X = int(os.environ.get("ENV_CHARUCO_SQUARES_X", "11"))
ENV_CHARUCO_SQUARES_Y = int(os.environ.get("ENV_CHARUCO_SQUARES_Y",  "8"))
ENV_CHARUCO_SQUARE_MM = float(os.environ.get("ENV_CHARUCO_SQUARE_MM", "40"))
ENV_CHARUCO_MARKER_MM = float(os.environ.get("ENV_CHARUCO_MARKER_MM",  "25"))
# Premier ID ArUco du board environnemental (les IDs 0-9 sont réservés aux pinces)
ENV_CHARUCO_FIRST_ID  = int(os.environ.get("ENV_CHARUCO_FIRST_ID",   "10"))

# IDs ArUco par défaut pour chaque pince (utilisés si absent des métadonnées)
GRIPPER_IDS_RIGHT: set = {4, 2, 1, 3, 0}
GRIPPER_IDS_LEFT:  set = {9, 7, 6, 8, 5}

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".h264"}

_CALIB_PATTERNS = [
    "{session}/calibration/{cam}.npz",
    "{session}/{cam}_calib.npz",
    "{session}/calibration.npz",
    "/nas/calibration/{cam}.npz",
    "/nas/calibration/default.npz",
]

_DICT_MAP: dict = {}
if HAS_CV2:
    _DICT_MAP = {
        "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_50":  cv2.aruco.DICT_6X6_50,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_7X7_50":  cv2.aruco.DICT_7X7_50,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    }

CSV_FIELDS = [
    "timestamp_ms",
    "gripper_side",
    "source",           # "stereo_charuco" | "head_aruco" | "final"
    "source_camera",
    "cx_norm", "cy_norm",
    # ── Position primaire (stéréo ChArUco pour stereo_charuco et final) ────
    "x_mm", "y_mm", "z_mm",
    "dx_mm", "dy_mm", "dz_mm", "displacement_mm",
    # ── Position secondaire head ArUco (remplie uniquement pour source=final) ─
    "head_x_mm", "head_y_mm", "head_z_mm",
    "crossval_dist_mm",         # distance 3D entre estimation stéréo et head
    "validated",                # True uniquement pour source=final
    # ── Coordonnées monde (repère board ChArUco environnemental 11×8) ──────
    "x_world_mm", "y_world_mm", "z_world_mm",
    "env_n_corners",            # nb de coins ChArUco env détectés dans la frame
    # ── Qualité de la détection ────────────────────────────────────────────
    "imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg",
    "n_markers",
    "n_charuco_corners",
    "inter_marker_error_mm",
    "confidence",
    "method",           # "charuco_pnp" | "board_pnp" | "multi_avg" | "single_pnp" | "pixel"
]


# ═════════════════════════════════════════════════════════════════════════════
# ORIENTATION — depuis sensors/{side}.jsonl (quaternion direct)
# ═════════════════════════════════════════════════════════════════════════════

def _quat_to_euler(q: list) -> Tuple[float, float, float]:
    """
    Convertit un quaternion [w, x, y, z] en angles d'Euler (degrés).
    Retourne (roll, pitch, yaw). Retourne (0,0,0) si quaternion nul/invalide.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Roll (axe X)
    sinr = 2.0 * (w*x + y*z)
    cosr = 1.0 - 2.0 * (x*x + y*y)
    roll = math.degrees(math.atan2(sinr, cosr))

    # Pitch (axe Y) — clamp pour gimbal lock
    sinp = max(-1.0, min(1.0, 2.0 * (w*y - z*x)))
    pitch = math.degrees(math.asin(sinp))

    # Yaw (axe Z)
    siny = 2.0 * (w*z + x*y)
    cosy = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw


def _load_sensor_orient(session_path: str, side: str) -> Optional[dict]:
    """
    Charge l'orientation d'une pince depuis sensors/{side}.jsonl.

    Chaque ligne :
      host_time_sec  → timestamp Unix absolu (× 1000 = ms)
      q              → quaternion [w, x, y, z]

    Les entrées avec q=[0,0,0,0] sont ignorées (capteur non initialisé).

    Retourne None si absent ou sans données valides, sinon :
      { 'ts_ms', 'roll_deg', 'pitch_deg', 'yaw_deg', 'n_valid', 'n_total' }
    """
    path = os.path.join(session_path, "sensors", f"{side}.jsonl")
    if not os.path.isfile(path):
        return None

    ts_list, roll_list, pitch_list, yaw_list = [], [], [], []
    n_total = 0

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n_total += 1

                ts = d.get("host_time_sec")
                q  = d.get("q")
                if ts is None or q is None or len(q) != 4:
                    continue

                # Ignorer les quaternions nuls (capteur non initialisé)
                if all(abs(v) < 1e-9 for v in q):
                    continue

                roll, pitch, yaw = _quat_to_euler(q)
                ts_list.append(float(ts) * 1000.0)  # s → ms
                roll_list.append(roll)
                pitch_list.append(pitch)
                yaw_list.append(yaw)

    except Exception as e:
        logger.debug("Erreur lecture sensor %s : %s", path, e)
        return None

    n_valid = len(ts_list)
    if n_valid == 0:
        logger.debug("sensors/%s.jsonl : 0 quaternion valide sur %d entrées", side, n_total)
        return None

    logger.info("Capteur %s : %d/%d quaternions valides", side, n_valid, n_total)
    return {
        "ts_ms":     np.array(ts_list),
        "roll_deg":  np.array(roll_list),
        "pitch_deg": np.array(pitch_list),
        "yaw_deg":   np.array(yaw_list),
        "n_valid":   n_valid,
        "n_total":   n_total,
    }


def _imu_at(orient: Optional[dict], ts_ms: float) -> Tuple[float, float, float]:
    """Interpolation de l'orientation à un timestamp (ms). Retourne (0,0,0) si absent."""
    if orient is None or len(orient["ts_ms"]) == 0:
        return 0.0, 0.0, 0.0
    t = orient["ts_ms"]
    r = float(np.interp(ts_ms, t, orient["roll_deg"]))
    p = float(np.interp(ts_ms, t, orient["pitch_deg"]))
    y = float(np.interp(ts_ms, t, orient["yaw_deg"]))
    return r, p, y


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

def _load_calibration(session_path: str, cam_name: str):
    # "wall" est un nom alternatif toléré pour la caméra fixe "head".
    candidates = ("head", "wall") if cam_name == "head" else (cam_name,)
    for cand in candidates:
        for pattern in _CALIB_PATTERNS:
            path = pattern.format(session=session_path, cam=cand)
            if not os.path.isfile(path):
                continue
            try:
                data = np.load(path)
                keys = set(data.files)
                cam_key  = next((k for k in ("camera_matrix", "K", "mtx") if k in keys), None)
                dist_key = next((k for k in ("dist_coeffs",   "dist", "D") if k in keys), None)
                if cam_key and dist_key:
                    return data[cam_key].astype(np.float64), data[dist_key].astype(np.float64), True
            except Exception as e:
                logger.debug("Erreur calib %s : %s", path, e)
    return None, None, False


def _pinhole(w: int, h: int):
    fx = 0.8 * w
    K  = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    D  = np.zeros((5, 1), dtype=np.float64)
    return K, D


# ═════════════════════════════════════════════════════════════════════════════
# ARUCO — détection et pose
# ═════════════════════════════════════════════════════════════════════════════

def _make_detector():
    dict_id    = _DICT_MAP.get(ARUCO_DICT_NAME, cv2.aruco.DICT_6X6_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


def _make_charuco_board():
    """
    Crée un board ChArUco CHARUCO_SQUARES_X × CHARUCO_SQUARES_Y avec IDs 0…N-1.
    Compatible OpenCV ≥ 4.6 (ancien et nouveau constructeur).
    """
    dict_id    = _DICT_MAP.get(ARUCO_DICT_NAME, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    sq_m = CHARUCO_SQUARE_MM / 1000.0
    mk_m = CHARUCO_MARKER_MM / 1000.0
    try:
        return cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), sq_m, mk_m, aruco_dict
        )
    except AttributeError:
        return cv2.aruco.CharucoBoard_create(
            CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y, sq_m, mk_m, aruco_dict
        )


def _make_env_charuco_board():
    """
    Board ChArUco environnemental ENV_CHARUCO_SQUARES_X × ENV_CHARUCO_SQUARES_Y
    (défaut 11×8 = ~44 marqueurs, IDs commençant à ENV_CHARUCO_FIRST_ID).

    Les IDs du board OpenCV sont 0-based en interne ; le décalage est géré dans
    _detect_env_charuco_pose() par remapping avant l'interpolation.
    Compatible OpenCV ≥ 4.6.
    """
    dict_id    = _DICT_MAP.get(ARUCO_DICT_NAME, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    sq_m = ENV_CHARUCO_SQUARE_MM / 1000.0
    mk_m = ENV_CHARUCO_MARKER_MM / 1000.0
    try:
        return cv2.aruco.CharucoBoard(
            (ENV_CHARUCO_SQUARES_X, ENV_CHARUCO_SQUARES_Y), sq_m, mk_m, aruco_dict
        )
    except AttributeError:
        return cv2.aruco.CharucoBoard_create(
            ENV_CHARUCO_SQUARES_X, ENV_CHARUCO_SQUARES_Y, sq_m, mk_m, aruco_dict
        )


def _detect_env_charuco_pose(gray: np.ndarray,
                              all_corners: list, all_ids: list,
                              env_board, K: np.ndarray, D: np.ndarray
                              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Localise la caméra dans le repère de la planche environnementale ChArUco 11×8.

    Filtre dans all_corners/all_ids les marqueurs appartenant au board env
    (id >= ENV_CHARUCO_FIRST_ID), reméppe les IDs en base 0, puis enchaîne :
      interpolateCornersCharuco → coins subpixel du damier
      estimatePoseCharucoBoard  → pose caméra / board (rvec, tvec)

    Retourne (rvec, tvec, n_corners) ou (None, None, 0) si détection insuffisante.
    Les positions tvec sont en mètres (convention OpenCV).
    """
    env_corners, env_ids_raw = [], []
    for c, mid in zip(all_corners, all_ids):
        if mid >= ENV_CHARUCO_FIRST_ID:
            env_corners.append(c.reshape(1, 4, 2).astype(np.float32))
            env_ids_raw.append(mid - ENV_CHARUCO_FIRST_ID)

    if not env_corners:
        return None, None, 0

    env_ids_arr = np.array([[i] for i in env_ids_raw], dtype=np.int32)

    try:
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            env_corners, env_ids_arr, gray, env_board, K, D
        )
    except Exception:
        try:
            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                env_corners, env_ids_arr, gray, env_board
            )
        except Exception as e:
            logger.debug("env interpolateCornersCharuco : %s", e)
            return None, None, 0

    if retval < 4 or ch_corners is None:   # minimum 4 coins pour une pose fiable
        return None, None, int(retval)

    try:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, env_board, K, D, None, None
        )
    except Exception as e:
        logger.debug("env estimatePoseCharucoBoard : %s", e)
        return None, None, int(retval)

    if not ok:
        return None, None, int(retval)

    return rvec, tvec, int(retval)


def _to_world(x_cam_mm: float, y_cam_mm: float, z_cam_mm: float,
              rvec: np.ndarray, tvec: np.ndarray
              ) -> Tuple[float, float, float]:
    """
    Transforme une position en coordonnées caméra → repère monde (board env).

    OpenCV solvePnP : P_cam = R · P_obj + t  ⟹  P_obj = Rᵀ · (P_cam − t)
    tvec est en mètres (sortie d'estimatePoseCharucoBoard), on travaille en mm.
    """
    R, _ = cv2.Rodrigues(rvec)
    P_cam_m = np.array([[x_cam_mm], [y_cam_mm], [z_cam_mm]]) / 1000.0
    P_world_m = R.T @ (P_cam_m - tvec)
    w = P_world_m.flatten() * 1000.0
    return round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 1)


def _compute_stable_env_pose(
    video_path: str, cam_name: str, session_path: str, env_board,
    max_detections: int = 60,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Estime la pose stable de la caméra dans le repère du board environnemental
    en scannant la vidéo à ~1 fps (indépendamment du tracking des grippers).

    Principe : on cherche le board env dans TOUTES les frames échantillonnées,
    y compris celles où les grippers ne sont pas visibles, pour maximiser le
    nombre de détections et obtenir une pose robuste via la médiane.

    Retourne (rvec_med, tvec_med, n_detections) ou (None, None, 0).
    tvec est en mètres (convention OpenCV / estimatePoseCharucoBoard).
    """
    if not HAS_CV2 or not os.path.isfile(video_path):
        return None, None, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    # Échantillonnage à ~1 fps (suffisant pour une pose statique)
    step = max(1, int(fps_vid))

    K, D, _ = _load_calibration(session_path, cam_name)
    if K is None:
        K, D = _pinhole(frame_w, frame_h)

    detector = _make_detector()
    rvecs, tvecs = [], []
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_corners, all_ids = _detect_markers_multipass(gray, detector)
            if all_corners:
                rvec, tvec, n_c = _detect_env_charuco_pose(
                    gray, all_corners, all_ids, env_board, K, D
                )
                if rvec is not None:
                    rvecs.append(rvec.flatten())
                    tvecs.append(tvec.flatten())
                    if len(tvecs) >= max_detections:
                        break
        frame_no += 1

    cap.release()

    if not tvecs:
        logger.debug("_compute_stable_env_pose %s : board env non détecté (%d frames)",
                     cam_name, frame_no // step + 1)
        return None, None, 0

    tvec_med = np.median(tvecs, axis=0).reshape(3, 1)
    rvec_med = np.median(rvecs, axis=0).reshape(3, 1)
    t_mm = tvec_med.flatten() * 1000
    logger.info(
        "Pose env stable %s : %d détections | t=[%.0f, %.0f, %.0f] mm",
        cam_name, len(tvecs), t_mm[0], t_mm[1], t_mm[2],
    )
    return rvec_med, tvec_med, len(tvecs)


def _detect_charuco_pose(gray: np.ndarray, corners_list: list, ids: list,
                         board, K: np.ndarray, D: np.ndarray,
                         id_offset: int = 0) -> dict:
    """
    Pose du gripper par interpolation ChArUco (subpixel) → solvePnP board global.

    Principe :
      1. Les IDs détectés sont décalés de id_offset pour correspondre aux IDs
         0-based du board (ex : left_ids 5-9 → 0-4 avec offset=5).
      2. interpolateCornersCharuco affine les coins du damier à la précision subpixel.
      3. estimatePoseCharucoBoard résout la pose 6DOF sur l'ensemble des coins.

    Retourne {} si le nombre de coins ChArUco est insuffisant (< 1).
    """
    if not corners_list:
        return {}

    corners_arr = [c.reshape(1, 4, 2).astype(np.float32) for c in corners_list]
    ids_arr     = np.array([[mid - id_offset] for mid in ids], dtype=np.int32)

    try:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners_arr, ids_arr, gray, board, K, D
        )
    except Exception:
        try:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners_arr, ids_arr, gray, board
            )
        except Exception as e:
            logger.debug("interpolateCornersCharuco : %s", e)
            return {}

    if retval < 1 or charuco_corners is None:
        return {}

    try:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, D, None, None
        )
    except Exception as e:
        logger.debug("estimatePoseCharucoBoard : %s", e)
        return {}

    if not ok:
        return {}

    tvec_mm = tvec.flatten() * 1000.0
    n_c     = int(retval)
    conf    = min(1.0, 0.45 + 0.14 * n_c)   # 4 coins → ~1.0

    return {
        "x_mm":                  round(float(tvec_mm[0]), 1),
        "y_mm":                  round(float(tvec_mm[1]), 1),
        "z_mm":                  round(float(tvec_mm[2]), 1),
        "rvec":                  rvec,
        "n_markers":             len(ids),
        "n_charuco_corners":     n_c,
        "inter_marker_error_mm": None,
        "confidence":            round(conf, 3),
        "method":                "charuco_pnp",
    }


def _detect_markers_multipass(gray: np.ndarray, detector) -> Tuple[list, list]:
    """
    Détection multi-passes ArUco avec plusieurs prétraitements pour maximiser
    le nombre de marqueurs détectés et la qualité des coins.

    Passes :
      1. Image originale
      2. CLAHE doux  (clipLimit=2, tile 8×8)
      3. CLAHE fort  (clipLimit=4, tile 4×4)
      4. Débruitage gaussien puis CLAHE doux
      5. Filtre de netteté (sharpen)

    Pour chaque marqueur détecté dans n'importe quelle passe :
      - Affinage subpixel des coins (cornerSubPix)
      - On conserve la détection avec le plus grand périmètre apparent
        (= marqueur le plus net / le plus proche → mesure la plus fiable)

    Retourne (corners_list, ids_list) avec au plus un candidat par ID.
    """
    clahe_soft = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_hard = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    sharpen_k  = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

    passes = [
        gray,
        clahe_soft.apply(gray),
        clahe_hard.apply(gray),
        clahe_soft.apply(cv2.GaussianBlur(gray, (3, 3), 0)),
        cv2.filter2D(gray, -1, sharpen_k).clip(0, 255).astype(np.uint8),
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    best: dict = {}  # id_int → (corners_1x4x2, pixel_size)

    for img in passes:
        corners_list, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        for corners, mid in zip(corners_list, ids.flatten()):
            c = corners.reshape(-1, 1, 2).astype(np.float32)
            try:
                cv2.cornerSubPix(img, c, (3, 3), (-1, -1), criteria)
            except cv2.error:
                pass
            corners_ref = c.reshape(1, 4, 2)
            pix_sz  = _pixel_size(corners_ref)
            mid_int = int(mid)
            if mid_int not in best or pix_sz > best[mid_int][1]:
                best[mid_int] = (corners_ref, pix_sz)

    if not best:
        return [], []
    return [v[0] for v in best.values()], list(best.keys())


def _marker_center(corners) -> Tuple[float, float]:
    c = corners.reshape(4, 2).astype(np.float64)
    return float(c[:, 0].mean()), float(c[:, 1].mean())


def _pixel_size(corners) -> float:
    c = corners.reshape(4, 2).astype(np.float64)
    return float(sum(np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)) / 4.0)


def _single_pnp(corners, marker_size_m: float, K, D):
    """solvePnP IPPE_SQUARE → (rvec, tvec_mm) ou (None, None)."""
    half = marker_size_m / 2.0
    obj  = np.array([
        [-half,  half, 0], [ half,  half, 0],
        [ half, -half, 0], [-half, -half, 0],
    ], dtype=np.float64)
    img  = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    return rvec, tvec * 1000.0


def _load_board_layout(meta: dict, side: str) -> dict:
    """
    Charge le layout du board depuis metadata.
    Clé attendue : gripper_{side}_board_layout → {id: [x_mm, y_mm]}.
    """
    if not meta:
        return {}
    raw = meta.get(f"gripper_{side}_board_layout", {})
    if not raw:
        return {}
    try:
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    except Exception:
        return {}


def _pose_multi_marker(corners_list: list, ids: list, K, D,
                       marker_size_m: float, board_layout: dict) -> dict:
    """
    Calcule la pose d'un gripper depuis N marqueurs ArUco visibles.

    Méthode ① board_pnp — board layout connu ET ≥ 2 marqueurs :
      solvePnP global sur tous les coins de tous les marqueurs.
      Meilleure précision car système sur-contraint.

    Méthode ② multi_avg — pas de layout :
      solvePnP par marqueur → médiane pondérée des tvec.
      Distance inter-marqueurs en pixels vs. 3D comme métrique de qualité :
        d_3d_ij estimée = d_px_ij × z_median / f
        erreur = |d_3d_pnp_ij − d_3d_estimée_ij|

    Retourne un dict { x_mm, y_mm, z_mm, rvec, n_markers,
                       inter_marker_error_mm, confidence, method }.
    """
    marker_size_mm = marker_size_m * 1000.0
    half_mm        = marker_size_mm / 2.0

    # ── Méthode ① : solvePnP global avec board layout ────────────────────────
    if board_layout and len(corners_list) >= 2:
        obj_pts, img_pts, used = [], [], 0
        for corners, mid in zip(corners_list, ids):
            if mid not in board_layout:
                continue
            bx, by = board_layout[mid]
            obj_pts.extend([
                [bx - half_mm,  by + half_mm, 0],
                [bx + half_mm,  by + half_mm, 0],
                [bx + half_mm,  by - half_mm, 0],
                [bx - half_mm,  by - half_mm, 0],
            ])
            img_pts.extend(corners.reshape(4, 2).tolist())
            used += 1

        if used >= 2:
            obj_arr = np.array(obj_pts, dtype=np.float64)
            img_arr = np.array(img_pts, dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                obj_arr, img_arr, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, K, D)
                reproj  = float(np.linalg.norm(
                    proj.reshape(-1, 2) - img_arr, axis=1).mean())
                conf = min(1.0, max(0.1, 1.0 - reproj / 5.0)) * (1.0 if used >= 3 else 0.85)
                z    = float(tvec[2, 0]) * 1000.0
                return {
                    "x_mm": float(tvec[0, 0]) * 1000.0,
                    "y_mm": float(tvec[1, 0]) * 1000.0,
                    "z_mm": z,
                    "rvec": rvec,
                    "n_markers":             used,
                    "inter_marker_error_mm": round(reproj * z / K[0, 0], 2) if z > 0 else None,
                    "confidence":            round(conf, 3),
                    "method":                "board_pnp",
                }

    # ── Méthode ② : solvePnP par marqueur + médiane pondérée ─────────────────
    poses   = []   # (tvec_mm_flat, rvec)
    centers = []   # (cx, cy) pixels

    for corners, mid in zip(corners_list, ids):
        rvec, tvec_mm = _single_pnp(corners, marker_size_m, K, D)
        if tvec_mm is not None:
            poses.append((tvec_mm.flatten(), rvec))
        cx, cy = _marker_center(corners)
        centers.append((cx, cy))

    if poses:
        xs = [p[0][0] for p in poses]
        ys = [p[0][1] for p in poses]
        zs = [p[0][2] for p in poses]
        x_mm     = float(np.median(xs))
        y_mm     = float(np.median(ys))
        z_mm     = float(np.median(zs))
        best_rvec = min(poses, key=lambda p: p[0][2])[1]

        # Erreur inter-marqueurs : distance 3D estimée vs distance pnp
        inter_err = None
        if len(centers) >= 2 and z_mm > 0:
            f    = K[0, 0]
            errs = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    d_px      = float(np.linalg.norm(
                        np.array(centers[i]) - np.array(centers[j])))
                    d_est     = d_px * z_mm / f   # distance 3D estimée depuis profondeur médiane
                    if i < len(poses) and j < len(poses):
                        d_pnp = float(np.linalg.norm(poses[i][0] - poses[j][0]))
                        errs.append(abs(d_est - d_pnp))
            inter_err = round(float(np.mean(errs)), 2) if errs else None

        n    = len(poses)
        conf = min(1.0, 0.4 + 0.15 * n) * (1.0 if n >= 3 else 0.8)
        return {
            "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
            "rvec": best_rvec,
            "n_markers":             n,
            "inter_marker_error_mm": inter_err,
            "confidence":            round(conf, 3),
            "method":                "multi_avg" if n > 1 else "single_pnp",
        }

    # ── Fallback : profondeur depuis taille de pixel ───────────────────────────
    if corners_list:
        px_sz = _pixel_size(corners_list[0])
        if px_sz > 1:
            z_px  = (marker_size_mm * K[0, 0]) / px_sz
            cx, cy = _marker_center(corners_list[0])
            return {
                "x_mm": (cx - K[0, 2]) * z_px / K[0, 0],
                "y_mm": (cy - K[1, 2]) * z_px / K[1, 1],
                "z_mm": z_px,
                "rvec": None,
                "n_markers": 1,
                "inter_marker_error_mm": None,
                "confidence": 0.15,
                "method": "pixel",
            }

    return {}


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS VIDÉO / JSONL
# ═════════════════════════════════════════════════════════════════════════════

def _read_timestamps(jsonl_path: str) -> List[float]:
    """
    Lit les timestamps depuis un JSONL caméra.
    Supporte :
      capture_timestamp_sec (s) → × 1000 = ms   [format actuel]
      capture_time          (ms)                 [format NAS]
    """
    times = []
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return times
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("capture_time")
                if t is None:
                    t_sec = d.get("capture_timestamp_sec")
                    if t_sec is not None:
                        t = float(t_sec) * 1000.0
                if t is not None:
                    times.append(float(t))
    except Exception as e:
        logger.debug("JSONL %s : %s", jsonl_path, e)
    return times


def _find_files(cam_dir: str, cam_name: str):
    """Retourne (video_path | None, jsonl_path | None) dans cam_dir.
    Tolère "wall" comme nom alternatif de la caméra fixe "head"."""
    candidates = ("head", "wall") if cam_name == "head" else (cam_name,)
    jsonl = None
    for cand in candidates:
        p = os.path.join(cam_dir, f"{cand}.jsonl")
        if os.path.isfile(p):
            jsonl = p
            break
    video = None
    for cand in candidates:
        for ext in _VIDEO_EXTS:
            p = os.path.join(cam_dir, f"{cand}{ext}")
            if os.path.isfile(p):
                video = p
                break
        if video:
            break
    return video, jsonl


def _cameras_dir(session_path: str) -> Optional[str]:
    """Retourne le répertoire caméras (cameras/ puis videos/)."""
    for d in ("cameras", "videos"):
        p = os.path.join(session_path, d)
        if os.path.isdir(p):
            return p
    return None


# ═════════════════════════════════════════════════════════════════════════════
# DÉPLACEMENT INTER-FRAMES
# ═════════════════════════════════════════════════════════════════════════════

def _add_displacement(track: list) -> list:
    """
    Ajoute dx_mm / dy_mm / dz_mm / displacement_mm à chaque point de trajectoire
    en comparant avec le point précédent (interpolation au pas d'échantillonnage).
    Le premier point reçoit des déplacements nuls.
    """
    prev = None
    for pt in track:
        if prev is None:
            pt["dx_mm"] = pt["dy_mm"] = pt["dz_mm"] = pt["displacement_mm"] = 0.0
        else:
            dx = pt["x_mm"] - prev["x_mm"]
            dy = pt["y_mm"] - prev["y_mm"]
            dz = pt["z_mm"] - prev["z_mm"]
            pt["dx_mm"]           = round(dx, 2)
            pt["dy_mm"]           = round(dy, 2)
            pt["dz_mm"]           = round(dz, 2)
            pt["displacement_mm"] = round(math.sqrt(dx*dx + dy*dy + dz*dz), 2)
        prev = pt
    return track


# ═════════════════════════════════════════════════════════════════════════════
# TRACKING — source stéréo (caméra left → pince gauche, right → pince droite)
# ═════════════════════════════════════════════════════════════════════════════

def _track_stereo_camera(video_path: str, jsonl_path: str,
                         cam_name: str, session_path: str,
                         target_ids: Optional[set],
                         board_layout: dict,
                         imu_orient: Optional[dict],
                         charuco_board=None,
                         id_offset: int = 0,
                         env_board=None,
                         stable_env_pose: Optional[Tuple] = None) -> list:
    """
    Trackle une caméra dédiée à UNE pince.

    Pour chaque frame échantillonnée :
      1. Détecte tous les marqueurs ArUco (une seule passe multi-pré-traitement)
      2. Board env ChArUco 11×8 : localise la caméra dans le repère monde
         → world coordinates de la pince calculées si pose env disponible
      3. Filtre les marqueurs par target_ids (pince concernée)
      4. Pose ChArUco pince (subpixel) ou fallback ArUco multi-marqueurs

    Note : le déplacement inter-frames est calculé en aval par _add_displacement().
    Retourne une liste de dicts par point de trajectoire.
    """
    timestamps = _read_timestamps(jsonl_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Impossible d'ouvrir %s", video_path)
        return []

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0

    K, D, _ = _load_calibration(session_path, cam_name)
    if K is None:
        K, D = _pinhole(frame_w, frame_h)

    step     = max(1, int(fps_vid / SAMPLE_FPS))
    detector = _make_detector()
    marker_m = MARKER_SIZE_MM / 1000.0
    results  = []
    frame_no = 0
    env_detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % step == 0:
            ts = timestamps[frame_no] if frame_no < len(timestamps) else None
            if ts is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Détection unique : marqueurs pince + board environnemental
                all_corners, all_ids = _detect_markers_multipass(gray, detector)

                # ── Localisation caméra via board environnemental ─────────────
                cam_rvec = cam_tvec = None
                env_n_corners = 0
                if env_board is not None and all_corners:
                    cam_rvec, cam_tvec, env_n_corners = _detect_env_charuco_pose(
                        gray, all_corners, all_ids, env_board, K, D
                    )
                    if cam_rvec is not None:
                        env_detected_count += 1
                # Fallback : pose stable pré-calculée sur toute la vidéo
                if cam_rvec is None and stable_env_pose is not None:
                    cam_rvec, cam_tvec = stable_env_pose[0], stable_env_pose[1]

                # ── Filtrage marqueurs pince ──────────────────────────────────
                filtered_c, filtered_ids = [], []
                for c, mid in zip(all_corners, all_ids):
                    if target_ids is None or mid in target_ids:
                        filtered_c.append(c)
                        filtered_ids.append(mid)

                if filtered_c:
                    # ChArUco pince en premier (subpixel, plus précis)
                    pose = {}
                    if charuco_board is not None:
                        pose = _detect_charuco_pose(
                            gray, filtered_c, filtered_ids,
                            charuco_board, K, D, id_offset,
                        )
                    # Fallback ArUco multi-marqueurs
                    if not pose:
                        pose = _pose_multi_marker(
                            filtered_c, filtered_ids, K, D, marker_m, board_layout
                        )
                    if pose:
                        all_cx = [_marker_center(c)[0] for c in filtered_c]
                        all_cy = [_marker_center(c)[1] for c in filtered_c]
                        roll, pitch, yaw = _imu_at(imu_orient, float(ts))
                        # ── World coordinates (si board env localisé) ─────────
                        x_w = y_w = z_w = None
                        if cam_rvec is not None:
                            x_w, y_w, z_w = _to_world(
                                pose["x_mm"], pose["y_mm"], pose["z_mm"],
                                cam_rvec, cam_tvec,
                            )
                        results.append({
                            "timestamp_ms":          round(float(ts), 1),
                            "cx_norm":               round(float(np.mean(all_cx)) / frame_w, 4),
                            "cy_norm":               round(float(np.mean(all_cy)) / frame_h, 4),
                            "x_mm":                  round(pose.get("x_mm") or 0, 1),
                            "y_mm":                  round(pose.get("y_mm") or 0, 1),
                            "z_mm":                  round(pose.get("z_mm") or 0, 1),
                            "x_world_mm":            x_w,
                            "y_world_mm":            y_w,
                            "z_world_mm":            z_w,
                            "env_n_corners":         env_n_corners,
                            "imu_roll_deg":          round(roll,  2),
                            "imu_pitch_deg":         round(pitch, 2),
                            "imu_yaw_deg":           round(yaw,   2),
                            "n_markers":             pose.get("n_markers", 0),
                            "n_charuco_corners":     pose.get("n_charuco_corners", 0),
                            "inter_marker_error_mm": pose.get("inter_marker_error_mm"),
                            "confidence":            pose.get("confidence", 0.0),
                            "method":                pose.get("method", ""),
                        })

        frame_no += 1

    cap.release()
    logger.info("stéréo %s : %d poses | env board: %d/%d frames (step=%d)",
                cam_name, len(results), env_detected_count,
                frame_no // step + 1, step)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# TRACKING — caméra HEAD (voit les deux pinces simultanément)
# ═════════════════════════════════════════════════════════════════════════════

def _track_head_camera(video_path: str, jsonl_path: str,
                       session_path: str,
                       left_ids: Optional[set],
                       right_ids: Optional[set],
                       board_layout_left: dict,
                       board_layout_right: dict,
                       imu_orient_left: Optional[dict],
                       imu_orient_right: Optional[dict],
                       env_board=None,
                       stable_env_pose: Optional[Tuple] = None) -> dict:
    """
    Trackle la caméra head pour les deux pinces simultanément (ArUco IDs 0-9).

    Sur chaque frame :
      1. Détecte tous les marqueurs ArUco (une seule passe)
      2. Board env ChArUco 11×8 → pose caméra dans le repère monde
      3. Pour chaque pince (left/right) : solvePnP multi-marqueurs ArUco
         → world coordinates si board env localisé

    Retourne { 'left': [dicts], 'right': [dicts], 'stats': {...} }.
    """
    timestamps = _read_timestamps(jsonl_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Impossible d'ouvrir head %s", video_path)
        return {"left": [], "right": []}

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0

    K, D, _ = _load_calibration(session_path, "head")
    if K is None:
        K, D = _pinhole(frame_w, frame_h)

    step     = max(1, int(fps_vid / SAMPLE_FPS))
    detector = _make_detector()
    marker_m = MARKER_SIZE_MM / 1000.0
    results  = {"left": [], "right": []}
    frames_sampled = frames_with_left = frames_with_right = env_detected_count = 0
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % step == 0:
            ts = timestamps[frame_no] if frame_no < len(timestamps) else None
            if ts is not None:
                frames_sampled += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_corners, all_ids = _detect_markers_multipass(gray, detector)

                # ── Localisation caméra head via board environnemental ────────
                cam_rvec = cam_tvec = None
                env_n_corners = 0
                if env_board is not None and all_corners:
                    cam_rvec, cam_tvec, env_n_corners = _detect_env_charuco_pose(
                        gray, all_corners, all_ids, env_board, K, D
                    )
                    if cam_rvec is not None:
                        env_detected_count += 1
                if cam_rvec is None and stable_env_pose is not None:
                    cam_rvec, cam_tvec = stable_env_pose[0], stable_env_pose[1]

                frame_has_left = frame_has_right = False
                if all_corners:
                    for side, target_ids, board_layout, imu_orient in [
                        ("left",  left_ids,  board_layout_left,  imu_orient_left),
                        ("right", right_ids, board_layout_right, imu_orient_right),
                    ]:
                        filtered_c, filtered_ids = [], []
                        for c, mid in zip(all_corners, all_ids):
                            if target_ids is None or mid in target_ids:
                                filtered_c.append(c)
                                filtered_ids.append(mid)

                        if filtered_c:
                            if side == "left":
                                frame_has_left = True
                            else:
                                frame_has_right = True

                            pose = _pose_multi_marker(
                                filtered_c, filtered_ids, K, D, marker_m, board_layout
                            )
                            if pose:
                                all_cx = [_marker_center(c)[0] for c in filtered_c]
                                all_cy = [_marker_center(c)[1] for c in filtered_c]
                                roll, pitch, yaw = _imu_at(
                                    imu_orient_left if side == "left" else imu_orient_right,
                                    float(ts)
                                )
                                x_w = y_w = z_w = None
                                if cam_rvec is not None:
                                    x_w, y_w, z_w = _to_world(
                                        pose["x_mm"], pose["y_mm"], pose["z_mm"],
                                        cam_rvec, cam_tvec,
                                    )
                                results[side].append({
                                    "timestamp_ms":          round(float(ts), 1),
                                    "cx_norm":               round(float(np.mean(all_cx)) / frame_w, 4),
                                    "cy_norm":               round(float(np.mean(all_cy)) / frame_h, 4),
                                    "x_mm":                  round(pose.get("x_mm") or 0, 1),
                                    "y_mm":                  round(pose.get("y_mm") or 0, 1),
                                    "z_mm":                  round(pose.get("z_mm") or 0, 1),
                                    "x_world_mm":            x_w,
                                    "y_world_mm":            y_w,
                                    "z_world_mm":            z_w,
                                    "env_n_corners":         env_n_corners,
                                    "imu_roll_deg":          round(roll,  2),
                                    "imu_pitch_deg":         round(pitch, 2),
                                    "imu_yaw_deg":           round(yaw,   2),
                                    "n_markers":             pose.get("n_markers", 0),
                                    "n_charuco_corners":     0,
                                    "inter_marker_error_mm": pose.get("inter_marker_error_mm"),
                                    "confidence":            pose.get("confidence", 0.0),
                                    "method":                pose.get("method", ""),
                                })

                if frame_has_left:
                    frames_with_left += 1
                if frame_has_right:
                    frames_with_right += 1

        frame_no += 1

    cap.release()

    n = frames_sampled or 1
    detection_stats = {
        "frames_sampled":    frames_sampled,
        "frames_with_left":  frames_with_left,
        "frames_with_right": frames_with_right,
        "frames_with_any":   frames_with_left + frames_with_right - min(frames_with_left, frames_with_right),
        "left_rate":         round(frames_with_left  / n, 4),
        "right_rate":        round(frames_with_right / n, 4),
        "any_rate":          round((frames_with_left + frames_with_right - min(frames_with_left, frames_with_right)) / n, 4),
    }

    env_rate = round(env_detected_count / (frames_sampled or 1), 4)
    detection_stats["env_detected_frames"] = env_detected_count
    detection_stats["env_rate"]            = env_rate

    logger.info(
        "head : left=%d poses, right=%d poses sur %d frames | "
        "env board: %d frames (%.0f%%)",
        len(results["left"]), len(results["right"]), frames_sampled,
        env_detected_count, env_rate * 100,
    )
    results["stats"] = detection_stats
    return results


# ═════════════════════════════════════════════════════════════════════════════
# CORRÉLATION STÉRÉO ↔ HEAD
# ═════════════════════════════════════════════════════════════════════════════

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _optimal_lag_ms(a: np.ndarray, b: np.ndarray, dt_ms: float,
                    max_lag_ms: float = 1500.0) -> float:
    n       = len(a)
    a_z     = (a - a.mean()) / (a.std() + 1e-8)
    b_z     = (b - b.mean()) / (b.std() + 1e-8)
    corr    = np.correlate(a_z, b_z, mode="full")
    lags_ms = np.arange(-(n - 1), n) * dt_ms
    mask    = np.abs(lags_ms) <= max_lag_ms
    best    = int(np.argmax(np.abs(corr[mask])))
    return float(lags_ms[mask][best])


def _correlate_trajectories(stereo: list, head: list) -> dict:
    """
    Corrèle la trajectoire stéréo et la trajectoire head pour une pince.

    Interpole les deux trajectoires sur une grille temporelle commune,
    puis calcule Pearson par axe + lag optimal + RMSE.

    Retourne un dict avec pearson_x/y/z, lag_ms, rmse_x/y/z_mm, n_points, coherent.
    """
    empty = {
        "pearson_x": 0.0, "pearson_y": 0.0, "pearson_z": 0.0,
        "lag_ms": 0.0,
        "rmse_x_mm": None, "rmse_y_mm": None, "rmse_z_mm": None,
        "n_points": 0, "coherent": False,
    }

    if len(stereo) < 5 or len(head) < 5:
        return {**empty, "note": f"Pas assez de points (stereo={len(stereo)}, head={len(head)})"}

    def _extract(traj):
        t  = np.array([r["timestamp_ms"] for r in traj])
        xs = np.array([r["x_mm"] for r in traj], dtype=float)
        ys = np.array([r["y_mm"] for r in traj], dtype=float)
        zs = np.array([r["z_mm"] for r in traj], dtype=float)
        return t, xs, ys, zs

    st, sx, sy, sz = _extract(stereo)
    ht, hx, hy, hz = _extract(head)

    t_lo = max(st.min(), ht.min())
    t_hi = min(st.max(), ht.max())
    if t_hi - t_lo < 2000:
        return {**empty, "note": f"Overlap < 2s ({(t_hi - t_lo)/1000:.1f}s)"}

    dt_ms  = 1000.0 / SAMPLE_FPS
    t_grid = np.arange(t_lo, t_hi, dt_ms)

    si_x = np.interp(t_grid, st, sx)
    si_y = np.interp(t_grid, st, sy)
    si_z = np.interp(t_grid, st, sz)
    hi_x = np.interp(t_grid, ht, hx)
    hi_y = np.interp(t_grid, ht, hy)
    hi_z = np.interp(t_grid, ht, hz)

    lag_ms = _optimal_lag_ms(si_x, hi_x, dt_ms)
    px     = _pearson(si_x, hi_x)
    py     = _pearson(si_y, hi_y)
    pz     = _pearson(si_z, hi_z)
    avg    = (abs(px) + abs(py) + abs(pz)) / 3.0

    def _rmse(a, b):
        return round(float(np.sqrt(np.mean((a - b) ** 2))), 2)

    return {
        "pearson_x":  round(px, 4),
        "pearson_y":  round(py, 4),
        "pearson_z":  round(pz, 4),
        "lag_ms":     round(lag_ms, 1),
        "rmse_x_mm":  _rmse(si_x, hi_x),
        "rmse_y_mm":  _rmse(si_y, hi_y),
        "rmse_z_mm":  _rmse(si_z, hi_z),
        "n_points":   len(t_grid),
        "coherent":   bool(avg >= MIN_CROSS_CORR),
    }


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION CROISÉE — stéréo ChArUco × head ArUco
# ═════════════════════════════════════════════════════════════════════════════

def _cross_validate_trajectories(stereo: list, head: list,
                                  pose_stereo: Optional[Tuple] = None,
                                  pose_head:   Optional[Tuple] = None,
                                  max_dist_mm: float = CROSSVAL_MAX_DIST_MM
                                  ) -> Tuple[list, dict]:
    """
    Validation croisée point par point entre la trajectoire stéréo (ChArUco)
    et la trajectoire head (ArUco).

    Comparaison en coordonnées MONDE (repère board env) quand les poses caméra
    sont disponibles — ce qui est le seul mode valide, car les positions stéréo
    et head sont exprimées dans des espaces caméra différents.

    pose_stereo / pose_head : (rvec, tvec) issus de _compute_stable_env_pose,
      avec P_cam = R · P_world + t  ⟹  P_world = Rᵀ · (P_cam − t).

    Si les poses sont absentes, la comparaison se fait en espace caméra stéréo
    (résultat non significatif — utilisé uniquement en dernier recours).

    Retourne (final_track, stats).
    """
    empty_stats = {
        "n_stereo": len(stereo), "n_head": len(head),
        "n_tested": 0, "n_validated": 0,
        "agreement_rate": 0.0,
        "mean_dist_mm": None, "max_dist_mm_obs": None,
        "threshold_mm": max_dist_mm,
        "world_space": False,
    }
    if len(stereo) < 3 or len(head) < 3:
        return [], empty_stats

    # ── Préparation de la transformation vers le repère monde ────────────────
    use_world = pose_stereo is not None and pose_head is not None
    if use_world:
        R_s, _ = cv2.Rodrigues(pose_stereo[0])
        t_s    = pose_stereo[1]           # (3,1) en mètres
        R_h, _ = cv2.Rodrigues(pose_head[0])
        t_h    = pose_head[1]

        def _world(x_mm, y_mm, z_mm, R, t):
            """Coordonnées caméra (mm) → repère monde (mm)."""
            P = np.array([[x_mm], [y_mm], [z_mm]]) / 1000.0
            return (R.T @ (P - t)).flatten() * 1000.0

        # Déplacement entre caméras (pour log)
        cam_s_world = _world(0, 0, 0, R_s, t_s)
        cam_h_world = _world(0, 0, 0, R_h, t_h)
        cam_dist_mm = float(np.linalg.norm(cam_s_world - cam_h_world))
        logger.info(
            "Cross-val world-space | dist inter-caméras=%.0f mm | "
            "cam_stereo=[%.0f,%.0f,%.0f] cam_head=[%.0f,%.0f,%.0f] (mm)",
            cam_dist_mm,
            cam_s_world[0], cam_s_world[1], cam_s_world[2],
            cam_h_world[0], cam_h_world[1], cam_h_world[2],
        )
    else:
        logger.warning(
            "Cross-val : poses caméra manquantes → comparaison en espace caméra "
            "(distances non significatives)"
        )

    ht = np.array([r["timestamp_ms"] for r in head], dtype=float)
    hx = np.array([r["x_mm"]        for r in head], dtype=float)
    hy = np.array([r["y_mm"]        for r in head], dtype=float)
    hz = np.array([r["z_mm"]        for r in head], dtype=float)

    validated = []
    dists     = []

    for pt in stereo:
        ts = float(pt["timestamp_ms"])
        if ts < ht[0] or ts > ht[-1]:
            continue

        hx_i = float(np.interp(ts, ht, hx))
        hy_i = float(np.interp(ts, ht, hy))
        hz_i = float(np.interp(ts, ht, hz))

        if use_world:
            ws = _world(pt["x_mm"], pt["y_mm"], pt["z_mm"], R_s, t_s)
            wh = _world(hx_i, hy_i, hz_i, R_h, t_h)
            dist = float(np.linalg.norm(ws - wh))
        else:
            dist = math.sqrt(
                (pt["x_mm"] - hx_i) ** 2 +
                (pt["y_mm"] - hy_i) ** 2 +
                (pt["z_mm"] - hz_i) ** 2
            )
        dists.append(dist)

        if dist <= max_dist_mm:
            merged = {**pt,
                      "head_x_mm":        round(hx_i, 1),
                      "head_y_mm":        round(hy_i, 1),
                      "head_z_mm":        round(hz_i, 1),
                      "crossval_dist_mm": round(dist, 2),
                      "validated":        True}
            validated.append(merged)

    _add_displacement(validated)

    n_tested = len(dists)
    n_valid  = len(validated)
    stats = {
        "n_stereo":        len(stereo),
        "n_head":          len(head),
        "n_tested":        n_tested,
        "n_validated":     n_valid,
        "agreement_rate":  round(n_valid / n_tested, 4) if n_tested > 0 else 0.0,
        "mean_dist_mm":    round(float(np.mean(dists)), 2) if dists else None,
        "max_dist_mm_obs": round(float(np.max(dists)),  2) if dists else None,
        "threshold_mm":    max_dist_mm,
        "world_space":     use_world,
    }
    return validated, stats


# ═════════════════════════════════════════════════════════════════════════════
# CSV / JSON OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def _write_csv(session_path: str, rows: list) -> str:
    path = os.path.join(session_path, OUTPUT_CSV_NAME)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_correlation_json(session_path: str, corr: dict) -> str:
    path = os.path.join(session_path, "gripper_correlation.json")
    with open(path, "w") as f:
        json.dump(corr, f, indent=2, default=str)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

def run(session_path: str, meta: dict = None) -> dict:
    """
    Tracking des grippers en trois passes successives.

    Passe 1 — Stéréo ChArUco (source primaire, haute précision) :
      cameras/left  → pince gauche  (IDs ChArUco 5-9, offset 5 → board 0-4)
      cameras/right → pince droite  (IDs ChArUco 0-4, offset 0 → board 0-4)
      Méthode : interpolateCornersCharuco → estimatePoseCharucoBoard (subpixel)
      Fallback : solvePnP multi-marqueurs ArUco si ChArUco insuffisant.

    Passe 2 — Head ArUco (source secondaire, vue globale des deux pinces) :
      cameras/head → pince gauche (IDs ArUco 5-9) ET pince droite (IDs 0-4)
      Méthode : solvePnP multi-marqueurs ArUco uniquement (pas de ChArUco
      car les deux boards sont visibles simultanément et interfèrent).

    Passe 3 — Validation croisée (trajectoire finale à 100%) :
      Pour chaque frame stéréo : interpolation head au même timestamp,
      distance 3D calculée. Si distance ≤ CROSSVAL_MAX_DIST_MM → validé.
      Le déplacement inter-frames est recalculé sur la trajectoire finale seule.

    CSV de sortie (trois sections identifiées par la colonne source) :
      "stereo_charuco" — toutes les poses stéréo ChArUco (debug)
      "head_aruco"     — toutes les poses head ArUco (debug)
      "final"          — uniquement les points cross-validés (trajectoire à 100%)
    """
    if not HAS_CV2:
        return {"error": "OpenCV absent"}

    cam_dir = _cameras_dir(session_path)
    if cam_dir is None:
        return {"error": "Pas de dossier cameras/ ou videos/"}

    # ── Orientation IMU ────────────────────────────────────────────────────────
    orient_left  = _load_sensor_orient(session_path, "left")
    orient_right = _load_sensor_orient(session_path, "right")

    sensor_summary = {
        "left":  ({"n_valid": orient_left["n_valid"],  "n_total": orient_left["n_total"]}
                  if orient_left  else None),
        "right": ({"n_valid": orient_right["n_valid"], "n_total": orient_right["n_total"]}
                  if orient_right else None),
    }

    # ── IDs et board layouts (surchargeables par les métadonnées) ─────────────
    left_ids  = GRIPPER_IDS_LEFT.copy()
    right_ids = GRIPPER_IDS_RIGHT.copy()
    if meta:
        if meta.get("gripper_left_aruco_ids"):
            left_ids  = set(int(i) for i in meta["gripper_left_aruco_ids"])
        if meta.get("gripper_right_aruco_ids"):
            right_ids = set(int(i) for i in meta["gripper_right_aruco_ids"])

    board_layout_left  = _load_board_layout(meta, "left")
    board_layout_right = _load_board_layout(meta, "right")

    # ── Boards ChArUco ────────────────────────────────────────────────────────
    # Board pince : 3×3, IDs 0-4 (offset appliqué par caméra)
    charuco_board = _make_charuco_board() if HAS_CV2 else None
    # Board environnemental : 11×8, IDs ENV_CHARUCO_FIRST_ID+
    env_board     = _make_env_charuco_board() if HAS_CV2 else None

    # ══════════════════════════════════════════════════════════════════════════
    # PASSE 0 — Estimation des poses stables des caméras via le board env
    #
    # On scanne chaque vidéo à ~1 fps pour trouver le board environnemental et
    # en déduire la pose stable de chaque caméra dans le repère monde.
    # Ces poses sont utilisées :
    #   • comme fallback de localisation par frame dans les passes 1 et 2
    #   • pour la cross-validation en coordonnées monde (passe 3)
    # ══════════════════════════════════════════════════════════════════════════
    stable_env_poses = {}   # {cam_name: (rvec, tvec, n_det) | (None, None, 0)}

    head_video_pre, _ = _find_files(cam_dir, "head")
    for cam_name in ("left", "right", "head"):
        vid, _ = _find_files(cam_dir, cam_name)
        if vid and env_board:
            r, t, n = _compute_stable_env_pose(vid, cam_name, session_path, env_board)
            stable_env_poses[cam_name] = (r, t, n)
        else:
            stable_env_poses[cam_name] = (None, None, 0)

    # Déplacement inter-caméras (log informatif)
    if all(stable_env_poses[k][0] is not None for k in ("left", "head")):
        def _cam_origin(pose):
            R, _ = cv2.Rodrigues(pose[0]); t = pose[1]
            return (R.T @ (np.zeros((3, 1)) - t)).flatten() * 1000
        for pair in [("left", "head"), ("right", "head"), ("left", "right")]:
            a, b = pair
            if stable_env_poses[a][0] is not None and stable_env_poses[b][0] is not None:
                d = np.linalg.norm(_cam_origin(stable_env_poses[a]) -
                                   _cam_origin(stable_env_poses[b]))
                logger.info("Distance inter-caméras %s↔%s = %.0f mm", a, b, d)

    # ══════════════════════════════════════════════════════════════════════════
    # PASSE 1 — Stéréo ChArUco : left → pince gauche, right → pince droite
    # ══════════════════════════════════════════════════════════════════════════
    stereo_tracks  = {"left": [], "right": []}
    stereo_cameras = {}

    for side, cam_ids, id_off, orient, layout in [
        ("left",  left_ids,  CHARUCO_FIRST_LEFT,  orient_left,  board_layout_left),
        ("right", right_ids, CHARUCO_FIRST_RIGHT, orient_right, board_layout_right),
    ]:
        cam_video, cam_jsonl = _find_files(cam_dir, side)
        sp_pose = stable_env_poses.get(side)
        stable  = (sp_pose[0], sp_pose[1]) if sp_pose and sp_pose[0] is not None else None
        if cam_video:
            track = _track_stereo_camera(
                cam_video, cam_jsonl, side, session_path,
                cam_ids, layout, orient,
                charuco_board=charuco_board,
                id_offset=id_off,
                env_board=env_board,
                stable_env_pose=stable,
            )
            stereo_tracks[side] = _add_displacement(track)
            stereo_cameras[side] = side
            logger.info("Passe 1 — %s : %d poses ChArUco", side, len(track))

    # ══════════════════════════════════════════════════════════════════════════
    # PASSE 2 — Head ArUco : deux pinces simultanées (IDs 0-9)
    # ══════════════════════════════════════════════════════════════════════════
    head_tracks = {"left": [], "right": []}
    head_video, head_jsonl = _find_files(cam_dir, "head")
    has_head = bool(head_video)

    if head_video:
        hp = stable_env_poses.get("head")
        stable_head = (hp[0], hp[1]) if hp and hp[0] is not None else None
        head_tracks = _track_head_camera(
            head_video, head_jsonl, session_path,
            left_ids, right_ids,
            board_layout_left, board_layout_right,
            orient_left, orient_right,
            env_board=env_board,
            stable_env_pose=stable_head,
        )
        logger.info("Passe 2 — head ArUco : left=%d poses right=%d poses",
                    len(head_tracks["left"]), len(head_tracks["right"]))

    # ══════════════════════════════════════════════════════════════════════════
    # PASSE 3 — Validation croisée en coordonnées monde
    #
    # Les poses caméra (stéréo + head) permettent de ramener les positions de
    # chaque source dans le repère monde commun (board env) avant de calculer
    # la distance de disagreement.  Sans poses disponibles, fallback stéréo.
    # ══════════════════════════════════════════════════════════════════════════
    final_tracks    = {"left": [], "right": []}
    crossval_stats  = {}

    for side in ("left", "right"):
        stereo = stereo_tracks[side]
        head   = head_tracks[side]

        # Poses pour la cross-val : stéréo = caméra "side", head = caméra "head"
        sp = stable_env_poses.get(side)
        hp = stable_env_poses.get("head")
        pose_s = (sp[0], sp[1]) if sp and sp[0] is not None else None
        pose_h = (hp[0], hp[1]) if hp and hp[0] is not None else None

        if pose_s is not None and pose_h is not None and len(stereo) >= 3 and len(head) >= 3:
            # Cross-validation en world-space (mode normal)
            final, stats = _cross_validate_trajectories(
                stereo, head, pose_stereo=pose_s, pose_head=pose_h
            )
        elif stereo:
            # Pas de pose caméra disponible : stéréo ChArUco utilisé directement
            final = [{**pt, "validated": False,
                      "head_x_mm": None, "head_y_mm": None, "head_z_mm": None,
                      "crossval_dist_mm": None}
                     for pt in stereo]
            _add_displacement(final)
            n = len(stereo)
            stats = {
                "n_stereo": n, "n_head": len(head),
                "n_tested": 0, "n_validated": n,
                "agreement_rate": 1.0,
                "mean_dist_mm": None, "max_dist_mm_obs": None,
                "threshold_mm": CROSSVAL_MAX_DIST_MM,
                "world_space": False,
                "note": "board env non détecté → stéréo utilisé comme final",
            }
            logger.info(
                "Passe 3 — %s : board env non localisé → stéréo final (%d poses)", side, n
            )
        else:
            final, stats = [], {
                "n_stereo": 0, "n_head": 0, "n_tested": 0, "n_validated": 0,
                "agreement_rate": 0.0, "mean_dist_mm": None, "max_dist_mm_obs": None,
                "threshold_mm": CROSSVAL_MAX_DIST_MM, "world_space": False,
            }

        final_tracks[side]   = final
        crossval_stats[side] = stats
        if stats.get("world_space"):
            logger.info(
                "Passe 3 — cross-val world %s : %d/%d validés (%.0f%%) | "
                "dist moy=%.1fmm max=%.1fmm seuil=%.0fmm",
                side, stats["n_validated"], stats["n_tested"],
                stats["agreement_rate"] * 100,
                stats["mean_dist_mm"] or 0,
                stats["max_dist_mm_obs"] or 0,
                stats["threshold_mm"],
            )

    # ── Corrélation Pearson stéréo ↔ head (métrique complémentaire) ───────────
    corr_left  = _correlate_trajectories(stereo_tracks["left"],  head_tracks["left"])
    corr_right = _correlate_trajectories(stereo_tracks["right"], head_tracks["right"])

    trajectory_correlation = {
        "left":          corr_left,
        "right":         corr_right,
        "both_coherent": corr_left["coherent"] and corr_right["coherent"],
        "any_coherent":  corr_left["coherent"] or  corr_right["coherent"],
    }

    # ── Assemblage du CSV (trois sections) ────────────────────────────────────
    csv_rows = []

    def _row(r, side, source, cam, *, validated=False):
        return {
            "timestamp_ms":          r["timestamp_ms"],
            "gripper_side":          side,
            "source":                source,
            "source_camera":         cam,
            "cx_norm":               r.get("cx_norm", ""),
            "cy_norm":               r.get("cy_norm", ""),
            "x_mm":                  r["x_mm"],
            "y_mm":                  r["y_mm"],
            "z_mm":                  r["z_mm"],
            "dx_mm":                 r.get("dx_mm",           0.0),
            "dy_mm":                 r.get("dy_mm",           0.0),
            "dz_mm":                 r.get("dz_mm",           0.0),
            "displacement_mm":       r.get("displacement_mm", 0.0),
            "head_x_mm":             r.get("head_x_mm",       ""),
            "head_y_mm":             r.get("head_y_mm",       ""),
            "head_z_mm":             r.get("head_z_mm",       ""),
            "crossval_dist_mm":      r.get("crossval_dist_mm",""),
            "validated":             validated,
            "x_world_mm":            r.get("x_world_mm",      ""),
            "y_world_mm":            r.get("y_world_mm",      ""),
            "z_world_mm":            r.get("z_world_mm",      ""),
            "env_n_corners":         r.get("env_n_corners",    0),
            "imu_roll_deg":          r.get("imu_roll_deg",  0.0),
            "imu_pitch_deg":         r.get("imu_pitch_deg", 0.0),
            "imu_yaw_deg":           r.get("imu_yaw_deg",   0.0),
            "n_markers":             r.get("n_markers",       0),
            "n_charuco_corners":     r.get("n_charuco_corners", 0),
            "inter_marker_error_mm": r.get("inter_marker_error_mm"),
            "confidence":            r.get("confidence",     0.0),
            "method":                r.get("method",          ""),
        }

    for side in ("left", "right"):
        for r in stereo_tracks[side]:
            csv_rows.append(_row(r, side, "stereo_charuco", f"cam_{side}"))
        for r in head_tracks[side]:
            csv_rows.append(_row(r, side, "head_aruco", "head"))
        for r in final_tracks[side]:
            csv_rows.append(_row(r, side, "final", "stereo+head", validated=True))

    csv_rows.sort(key=lambda r: (float(r["timestamp_ms"]), r["gripper_side"], r["source"]))

    csv_path  = None
    corr_path = None
    if csv_rows and WRITE_OUTPUTS:
        csv_path  = _write_csv(session_path, csv_rows)
        corr_path = _write_correlation_json(session_path, trajectory_correlation)
        n_final = sum(1 for r in csv_rows if r["source"] == "final")
        logger.info(
            "%s — CSV: %d lignes totales (%d final) | "
            "corr Pearson: left=%s right=%s",
            os.path.basename(session_path), len(csv_rows), n_final,
            "OK" if corr_left["coherent"]  else "KO",
            "OK" if corr_right["coherent"] else "KO",
        )

    head_det_stats = head_tracks.get("stats", {})

    env_pose_summary = {
        k: {"n_detections": v[2], "available": v[0] is not None}
        for k, v in stable_env_poses.items()
    }

    return {
        "imu_available":          orient_left is not None or orient_right is not None,
        "sensor_summary":         sensor_summary,
        "stereo_cameras":         stereo_cameras,
        "head_camera":            has_head,
        # Passe 0 — poses stables board env
        "env_pose_summary":       env_pose_summary,
        # Passe 1 — stéréo ChArUco
        "stereo_left_poses":      len(stereo_tracks["left"]),
        "stereo_right_poses":     len(stereo_tracks["right"]),
        # Passe 2 — head ArUco
        "head_left_poses":        len(head_tracks["left"]),
        "head_right_poses":       len(head_tracks["right"]),
        "head_detection_stats":   head_det_stats,
        # Passe 3 — trajectoire finale cross-validée
        "final_left_poses":       len(final_tracks["left"]),
        "final_right_poses":      len(final_tracks["right"]),
        "crossval_stats":         crossval_stats,
        # Métriques globales
        "left_detected":          bool(final_tracks["left"]  or stereo_tracks["left"]),
        "right_detected":         bool(final_tracks["right"] or stereo_tracks["right"]),
        "trajectory_correlation": trajectory_correlation,
        "cross_validated":        bool(final_tracks["left"] or final_tracks["right"]),
        "n_frames_tracked":       len(csv_rows),
        "csv_path":               csv_path,
        "correlation_json_path":  corr_path,
    }
