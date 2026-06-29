"""
checks.py — Contrôles techniques appliqués à chaque session.

Chaque check retourne : {"ok": bool, "detail": str, "value": any}

Checks structurels (1–7, toujours actifs) :
  1. folder_exists          — dossier session présent sur le volume NAS
  2. metadata_valid         — metadata.json présent et JSON valide
  3. files_present          — au moins un fichier de données
  4. no_empty_files         — aucun fichier de taille 0
  5. min_size               — taille totale > MIN_SESSION_BYTES
  6. duration_ok            — durée dans [MIN_DURATION_S, MAX_DURATION_S]
  7. scenario_match         — scenario_id cohérent entre DB et metadata

Check capture (8, toujours actif si analysis.json présent) :
  8. analysis_report        — rapport de capture : errors/warnings V4L2, FPS, drift,
                              sync camera/sensor. Erreurs = rejet critique.
                              Données utilisées par le scoring (sync_inter_cameras,
                              video_stability) en priorité sur les métriques JSONL.

Checks flux (9–11, actifs si numpy+pandas installés) :
  9. frame_drops            — détection des drops par caméra via timestamps JSONL
  10. temporal_drift        — dérive d'horloge caméras vs FPS nominal
  11. gripper_flux_coherence — cohérence capteur gripper ↔ flux optique caméra (NCC)

Checks vision (12–13, actif si OpenCV disponible et ENABLE_VISION_CHECKS=true) :
  12. gripper_tracking      — tracking complet des pinces : ArUco dans fix_*, ChArUco dans
                              gripper_*, cross-validation des trajectoires, CSV de sortie
  13. gripper_label_inversion — détecte si les caméras left/right sont étiquetées à l'envers
                              en comparant le flux optique pince ↔ moitiés gauche/droite du head
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Variables d'environnement ────────────────────────────────────────────────
NAS_DIR        = os.environ.get("NAS_SESSIONS_DIR",  "/nas/sessions")
MIN_DURATION_S = float(os.environ.get("MIN_DURATION_S",    "10"))
MAX_DURATION_S = float(os.environ.get("MAX_DURATION_S",    "7200"))
MIN_SIZE_BYTES = int(os.environ.get("MIN_SESSION_BYTES",   "1024"))
ENABLE_VISION  = os.environ.get("ENABLE_VISION_CHECKS",   "true").lower() == "true"

# Seuils checks flux
MAX_FRAME_DROP_PCT    = float(os.environ.get("MAX_FRAME_DROP_PCT",    "10.0"))
MAX_SEVERE_DROPS      = int(os.environ.get("MAX_SEVERE_DROPS",        "5"))
MAX_DRIFT_FPS         = float(os.environ.get("MAX_DRIFT_FPS",         "1.0"))
MIN_GRIPPER_COHERENCE = float(os.environ.get("MIN_GRIPPER_COHERENCE", "0.30"))

# Seuils checks vision
MIN_ARUCO_DETECTION_RATE   = float(os.environ.get("MIN_ARUCO_DETECTION_RATE",   "0.30"))
MIN_CHARUCO_DETECTION_RATE = float(os.environ.get("MIN_CHARUCO_DETECTION_RATE", "0.50"))
MIN_CROSS_CORR_OK          = float(os.environ.get("TRACKING_MIN_CROSS_CORR",    "0.70"))

# Score de qualité minimum pour qu'une session passe (0-100)
MIN_QUALITY_SCORE = float(os.environ.get("MIN_QUALITY_SCORE", "40.0"))
# Checks binaires : un seul KO = session immédiatement rejetée
CRITICAL_CHECKS = {"folder_exists", "files_present", "no_empty_files", "analysis_report"}

# Seuils check inversion labels (check 12)
LABEL_SAMPLE_FPS       = float(os.environ.get("LABEL_CHECK_SAMPLE_FPS",   "2.0"))
LABEL_INVERSION_MARGIN = float(os.environ.get("LABEL_INVERSION_MIN_DIFF", "0.12"))
LABEL_FLOW_W           = int(os.environ.get("LABEL_FLOW_WIDTH",  "320"))
LABEL_FLOW_H           = int(os.environ.get("LABEL_FLOW_HEIGHT", "200"))

# ── Imports optionnels ───────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy/pandas non disponibles — checks flux (8–10) désactivés")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV non disponible — checks vision (11–12) désactivés")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ok(detail="", value=None):
    return {"ok": True,  "detail": str(detail), "value": value}

def _fail(detail="", value=None):
    return {"ok": False, "detail": str(detail), "value": value}

def _skip(reason=""):
    return {"ok": True,  "detail": f"[IGNORÉ] {reason}", "value": None}


def _read_analysis_json(session_path: str) -> dict | None:
    """Lit et retourne analysis.json de la session, ou None si absent/invalide."""
    path = os.path.join(session_path, "analysis.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Impossible de lire analysis.json pour %s : %s", session_path, e)
        return None


def _cameras_dir(session_path: str):
    """Retourne le répertoire caméras : essaie cameras/ puis videos/."""
    for d in ("cameras", "videos"):
        p = os.path.join(session_path, d)
        if os.path.isdir(p):
            return p
    return None


def _find_video_file(directory: str, cam_name: str):
    """Cherche un fichier vidéo pour la caméra donnée dans le répertoire.
    Tolère "wall" comme nom alternatif de la caméra fixe "head"."""
    candidates = ("head", "wall") if cam_name == "head" else (cam_name,)
    for cand in candidates:
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".h264"):
            p = os.path.join(directory, f"{cand}{ext}")
            if os.path.isfile(p):
                return p
    return None


# ═════════════════════════════════════════════════════════════════════════════
# CHECKS STRUCTURELS (1–7)
# ═════════════════════════════════════════════════════════════════════════════

def _check_folder(session_path):
    if os.path.isdir(session_path):
        return _ok(session_path)
    return _fail(f"Dossier introuvable : {session_path}")


def _check_metadata(session_path):
    """Retourne (check_result, meta_dict | None).
    Supporte metadata.json (format NAS) et config.json (format généré).
    """
    for fname in ("metadata.json", "config.json"):
        meta_path = os.path.join(session_path, fname)
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return _ok(f"{fname} valide", list(meta.keys())[:5]), meta
        except Exception as e:
            return _fail(f"{fname} invalide : {e}"), None

    # Essaie de combiner mission.json + config.json pour le format généré
    combined = {}
    for fname in ("mission.json", "config.json"):
        p = os.path.join(session_path, fname)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    combined.update(json.load(f))
            except Exception:
                pass
    if combined:
        return _ok("mission.json+config.json valides", list(combined.keys())[:5]), combined

    return _fail("metadata.json / config.json absents"), None


def _scan_files(session_path):
    """Retourne (all_files, empty_files, total_bytes)."""
    all_files, empty_files, total_bytes = [], [], 0
    try:
        for root, _, files in os.walk(session_path):
            for fname in files:
                if fname in ("metadata.json", "config.json", "mission.json",
                             "treatment.json", "analysis.json", "result.json"):
                    continue
                fpath = os.path.join(root, fname)
                size  = os.path.getsize(fpath)
                all_files.append(fname)
                total_bytes += size
                if size == 0:
                    empty_files.append(fname)
    except Exception as e:
        logger.warning("Erreur scan %s : %s", session_path, e)
    return all_files, empty_files, total_bytes


def _check_duration(meta, db_dur):
    duration = None
    if meta and isinstance(meta.get("duration_seconds"), (int, float)):
        duration = float(meta["duration_seconds"])
    elif db_dur is not None:
        duration = float(db_dur)
    if duration is None:
        return _ok("Durée non disponible — ignoré")
    if MIN_DURATION_S <= duration <= MAX_DURATION_S:
        return _ok(f"{duration:.1f}s", duration)
    return _fail(
        f"Hors limites : {duration:.1f}s (attendu {MIN_DURATION_S}–{MAX_DURATION_S}s)",
        duration,
    )


def _check_scenario(meta, db_scen):
    if meta and db_scen:
        meta_scen = meta.get("scenario_id") or meta.get("scenario")
        if meta_scen and str(meta_scen) == str(db_scen):
            return _ok(f"scenario_id={db_scen}")
        if meta_scen:
            return _fail(f"DB={db_scen} ≠ metadata={meta_scen}")
    return _ok("Vérification scénario ignorée (données manquantes)")


# ═════════════════════════════════════════════════════════════════════════════
# CHECKS FLUX (8–10) — nécessitent numpy + pandas
# ═════════════════════════════════════════════════════════════════════════════

def _read_jsonl_timestamps(jsonl_path):
    """Retourne une liste triée de timestamps (ms) depuis un fichier .jsonl caméra.
    Supporte capture_time (ms, format NAS) et capture_timestamp_sec (s, format généré).
    """
    times = []
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("capture_time")
                if t is None:
                    t_sec = d.get("capture_timestamp_sec")
                    if t_sec is not None:
                        t = float(t_sec) * 1000.0  # convertir secondes → ms
                if t is not None:
                    times.append(float(t))
    except Exception as e:
        logger.debug("Erreur JSONL %s : %s", jsonl_path, e)
    return sorted(times)


def _check_frame_drops(session_path, meta):
    """
    Détecte les frame drops sur toutes les caméras à partir des timestamps JSONL.

    Un drop est un gap > 2.5× l'intervalle nominal (ex: >83ms à 30fps).
    Un drop sévère est un gap > 10× l'intervalle (ex: >333ms à 30fps).
    """
    if not HAS_NUMPY:
        return _skip("numpy non disponible")

    cam_dir = _cameras_dir(session_path)
    if cam_dir is None:
        return _skip("Pas de répertoire caméras")

    fps_nominal      = float((meta or {}).get("fps", (meta or {}).get("frame_rate", 30)))
    expected_ms      = 1000.0 / fps_nominal
    drop_thresh_ms   = expected_ms * 2.5
    severe_thresh_ms = expected_ms * 10.0

    cameras = {}
    for fname in sorted(os.listdir(cam_dir)):
        if not fname.endswith(".jsonl"):
            continue
        cam   = fname[:-6]
        times = _read_jsonl_timestamps(os.path.join(cam_dir, fname))
        if len(times) < 2:
            cameras[cam] = {"frames": len(times), "drops": 0, "severe_drops": 0, "drop_pct": 0.0}
            continue

        gaps     = np.diff(times)
        n_drops  = int((gaps > drop_thresh_ms).sum())
        n_severe = int((gaps > severe_thresh_ms).sum())
        max_gap  = float(gaps.max())
        drop_pct = n_drops / len(gaps) * 100.0

        cameras[cam] = {
            "frames":       len(times),
            "drops":        n_drops,
            "severe_drops": n_severe,
            "max_gap_ms":   round(max_gap, 1),
            "drop_pct":     round(drop_pct, 2),
        }

    if not cameras:
        return _skip("Aucun fichier .jsonl dans le répertoire caméras")

    total_drops  = sum(c["drops"]        for c in cameras.values())
    total_severe = sum(c["severe_drops"] for c in cameras.values())
    worst_cam    = max(cameras, key=lambda k: cameras[k]["drop_pct"])
    worst_pct    = cameras[worst_cam]["drop_pct"]
    worst_gap    = cameras[worst_cam].get("max_gap_ms", 0)

    detail = (
        f"{total_drops} drops ({total_severe} sévères) sur {len(cameras)} cam(s) — "
        f"pire: {worst_cam} {worst_pct:.1f}% (gap max {worst_gap:.0f}ms)"
    )
    if total_severe > MAX_SEVERE_DROPS or worst_pct > MAX_FRAME_DROP_PCT:
        return _fail(detail, cameras)
    return _ok(detail, cameras)


def _check_temporal_drift(session_path, meta):
    """
    Détecte la dérive temporelle en comparant le FPS effectif de chaque caméra
    avec le FPS nominal. Analyse aussi la variation locale (fenêtres 10s).
    """
    if not HAS_NUMPY:
        return _skip("numpy non disponible")

    cam_dir = _cameras_dir(session_path)
    if cam_dir is None:
        return _skip("Pas de répertoire caméras")

    fps_nominal = float((meta or {}).get("fps", (meta or {}).get("frame_rate", 30)))

    drift_per_cam = {}
    for fname in sorted(os.listdir(cam_dir)):
        if not fname.endswith(".jsonl"):
            continue
        cam   = fname[:-6]
        times = _read_jsonl_timestamps(os.path.join(cam_dir, fname))
        if len(times) < 10:
            continue

        t = np.array(times)
        duration_ms = t[-1] - t[0]
        if duration_ms < 1000:
            continue

        fps_actual = (len(t) - 1) / (duration_ms / 1000.0)
        fps_delta  = fps_actual - fps_nominal

        window_ms  = 10_000.0
        local_fpss = []
        t_start    = t[0]
        while t_start + window_ms <= t[-1]:
            t_end = t_start + window_ms
            sub   = t[(t >= t_start) & (t < t_end)]
            if len(sub) > 5:
                local_fpss.append((len(sub) - 1) / ((sub[-1] - sub[0]) / 1000.0))
            t_start += window_ms / 2

        local_arr  = np.array(local_fpss) if local_fpss else np.array([fps_actual])
        fps_std    = float(local_arr.std())
        fps_trend  = float(local_arr[-1] - local_arr[0]) if len(local_arr) > 1 else 0.0

        drift_per_cam[cam] = {
            "fps_nominal":   fps_nominal,
            "fps_actual":    round(fps_actual,  3),
            "fps_delta":     round(fps_delta,   3),
            "fps_std_local": round(fps_std,     3),
            "fps_trend":     round(fps_trend,   3),
            "drift_ms_per_s": round(abs(fps_delta) / fps_nominal * 1000.0, 2),
        }

    if not drift_per_cam:
        return _skip("Aucune caméra avec données suffisantes")

    max_abs_delta = max(abs(v["fps_delta"]) for v in drift_per_cam.values())
    worst_cam     = max(drift_per_cam, key=lambda k: abs(drift_per_cam[k]["fps_delta"]))
    worst_ms_s    = drift_per_cam[worst_cam]["drift_ms_per_s"]

    detail = (
        f"Drift max: {drift_per_cam[worst_cam]['fps_delta']:+.3f} fps "
        f"({worst_ms_s:.1f} ms/s) — {worst_cam}"
    )
    if max_abs_delta > MAX_DRIFT_FPS:
        return _fail(detail, drift_per_cam)
    return _ok(detail, drift_per_cam)


def _check_gripper_flux_coherence(session_path):
    """
    Vérifie la cohérence entre le capteur gripper (gripper_{side}_data.csv)
    et le flux optique de la caméra gripper correspondante (videos/gripper_{side}_flux.csv).

    Méthode : NCC (normalized cross-correlation) avec lag variable ±1.5s.
    Corrélation |r| < MIN_GRIPPER_COHERENCE → KO.
    """
    if not HAS_NUMPY:
        return _skip("numpy non disponible")

    results = {}

    for side in ["left", "right"]:
        gripper_csv = os.path.join(session_path, f"gripper_{side}_data.csv")
        cam_dir = _cameras_dir(session_path)
        flux_csv = os.path.join(cam_dir, f"gripper_{side}_flux.csv") if cam_dir else None

        if not os.path.isfile(gripper_csv) or not flux_csv or not os.path.isfile(flux_csv):
            continue

        try:
            g_df = pd.read_csv(gripper_csv)
            f_df = pd.read_csv(flux_csv)
        except Exception as e:
            results[side] = {"error": f"Lecture CSV : {e}"}
            continue

        g_ts  = next((c for c in ["timestamp_ms", "timestamp", "t", "time_ms", "ts"]
                      if c in g_df.columns), None)
        g_val = next((c for c in ["position", "aperture", "force", "state", "grip", "value"]
                      if c in g_df.columns), None)
        f_ts  = next((c for c in ["timestamp_abs_ms", "timestamp_ms", "timestamp", "t"]
                      if c in f_df.columns), None)
        f_val = next((c for c in ["mean_magnitude", "flow_magnitude", "magnitude", "flux", "flow"]
                      if c in f_df.columns), None)

        if not g_ts or not g_val or not f_ts or not f_val:
            results[side] = {
                "error":        "Colonnes non trouvées",
                "gripper_cols": list(g_df.columns),
                "flux_cols":    list(f_df.columns),
            }
            continue

        g_t = g_df[g_ts].values.astype(float)
        g_v = g_df[g_val].values.astype(float)
        f_t = f_df[f_ts].values.astype(float)
        f_v = f_df[f_val].values.astype(float)

        t_lo = max(g_t.min(), f_t.min())
        t_hi = min(g_t.max(), f_t.max())
        if t_hi - t_lo < 2000:
            results[side] = {"error": "Overlap temporel < 2s", "overlap_ms": round(t_hi - t_lo)}
            continue

        t_grid   = np.arange(t_lo, t_hi, 33.33)
        g_interp = np.interp(t_grid, g_t, g_v)
        f_interp = np.interp(t_grid, f_t, f_v)

        g_range = float(g_interp.max() - g_interp.min())
        f_range = float(f_interp.max() - f_interp.min())

        if g_range < 1e-6:
            results[side] = {"warning": "Capteur gripper plat — pas de mouvement", "gripper_range": g_range}
            continue
        if f_range < 1e-6:
            results[side] = {"warning": "Flux optique plat — caméra statique?", "flux_range": f_range}
            continue

        n     = len(g_interp)
        g_z   = (g_interp - g_interp.mean()) / (g_interp.std() * n + 1e-8)
        f_z   = (f_interp - f_interp.mean()) / (f_interp.std()     + 1e-8)
        corr  = np.correlate(f_z, g_z, mode="full")
        lags  = np.arange(-(n - 1), n)

        max_lag_samp = int(1.5 * 30)
        mask         = np.abs(lags) <= max_lag_samp
        corr_lim     = corr[mask]
        lags_lim     = lags[mask]

        best_idx = int(np.argmax(np.abs(corr_lim)))
        best_r   = float(corr_lim[best_idx])
        best_lag = int(lags_lim[best_idx])

        ncc0 = float(corr[n - 1])
        if abs(ncc0) >= 0.80 * abs(best_r):
            best_r   = ncc0
            best_lag = 0

        results[side] = {
            "correlation":   round(best_r,           4),
            "lag_ms":        round(best_lag * 33.33,  1),
            "gripper_range": round(g_range,           6),
            "flux_range":    round(f_range,           6),
        }

    if not results:
        return _skip("Aucune paire gripper+flux trouvée")

    valid = {k: v for k, v in results.items() if "correlation" in v}
    if not valid:
        detail = "; ".join(
            f"{s}: {v.get('error', v.get('warning', '?'))}" for s, v in results.items()
        )
        return _ok(f"Pas de corrélation calculable — {detail}", results)

    min_r   = min(abs(v["correlation"]) for v in valid.values())
    summary = ", ".join(
        f"{s}: r={v['correlation']:.3f} lag={v['lag_ms']}ms" for s, v in valid.items()
    )
    detail  = f"Cohérence gripper-flux — {summary}"

    if min_r < MIN_GRIPPER_COHERENCE:
        return _fail(detail, results)
    return _ok(detail, results)


# ═════════════════════════════════════════════════════════════════════════════
# CHECK VISION (11) — tracking complet des grippers
# ═════════════════════════════════════════════════════════════════════════════

def _check_gripper_tracking(session_path, meta):
    """
    Lance le tracking multi-sources des grippers via gripper_tracking.py :
      • IMU                → orientation par filtre complémentaire
      • Caméras left/right → trajectoire stéréo (multi-marqueurs ArUco)
      • Caméra head        → trajectoires des deux pinces simultanément
      • Corrélation        → Pearson/RMSE stéréo ↔ head par axe
      • Génère gripper_tracking.csv + gripper_correlation.json
    """
    if not HAS_CV2:
        return _skip("OpenCV non disponible")

    try:
        import gripper_tracking
        result = gripper_tracking.run(session_path, meta)
    except Exception as e:
        logger.exception("Erreur gripper_tracking pour %s", session_path)
        return _ok(f"Erreur gripper_tracking : {e}")

    if "error" in result:
        return _skip(result["error"])

    n_frames = result.get("n_frames_tracked", 0)
    corr     = result.get("trajectory_correlation", {})
    corr_l   = corr.get("left",  {})
    corr_r   = corr.get("right", {})

    imu_tag  = "IMU✓" if result.get("imu_available") else "IMU✗"
    head_tag = "head✓" if result.get("head_camera")  else "head✗"

    stereo_summary = (
        f"stéréo: left={result.get('stereo_left_poses',0)} "
        f"right={result.get('stereo_right_poses',0)}"
    )
    head_summary = (
        f"head: left={result.get('head_left_poses',0)} "
        f"right={result.get('head_right_poses',0)}"
    )

    def _corr_tag(c):
        px = c.get("pearson_x", 0)
        py = c.get("pearson_y", 0)
        pz = c.get("pearson_z", 0)
        ok = "✓" if c.get("coherent") else "✗"
        return f"r=({px:.2f},{py:.2f},{pz:.2f}) lag={c.get('lag_ms',0):.0f}ms {ok}"

    corr_summary = (
        f"corr stéréo↔head: left [{_corr_tag(corr_l)}] | right [{_corr_tag(corr_r)}]"
    )

    detail = f"{n_frames} frames | {imu_tag} {head_tag} | {stereo_summary} | {head_summary} | {corr_summary}"
    if result.get("csv_path"):
        detail += f" → {os.path.basename(result['csv_path'])}"

    if not result.get("left_detected") and not result.get("right_detected"):
        return _fail(f"Aucun gripper détecté — {detail}", result)

    return _ok(detail, result)


# ═════════════════════════════════════════════════════════════════════════════
# CHECK VISION (12) — détection d'inversion des labels left/right
# ═════════════════════════════════════════════════════════════════════════════

def _load_frame_timestamps(jsonl_path: str) -> list:
    """Retourne une liste triée de (frame_index, timestamp_sec) depuis un JSONL caméra."""
    frames = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                fi = d.get("frame_index")
                ts = d.get("capture_timestamp_sec")
                if ts is None:
                    # Format NAS : capture_time en ms → convertir en sec
                    t_ms = d.get("capture_time")
                    if t_ms is not None:
                        ts = float(t_ms) / 1000.0
                if fi is not None and ts is not None:
                    frames.append((int(fi), float(ts)))
    except Exception as e:
        logger.debug("Erreur JSONL %s : %s", jsonl_path, e)
    return sorted(frames, key=lambda x: x[0])


def _compute_flow_series(video_path: str, frame_timestamps: list,
                         sample_fps: float = 2.0, split: bool = False):
    """
    Calcule la magnitude du flux optique d'une vidéo à intervalles réguliers.

    frame_timestamps : liste de (frame_index, timestamp_sec)
    split            : si True, retourne les magnitudes de la moitié gauche et droite
                       séparément (pour la caméra head)

    Retourne :
      split=False → (times_sec, magnitudes)
      split=True  → (times_sec, left_magnitudes, right_magnitudes)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Impossible d'ouvrir %s", video_path)
        return ([], [], []) if split else ([], [])

    fps_vid    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step       = max(1, int(fps_vid / sample_fps))
    ts_map     = {fi: ts for fi, ts in frame_timestamps}
    first_ts   = frame_timestamps[0][1] if frame_timestamps else 0.0

    times       = []
    mags_full   = []
    mags_left   = []
    mags_right  = []
    prev_gray   = None
    frame_no    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % step == 0:
            small = cv2.resize(frame, (LABEL_FLOW_W, LABEL_FLOW_H))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                ts = ts_map.get(frame_no)
                if ts is None:
                    ts = first_ts + frame_no / fps_vid

                times.append(ts)
                if split:
                    w = mag.shape[1]
                    mags_left.append(float(mag[:, :w // 2].mean()))
                    mags_right.append(float(mag[:, w // 2:].mean()))
                else:
                    mags_full.append(float(mag.mean()))

            prev_gray = gray

        frame_no += 1

    cap.release()
    logger.debug("flux optique %s : %d points (step=%d)", video_path, len(times), step)

    if split:
        return times, mags_left, mags_right
    return times, mags_full


def _pearson(a, b) -> float:
    """Corrélation de Pearson entre deux arrays numpy."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _check_gripper_label_inversion(session_path):
    """
    Check 12 — Détecte si les labels left/right des caméras pinces sont inversés.

    Méthode :
      1. Calcule le flux optique de cameras/left.mp4 et cameras/right.mp4 (magnitude totale)
      2. Calcule le flux optique de cameras/head.mp4 séparé en moitié gauche et moitié droite
      3. Synchronise les séries temporelles sur une grille commune
      4. Corrèle chaque caméra pince avec les deux moitiés du head :
           - left devrait corréler plus avec la moitié GAUCHE du head
           - right devrait corréler plus avec la moitié DROITE du head
      5. Si le pattern est inversé (left corrèle plus avec la droite du head) → inversion détectée

    La corrélation est calculée sur les magnitudes de flux (intensité du mouvement),
    indépendante de la direction et donc insensible au sens d'installation des caméras.
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return _skip("OpenCV ou numpy non disponible")

    cam_dir = _cameras_dir(session_path)
    if cam_dir is None:
        return _skip("Répertoire caméras introuvable")

    head_video  = _find_video_file(cam_dir, "head")
    left_video  = _find_video_file(cam_dir, "left")
    right_video = _find_video_file(cam_dir, "right")

    if not head_video:
        return _skip("Vidéo head manquante")
    if not left_video and not right_video:
        return _skip("Vidéos left et right toutes les deux manquantes")

    # Charger les timestamps depuis les JSONL ("wall" toléré comme alias de "head")
    head_jsonl = next(
        (os.path.join(cam_dir, f"{n}.jsonl") for n in ("head", "wall")
         if os.path.isfile(os.path.join(cam_dir, f"{n}.jsonl"))), None,
    )
    head_ts  = _load_frame_timestamps(head_jsonl) if head_jsonl else []
    left_ts  = _load_frame_timestamps(os.path.join(cam_dir, "left.jsonl"))  if os.path.isfile(os.path.join(cam_dir, "left.jsonl"))  else []
    right_ts = _load_frame_timestamps(os.path.join(cam_dir, "right.jsonl")) if os.path.isfile(os.path.join(cam_dir, "right.jsonl")) else []

    # Calculer les séries de flux optique
    t_h, h_left_mags, h_right_mags = _compute_flow_series(
        head_video, head_ts, LABEL_SAMPLE_FPS, split=True
    )

    if len(t_h) < 5:
        return _skip(f"Pas assez de frames head analysées ({len(t_h)} points)")

    t_h_arr    = np.array(t_h)
    h_left_arr = np.array(h_left_mags)
    h_right_arr = np.array(h_right_mags)

    results = {}

    for side, video, timestamps in [
        ("left",  left_video,  left_ts),
        ("right", right_video, right_ts),
    ]:
        if not video:
            continue

        t_cam, cam_mags = _compute_flow_series(video, timestamps, LABEL_SAMPLE_FPS, split=False)

        if len(t_cam) < 5:
            results[side] = {"skipped": f"Pas assez de frames ({len(t_cam)} points)"}
            continue

        # Plage temporelle commune
        t_lo = max(t_h_arr.min(), min(t_cam))
        t_hi = min(t_h_arr.max(), max(t_cam))
        if t_hi - t_lo < 5.0:
            results[side] = {"skipped": f"Overlap temporel trop court ({t_hi - t_lo:.1f}s)"}
            continue

        dt     = 1.0 / LABEL_SAMPLE_FPS
        t_grid = np.arange(t_lo, t_hi, dt)

        head_l = np.interp(t_grid, t_h_arr, h_left_arr)
        head_r = np.interp(t_grid, t_h_arr, h_right_arr)
        cam_m  = np.interp(t_grid, np.array(t_cam), np.array(cam_mags))

        r_left  = _pearson(cam_m, head_l)   # corrélation avec moitié gauche du head
        r_right = _pearson(cam_m, head_r)   # corrélation avec moitié droite du head
        margin  = r_left - r_right           # positif = cohérent (left corrèle avec gauche)

        expected_side = side  # "left" devrait corréler avec head_left, idem pour "right"
        if side == "left":
            consistent = r_left > r_right
        else:
            consistent = r_right > r_left
            margin = r_right - r_left  # pour "right", margin > 0 = cohérent

        results[side] = {
            "corr_head_left":  round(r_left,     3),
            "corr_head_right": round(r_right,    3),
            "margin":          round(margin,     3),
            "consistent":      bool(consistent),
            "n_points":        len(t_grid),
        }

    if not results:
        return _skip("Aucune série de flux calculée")

    # Ignorer les sides sans données
    valid = {k: v for k, v in results.items() if "consistent" in v}
    if not valid:
        skipped_reasons = "; ".join(
            f"{k}: {v.get('skipped', '?')}" for k, v in results.items()
        )
        return _skip(f"Calcul impossible — {skipped_reasons}")

    # Construire le résumé
    summary_parts = []
    for side, v in valid.items():
        marker = "OK" if v["consistent"] else "INVERSION?"
        summary_parts.append(
            f"{side}: corr_L={v['corr_head_left']:.2f} corr_R={v['corr_head_right']:.2f} "
            f"margin={v['margin']:+.2f} [{marker}]"
        )
    summary = " | ".join(summary_parts)

    # Verdict : inversion confirmée si TOUS les gripper valides sont incohérents
    # avec une marge significative (évite les faux positifs sur signal faible)
    all_inconsistent = all(not v["consistent"] for v in valid.values())
    all_margins_significant = all(abs(v["margin"]) >= LABEL_INVERSION_MARGIN for v in valid.values())
    any_margin_significant  = any(abs(v["margin"]) >= LABEL_INVERSION_MARGIN for v in valid.values())

    if all_inconsistent and all_margins_significant:
        return _fail(
            f"Labels pinces probablement inversés (left↔right) — {summary}",
            results,
        )
    elif all_inconsistent and any_margin_significant:
        # Incohérent mais signal faible : avertissement non bloquant
        return _ok(
            f"[SUSPECT] Labels pinces possiblement inversés — {summary}",
            results,
        )
    else:
        return _ok(f"Labels pinces cohérents — {summary}", results)


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 13 — rapport analysis.json (données de capture)
# ═════════════════════════════════════════════════════════════════════════════

def _check_analysis_report(session_path: str) -> dict:
    """
    Check 13 — Lit analysis.json et vérifie les erreurs de capture.

    ok=False si errors[] est non vide (bloquant, cohérent avec la politique livraison).
    Les warnings sont signalés dans le détail mais ne bloquent pas.
    La valeur retournée est le contenu complet de analysis.json pour le scoring.
    """
    analysis = _read_analysis_json(session_path)
    if analysis is None:
        return _skip("analysis.json absent")

    errors   = [str(e) for e in analysis.get("errors",   [])]
    warnings = [str(w) for w in analysis.get("warnings", [])]

    parts = []
    if errors:
        parts.append(f"{len(errors)} erreur(s)")
    if warnings:
        parts.append(f"{len(warnings)} avertissement(s)")
    if not parts:
        parts.append("aucun problème")

    detail = "Capture : " + ", ".join(parts)
    if errors:
        return _fail(detail, analysis)
    return _ok(detail, analysis)


# ═════════════════════════════════════════════════════════════════════════════
# SCORING — score de qualité de session 0-100
# ═════════════════════════════════════════════════════════════════════════════

def _temporal_overlap_ratio(session_path: str):
    """
    Ratio overlap/union des plages temporelles de toutes les caméras.
    1.0 = toutes les caméras couvrent exactement la même fenêtre temporelle.
    """
    cam_dir = _cameras_dir(session_path)
    if not cam_dir:
        return None
    ranges = []
    for fname in sorted(os.listdir(cam_dir)):
        if not fname.endswith(".jsonl"):
            continue
        times = _read_jsonl_timestamps(os.path.join(cam_dir, fname))
        if len(times) >= 2:
            ranges.append((times[0], times[-1]))
    if len(ranges) < 2:
        return None
    overlap = max(0.0, min(r[1] for r in ranges) - max(r[0] for r in ranges))
    union   = max(r[1] for r in ranges) - min(r[0] for r in ranges)
    return overlap / union if union > 0 else None


def _compute_quality_score(checks_result: dict, session_path: str) -> dict:
    """
    Score de qualité de session 0-100, décomposé en 8 critères pondérés.

    Critère                     Poids  Ce que ça mesure
    ─────────────────────────────────────────────────────────────────────────
    1. sync_inter_cameras         20%  Drift relatif inter-caméras (analysis.json
                                       drift_check.pairs en priorité, sinon fps_delta)
    2. video_stability            20%  Frame drops JSONL + sequence_gaps kernel (analysis.json)
    3. gripper_detectability      20%  % frames head avec ≥1 marqueur gripper
    4. gripper_coverage           10%  Équilibre left/right dans la head camera
    5. imu_quality                10%  % quaternions capteurs valides (non nuls)
    6. data_integrity              8%  Fichiers présents, non vides, metadata valide
    7. temporal_coverage           7%  Overlap temporel inter-caméras
    8. session_duration            5%  Durée dans la plage valide
    """
    WEIGHTS = {
        "sync_inter_cameras":    0.20,
        "video_stability":       0.20,
        "gripper_detectability": 0.20,
        "gripper_coverage":      0.10,
        "imu_quality":           0.10,
        "data_integrity":        0.08,
        "temporal_coverage":     0.07,
        "session_duration":      0.05,
    }

    s = {}  # scores 0-100 par critère

    # Données analysis.json (mesures directes de capture, plus fiables)
    analysis = (checks_result.get("analysis_report", {}).get("value")
                or _read_analysis_json(session_path))

    # ── 1. Synchronisation inter-caméras ─────────────────────────────────────
    # Priorité à analysis.json drift_check.pairs (dérive relative mesurée directement
    # par les compteurs V4L2, indépendante du fps_nominal configuré).
    drift_val = checks_result.get("temporal_drift", {}).get("value") or {}
    _sync_set = False
    if analysis:
        pairs = analysis.get("drift_check", {}).get("pairs", {})
        if pairs:
            max_rel = max(abs(v.get("relative_drift_ms_per_min", 0)) for v in pairs.values())
            # 0 ms/min = parfait (100), 10 ms/min = score 50, 20 ms/min = 0
            s["sync_inter_cameras"] = round(max(0.0, 100.0 - max_rel * 5.0), 1)
            _sync_set = True
    if not _sync_set:
        if isinstance(drift_val, dict) and len(drift_val) >= 2:
            deltas = [v.get("fps_delta", 0) for v in drift_val.values()]
            spread = max(deltas) - min(deltas)
            s["sync_inter_cameras"] = max(0.0, 100.0 - spread * 60.0)
        else:
            s["sync_inter_cameras"] = 50.0

    # ── 2. Stabilité vidéo ───────────────────────────────────────────────────
    drop_val = checks_result.get("frame_drops", {}).get("value") or {}
    if isinstance(drop_val, dict) and drop_val:
        worst_pct   = max(c.get("drop_pct",     0) for c in drop_val.values())
        n_severe    = sum(c.get("severe_drops", 0) for c in drop_val.values())
        drop_score  = max(0.0, 100.0 - worst_pct * 4.0 - n_severe * 3.0)
    else:
        drop_score  = 50.0

    if isinstance(drift_val, dict) and drift_val:
        max_std    = max(v.get("fps_std_local", 0) for v in drift_val.values())
        std_score  = max(0.0, 100.0 - max_std * 30.0)
    else:
        std_score  = 50.0

    # Complément analysis.json : sequence_gaps (drops kernel V4L2)
    if analysis:
        cams_a = analysis.get("fps_check", {}).get("cameras", {})
        if cams_a:
            total_est  = sum((c.get("measured_fps", 0) or c.get("expected_fps", 0)) *
                             c.get("duration_sec", 0) for c in cams_a.values())
            total_gaps = sum(c.get("sequence_gaps",  0) for c in cams_a.values())
            total_qdrop = sum(c.get("queue_drops",   0) for c in cams_a.values())
            gap_pct     = total_gaps / total_est * 100.0 if total_est > 0 else 0.0
            kernel_score = max(0.0, 100.0 - gap_pct * 2.0 - total_qdrop * 5.0)
            s["video_stability"] = round(0.45 * drop_score + 0.25 * std_score + 0.30 * kernel_score, 1)
        else:
            s["video_stability"] = round(0.65 * drop_score + 0.35 * std_score, 1)
    else:
        s["video_stability"] = round(0.65 * drop_score + 0.35 * std_score, 1)

    # ── 3. Détectabilité grippers (taux ArUco head) ───────────────────────────
    gt_val    = checks_result.get("gripper_tracking", {}).get("value") or {}
    head_det  = gt_val.get("head_detection_stats", {})
    any_rate  = head_det.get("any_rate")
    if any_rate is not None:
        s["gripper_detectability"] = round(any_rate * 100.0, 1)
    elif gt_val.get("head_left_poses", 0) + gt_val.get("head_right_poses", 0) > 0:
        s["gripper_detectability"] = 20.0   # détecté mais pas de stats complètes
    else:
        s["gripper_detectability"] = 0.0

    # ── 4. Équilibre left/right dans la head camera ───────────────────────────
    left_rate  = head_det.get("left_rate",  0.0)
    right_rate = head_det.get("right_rate", 0.0)
    if left_rate > 0 and right_rate > 0:
        balance = min(left_rate, right_rate) / max(left_rate, right_rate)
        s["gripper_coverage"] = round(balance * 100.0, 1)
    elif left_rate > 0 or right_rate > 0:
        s["gripper_coverage"] = 25.0        # une seule pince visible
    else:
        s["gripper_coverage"] = 0.0

    # ── 5. Qualité IMU ────────────────────────────────────────────────────────
    sensor_summary = gt_val.get("sensor_summary", {}) or {}
    imu_rates = []
    for side_data in sensor_summary.values():
        if side_data and side_data.get("n_total", 0) > 0:
            imu_rates.append(side_data["n_valid"] / side_data["n_total"])
    s["imu_quality"] = round(sum(imu_rates) / len(imu_rates) * 100.0, 1) if imu_rates else 0.0

    # ── 6. Intégrité des données ──────────────────────────────────────────────
    integrity_keys = ["folder_exists", "metadata_valid", "files_present",
                      "no_empty_files", "min_size"]
    ok_count = sum(1 for k in integrity_keys
                   if checks_result.get(k, {}).get("ok", False))
    s["data_integrity"] = round(ok_count / len(integrity_keys) * 100.0, 1)

    # ── 7. Couverture temporelle inter-caméras ────────────────────────────────
    overlap = _temporal_overlap_ratio(session_path)
    if overlap is not None:
        s["temporal_coverage"] = round(overlap * 100.0, 1)
    else:
        s["temporal_coverage"] = 50.0

    # ── 8. Durée session ──────────────────────────────────────────────────────
    dur_check = checks_result.get("duration_ok", {})
    dur_val   = dur_check.get("value")
    if dur_val is not None:
        if MIN_DURATION_S <= float(dur_val) <= MAX_DURATION_S:
            s["session_duration"] = 100.0
        else:
            s["session_duration"] = 0.0
    else:
        s["session_duration"] = 50.0

    # ── Score global ──────────────────────────────────────────────────────────
    for k in s:
        s[k] = round(float(s[k]), 1)

    total = round(sum(s[k] * WEIGHTS[k] for k in WEIGHTS), 1)

    grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D" if total >= 40 else "F"

    return {
        "score": total,
        "grade": grade,
        "criteria": {
            k: {
                "score":      s[k],
                "weight_pct": round(WEIGHTS[k] * 100),
            }
            for k in WEIGHTS
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# ÉCRITURE treatment.json
# ═════════════════════════════════════════════════════════════════════════════

_CHECK_LABELS = {
    "folder_exists":          "Dossier session",
    "metadata_valid":         "Métadonnées",
    "files_present":          "Fichiers de données",
    "no_empty_files":         "Fichiers vides",
    "min_size":               "Taille minimale",
    "duration_ok":            "Durée",
    "scenario_match":         "Scénario",
    "analysis_report":        "Rapport de capture",
    "frame_drops":            "Frame drops",
    "temporal_drift":         "Dérive temporelle",
    "gripper_flux_coherence": "Cohérence capteur-flux",
    "gripper_tracking":       "Tracking gripper (ArUco)",
    "gripper_label_inversion":"Inversion labels pinces",
}


def _build_conclusions(checks_result: dict) -> tuple:
    """Génère des listes conclusions/warnings/errors lisibles depuis les résultats de checks."""
    from datetime import datetime, timezone

    conclusions = []
    warnings    = []
    errors      = []

    for name, check in checks_result.items():
        label  = _CHECK_LABELS.get(name, name)
        ok     = check.get("ok", True)
        detail = check.get("detail", "")

        if detail.startswith("[IGNORÉ]"):
            continue

        suspect = "SUSPECT" in detail or "possiblement" in detail.lower()

        if not ok:
            errors.append(f"{label} : {detail}")
        elif suspect:
            warnings.append(f"{label} : {detail}")
        else:
            conclusions.append(f"{label} : {detail}")

    return conclusions, warnings, errors


def write_treatment_json(session_path: str, session_id: str, result: dict) -> str:  # noqa: E501
    """
    Écrit treatment.json dans le dossier de la session avec tous les résultats
    et les conclusions du traitement.

    Retourne le chemin du fichier créé.
    """
    from datetime import datetime, timezone

    checks_result = result.get("checks", {})
    passed        = result.get("passed", False)
    file_count    = result.get("file_count", 0)
    size_bytes    = result.get("size_bytes", 0)

    conclusions, warnings, errors = _build_conclusions(checks_result)

    verdict = "passed" if passed else "failed"
    if passed and warnings:
        verdict = "warning"

    # Résumé de la durée si disponible
    duration = None
    dur_check = checks_result.get("duration_ok", {})
    if isinstance(dur_check.get("value"), (int, float)):
        duration = dur_check["value"]

    quality = result.get("quality") or _compute_quality_score(checks_result, session_path)

    treatment = {
        "version":      "1.1",
        "session_id":   session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed":       passed,
        "verdict":      verdict,
        "quality": {
            "score": quality["score"],
            "grade": quality["grade"],
            "criteria": quality["criteria"],
        },
        "stats": {
            "file_count":       file_count,
            "size_bytes":       size_bytes,
            "duration_seconds": duration,
        },
        "checks":      checks_result,
        "conclusions": conclusions,
        "warnings":    warnings,
        "errors":      errors,
    }

    out_path = os.path.join(session_path, "treatment.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(treatment, f, indent=2, default=str, ensure_ascii=False)
        logger.info("treatment.json écrit : %s", out_path)
    except Exception as e:
        logger.error("Impossible d'écrire treatment.json pour %s : %s", session_path, e)
        raise

    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR
# ═════════════════════════════════════════════════════════════════════════════

def resolve_session_path(session: dict) -> str:
    """
    Resolve the NAS path for a session using multiple fallback strategies:
      1. NAS_DIR / session_folder (from DB) — direct or normalized
      2. NAS_DIR / session_YYYYMMDD_HHMMSS  (derived from sess_YYYYMMDD_HHMMSS_<hash>)
      3. glob NAS_DIR / session_YYYYMMDD_HHMMSS*  (same-second collisions)
      4. Old-format folder: ./data/YYYY-MM-DD_HH-MM-SS_NNNNN → session_YYYYMMDD_HHMMSS
    Returns the path (may not exist; caller checks with folder_exists).
    """
    sid    = session["session_id"]
    folder = session.get("session_folder") or sid

    candidate = os.path.join(NAS_DIR, folder)
    if os.path.isdir(candidate):
        return candidate

    # Fallback 2–3 : extrait le timestamp depuis le session_id
    m = re.match(r"^sess_(\d{8}_\d{6})(?:_[0-9a-f]+)?$", sid)
    if m:
        ts = m.group(1)
        direct = os.path.join(NAS_DIR, f"session_{ts}")
        if os.path.isdir(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(NAS_DIR, f"session_{ts}*")))
        if matches:
            return matches[0]

    # Fallback 4 : ancien format de dossier YYYY-MM-DD_HH-MM-SS_NNNNN
    # (ex: ./data/2026-05-18_18-59-35_000053 → session_20260518_185935)
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", folder)
    if m2:
        ts2 = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}_{m2.group(4)}{m2.group(5)}{m2.group(6)}"
        direct2 = os.path.join(NAS_DIR, f"session_{ts2}")
        if os.path.isdir(direct2):
            return direct2
        matches2 = sorted(glob.glob(os.path.join(NAS_DIR, f"session_{ts2}*")))
        if matches2:
            return matches2[0]

    return candidate


def run_checks(session: dict, session_path_override: str = None) -> dict:
    sid    = session["session_id"]
    db_dur = session.get("duration_seconds")
    db_scen = session.get("scenario_id")

    checks       = {}
    meta         = None

    # Résolution du chemin : override > smart resolution > NAS_DIR/folder
    if session_path_override:
        session_path = session_path_override
    else:
        session_path = resolve_session_path(session)

    # ── 1. Dossier ────────────────────────────────────────────────────────────
    checks["folder_exists"] = _check_folder(session_path)
    if not checks["folder_exists"]["ok"]:
        return _result(checks, 0, 0, "")

    # ── 2. Metadata ───────────────────────────────────────────────────────────
    checks["metadata_valid"], meta = _check_metadata(session_path)

    # ── 3–4. Fichiers ─────────────────────────────────────────────────────────
    all_files, empty_files, total_bytes = _scan_files(session_path)
    file_count = len(all_files)

    checks["files_present"] = (
        _ok(f"{file_count} fichier(s)", file_count) if file_count > 0
        else _fail("Aucun fichier de données", 0)
    )
    checks["no_empty_files"] = (
        _ok("Aucun fichier vide") if not empty_files
        else _fail(f"{len(empty_files)} fichier(s) vide(s) : {empty_files[:3]}", empty_files)
    )

    # ── 5. Taille ─────────────────────────────────────────────────────────────
    checks["min_size"] = (
        _ok(f"{total_bytes:,} bytes", total_bytes) if total_bytes >= MIN_SIZE_BYTES
        else _fail(f"{total_bytes} bytes < {MIN_SIZE_BYTES}", total_bytes)
    )

    # ── 6. Durée ──────────────────────────────────────────────────────────────
    checks["duration_ok"] = _check_duration(meta, db_dur)

    # ── 7. Scénario ───────────────────────────────────────────────────────────
    checks["scenario_match"] = _check_scenario(meta, db_scen)

    # ── 8. Rapport de capture (analysis.json) ────────────────────────────────
    checks["analysis_report"] = _check_analysis_report(session_path)
    if not checks["analysis_report"]["ok"]:
        return _result(checks, file_count, total_bytes, session_path)

    # ── 9. Frame drops ────────────────────────────────────────────────────────
    checks["frame_drops"] = _check_frame_drops(session_path, meta)

    # ── 10. Dérive temporelle ─────────────────────────────────────────────────
    checks["temporal_drift"] = _check_temporal_drift(session_path, meta)

    # ── 11. Cohérence gripper ↔ flux optique ──────────────────────────────────
    checks["gripper_flux_coherence"] = _check_gripper_flux_coherence(session_path)

    # ── 12. Tracking gripper (ArUco fix_* + ChArUco gripper_* + CSV) ──────────
    if ENABLE_VISION and HAS_CV2:
        checks["gripper_tracking"] = _check_gripper_tracking(session_path, meta)
    else:
        reason = "ENABLE_VISION_CHECKS=false" if not ENABLE_VISION else "OpenCV non installé"
        checks["gripper_tracking"] = _skip(reason)

    # ── 13. Inversion labels pinces (flux optique left/right vs head) ─────────
    if ENABLE_VISION and HAS_CV2 and HAS_NUMPY:
        checks["gripper_label_inversion"] = _check_gripper_label_inversion(session_path)
    else:
        reason = "ENABLE_VISION_CHECKS=false" if not ENABLE_VISION else "OpenCV/numpy non installé"
        checks["gripper_label_inversion"] = _skip(reason)

    return _result(checks, file_count, total_bytes, session_path)


def _result(checks: dict, file_count: int, size_bytes: int, session_path: str = "") -> dict:
    critical_failed = any(not checks.get(k, {}).get("ok", True) for k in CRITICAL_CHECKS)
    quality = _compute_quality_score(checks, session_path) if session_path else {"score": 0.0, "grade": "F", "criteria": {}}
    passed = not critical_failed and quality["score"] >= MIN_QUALITY_SCORE
    return {
        "checks":       checks,
        "passed":       passed,
        "file_count":   file_count,
        "size_bytes":   size_bytes,
        "quality":      quality,
        "session_path": session_path,
    }
