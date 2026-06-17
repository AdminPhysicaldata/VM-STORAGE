#!/usr/bin/env python3
"""Vérifie que les vidéos left/right sont bien nommées en détectant les
marqueurs ArUco des pinces (id 244 et 255) et en comparant l'écartement
mesuré dans l'image avec la courbe "Opening_width" des fichiers sensors/.

Contexte : seules les caméras poignet (gripper) voient les deux marqueurs
244/255 collés sur les mors de la pince. La caméra "head" ne doit jamais
les voir. Si un nommage a été inversé (ex : la vidéo nommée "left" est en
fait celle montée à droite), l'écartement mesuré sur l'image doit corréler
avec sensors/right.jsonl et pas avec sensors/left.jsonl — ce script détecte
exactement ce genre d'inversion.

Étapes :
  1. Échantillonne les 3 vidéos (cameras/*.mp4) et détecte les marqueurs
     ArUco id 244 et 255 sur chaque frame échantillonnée.
  2. Classe chaque vidéo : "gripper" (voit les 2 marqueurs sur une part
     significative des frames) ou "head" (ne les voit presque jamais).
     Doit donner exactement 2 "gripper" + 1 "head" — sinon alerte.
  3. Pour les 2 vidéos "gripper", calcule la distance pixel entre les deux
     marqueurs à chaque frame où ils sont détectés ensemble → courbe
     d'écartement au cours du temps.
  4. Charge sensors/left.jsonl et sensors/right.jsonl (champ Opening_width)
     et corrèle chaque courbe vidéo aux deux courbes capteur.
  5. L'appariement (vidéo → capteur) qui maximise la corrélation totale
     donne l'identité réelle de chaque vidéo gripper. Si ça contredit le
     nom de fichier actuel → anomalie signalée (et corrigeable via --apply).
  6. Sauve un graphique comparatif (PNG) pour vérification visuelle.

Dépendances : opencv-contrib-python, numpy, matplotlib
    uv add opencv-contrib-python numpy matplotlib

Usage :
    python3 detect_charuco_lr.py --session ../session_20260605_190710
    python3 detect_charuco_lr.py --session ../session_20260605_190710 --plot out.png
    python3 detect_charuco_lr.py --session ../session_20260605_190710 --apply
    python3 detect_charuco_lr.py /media/qbee/T9/sessions/   # scan d'un répertoire
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"Dépendance manquante : {exc}\n"
          f"  uv add opencv-contrib-python numpy matplotlib", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_camera_names import _replace_in_json  # réutilisé pour les corrections JSON

_MARKER_IDS = (244, 255)
_DEFAULT_DICT = "DICT_4X4_1000"
_GRIPPER_MIN_RATIO = 0.05   # >= 5% des frames échantillonnées voient les 2 marqueurs → "gripper"
_HEAD_MAX_RATIO = 0.02      # <= 2% → "head" (bruit/faux positifs tolérés)
_CLASSIFY_FPS = 1.0         # passe légère : sert juste à trier head vs gripper
_CURVE_FPS = 5.0            # passe dense : uniquement sur les 2 vidéos confirmées "gripper"
_MIN_CONFIDENT_CORR = 0.15  # |corr| minimum pour considérer un appariement fiable (cf. analyze_session)


# ─── Détecteur ArUco (compat anciennes/nouvelles API OpenCV) ────────────────

def _make_detector(dict_name: str):
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    else:
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
        params = cv2.aruco.DetectorParameters_create()
        return lambda gray: cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)[:2]


# ─── Échantillonnage d'une vidéo ─────────────────────────────────────────────

@dataclass
class FrameSample:
    frame_index: int
    centers: dict  # marker_id -> (x, y)


def _sample_video(path: Path, detector, sample_fps: float) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))

    samples: list[FrameSample] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detector(gray)
                centers = {}
                if ids is not None:
                    for c, i in zip(corners, ids.flatten()):
                        i = int(i)
                        if i in _MARKER_IDS:
                            centers[i] = c.reshape(-1, 2).mean(axis=0)
                samples.append(FrameSample(frame_index=idx, centers=centers))
        idx += 1
    cap.release()
    return samples


def _frame_timestamps(jsonl_path: Path) -> dict:
    """frame_index -> capture_timestamp_sec, lu dans cameras/<name>.jsonl."""
    out = {}
    try:
        with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out[rec["frame_index"]] = float(rec["capture_timestamp_sec"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    except OSError:
        pass
    return out


# ─── Classification gripper vs head ─────────────────────────────────────────

def _both_markers_ratio(samples: list[FrameSample]) -> float:
    if not samples:
        return 0.0
    both = sum(1 for s in samples if all(m in s.centers for m in _MARKER_IDS))
    return both / len(samples)


def _classify(ratio: float) -> str:
    if ratio >= _GRIPPER_MIN_RATIO:
        return "gripper"
    if ratio <= _HEAD_MAX_RATIO:
        return "head"
    return "ambiguous"


# ─── Courbe d'écartement (distance pixel entre les 2 marqueurs) ────────────

def _opening_curve(samples: list[FrameSample], frame_ts: dict) -> tuple:
    times, dists = [], []
    for s in samples:
        if all(m in s.centers for m in _MARKER_IDS) and s.frame_index in frame_ts:
            p0, p1 = s.centers[_MARKER_IDS[0]], s.centers[_MARKER_IDS[1]]
            dists.append(float(np.linalg.norm(p0 - p1)))
            times.append(frame_ts[s.frame_index])
    return np.array(times), np.array(dists)


# ─── Courbe capteur (Opening_width) ──────────────────────────────────────────

def _sensor_curve(jsonl_path: Path) -> tuple:
    times, widths = [], []
    try:
        with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    times.append(float(rec["host_time_sec"]))
                    widths.append(float(rec["Opening_width"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    except OSError:
        pass
    return np.array(times), np.array(widths)


# ─── Corrélation entre une courbe vidéo et une courbe capteur ──────────────

def _correlate(t_video, v_video, t_sensor, v_sensor) -> float:
    """Pearson, après ré-échantillonnage de la courbe capteur sur les temps vidéo."""
    if len(t_video) < 5 or len(t_sensor) < 5:
        return float("nan")
    lo = max(t_video.min(), t_sensor.min())
    hi = min(t_video.max(), t_sensor.max())
    mask = (t_video >= lo) & (t_video <= hi)
    if mask.sum() < 5:
        return float("nan")
    tv, vv = t_video[mask], v_video[mask]
    order = np.argsort(t_sensor)
    v_resampled = np.interp(tv, t_sensor[order], v_sensor[order])
    if vv.std() == 0 or v_resampled.std() == 0:
        return float("nan")
    return float(np.corrcoef(vv, v_resampled)[0, 1])


# ─── Analyse d'une session ───────────────────────────────────────────────────

@dataclass
class VideoFinding:
    current_name: str
    role: str                  # "gripper" / "head" / "ambiguous" / "unreadable"
    ratio: float
    times: object = None
    dists: object = None
    inferred_name: str | None = None
    corr_left: float = float("nan")
    corr_right: float = float("nan")
    confident: bool = True      # False = signal de corrélation trop faible pour conclure


def analyze_session(
    session_dir: Path,
    dict_name: str = _DEFAULT_DICT,
    classify_fps: float = _CLASSIFY_FPS,
    curve_fps: float = _CURVE_FPS,
) -> list[VideoFinding]:
    """
    Coût optimisé pour tourner sur des dizaines de milliers de sessions :
    chaque vidéo n'est décodée densément qu'une seule fois, et seulement si
    nécessaire.

      1. Passe légère (classify_fps, ~1 Hz) sur les 3 vidéos → suffit pour
         estimer le ratio de frames avec les 2 marqueurs et trier head/gripper.
      2. Une vidéo classée "head" en passe légère s'arrête là (jamais
         redécodée) — c'est la grande majorité du gain, puisque 1/3 des
         vidéos d'une session n'ont structurellement aucune chance de voir
         les marqueurs.
      3. Seules les vidéos qui ressemblent à un gripper (ou ambiguës) sont
         redécodées une fois en dense (curve_fps, ~5 Hz) pour confirmer le
         rôle et obtenir la courbe d'écartement précise.
    """
    cameras_dir = session_dir / "cameras"
    sensors_dir = session_dir / "sensors"
    detector = _make_detector(dict_name)

    findings: list[VideoFinding] = []
    for video_path in sorted(cameras_dir.glob("*.mp4")):
        name = video_path.stem
        sparse = _sample_video(video_path, detector, classify_fps)
        if not sparse:
            # Vidéo illisible/corrompue (0 octet, conteneur cassé, upload tronqué) :
            # ne JAMAIS la confondre avec "head" (qui a aussi un ratio de 0%),
            # sinon une vidéo cassée serait classée "correcte" à tort.
            findings.append(VideoFinding(current_name=name, role="unreadable", ratio=0.0))
            continue

        prelim_ratio = _both_markers_ratio(sparse)
        if _classify(prelim_ratio) == "head":
            findings.append(VideoFinding(current_name=name, role="head", ratio=prelim_ratio))
            continue

        # Candidat gripper (ou ambigu) : on confirme avec une passe dense.
        dense = _sample_video(video_path, detector, curve_fps)
        ratio = _both_markers_ratio(dense) if dense else 0.0
        role = _classify(ratio) if dense else "unreadable"
        finding = VideoFinding(current_name=name, role=role, ratio=ratio)
        if role == "gripper":
            frame_ts = _frame_timestamps(cameras_dir / f"{name}.jsonl")
            finding.times, finding.dists = _opening_curve(dense, frame_ts)
        findings.append(finding)

    gripper = [f for f in findings if f.role == "gripper"]
    heads = [f for f in findings if f.role == "head"]
    if len(gripper) == 2:
        t_left, v_left = _sensor_curve(sensors_dir / "left.jsonl")
        t_right, v_right = _sensor_curve(sensors_dir / "right.jsonl")
        for f in gripper:
            f.corr_left = _correlate(f.times, f.dists, t_left, v_left)
            f.corr_right = _correlate(f.times, f.dists, t_right, v_right)

        a, b = gripper
        # Deux appariements possibles ; on choisit celui qui maximise la corrélation totale.
        score_ab = (a.corr_left if not np.isnan(a.corr_left) else -1) + \
                   (b.corr_right if not np.isnan(b.corr_right) else -1)
        score_ba = (a.corr_right if not np.isnan(a.corr_right) else -1) + \
                   (b.corr_left if not np.isnan(b.corr_left) else -1)
        if score_ab >= score_ba:
            pairing = ("left", "right")
            winning_corrs = (a.corr_left, b.corr_right)
        else:
            pairing = ("right", "left")
            winning_corrs = (a.corr_right, b.corr_left)

        # Avec ~300-400 échantillons bruités (détection ArUco frame par frame,
        # pas de filtrage sub-pixel), le bruit de fond d'une corrélation de Pearson
        # tourne autour de ±0.06 (1/sqrt(n)). En dessous de _MIN_CONFIDENT_CORR, le
        # "gagnant" choisi entre les deux appariements n'est pas distinguable du
        # bruit : mieux vaut ne RIEN affirmer que de renommer sur une fausse piste.
        best = max((abs(c) for c in winning_corrs if not np.isnan(c)), default=0.0)
        confident = best >= _MIN_CONFIDENT_CORR
        a.confident = b.confident = confident
        if confident:
            a.inferred_name, b.inferred_name = pairing

        # La 3ᵉ vidéo (rôle "head") doit elle aussi être renommée "head" si besoin —
        # sinon le renommage des 2 gripper pourrait écraser son fichier (collision).
        if confident and len(heads) == 1:
            heads[0].inferred_name = "head"

    return findings


# ─── Rapport texte ────────────────────────────────────────────────────────

def print_report(session_name: str, findings: list[VideoFinding]) -> bool:
    """Retourne True si une anomalie de nommage left/right est détectée."""
    print(f"\n{session_name}")
    anomaly = False
    for f in findings:
        marker = f"ratio_2_markers={f.ratio:.1%}"
        mismatch = f.inferred_name is not None and f.inferred_name != f.current_name
        anomaly = anomaly or mismatch
        flag = "  ⚠ MAL NOMMÉE" if mismatch else ""
        if f.role == "gripper":
            extra = f"corr(left)={f.corr_left:.2f} corr(right)={f.corr_right:.2f}"
            if not f.confident:
                verdict = "signal trop faible pour conclure (vérification manuelle conseillée)"
            else:
                verdict = f"identité réelle probable : {f.inferred_name}"
            print(f"  {f.current_name:<8} [{f.role:<9}] {marker}  {extra}  → {verdict}{flag}")
        else:
            suffix = f"  → identité réelle probable : {f.inferred_name}{flag}" if f.inferred_name else ""
            print(f"  {f.current_name:<8} [{f.role:<9}] {marker}{suffix}")
    n_gripper = sum(1 for f in findings if f.role == "gripper")
    n_unreadable = sum(1 for f in findings if f.role == "unreadable")
    if n_unreadable:
        anomaly = True
        print(f"  ⚠ {n_unreadable} vidéo(s) illisible(s)/corrompue(s) — vérification manuelle requise")
    if n_gripper != 2:
        anomaly = True
        print(f"  ⚠ {n_gripper} vidéo(s) classée(s) 'gripper' (attendu : 2) — vérification manuelle requise")
    return anomaly


# ─── Graphique comparatif ────────────────────────────────────────────────────

def save_plot(session_dir: Path, findings: list[VideoFinding], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensors_dir = session_dir / "sensors"
    t_left, v_left = _sensor_curve(sensors_dir / "left.jsonl")
    t_right, v_right = _sensor_curve(sensors_dir / "right.jsonl")

    gripper = [f for f in findings if f.role == "gripper"]
    fig, axes = plt.subplots(len(gripper) or 1, 1, figsize=(10, 4 * (len(gripper) or 1)), squeeze=False)
    for ax, f in zip(axes[:, 0], gripper):
        ax2 = ax.twinx()
        ax.plot(f.times, f.dists, color="tab:blue", label=f"distance pixel ({f.current_name}.mp4)")
        ax2.plot(t_left, v_left, color="tab:green", alpha=0.6, label="sensors/left Opening_width")
        ax2.plot(t_right, v_right, color="tab:red", alpha=0.6, label="sensors/right Opening_width")
        ax.set_title(f"{f.current_name}.mp4  →  identité probable : {f.inferred_name}")
        ax.set_ylabel("distance marqueurs (px)", color="tab:blue")
        ax2.set_ylabel("Opening_width (capteur)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  graphique → {out_path}")


# ─── Correction (renommage) ──────────────────────────────────────────────────

def apply_fix(session_dir: Path, findings: list[VideoFinding]) -> None:
    # IMPORTANT : on inclut TOUTES les vidéos dont l'identité inférée diffère du
    # nom actuel — pas seulement les "gripper". Si on omettait la vidéo "head"
    # lors d'un swap head<->left, le renommage des 2 gripper écraserait
    # silencieusement le fichier qui occupe déjà le nom cible (perte de données).
    renames = {
        f.current_name: f.inferred_name
        for f in findings
        if f.inferred_name and f.inferred_name != f.current_name
    }
    if not renames:
        print("  rien à corriger.")
        return

    targets = list(renames.values())
    if len(set(targets)) != len(targets):
        print(f"  [ERREUR] mapping non-permutation, abandon : {renames}", file=sys.stderr)
        return

    cameras_dir = session_dir / "cameras"
    # Renommage en 2 temps via noms temporaires pour gérer le swap left<->right.
    tmp_suffix = ".tmp_charuco_swap"
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
        except (OSError, json.JSONDecodeError):
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
    p.add_argument("--classify-fps", type=float, default=_CLASSIFY_FPS,
                   help=f"Fréquence d'échantillonnage de la passe légère head/gripper, en Hz "
                        f"(défaut : {_CLASSIFY_FPS})")
    p.add_argument("--curve-fps", type=float, default=_CURVE_FPS,
                   help=f"Fréquence d'échantillonnage de la passe dense (courbe d'écartement), "
                        f"uniquement sur les vidéos confirmées gripper (défaut : {_CURVE_FPS})")
    p.add_argument("--plot", type=Path, metavar="PNG",
                   help="Sauver un graphique comparatif (uniquement avec --session)")
    p.add_argument("--apply", action="store_true",
                   help="Appliquer le renommage si une inversion left/right est détectée")
    args = p.parse_args()

    if args.session:
        session_dir = args.session.resolve()
        findings = analyze_session(session_dir, args.dict_name, args.classify_fps, args.curve_fps)
        anomaly = print_report(session_dir.name, findings)
        if args.plot:
            save_plot(session_dir, findings, args.plot)
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
            findings = analyze_session(session_dir, args.dict_name, args.classify_fps, args.curve_fps)
            if print_report(session_dir.name, findings):
                anomalies += 1
                if args.apply:
                    apply_fix(session_dir, findings)
        print(f"\n{'─' * 50}")
        print(f"Sessions analysées : {len(sessions)}")
        print(f"Anomalies left/right : {anomalies}")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
