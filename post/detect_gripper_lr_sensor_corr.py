#!/usr/bin/env python3
"""Détermine si les vidéos cameras/left.mp4 et cameras/right.mp4 sont
nommées dans le bon sens, en corrélant le mouvement visible dans chaque
vidéo avec la vitesse d'ouverture/fermeture mesurée par les capteurs
sensors/left.jsonl et sensors/right.jsonl.

Contexte : detect_charuco_lr.py a déjà tenté d'utiliser la distance pixel
entre 2 marqueurs ArUco fixes comme proxy de l'écartement des pinces, et a
conclu (sur UNE session) que c'était trop bruité pour distinguer left de
right de façon fiable. Constat supplémentaire ici : les marqueurs visés
(244/255) n'apparaissent même pas dans les vidéos de ce jeu de données.

Approche retenue — flux optique global ↔ vitesse d'ouverture capteur :
  - magnitude du flux optique de la vidéo (mouvement global, agnostique au
    placement exact des marqueurs)
  - |d(Opening_width)/dt| du capteur (à quel rythme la pince bouge)
  - corrélation à décalage optimal (fenêtre de lag restreinte, le délai
    caméra/capteur étant de l'ordre de la latence de capture, pas de
    plusieurs secondes)

Diagnostic PAR SESSION (le câblage caméra↔capteur peut varier d'une session
à l'autre — pas d'hypothèse de biais systématique global) :
  Pour chaque session, la marge observée (score_same - score_swap) est
  comparée à une distribution nulle obtenue par test de permutation : les
  deux séries capteur (left + right) sont décalées circulairement, ensemble
  et du même tirage aléatoire (ce qui détruit leur alignement avec les
  vidéos tout en conservant leur corrélation mutuelle, due à la même
  tâche), répété _N_PERMUTATIONS fois. La session n'est tranchée ("same"/
  "swap") que si p < _SIGNIFICANT_P ; sinon elle est rapportée comme
  "inconclusive" (durée insuffisante / mouvement insuffisant pour ce
  signal) plutôt que de deviner.

Usage :
    python3 detect_gripper_lr_sensor_corr.py --session ../session_xxx
    python3 detect_gripper_lr_sensor_corr.py /media/qbee/T9/sessions/ --report report.jsonl
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

_SAMPLE_FPS = 8.0
_FLOW_W, _FLOW_H = 320, 200
_MAX_LAG_SEC = 0.5
_LAG_STEP_SEC = 0.05
_MIN_OVERLAP_SEC = 2.0
_N_PERMUTATIONS = 300       # tirages — passe rapide (filtre grossier p<0.05 sur toutes les sessions)
_N_PERMUTATIONS_REFINE = 5000  # tirages — passe de raffinement (uniquement les candidats de la passe rapide)
_PERM_GUARD_SEC = 1.0       # exclut les décalages proches de 0 (auto-corrélation triviale)
_SIGNIFICANT_P = 0.05       # seuil pour qu'une session soit tranchée individuellement


# ─── Séries temporelles ──────────────────────────────────────────────────

def _flow_series(video_path: Path, sample_fps: float = _SAMPLE_FPS) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude moyenne du flux optique (Farneback) entre frames consécutives
    échantillonnées. Retourne (timestamps_relatifs_sec, magnitudes)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))

    times, mags = [], []
    prev_gray = None
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step == 0:
            small = cv2.resize(frame, (_FLOW_W, _FLOW_H))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mags.append(float(mag.mean()))
                times.append(frame_no / fps)
            prev_gray = gray
        frame_no += 1
    cap.release()
    return np.array(times), np.array(mags)


def _camera_first_ts(jsonl_path: Path) -> Optional[float]:
    """Timestamp absolu (capture_timestamp_sec) de la première frame, pour
    convertir les timestamps relatifs de _flow_series en absolus."""
    try:
        with open(jsonl_path) as f:
            d = json.loads(f.readline())
        return float(d["capture_timestamp_sec"])
    except Exception:
        return None


def _sensor_opening_derivative(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lit sensors/{side}.jsonl et retourne (timestamps_milieu, |dOpening_width/dt|).
    Déduplique les timestamps identiques (résolution du capteur) avant dérivation."""
    ts, ow = [], []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("host_time_sec")
                w = d.get("Opening_width")
                if t is not None and w is not None:
                    ts.append(float(t))
                    ow.append(float(w))
    except Exception:
        return np.array([]), np.array([])

    if len(ts) < 3:
        return np.array([]), np.array([])

    ts_arr = np.array(ts)
    ow_arr = np.array(ow)
    order = np.argsort(ts_arr)
    ts_arr, ow_arr = ts_arr[order], ow_arr[order]
    uniq_t, idx = np.unique(ts_arr, return_index=True)
    ow_arr = ow_arr[idx]

    dt = np.diff(uniq_t)
    valid = dt > 0
    if valid.sum() < 3:
        return np.array([]), np.array([])
    dv = np.abs(np.diff(ow_arr)[valid]) / dt[valid]
    tm = ((uniq_t[1:] + uniq_t[:-1]) / 2.0)[valid]
    return tm, dv


# ─── Corrélation à décalage optimal ─────────────────────────────────────

def _best_abs_corr(
    cam_t: np.ndarray, cam_v: np.ndarray,
    sen_t: np.ndarray, sen_v: np.ndarray,
    max_lag: float = _MAX_LAG_SEC, lag_step: float = _LAG_STEP_SEC,
    grid_dt: float = 0.05,
) -> tuple[float, float, int]:
    """Corrélation de Pearson maximisée en valeur absolue sur une fenêtre de
    décalage temporel restreinte. Retourne (r, lag_sec, n_points_grille)."""
    if len(cam_t) < 5 or len(sen_t) < 5:
        return 0.0, 0.0, 0
    lo = max(cam_t.min(), sen_t.min())
    hi = min(cam_t.max(), sen_t.max())
    if hi - lo < _MIN_OVERLAP_SEC:
        return 0.0, 0.0, 0

    grid = np.arange(lo, hi, grid_dt)
    a = np.interp(grid, cam_t, cam_v)
    if a.std() < 1e-9:
        return 0.0, 0.0, len(grid)
    a_z = (a - a.mean()) / (a.std() + 1e-9)

    best_r, best_lag = 0.0, 0.0
    for lag in np.arange(-max_lag, max_lag + lag_step, lag_step):
        b = np.interp(grid - lag, sen_t, sen_v)
        if b.std() < 1e-9:
            continue
        b_z = (b - b.mean()) / (b.std() + 1e-9)
        r = float(np.corrcoef(a_z, b_z)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, float(lag)
    return best_r, best_lag, len(grid)


def _margin_for_shift(
    flow_t_left: np.ndarray, flow_v_left: np.ndarray,
    flow_t_right: np.ndarray, flow_v_right: np.ndarray,
    sen_t_left: np.ndarray, sen_v_left: np.ndarray,
    sen_t_right: np.ndarray, sen_v_right: np.ndarray,
) -> float:
    r_LL, _, _ = _best_abs_corr(flow_t_left, flow_v_left, sen_t_left, sen_v_left)
    r_LR, _, _ = _best_abs_corr(flow_t_left, flow_v_left, sen_t_right, sen_v_right)
    r_RL, _, _ = _best_abs_corr(flow_t_right, flow_v_right, sen_t_left, sen_v_left)
    r_RR, _, _ = _best_abs_corr(flow_t_right, flow_v_right, sen_t_right, sen_v_right)
    return (abs(r_LL) + abs(r_RR)) - (abs(r_LR) + abs(r_RL))


def _circular_shift(t: np.ndarray, v: np.ndarray, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Décale circulairement une série temporelle de `shift` secondes, en
    bouclant sur sa propre durée. Préserve l'autocorrélation interne de la
    série (donc sa structure de bruit) tout en détruisant son alignement
    réel avec les autres séries — c'est l'hypothèse nulle du test."""
    t0, t1 = t.min(), t.max()
    period = t1 - t0
    if period <= 0:
        return t, v
    t_shifted = t0 + ((t - t0 + shift) % period)
    order = np.argsort(t_shifted)
    return t_shifted[order], v[order]


def _permutation_pvalue(
    flow_t_left: np.ndarray, flow_v_left: np.ndarray,
    flow_t_right: np.ndarray, flow_v_right: np.ndarray,
    sen_t_left: np.ndarray, sen_v_left: np.ndarray,
    sen_t_right: np.ndarray, sen_v_right: np.ndarray,
    observed_margin: float,
    n_perm: int = _N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Test de permutation par décalage circulaire : décale les DEUX séries
    capteur (left + right) du même tirage aléatoire à chaque itération — ce
    qui détruit leur alignement temporel avec les vidéos tout en conservant
    la relation entre les deux capteurs eux-mêmes (mouvements corrélés du
    même tâche). Retourne la fraction de tirages où |marge nulle| >= |marge
    observée| (p-value bilatérale, test exact par permutation)."""
    period_l = sen_t_left.max() - sen_t_left.min()
    period_r = sen_t_right.max() - sen_t_right.min()
    period = min(period_l, period_r)
    if period <= 2 * _PERM_GUARD_SEC:
        return 1.0  # session trop courte pour un test de permutation valide

    rng = np.random.default_rng(seed)
    low, high = _PERM_GUARD_SEC, period - _PERM_GUARD_SEC
    shifts = rng.uniform(low, high, size=n_perm)

    n_extreme = 0
    for shift in shifts:
        sl_t, sl_v = _circular_shift(sen_t_left, sen_v_left, float(shift))
        sr_t, sr_v = _circular_shift(sen_t_right, sen_v_right, float(shift))
        null_margin = _margin_for_shift(
            flow_t_left, flow_v_left, flow_t_right, flow_v_right,
            sl_t, sl_v, sr_t, sr_v,
        )
        if abs(null_margin) >= abs(observed_margin):
            n_extreme += 1
    return (1 + n_extreme) / (1 + n_perm)


# ─── Analyse d'une session ───────────────────────────────────────────────

@dataclass
class SessionCorr:
    name: str
    c_LL: float  # |corr| left.mp4  ↔ sensors/left.jsonl
    c_LR: float  # |corr| left.mp4  ↔ sensors/right.jsonl
    c_RL: float  # |corr| right.mp4 ↔ sensors/left.jsonl
    c_RR: float  # |corr| right.mp4 ↔ sensors/right.jsonl
    lag_LL: float
    lag_RR: float
    p_value: float

    @property
    def score_same(self) -> float:
        """Score de l'hypothèse 'le nommage actuel est correct'."""
        return self.c_LL + self.c_RR

    @property
    def score_swap(self) -> float:
        """Score de l'hypothèse 'left.mp4 et right.mp4 sont inversées'."""
        return self.c_LR + self.c_RL

    @property
    def margin(self) -> float:
        return self.score_same - self.score_swap

    @property
    def vote(self) -> str:
        return "same" if self.margin > 0 else "swap"

    @property
    def significant(self) -> bool:
        return self.p_value < _SIGNIFICANT_P

    @property
    def verdict(self) -> str:
        """Décision exploitable pour CETTE session : tranchée seulement si
        le test de permutation est significatif, sinon 'inconclusive' —
        à ne jamais traiter comme 'same' par défaut."""
        if not self.significant:
            return "inconclusive"
        return self.vote


def analyze_session(session_dir: Path, n_perm: int = _N_PERMUTATIONS) -> Optional[SessionCorr]:
    cam_dir = session_dir / "cameras"
    sens_dir = session_dir / "sensors"
    left_video = cam_dir / "left.mp4"
    right_video = cam_dir / "right.mp4"
    left_sensor = sens_dir / "left.jsonl"
    right_sensor = sens_dir / "right.jsonl"

    if not (left_video.is_file() and right_video.is_file()
            and left_sensor.is_file() and right_sensor.is_file()):
        return None

    t0_left = _camera_first_ts(cam_dir / "left.jsonl")
    t0_right = _camera_first_ts(cam_dir / "right.jsonl")
    if t0_left is None or t0_right is None:
        return None

    flow_t_left, flow_v_left = _flow_series(left_video)
    flow_t_right, flow_v_right = _flow_series(right_video)
    if len(flow_t_left) < 5 or len(flow_t_right) < 5:
        return None
    flow_t_left = flow_t_left + t0_left
    flow_t_right = flow_t_right + t0_right

    sen_t_left, sen_v_left = _sensor_opening_derivative(left_sensor)
    sen_t_right, sen_v_right = _sensor_opening_derivative(right_sensor)
    if len(sen_t_left) < 5 or len(sen_t_right) < 5:
        return None

    r_LL, lag_LL, _ = _best_abs_corr(flow_t_left, flow_v_left, sen_t_left, sen_v_left)
    r_LR, _, _ = _best_abs_corr(flow_t_left, flow_v_left, sen_t_right, sen_v_right)
    r_RL, _, _ = _best_abs_corr(flow_t_right, flow_v_right, sen_t_left, sen_v_left)
    r_RR, lag_RR, _ = _best_abs_corr(flow_t_right, flow_v_right, sen_t_right, sen_v_right)

    margin = (abs(r_LL) + abs(r_RR)) - (abs(r_LR) + abs(r_RL))
    p_value = _permutation_pvalue(
        flow_t_left, flow_v_left, flow_t_right, flow_v_right,
        sen_t_left, sen_v_left, sen_t_right, sen_v_right,
        observed_margin=margin,
        n_perm=n_perm,
        seed=hash(session_dir.name) & 0xFFFFFFFF,
    )

    return SessionCorr(
        name=session_dir.name,
        c_LL=abs(r_LL), c_LR=abs(r_LR), c_RL=abs(r_RL), c_RR=abs(r_RR),
        lag_LL=lag_LL, lag_RR=lag_RR, p_value=p_value,
    )


def _analyze_one(session_dir_str: str, n_perm: int = _N_PERMUTATIONS) -> Optional[dict]:
    try:
        result = analyze_session(Path(session_dir_str), n_perm=n_perm)
    except Exception as exc:  # noqa: BLE001 — une session pourrie ne doit pas planter le batch
        return {"name": Path(session_dir_str).name, "error": repr(exc)}
    if result is None:
        return None
    return {
        "name": result.name,
        "c_LL": round(result.c_LL, 4), "c_LR": round(result.c_LR, 4),
        "c_RL": round(result.c_RL, 4), "c_RR": round(result.c_RR, 4),
        "lag_LL": round(result.lag_LL, 3), "lag_RR": round(result.lag_RR, 3),
        "margin": round(result.margin, 4),
        "vote": result.vote,
        "p_value": round(result.p_value, 4),
        "verdict": result.verdict,   # "same" | "swap" | "inconclusive"
    }


# ─── Agrégation multi-sessions ───────────────────────────────────────────

def _bh_fdr_significant(p_values: list[float], q: float = _SIGNIFICANT_P) -> list[bool]:
    """Correction de Benjamini-Hochberg (FDR) sur une famille de p-values.

    Tester ~230 sessions à p<0.05 chacune produit ~11 faux positifs attendus
    par pur hasard même si aucune session n'est réellement inversée. Sans
    cette correction, une seule détection isolée parmi des dizaines de
    'same' ne permet pas de distinguer un vrai câblage inversé d'un faux
    positif statistique — exactement le piège à éviter avant de qualifier
    une session de 'à corriger'.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    max_rank_passing = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / n) * q:
            max_rank_passing = rank
    sig = [False] * n
    for rank, idx in enumerate(order, start=1):
        if rank <= max_rank_passing:
            sig[idx] = True
    return sig


def _print_aggregate(rows: list[dict]) -> None:
    valid = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    print(f"\n{'─' * 60}")
    print(f"Sessions analysées : {len(rows)}  (exploitables : {len(valid)}, erreurs : {len(errors)})")
    if not valid:
        print("Aucune session exploitable.")
        return

    decided = [r for r in valid if r["verdict"] != "inconclusive"]
    inconclusive = [r for r in valid if r["verdict"] == "inconclusive"]
    n_same = sum(1 for r in decided if r["verdict"] == "same")
    n_swap = sum(1 for r in decided if r["verdict"] == "swap")

    print(f"\nDiagnostic PAR SESSION (test de permutation, p<{_SIGNIFICANT_P} avant correction) :")
    print(f"  Tranchées 'nommage correct'  : {n_same}/{len(valid)}")
    print(f"  Tranchées 'nommage inversé'  : {n_swap}/{len(valid)}")
    print(f"  Non concluantes (p>={_SIGNIFICANT_P}, session trop courte/bruitée) : {len(inconclusive)}/{len(valid)}")

    # ── Correction FDR sur l'ensemble des sessions exploitables ─────────────
    sig_fdr = _bh_fdr_significant([r["p_value"] for r in valid])
    swap_fdr = [r for r, s in zip(valid, sig_fdr) if s and r["verdict"] == "swap"]
    same_fdr = [r for r, s in zip(valid, sig_fdr) if s and r["verdict"] == "same"]
    print(f"\nAprès correction FDR (Benjamini-Hochberg, q<{_SIGNIFICANT_P}, {len(valid)} tests) :")
    print(f"  'nommage correct' robuste au test multiple : {len(same_fdr)}/{len(valid)}")
    print(f"  'nommage inversé' robuste au test multiple : {len(swap_fdr)}/{len(valid)}")

    if swap_fdr:
        print(f"\n  Sessions à corriger (nommage inversé, significatif APRÈS correction FDR) :")
        for r in sorted(swap_fdr, key=lambda x: x["p_value"])[:30]:
            print(f"    {r['name']}  margin={r['margin']:+.3f}  p={r['p_value']:.4f}")
        if len(swap_fdr) > 30:
            print(f"    … et {len(swap_fdr) - 30} de plus (voir le rapport JSONL complet)")
    elif n_swap:
        print(f"\n  {n_swap} session(s) flaguée(s) 'inversée' avant correction, mais aucune ne")
        print(f"  survit à la correction FDR — compatible avec du bruit statistique pur")
        print(f"  (~{len(valid) * _SIGNIFICANT_P:.0f} faux positifs attendus sur {len(valid)} tests à p<{_SIGNIFICANT_P}).")
        print(f"  Ne pas corriger ces sessions sur cette seule base.")

    # ── Verdict agrégé (à titre indicatif seulement — le câblage peut varier
    #    d'une session à l'autre, donc ceci ne remplace pas le diagnostic
    #    par session ci-dessus) ─────────────────────────────────────────────
    sum_margin = sum(r["margin"] for r in valid)
    print(f"\n{'─' * 60}")
    print(f"[Indicatif] Marge cumulée toutes sessions confondues : {sum_margin:+.4f}")
    print("[Indicatif] Ne reflète qu'une tendance globale — ne pas l'utiliser pour")
    print("            corriger une session individuelle : voir le verdict par")
    print("            session ci-dessus pour ça.")

    if errors:
        print(f"\n{len(errors)} session(s) en erreur :")
        for e in errors[:10]:
            print(f"  {e['name']} : {e['error']}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path, help="Répertoire contenant plusieurs sessions")
    p.add_argument("--session", type=Path, help="Analyser une seule session")
    p.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--report", type=Path, metavar="JSONL", help="Écrire un rapport JSONL (une ligne par session)")
    args = p.parse_args()

    if args.session:
        result = _analyze_one(str(args.session.resolve()))
        if result is None:
            print(f"{args.session.name} : données insuffisantes (vidéos/capteurs manquants ou trop courts)")
            return 0
        print(json.dumps(result, indent=2, ensure_ascii=False))
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

    print(f"{len(sessions)} sessions, {args.workers} workers…\n")
    rows: list[dict] = []
    report_fh = args.report.open("w", encoding="utf-8") if args.report else None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # ── Passe 1 (rapide) : filtre grossier p<0.05 sur toutes les sessions ──
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

        # ── Passe 2 (raffinement) : ré-évalue uniquement les candidats de la
        #    passe 1 avec beaucoup plus de tirages, pour obtenir des p-values
        #    assez fines pour survivre à la correction FDR multi-sessions
        #    (300 tirages plafonnent à p=1/301, trop grossier pour ~200+ tests). ──
        candidates = [r["name"] for r in rows if "error" not in r and r["p_value"] < _SIGNIFICANT_P]
        if candidates:
            print(f"Raffinement de {len(candidates)} session(s) candidate(s) ({_N_PERMUTATIONS_REFINE} tirages)…")
            by_name = {Path(s).name: s for s in sessions}
            futures = {
                pool.submit(_analyze_one, str(by_name[name]), _N_PERMUTATIONS_REFINE): name
                for name in candidates
            }
            refined_by_name = {}
            for done2, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if result is not None:
                    refined_by_name[result["name"]] = result
                if done2 % 10 == 0 or done2 == len(candidates):
                    print(f"  … {done2}/{len(candidates)}", end="\r")
            print()
            rows = [refined_by_name.get(r["name"], r) for r in rows]

    if report_fh:
        for r in rows:
            report_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        report_fh.close()

    _print_aggregate(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
