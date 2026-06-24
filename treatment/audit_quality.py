#!/usr/bin/env python3
"""
audit_quality.py — Audit en lecture seule de la qualité des sessions, à l'échelle.

Pour chaque session, agrège TROIS couches de vérification, toutes en
dry-run (aucune écriture dans les dossiers de session) :

  1. checks.py (score 0-100, grade A-F)            — treatment-worker
  2. post/ pipeline (intégrité, sync, shuffle,      — vm-storage/post/
     charuco head/gripper, left/right marker-distance)
  3. validate_session (structure SessionsToMistral) — sessions-uploader

Conçu pour tourner sur des dizaines de milliers de sessions (cf. besoin
~50 000 sessions, plusieurs jours de calcul) :

  - Multiprocessing (CPU-bound : numpy + décodage vidéo OpenCV).
  - Cache externe (--cache fichier.json) : ne réanalyse jamais une session
    dont l'empreinte (taille+mtime des fichiers) n'a pas changé depuis le
    dernier run → un run interrompu/relancé reprend où il s'était arrêté.
    Aucun fichier n'est jamais écrit dans les dossiers de session.
  - Interface ASCII terminal (curses) en direct : score moyen, distribution
    des grades A-F, moyennes par critère, problèmes les plus fréquents.
    Touche 'q' pour arrêter proprement (le CSV/cache déjà écrits restent
    valides). Retombe en mode texte simple si le terminal ne supporte pas
    curses (--no-ui, sortie redirigée vers un fichier, cron...).
  - CSV écrit en continu (append) : on peut le `tail -f` ou l'ouvrir dans un
    tableur pendant que le run tourne, sans attendre la fin.

Usage :
  python audit_quality.py --dir /data/sessions --cache audit_cache.json --csv audit.csv
  python audit_quality.py --dir /data/sessions --workers 16 --skip-charuco
  python audit_quality.py --dir /data/sessions --cache audit_cache.json   # relance = reprise
  python audit_quality.py --dir /data/sessions --no-ui > audit.log        # mode texte (cron/log)
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "post"))
os.environ.setdefault("NAS_SESSIONS_DIR", "")

import checks  # noqa: E402

try:
    import verify_integrity
    import verify_camera_sync
    import diagnose_shuffle
    HAS_POST_BASE = True
except ImportError as e:
    HAS_POST_BASE = False
    logger.warning("Modules post/ (intégrité/sync/shuffle) non disponibles : %s", e)

try:
    import detect_charuco_lr
    import detect_gripper_lr_marker_distance as lr_check
    HAS_POST_VISION = True
except ImportError as e:
    HAS_POST_VISION = False
    logger.warning("Modules post/ vision (charuco/lr) non disponibles : %s", e)


GRADES = ("A", "B", "C", "D", "F")

CSV_FIELDNAMES = [
    "session_id", "path", "fingerprint", "score", "grade", "passed", "duration_s",
    "treatment_issues", "post_status", "post_issues", "mistral_issues",
]


# ═════════════════════════════════════════════════════════════════════════════
# Métadonnées de session (réutilisé par checks.run_checks)
# ═════════════════════════════════════════════════════════════════════════════

def _read_json_safe(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_duration(session_path: str) -> Optional[float]:
    analysis = _read_json_safe(os.path.join(session_path, "analysis.json"))
    if analysis:
        cams = analysis.get("fps_check", {}).get("cameras", {})
        durs = [c["duration_sec"] for c in cams.values() if c.get("duration_sec")]
        if durs:
            durs.sort()
            return durs[len(durs) // 2]
    for fname in ("metadata.json", "config.json"):
        meta = _read_json_safe(os.path.join(session_path, fname))
        if isinstance(meta.get("duration_seconds"), (int, float)):
            return float(meta["duration_seconds"])
    return None


def _get_scenario(session_path: str) -> Optional[str]:
    mission = _read_json_safe(os.path.join(session_path, "mission.json"))
    if mission.get("scenario_id"):
        return str(mission["scenario_id"])
    if mission.get("name"):
        return str(mission["name"])
    meta = _read_json_safe(os.path.join(session_path, "metadata.json"))
    sid = meta.get("scenario_id") or meta.get("scenario")
    return str(sid) if sid else None


def build_session_dict(session_path: str) -> dict:
    session_id = os.path.basename(session_path.rstrip("/"))
    return {
        "session_id":       session_id,
        "session_folder":   session_path,
        "duration_seconds": _get_duration(session_path),
        "scenario_id":      _get_scenario(session_path),
    }


def find_sessions(search_dir: str, pattern: str = None) -> list[str]:
    if not os.path.isdir(search_dir):
        return []
    out = []
    for entry in sorted(os.listdir(search_dir)):
        full_path = os.path.join(search_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if pattern and not fnmatch.fnmatch(entry, pattern):
            continue
        out.append(full_path)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Couche 2 — post/ pipeline (intégrité, sync, shuffle, charuco, left/right)
# Reprend la logique de post/run_pipeline.py::_process_one, en lecture seule
# stricte (apply=False, et on n'écrit jamais .postcheck.json dans la session).
# ═════════════════════════════════════════════════════════════════════════════

def run_post_checks(session_path: str, run_charuco: bool, run_lr_check: bool,
                     charuco_sample_fps: float, lr_n_samples: int) -> tuple[str, list[str]]:
    """Retourne (status, lines) avec status in {"OK", "ANOMALY", "ERROR", "SKIPPED"}."""
    if not HAS_POST_BASE:
        return "SKIPPED", ["modules post/ indisponibles"]

    session_dir = Path(session_path)
    lines: list[str] = []
    try:
        sync_issues = verify_camera_sync.check_session(session_dir)
        for issue in sync_issues:
            lines.append(f"[sync] {issue}")
        is_clean = not sync_issues

        integrity = verify_integrity.check_session(session_dir)
        for subdir, names in sorted(integrity.extra.items()):
            lines.append(f"[en trop] {subdir}/ {sorted(names)}")
        for subdir, names in sorted(integrity.missing.items()):
            lines.append(f"[manquant] {subdir}/ {sorted(names)}")
        for subdir, names in sorted(integrity.corrupt.items()):
            lines.append(f"[corrompu] {subdir}/ {sorted(names)}")
        is_clean = is_clean and integrity.is_clean

        if (session_dir / "config.json").is_file():
            shuffle_report = diagnose_shuffle.analyze_session(session_dir)
            for finding in shuffle_report.findings:
                lines.append(f"[shuffle/{finding.confidence}] {finding.camera.name} : {', '.join(finding.reasons)}")
            is_clean = is_clean and not shuffle_report.findings

        if run_charuco and HAS_POST_VISION and (session_dir / "cameras").is_dir():
            findings = detect_charuco_lr.analyze_session(session_dir, sample_fps=charuco_sample_fps)
            charuco_anomaly = any(f.mismatch or f.role == "unreadable" for f in findings)
            for f in findings:
                if f.mismatch:
                    lines.append(f"[charuco] {f.current_name}.mp4 incohérent (contenu='{f.role}')")
                elif f.role == "unreadable":
                    lines.append(f"[charuco] {f.current_name}.mp4 illisible/corrompue")
            is_clean = is_clean and not charuco_anomaly

        if run_lr_check and HAS_POST_VISION and (session_dir / "cameras").is_dir() and (session_dir / "sensors").is_dir():
            lr_result = lr_check.quick_analyze_session(session_dir, n_samples=lr_n_samples)
            verdict = lr_result.verdict if lr_result is not None else None
            if lr_result is not None and verdict.startswith("inconclusive"):
                full_result = lr_check.analyze_session(session_dir)
                if full_result is not None:
                    lr_result, verdict = full_result, full_result.verdict
            if lr_result is not None and verdict == "swap":
                lines.append(f"[lr] left/right probablement inversés (verdict={verdict})")
                is_clean = is_clean and False

        return ("OK" if is_clean else "ANOMALY"), lines

    except Exception as e:
        logger.exception("Erreur post-checks sur %s", session_dir.name)
        return "ERROR", [f"[erreur interne] {e!r}"]


# ═════════════════════════════════════════════════════════════════════════════
# Couche 3 — validate_session (structure attendue par SessionsToMistral.py)
# Copié localement (logique pure, sans réseau) pour ne pas dépendre de
# `requests` ni des constantes d'upload du module original.
# ═════════════════════════════════════════════════════════════════════════════

_MP4_MIN_BYTES = 100_000  # 100 KB


def validate_mistral_structure(session_dir: Path) -> list[str]:
    issues: list[str] = []

    result_path = session_dir / "result.json"
    if not result_path.exists():
        issues.append("result.json manquant")
    else:
        try:
            res = json.loads(result_path.read_text(encoding="utf-8"))
            if str(res.get("result", "")).upper() != "SUCCESS":
                issues.append(f"result.json non-SUCCESS (valeur : '{res.get('result')}')")
        except Exception as exc:
            issues.append(f"result.json illisible : {exc}")

    config_path = session_dir / "config.json"
    expected_cameras: list[str] = []
    expected_sensors: list[str] = []
    if not config_path.exists():
        issues.append("config.json manquant")
    else:
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            for cam in cfg.get("cameras", []):
                name = cam.get("name")
                if not name:
                    continue
                expected_cameras.append(name)
                if cam.get("error"):
                    issues.append(f"caméra '{name}' : erreur hardware ({cam['error']})")
            for sen in cfg.get("sensors", []):
                if sen.get("name"):
                    expected_sensors.append(sen["name"])
        except Exception as exc:
            issues.append(f"config.json illisible : {exc}")

    if not (session_dir / "mission.json").exists():
        issues.append("mission.json manquant")

    analysis_path = session_dir / "analysis.json"
    if analysis_path.exists():
        try:
            data = json.loads(analysis_path.read_text(encoding="utf-8"))
            sync = data.get("sync_check", {})
            if isinstance(sync.get("ok"), bool) and not sync["ok"]:
                issues.append(f"sync_check échoué — delta={sync.get('delta_sec', '?')}s")
            errors = data.get("errors") or []
            for e in errors:
                issues.append(f"analysis.json erreur : {e}")
        except Exception:
            pass

    cam_dir = session_dir / "cameras"
    for name in expected_cameras:
        mp4 = cam_dir / f"{name}.mp4"
        jsonl = cam_dir / f"{name}.jsonl"
        if not mp4.exists():
            issues.append(f"cameras/{name}.mp4 manquant")
        elif mp4.stat().st_size < _MP4_MIN_BYTES:
            issues.append(f"cameras/{name}.mp4 trop petit ({mp4.stat().st_size} octets)")
        if not jsonl.exists():
            issues.append(f"cameras/{name}.jsonl manquant")
        elif jsonl.stat().st_size == 0:
            issues.append(f"cameras/{name}.jsonl vide")

    resampled = cam_dir / "resampled_30hz.jsonl"
    if expected_cameras:
        if not resampled.exists():
            issues.append("cameras/resampled_30hz.jsonl manquant")
        elif resampled.stat().st_size == 0:
            issues.append("cameras/resampled_30hz.jsonl vide")

    sen_dir = session_dir / "sensors"
    for name in expected_sensors:
        jsonl = sen_dir / f"{name}.jsonl"
        if not jsonl.exists():
            issues.append(f"sensors/{name}.jsonl manquant")
        elif jsonl.stat().st_size == 0:
            issues.append(f"sensors/{name}.jsonl vide")

    return issues


# ═════════════════════════════════════════════════════════════════════════════
# Catégorisation des problèmes (pour les compteurs agrégés / le tableau de bord)
# ═════════════════════════════════════════════════════════════════════════════

_POST_TAG_RE = re.compile(r"^\[([^\]]+)\]")

_MISTRAL_PATTERNS = [
    (re.compile(r"^result\.json"), "result.json"),
    (re.compile(r"^config\.json"), "config.json"),
    (re.compile(r"^mission\.json"), "mission.json"),
    (re.compile(r"^caméra .* erreur hardware"), "caméra hardware"),
    (re.compile(r"^sync_check"), "sync_check"),
    (re.compile(r"^analysis\.json erreur"), "analysis.json erreur"),
    (re.compile(r"^cameras/.*\.mp4 manquant"), "mp4 manquant"),
    (re.compile(r"^cameras/.*\.mp4 trop petit"), "mp4 trop petit"),
    (re.compile(r"^cameras/.*\.jsonl manquant"), "camera jsonl manquant"),
    (re.compile(r"^cameras/.*\.jsonl vide"), "camera jsonl vide"),
    (re.compile(r"^cameras/resampled_30hz"), "resampled_30hz"),
    (re.compile(r"^sensors/.*\.jsonl manquant"), "sensor jsonl manquant"),
    (re.compile(r"^sensors/.*\.jsonl vide"), "sensor jsonl vide"),
]


def _extract_post_tags(lines: list[str]) -> list[str]:
    tags = []
    for line in lines:
        m = _POST_TAG_RE.match(line)
        tags.append(m.group(1) if m else "autre")
    return tags


def _extract_mistral_tags(issues: list[str]) -> list[str]:
    tags = []
    for issue in issues:
        tag = next((t for rx, t in _MISTRAL_PATTERNS if rx.match(issue)), "autre")
        tags.append(tag)
    return tags


# ═════════════════════════════════════════════════════════════════════════════
# Empreinte de session (pour le cache externe) — même principe que
# post/run_pipeline.py::_fingerprint, mais le cache vit hors des sessions.
# ═════════════════════════════════════════════════════════════════════════════

def fingerprint_session(session_dir: Path) -> str:
    from hashlib import sha1
    parts: list[str] = []
    for sub in (".", "cameras", "sensors"):
        d = session_dir if sub == "." else session_dir / sub
        if not d.is_dir():
            continue
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            parts.append(f"{sub}/{entry.name}:{st.st_size}:{int(st.st_mtime)}")
    return sha1("|".join(parts).encode()).hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# Audit d'une session — combine les 3 couches
# ═════════════════════════════════════════════════════════════════════════════

def _init_worker():
    """Limite chaque worker à 1 thread OpenCV interne pour éviter la
    sur-souscription CPU (N processus x M threads OpenCV chacun)."""
    if checks.HAS_CV2:
        checks.cv2.setNumThreads(1)


def audit_session(session_path: str, run_charuco: bool = True, run_lr_check: bool = True,
                   charuco_sample_fps: float = 2.0, lr_n_samples: int = 10) -> dict:
    """Lance les 3 couches de checks. N'écrit jamais sur disque."""
    session_path = str(Path(session_path).resolve())
    session_dir = Path(session_path)
    session = build_session_dict(session_path)
    sid = session["session_id"]

    t0 = time.monotonic()
    fingerprint = fingerprint_session(session_dir)

    try:
        result = checks.run_checks(session, session_path_override=session_path)
        quality = result.get("quality", {})
        failed = [k for k, v in result["checks"].items() if not v.get("ok", True)]
        treatment_detail = "; ".join(f"{k}: {result['checks'][k].get('detail', '')}" for k in failed)
        score, grade, passed = quality.get("score", 0.0), quality.get("grade", "F"), result["passed"]
        criteria = quality.get("criteria", {})
    except Exception as e:
        logger.exception("Erreur checks.py sur %s", sid)
        score, grade, passed = 0.0, "F", False
        treatment_detail = f"EXCEPTION: {e}"
        failed, criteria = [], {}

    post_status, post_lines = run_post_checks(
        session_path, run_charuco, run_lr_check, charuco_sample_fps, lr_n_samples
    )
    mistral_issues = validate_mistral_structure(session_dir)

    elapsed = time.monotonic() - t0
    return {
        "session_id":        sid,
        "path":               session_path,
        "fingerprint":        fingerprint,
        "score":              score,
        "grade":              grade,
        "passed":             passed,
        "duration_s":         round(elapsed, 2),
        "treatment_issues":   treatment_detail,
        "post_status":        post_status,
        "post_issues":        "; ".join(post_lines),
        "mistral_issues":     "; ".join(mistral_issues),
        # champs structurés, utilisés pour les agrégats live (pas dans le CSV)
        "criteria":           criteria,
        "failed_checks_list": failed,
        "post_tags_list":     _extract_post_tags(post_lines),
        "mistral_tags_list":  _extract_mistral_tags(mistral_issues),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Cache externe (jamais dans les dossiers de session)
# ═════════════════════════════════════════════════════════════════════════════

def load_cache(path: Optional[str]) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Cache illisible (%s), redémarrage à vide", path)
        return {}


def save_cache(path: Optional[str], cache: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def _open_csv(path: Optional[str]):
    if not path:
        return None, None
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    return fh, writer


# ═════════════════════════════════════════════════════════════════════════════
# Statistiques agrégées en direct
# ═════════════════════════════════════════════════════════════════════════════

class Stats:
    def __init__(self):
        self.done = 0
        self.grade_counts: Counter = Counter()
        self.score_sum = 0.0
        self.criteria_sum: Counter = Counter()
        self.criteria_count: Counter = Counter()
        self.issue_counts: Counter = Counter()
        self.n_post_anomaly = 0
        self.n_post_error = 0
        self.n_mistral_issue = 0

    def add(self, row: dict) -> None:
        self.done += 1
        self.grade_counts[row.get("grade", "F")] += 1
        self.score_sum += row.get("score", 0.0) or 0.0

        for k, v in (row.get("criteria") or {}).items():
            sc = v.get("score", 0.0) if isinstance(v, dict) else v
            self.criteria_sum[k] += sc
            self.criteria_count[k] += 1

        for tag in row.get("failed_checks_list") or []:
            self.issue_counts[f"checks.py: {tag}"] += 1
        for tag in row.get("post_tags_list") or []:
            self.issue_counts[f"post/: {tag}"] += 1
        for tag in row.get("mistral_tags_list") or []:
            self.issue_counts[f"mistral: {tag}"] += 1

        if row.get("post_status") == "ANOMALY":
            self.n_post_anomaly += 1
        elif row.get("post_status") == "ERROR":
            self.n_post_error += 1
        if row.get("mistral_issues"):
            self.n_mistral_issue += 1

    def avg_score(self) -> float:
        return self.score_sum / self.done if self.done else 0.0

    def criteria_averages(self) -> dict:
        return {k: self.criteria_sum[k] / self.criteria_count[k]
                for k in self.criteria_sum if self.criteria_count[k]}

    def top_issues(self, n: int = 10) -> list:
        return self.issue_counts.most_common(n)


# ═════════════════════════════════════════════════════════════════════════════
# Affichage — interface ASCII terminal (curses) ou texte simple en fallback
# ═════════════════════════════════════════════════════════════════════════════

def _render_dashboard(stdscr, stats: Stats, total: int, t_start: float, label: str) -> None:
    import curses

    stdscr.erase()
    maxy, maxx = stdscr.getmaxyx()

    def put(y: int, x: int, text: str, attr: int = 0) -> None:
        if 0 <= y < maxy - 1 and 0 <= x < maxx:
            try:
                stdscr.addstr(y, x, text[: max(0, maxx - x - 1)], attr)
            except curses.error:
                pass

    elapsed = time.monotonic() - t_start
    rate = stats.done / elapsed if elapsed > 0 else 0.0
    eta_min = (total - stats.done) / rate / 60.0 if rate > 0 else 0.0
    pct = stats.done / total * 100 if total else 0.0

    put(0, 0, f" AUDIT QUALITÉ — {label} ".center(maxx - 1, "─"), curses.A_BOLD)
    put(1, 1, f"{stats.done}/{total} ({pct:5.1f}%)  |  {rate:5.2f} sess/s  |  "
              f"ETA {eta_min:6.0f} min  |  [q] arrêter proprement")

    bar_w = max(10, maxx - 4)
    filled = int(bar_w * pct / 100)
    put(2, 1, "[" + "#" * filled + "-" * (bar_w - filled) + "]")

    put(4, 1, f"Score moyen global : {stats.avg_score():5.1f} / 100", curses.A_BOLD)

    row = 6
    put(row, 1, "Distribution des grades :", curses.A_UNDERLINE)
    row += 1
    bw = 30
    for g in GRADES:
        n = stats.grade_counts.get(g, 0)
        pctg = n / stats.done * 100 if stats.done else 0.0
        f = int(bw * pctg / 100)
        put(row, 3, f"{g}  [{'#' * f}{'-' * (bw - f)}]  {pctg:5.1f}%  ({n})")
        row += 1

    row += 1
    put(row, 1, "Moyennes par critère (checks.py) :", curses.A_UNDERLINE)
    row += 1
    items = sorted(stats.criteria_averages().items())
    for i in range(0, len(items), 2):
        line = ""
        for k, v in items[i:i + 2]:
            line += f"{k:<24} {v:5.1f}   "
        put(row, 3, line)
        row += 1

    row += 1
    put(row, 1, "Problèmes les plus fréquents :", curses.A_UNDERLINE)
    row += 1
    top = stats.top_issues(10)
    if not top:
        put(row, 3, "(aucun pour le moment)")
        row += 1
    for i in range(0, len(top), 2):
        line = ""
        for name, cnt in top[i:i + 2]:
            line += f"{name[:28]:<28} {cnt:>5}   "
        put(row, 3, line)
        row += 1

    row += 1
    put(row, 1,
        f"Anomalies post/ : {stats.n_post_anomaly}   |  Erreurs post/ : {stats.n_post_error}   |  "
        f"Problèmes structure (mistral) : {stats.n_mistral_issue}", curses.A_BOLD)

    stdscr.refresh()


def _print_live_plain(stats: Stats, total: int, t_start: float) -> None:
    elapsed = time.monotonic() - t_start
    rate = stats.done / elapsed if elapsed > 0 else 0.0
    eta_min = (total - stats.done) / rate / 60.0 if rate > 0 else 0.0
    grade_pct = " ".join(
        f"{g}={stats.grade_counts.get(g, 0) / stats.done * 100:.0f}%" for g in GRADES
    ) if stats.done else ""
    print(
        f"  [{stats.done}/{total}] {grade_pct} | post-anomalies={stats.n_post_anomaly} "
        f"mistral-issues={stats.n_mistral_issue} | {rate:.2f} sess/s, ETA {eta_min:.0f} min",
        end="\r", flush=True,
    )


class Reporter:
    """Affiche la progression soit via curses (tableau de bord ASCII), soit
    en texte simple (\\r) si curses est indisponible/désactivé. Débit limité
    à --ui-interval secondes pour ne pas ralentir le run sur de petites
    sessions très rapides."""

    def __init__(self, stdscr, label: str, total: int, t_start: float, min_interval: float = 0.4):
        self.stdscr = stdscr
        self.label = label
        self.total = total
        self.t_start = t_start
        self.min_interval = min_interval
        self._last_draw = 0.0
        if self.stdscr is not None:
            self.stdscr.nodelay(True)

    def update(self, stats: Stats, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        if self.stdscr is not None:
            _render_dashboard(self.stdscr, stats, self.total, self.t_start, self.label)
        else:
            _print_live_plain(stats, self.total, self.t_start)

    def quit_requested(self) -> bool:
        if self.stdscr is None:
            return False
        import curses
        try:
            ch = self.stdscr.getch()
        except curses.error:
            return False
        return ch in (ord("q"), ord("Q"))


# ═════════════════════════════════════════════════════════════════════════════
# Cœur du traitement (appelé directement, ou via curses.wrapper)
# ═════════════════════════════════════════════════════════════════════════════

def _run(stdscr, args) -> tuple[list[dict], Stats, int, dict]:
    run_charuco = not args.skip_charuco
    run_lr_check = not args.skip_lr_check

    if args.session:
        session_dirs = [args.session]
    else:
        session_dirs = find_sessions(args.dir, args.pattern)

    if not session_dirs:
        logger.warning("Aucune session trouvée dans %s (pattern=%s)", args.dir, args.pattern)
        return [], Stats(), 0, {"total": 0, "n_cached": 0, "elapsed": 0.0, "stopped_early": False}

    cache = load_cache(args.cache)
    todo, skipped_rows = [], []
    for sd in session_dirs:
        fp = fingerprint_session(Path(sd))
        cached = cache.get(os.path.basename(sd.rstrip("/")))
        if cached and cached.get("fingerprint") == fp:
            skipped_rows.append(cached["row"])
        else:
            todo.append(sd)

    logger.info("%d session(s) au total — %d depuis le cache, %d à analyser",
                len(session_dirs), len(skipped_rows), len(todo))

    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    n_workers = max(1, min(n_workers, len(todo))) if todo else 1

    stats = Stats()
    rows = list(skipped_rows)
    for r in rows:
        stats.add(r)

    csv_fh, csv_writer = _open_csv(args.csv)

    total = len(session_dirs)
    t_start = time.monotonic()
    label = os.path.basename(args.dir.rstrip("/")) or args.dir
    reporter = Reporter(stdscr, label, total, t_start, min_interval=args.ui_interval)
    reporter.update(stats, force=True)

    last_cache_save = time.monotonic()
    stop_requested = False

    def _handle_row(row: dict) -> None:
        nonlocal last_cache_save
        rows.append(row)
        stats.add(row)
        if csv_writer:
            csv_writer.writerow(row)
            csv_fh.flush()
        cache[row["session_id"]] = {"fingerprint": row["fingerprint"], "row": row}
        reporter.update(stats)
        if args.cache and time.monotonic() - last_cache_save > 30:
            save_cache(args.cache, cache)
            last_cache_save = time.monotonic()

    try:
        if n_workers == 1:
            for sd in todo:
                if reporter.quit_requested():
                    stop_requested = True
                    break
                row = audit_session(sd, run_charuco, run_lr_check,
                                     args.charuco_sample_fps, args.lr_n_samples)
                _handle_row(row)
        elif todo:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
                futures = {
                    pool.submit(audit_session, sd, run_charuco, run_lr_check,
                                args.charuco_sample_fps, args.lr_n_samples): sd
                    for sd in todo
                }
                for future in as_completed(futures):
                    row = future.result()
                    _handle_row(row)
                    if reporter.quit_requested():
                        stop_requested = True
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
    finally:
        if args.cache:
            save_cache(args.cache, cache)
        if csv_fh:
            csv_fh.close()

    reporter.update(stats, force=True)
    elapsed = time.monotonic() - t_start
    n_failed = sum(1 for r in rows if not r.get("passed"))
    exit_code = 0 if (n_failed == 0 and stats.n_post_anomaly == 0 and not stop_requested) else 1
    meta = {
        "total": total, "n_cached": len(skipped_rows),
        "elapsed": elapsed, "stopped_early": stop_requested,
    }
    return rows, stats, exit_code, meta


def _print_summary(rows: list[dict], stats: Stats, meta: dict, args) -> None:
    total = meta.get("total", 0)
    if total == 0:
        return

    rows_sorted = sorted(rows, key=lambda r: r.get("score", 0.0))
    n_passed = sum(1 for r in rows if r.get("passed"))
    n_failed = len(rows) - n_passed
    extra = " (arrêté avant la fin — touche q)" if meta.get("stopped_early") else ""

    print("\n" + "=" * 100)
    print(f"AUDIT QUALITÉ — {len(rows)}/{total} session(s) "
          f"({meta.get('n_cached', 0)} depuis cache) en {meta.get('elapsed', 0.0):.1f}s{extra}")
    print(f"  Score moyen : {stats.avg_score():.1f}  |  Passées (checks.py) : {n_passed}  |  Échouées : {n_failed}")
    if rows:
        print("  Distribution : " + "  ".join(
            f"{g}={stats.grade_counts.get(g, 0)} ({stats.grade_counts.get(g, 0) / len(rows) * 100:.1f}%)"
            for g in GRADES
        ))
    crit_avgs = stats.criteria_averages()
    if crit_avgs:
        print("  Moyennes par critère :")
        for k, v in sorted(crit_avgs.items()):
            print(f"    {k:<24} {v:5.1f}")
    print(f"  Anomalies post/ (intégrité/sync/shuffle/charuco/lr) : {stats.n_post_anomaly}  "
          f"(erreurs : {stats.n_post_error})")
    print(f"  Problèmes structure (style SessionsToMistral)      : {stats.n_mistral_issue}")
    if stats.issue_counts:
        print("  Problèmes les plus fréquents :")
        for name, cnt in stats.top_issues(15):
            print(f"    {name:<40} {cnt}")
    print("=" * 100)

    print(f"\n{'PIRES SESSIONS (top ' + str(args.top) + ')':^100}")
    print(f"{'session_id':<35} {'score':>6} {'grade':>5} {'post':>8}  problèmes")
    print("-" * 100)
    for r in rows_sorted[: args.top]:
        problems = "; ".join(p for p in (r.get("treatment_issues"), r.get("post_issues"), r.get("mistral_issues")) if p)
        print(f"{r['session_id']:<35} {r['score']:>6.1f} {r['grade']:>5} {r.get('post_status', '?'):>8}  {problems[:80]}")


# ═════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Audite la qualité des sessions à l'échelle, sans écrire de fichier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir", "-d", required=True, help="Répertoire contenant les sessions")
    parser.add_argument("--pattern", "-p", default=None, help="Filtrer par nom (ex: session_2026*)")
    parser.add_argument("--session", "-s", default=None, help="Auditer une seule session")
    parser.add_argument("--csv", default=None, help="CSV de sortie, écrit en continu (append)")
    parser.add_argument("--cache", default=None,
                         help="Fichier de cache externe (JSON) pour reprendre un run interrompu "
                              "sans réanalyser les sessions inchangées")
    parser.add_argument("--top", type=int, default=20, help="Nombre de pires sessions à afficher (def: 20)")
    parser.add_argument("--workers", "-w", type=int, default=0,
                         help="Processus parallèles (def: tous les cœurs). 1 = séquentiel")
    parser.add_argument("--skip-charuco", action="store_true",
                         help="Sauter le check charuco head/gripper (coûteux, post/)")
    parser.add_argument("--skip-lr-check", action="store_true",
                         help="Sauter le check left/right marker-distance (coûteux, post/)")
    parser.add_argument("--charuco-sample-fps", type=float, default=2.0)
    parser.add_argument("--lr-n-samples", type=int, default=10)
    parser.add_argument("--ui-interval", type=float, default=0.4,
                         help="Intervalle minimum (s) entre deux rafraîchissements de l'affichage live")
    parser.add_argument("--no-ui", action="store_true",
                         help="Désactive le tableau de bord ASCII (curses), force le mode texte simple "
                              "(utile pour cron/logs/sortie redirigée)")
    args = parser.parse_args()

    use_ui = (not args.no_ui) and sys.stdout.isatty()
    if use_ui:
        try:
            import curses
        except ImportError:
            use_ui = False

    if use_ui:
        prev_level = logger.level
        logger.setLevel(logging.ERROR)  # curses possède l'écran : pas de logs qui le perturbent
        try:
            rows, stats, exit_code, meta = curses.wrapper(_run, args)
        finally:
            logger.setLevel(prev_level)
    else:
        rows, stats, exit_code, meta = _run(None, args)

    _print_summary(rows, stats, meta, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
