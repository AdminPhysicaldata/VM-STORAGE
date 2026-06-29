#!/usr/bin/env python3
"""Pipeline complet de nettoyage/validation/notation d'un répertoire de sessions.

Chaîne, pour chaque session, les contrôles dans cet ordre :

  1. fix_camera_names   — corrige les typos de noms (cameras/ et sensors/)
  2. verify_camera_sync — vérifie qu'un renommage éventuel (étape 1, ou un
                           swap appliqué par detect_charuco_lr.apply_fix) a
                           bien été propagé jusqu'à cameras/resampled_30hz.jsonl
                           (sinon la session reste désynchronisée sur les
                           ANCIENS noms — silencieusement inutilisable pour
                           toute corrélation caméra↔capteur)
  3. verify_integrity   — vérifie l'existence EXACTE des fichiers attendus
                           (+ détecte les fichiers présents mais corrompus)
  4. diagnose_shuffle    — détecte les sessions contaminées par un autre device
                           (alignement temporel, durée jsonl + durée vidéo ffprobe)
  5. detect_charuco_lr  — vérifie via les marqueurs ArUco 244/255 que "head"
                           ne voit jamais les pinces et que "left"/"right" les
                           voient toujours (ne tente PAS de distinguer left de
                           right entre eux : signal trop bruité en valeur
                           absolue, voir le docstring de detect_charuco_lr.py).
                           "ambiguous" (zone grise, fréquent sur vidéo basse
                           résolution/floue) n'est PAS bloquant — seul un
                           "mismatch" net ou une vidéo illisible le sont.
  6. detect_gripper_lr_marker_distance — tranche ce que l'étape 5 ne peut pas :
                           corrèle la distance des marqueurs 244/255 à
                           Opening_width (sensors/) pour confirmer que
                           cameras/left.mp4 correspond bien à sensors/left.jsonl
                           (et idem right) — r≈0.99 sur le bon appariement.
                           Passe rapide (~10 frames ciblées) puis, si
                           inconclusive, ESCALADE vers un scan complet (toute
                           la vidéo) avant de conclure — un échantillonnage
                           insuffisant ne prouve rien sur une vidéo difficile à
                           analyser. Seul un "swap" confiant (rapide ou complet)
                           rend la session anormale (corrigé auto en --apply) ;
                           "inconclusive" même après le scan complet reste OK :
                           l'absence de preuve n'est pas une preuve d'inversion.
  7. checks.py (score)  — note la session de 0 à 100 (grade A-F), 8 critères
                           pondérés (sync caméras, stabilité vidéo, détection
                           gripper, IMU, intégrité, couverture temporelle,
                           durée). Fusionné depuis treatment-worker : même
                           moteur de scoring que le pipeline de production,
                           pour avoir une note cohérente entre l'audit local
                           et le traitement en base. N'écrit rien d'autre que
                           le marqueur de cache (.postcheck.json) — jamais de
                           treatment.json dans la session (réservé au worker).
  8. structure (mistral) — vérification légère façon SessionsToMistral.py
                           (result.json=SUCCESS, tailles mp4, fichiers
                           capteurs...) : ce qui ferait rejeter la session à
                           l'envoi, détecté ici en amont plutôt qu'au moment
                           de l'upload.

Conçu pour tourner sur des dizaines de milliers de sessions :

  - Parallélisme par PROCESSUS (pas threads) : le décodage vidéo (charuco,
    scoring) est CPU-bound, un ProcessPoolExecutor exploite donc tous les
    cœurs sans être bridé par le GIL. Réglable via -j/--workers (défaut :
    tous les cœurs). Chaque worker limite OpenCV à 1 thread interne pour
    éviter la sur-souscription CPU (N processus x M threads chacun).
  - Cache persistant par session : un fichier .postcheck.json est écrit dans
    chaque session après son premier passage (statut structurel + score de
    qualité + lignes de détail). Au prochain run, si aucun fichier de la
    session n'a changé (empreinte taille+mtime) ET que les paramètres du
    pipeline sont identiques, la session est court-circuitée sans rien
    redécoder. Sur un dossier de sessions qui grossit en continu (le cas réel
    ici), ça transforme un "tout réanalyser chaque nuit" en "n'analyser que
    les sessions réellement nouvelles". --force ignore le cache.
  - Tableau de bord ASCII en direct (curses) : score moyen, distribution des
    grades A-F, moyennes par critère, problèmes les plus fréquents. Touche
    'q' pour arrêter proprement. Retombe automatiquement en lignes de
    progression texte simple si la sortie n'est pas un vrai terminal (cron,
    Docker, --report, --no-ui) — jamais de plantage curses en conteneur.
  - Isolation des erreurs : une session qui plante (vidéo illisible,
    exception inattendue) est rapportée en erreur et n'interrompt jamais le
    reste du lot.
  - Rapport JSONL optionnel (--report) pour audit/diff sans avoir à tout
    réimprimer sur stdout.

Usage :
    # Rapport complet, aucune modification, tableau de bord ASCII, tous les cœurs
    python3 run_pipeline.py /media/qbee/T9/sessions/

    # Gros volume : limiter les workers, écrire un rapport JSONL, mode texte (cron)
    python3 run_pipeline.py /media/qbee/T9/sessions/ -j 12 --report report.jsonl --no-ui

    # Une seule session, verbeux, jamais de cache
    python3 run_pipeline.py --session ../session_20260605_190710

    # Application réelle + tri, en ignorant le cache existant
    python3 run_pipeline.py /media/qbee/T9/sessions/ --apply --force \\
        --move-clean /media/qbee/T9/clean/ --move-bad /media/qbee/T9/quarantine/

    # Sauter le scoring qualité (post/ seul, comportement historique)
    python3 run_pipeline.py /media/qbee/T9/sessions/ --skip-quality
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sessions-uploader"))
os.environ.setdefault("NAS_SESSIONS_DIR", "")
# Le scoring qualité (checks.py → gripper_tracking.py, check #12) écrit par
# défaut gripper_tracking.csv/gripper_correlation.json dans la session — ce
# qui est le comportement voulu en production (treatment-worker), mais ferait
# échouer verify_integrity.py ici (fichiers "en trop") sur des sessions par
# ailleurs propres. Le post-pipeline reste un audit, donc on désactive cette
# écriture (cf. gripper_tracking.WRITE_OUTPUTS).
os.environ.setdefault("GRIPPER_TRACKING_WRITE_OUTPUT", "false")

import fix_camera_names
import verify_camera_sync
import verify_integrity
import diagnose_shuffle
import checks  # noqa: E402 — scoring qualité (ex treatment-worker)
import SessionsToMistral as mistral_uploader  # noqa: E402 — envoi (--send-mistral)

_PIPELINE_VERSION = 8  # bump à chaque changement de sémantique pour invalider le cache existant
_MARKER_NAME = ".postcheck.json"
_DEFAULT_WORKERS = os.cpu_count() or 4
_PROGRESS_EVERY = 200

GRADES = ("A", "B", "C", "D", "F")


# ─── Couche structure (façon SessionsToMistral.py) ───────────────────────────
# Copié localement (logique pure, sans réseau) pour ne pas dépendre de
# `requests` ni des constantes d'upload du module original.

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
            for e in data.get("errors") or []:
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


# ─── Envoi Mistral (--send-mistral) ───────────────────────────────────────────
# Réutilise les fonctions de sessions-uploader/SessionsToMistral.py (zip,
# upload, enregistrement BDD) au lieu de dupliquer cette logique : une session
# n'est envoyée que si CE pipeline (le plus complet : sync, shuffle, charuco,
# corrélation gripper, structure) l'a déclarée OK — le cron --max-sessions de
# sessions-uploader sur /data/sessions ne fait que la vérification structurelle
# légère de SessionsToMistral.validate_session et n'a pas connaissance de ces
# anomalies.

def send_session_to_mistral(session_dir: Path, sent_dir: Path, offline: bool) -> tuple[bool, str]:
    try:
        analysis, config, mission = mistral_uploader.read_session_metadata(session_dir, dry_run=False)
        duration = mistral_uploader.read_duration(session_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = mistral_uploader.zip_session(session_dir, Path(tmp_dir))
            zip_size = zip_path.stat().st_size
            ok = mistral_uploader.upload_zip_to_mistral(str(zip_path))
            zip_path.unlink(missing_ok=True)

        if not ok:
            if not offline:
                mistral_uploader.api_mark_send_failed_bulk([session_dir.name])
            return False, "upload Mistral échoué"

        mistral_uploader.move_session_to_sent(session_dir, sent_dir)

        if not offline:
            session_id = None
            if analysis is not None:
                registered = mistral_uploader.db_register_sessions_bulk([{
                    "folder_name": session_dir.name, "analysis": analysis,
                    "config": config, "mission": mission, "size_bytes": zip_size,
                }])
                session_id = registered.get(session_dir.name)
            ref = session_id or session_dir.name
            mistral_uploader.api_mark_sent_bulk(
                [{"session_ref": ref, "size_bytes": zip_size, "duration_seconds": duration}]
            )

        return True, f"envoyée à Mistral ({zip_size} octets)"
    except Exception as exc:
        return False, f"erreur envoi Mistral : {exc!r}"


# ─── Couche scoring qualité (checks.py, ex treatment-worker) ─────────────────

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


def build_quality_session_dict(session_path: str) -> dict:
    session_id = os.path.basename(session_path.rstrip("/"))
    return {
        "session_id":       session_id,
        "session_folder":   session_path,
        "duration_seconds": _get_duration(session_path),
        "scenario_id":      _get_scenario(session_path),
    }


def compute_quality(session_dir: Path) -> tuple[float, str, dict, list[str]]:
    """Retourne (score, grade, criteria, lines) via checks.run_checks()."""
    session_path = str(session_dir)
    session = build_quality_session_dict(session_path)
    result = checks.run_checks(session, session_path_override=session_path)
    quality = result.get("quality", {})
    score, grade = quality.get("score", 0.0), quality.get("grade", "F")
    criteria = quality.get("criteria", {})
    lines = [
        f"[score] {k}: {v.get('detail', '')}"
        for k, v in result["checks"].items() if not v.get("ok", True)
    ]
    return score, grade, criteria, lines


# ─── Empreinte de session (pour le cache) ────────────────────────────────────

def _fingerprint(session_dir: Path) -> str:
    """Empreinte bon marché (taille+mtime, pas le contenu) de tous les fichiers
    pertinents d'une session. Si elle ne change pas entre deux runs, on sait
    que rien n'a été modifié et on peut sauter le travail coûteux."""
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
            if not entry.is_file() or entry.name == _MARKER_NAME:
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            parts.append(f"{sub}/{entry.name}:{st.st_size}:{int(st.st_mtime)}")
    return sha1("|".join(parts).encode()).hexdigest()


def _config_key(apply: bool, run_charuco: bool, charuco_sample_fps: float,
                 run_lr_check: bool, lr_n_samples: int, run_quality: bool) -> str:
    # NB : "apply" n'entre pas dans la clé — un résultat "OK" en dry-run reste
    # valide en mode --apply (rien à appliquer). Voir _process_one pour la
    # logique d'invalidation qui dépend du statut ET de apply.
    return f"{_PIPELINE_VERSION}:{run_charuco}:{charuco_sample_fps}:{run_lr_check}:{lr_n_samples}:{run_quality}"


def _load_cache(session_dir: Path, fingerprint: str, config_key: str) -> dict | None:
    try:
        data = json.loads((session_dir / _MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fingerprint or data.get("config_key") != config_key:
        return None
    return data


def _save_cache(session_dir: Path, fingerprint: str, config_key: str, status: str,
                 lines: list[str], score: float, grade: str, criteria: dict,
                 mistral_issues: list[str]) -> None:
    marker = session_dir / _MARKER_NAME
    try:
        marker.write_text(json.dumps({
            "fingerprint": fingerprint,
            "config_key": config_key,
            "status": status,
            "lines": lines,
            "score": score,
            "grade": grade,
            "criteria": criteria,
            "mistral_issues": mistral_issues,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")
    except OSError:
        pass  # un cache qu'on n'arrive pas à écrire n'est pas bloquant


# ─── Traitement d'une session (fonction top-level → picklable pour le pool) ──

def _init_worker() -> None:
    """Limite chaque worker à 1 thread OpenCV interne pour éviter la
    sur-souscription CPU (N processus x M threads OpenCV chacun)."""
    if checks.HAS_CV2:
        checks.cv2.setNumThreads(1)


def _process_one(
    session_dir_str: str,
    apply: bool,
    run_charuco: bool,
    charuco_sample_fps: float,
    force: bool,
    run_lr_check: bool = True,
    lr_n_samples: int = 10,
    run_quality: bool = True,
) -> dict:
    """Tout ce qui touche une session, isolé pour tourner dans un worker
    séparé. Ne lève jamais — toute exception est convertie en statut ERROR
    pour ne jamais faire tomber le pool sur une session pourrie."""
    session_dir = Path(session_dir_str)
    name = session_dir.name
    config_key = _config_key(apply, run_charuco, charuco_sample_fps, run_lr_check, lr_n_samples, run_quality)

    try:
        fingerprint = _fingerprint(session_dir)
        cached = None if force else _load_cache(session_dir, fingerprint, config_key)
        # Un cache "OK" est valide quel que soit apply (rien à faire de toute façon).
        # Un cache "ANOMALY"/"ERROR" n'est réutilisable qu'en lecture (apply=False) :
        # si apply=True il faut réellement tenter la correction, pas juste relire
        # un ancien diagnostic jamais appliqué.
        if cached is not None and (cached["status"] == "OK" or not apply):
            return {
                "name": name, "status": cached["status"], "lines": cached["lines"], "cached": True,
                "score": cached.get("score", 0.0), "grade": cached.get("grade", "F"),
                "criteria": cached.get("criteria", {}), "mistral_issues": cached.get("mistral_issues", []),
            }

        lines: list[str] = []

        cam_fixer = fix_camera_names.SessionFixer(
            session_dir, frozenset(fix_camera_names._DEFAULT_EXPECTED), dry_run=not apply, subdir="cameras"
        )
        cam_fixer.fix()
        sens_fixer = fix_camera_names.SessionFixer(
            session_dir, frozenset({"left", "right"}), dry_run=not apply, subdir="sensors"
        )
        sens_fixer.fix()
        lines.extend(f"[noms] {l.strip()}" for l in (*cam_fixer.log, *sens_fixer.log))

        # Vérifie qu'un éventuel renommage left/right a bien été propagé jusqu'à
        # cameras/resampled_30hz.jsonl — sinon la session reste synchronisée sur les
        # ANCIENS noms et toute corrélation caméra↔capteur basée sur ses timestamps
        # devient silencieusement fausse (cf. fix_camera_names.fix_resampled_jsonl).
        sync_issues = verify_camera_sync.check_session(session_dir)
        for issue in sync_issues:
            lines.append(f"[sync]      {issue}")
        is_sync_ok = not sync_issues

        integrity = verify_integrity.check_session(session_dir)
        for subdir, names in sorted(integrity.extra.items()):
            lines.append(f"[en trop]   {subdir}/  {sorted(names)}")
        for subdir, names in sorted(integrity.missing.items()):
            lines.append(f"[manquant]  {subdir}/  {sorted(names)}")
        for subdir, names in sorted(integrity.corrupt.items()):
            lines.append(f"[corrompu]  {subdir}/  {sorted(names)}")
        is_clean = integrity.is_clean and is_sync_ok

        if (session_dir / "config.json").is_file():
            shuffle_report = diagnose_shuffle.analyze_session(session_dir)
            for finding in shuffle_report.findings:
                lines.append(f"[shuffle/{finding.confidence}] {finding.camera.name} : {', '.join(finding.reasons)}")
            # Toute suspicion de shuffle fait échouer la session, même en confiance LOW
            # (un seul signal) : sur ce rig, les sessions saines n'ont structurellement
            # aucun écart temporel de ce type — un LOW n'est donc pas du bruit tolérable
            # ici, contrairement à is_contaminated (qui ne se déclenche qu'à HIGH, ≥2 signaux).
            is_clean = is_clean and not shuffle_report.findings

        if run_charuco and (session_dir / "cameras").is_dir():
            import detect_charuco_lr  # importé seulement si nécessaire (évite la dépendance opencv sinon)
            findings = detect_charuco_lr.analyze_session(session_dir, sample_fps=charuco_sample_fps)
            # "ambiguous" (ratio entre les deux seuils) n'est PAS bloquant : sur des vidéos
            # basse résolution / avec flou de mouvement, le taux de détection des marqueurs
            # tombe naturellement dans cette zone grise même pour un gripper correctement
            # nommé — ça ne veut pas dire que le nommage est faux, juste que le signal est
            # faible. Seul un "mismatch" net (le rôle détecté contredit le nom) ou une vidéo
            # illisible restent des preuves suffisantes pour bloquer la session.
            charuco_anomaly = any(f.mismatch or f.role == "unreadable" for f in findings)
            if charuco_anomaly and apply:
                detect_charuco_lr.apply_fix(session_dir, findings)
                # Le renommage invalide l'empreinte ; on la recalcule pour le cache final.
                fingerprint = _fingerprint(session_dir)
            for f in findings:
                if f.mismatch:
                    lines.append(
                        f"[charuco]   {f.current_name}.mp4 incohérent : nommée '{f.current_name}' "
                        f"mais le contenu correspond à '{f.role}'"
                    )
                elif f.role == "unreadable":
                    lines.append(f"[charuco]   {f.current_name}.mp4 illisible/corrompue")
                elif f.role == "ambiguous":
                    lines.append(f"[charuco/info] {f.current_name}.mp4 : ni clairement gripper ni clairement head (non bloquant)")
            is_clean = is_clean and not charuco_anomaly

        # Vérifie que cameras/left.mp4 et right.mp4 correspondent bien à
        # sensors/left.jsonl et right.jsonl (et pas l'inverse) : corrélation
        # entre la distance des marqueurs ArUco 244/255 et Opening_width.
        #
        # Politique (révisée après un faux-positif massif sur un lot 640x480 à
        # fort flou de mouvement où le taux de détection par frame est
        # naturellement faible) :
        #   1. Passe rapide (~10 frames ciblées, quick_analyze_session) — résout
        #      la grande majorité des sessions en quelques secondes.
        #   2. Si "inconclusive", ESCALADE vers le scan complet (analyze_session,
        #      décode toute la vidéo) avant de conclure quoi que ce soit — un
        #      échantillon de 10 frames qui échoue ne prouve rien sur une vidéo
        #      difficile à analyser, il faut épuiser le signal disponible.
        #   3. Seul un verdict confiant "swap" (passe rapide OU complète) rend la
        #      session anormale — corrigé automatiquement en --apply. "same" et
        #      "inconclusive" (même après le scan complet) sont traités comme OK :
        #      l'absence de preuve n'est pas une preuve d'inversion.
        if run_lr_check and (session_dir / "cameras").is_dir() and (session_dir / "sensors").is_dir():
            import detect_gripper_lr_marker_distance as lr_check
            lr_result = lr_check.quick_analyze_session(session_dir, n_samples=lr_n_samples)
            verdict = lr_result.verdict if lr_result is not None else None
            escalated = False
            if lr_result is not None and verdict.startswith("inconclusive"):
                full_result = lr_check.analyze_session(session_dir)
                if full_result is not None:
                    lr_result, verdict, escalated = full_result, full_result.verdict, True

            if lr_result is None:
                lines.append("[lr]        vidéos/capteurs left+right manquants — vérification ignorée")
            else:
                if verdict == "swap" and apply:
                    fix_log = lr_check.apply_swap_fix(session_dir)
                    lines.extend(f"[lr-fix]    {l}" for l in fix_log)
                    fingerprint = _fingerprint(session_dir)
                    # Re-vérifie après correction : la prochaine analyse doit voir "same".
                    lr_result = lr_check.quick_analyze_session(session_dir, n_samples=lr_n_samples)
                    verdict = lr_result.verdict if lr_result is not None else "inconclusive (échec post-correction)"
                tag = "[lr/full]  " if escalated else "[lr]       "
                if verdict == "swap":
                    if hasattr(lr_result, "r_same_left"):  # QuickResult
                        detail = (f"r_same_left={lr_result.r_same_left:+.2f} r_same_right={lr_result.r_same_right:+.2f} "
                                  f"r_cross_left={lr_result.r_cross_left:+.2f} r_cross_right={lr_result.r_cross_right:+.2f}")
                    else:  # SessionResult
                        detail = (f"r_LL={lr_result.r_LL:+.2f} r_RR={lr_result.r_RR:+.2f} "
                                  f"r_LR={lr_result.r_LR:+.2f} r_RL={lr_result.r_RL:+.2f}")
                    lines.append(f"{tag} left/right probablement inversés ({detail})")
                elif verdict != "same":
                    lines.append(f"{tag} {verdict} (non bloquant)")
                is_clean = is_clean and verdict != "swap"

        # Score de qualité 0-100 / grade A-F (checks.py) — n'influence PAS
        # is_clean/status : c'est une dimension indépendante de la conformité
        # structurelle, affichée et cachée à part (cf. docstring point 7).
        score, grade, criteria = 0.0, "F", {}
        if run_quality:
            try:
                score, grade, criteria, q_lines = compute_quality(session_dir)
                lines.extend(q_lines)
            except Exception as e:
                lines.append(f"[score/erreur] {e!r}")

        mistral_issues = validate_mistral_structure(session_dir)
        for issue in mistral_issues:
            lines.append(f"[structure] {issue}")

        status = "OK" if is_clean else "ANOMALY"
        _save_cache(session_dir, fingerprint, config_key, status, lines, score, grade, criteria, mistral_issues)
        return {
            "name": name, "status": status, "lines": lines, "cached": False,
            "score": score, "grade": grade, "criteria": criteria, "mistral_issues": mistral_issues,
        }

    except Exception as exc:  # noqa: BLE001 — défense en profondeur, jamais de crash du pool
        return {
            "name": name, "status": "ERROR", "lines": [f"[erreur interne] {exc!r}"], "cached": False,
            "score": 0.0, "grade": "F", "criteria": {}, "mistral_issues": [],
        }


# ─── Statistiques agrégées en direct ──────────────────────────────────────────

_TAG_RE = re.compile(r"^\[([^\]]+)\]")


def _extract_tags(lines: list[str]) -> list[str]:
    tags = []
    for line in lines:
        m = _TAG_RE.match(line.strip())
        tags.append(m.group(1) if m else "autre")
    return tags


class Stats:
    def __init__(self):
        self.done = 0
        self.grade_counts: Counter = Counter()
        self.score_sum = 0.0
        self.criteria_sum: Counter = Counter()
        self.criteria_count: Counter = Counter()
        self.issue_counts: Counter = Counter()
        self.status_counts: Counter = Counter()
        self.n_mistral_issue = 0

    def add(self, row: dict) -> None:
        self.done += 1
        self.grade_counts[row.get("grade", "F")] += 1
        self.score_sum += row.get("score", 0.0) or 0.0
        self.status_counts[row.get("status", "ERROR")] += 1

        for k, v in (row.get("criteria") or {}).items():
            sc = v.get("score", 0.0) if isinstance(v, dict) else v
            self.criteria_sum[k] += sc
            self.criteria_count[k] += 1

        for tag in _extract_tags(row.get("lines") or []):
            self.issue_counts[tag] += 1

        if row.get("mistral_issues"):
            self.n_mistral_issue += 1

    def avg_score(self) -> float:
        return self.score_sum / self.done if self.done else 0.0

    def criteria_averages(self) -> dict:
        return {k: self.criteria_sum[k] / self.criteria_count[k]
                for k in self.criteria_sum if self.criteria_count[k]}

    def top_issues(self, n: int = 10) -> list:
        return self.issue_counts.most_common(n)


# ─── Affichage — tableau de bord ASCII (curses) ou texte simple en fallback ──

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

    put(0, 0, f" POST PIPELINE — {label} ".center(maxx - 1, "─"), curses.A_BOLD)
    put(1, 1, f"{stats.done}/{total} ({pct:5.1f}%)  |  {rate:5.2f} sess/s  |  "
              f"ETA {eta_min:6.0f} min  |  [q] arrêter proprement")

    bar_w = max(10, maxx - 4)
    filled = int(bar_w * pct / 100)
    put(2, 1, "[" + "#" * filled + "-" * (bar_w - filled) + "]")

    put(4, 1, f"Score moyen global : {stats.avg_score():5.1f} / 100"
              f"   |   OK={stats.status_counts.get('OK', 0)}  "
              f"ANOMALY={stats.status_counts.get('ANOMALY', 0)}  "
              f"ERROR={stats.status_counts.get('ERROR', 0)}", curses.A_BOLD)

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
    put(row, 1, f"Problèmes de structure (style SessionsToMistral) : {stats.n_mistral_issue}", curses.A_BOLD)

    stdscr.refresh()


def _print_live_plain(stats: Stats, total: int, t_start: float) -> None:
    elapsed = time.monotonic() - t_start
    rate = stats.done / elapsed if elapsed > 0 else 0.0
    eta_min = (total - stats.done) / rate / 60.0 if rate > 0 else 0.0
    grade_pct = " ".join(
        f"{g}={stats.grade_counts.get(g, 0) / stats.done * 100:.0f}%" for g in GRADES
    ) if stats.done else ""
    print(
        f"  … {stats.done}/{total}  {grade_pct}  ok={stats.status_counts.get('OK', 0)} "
        f"anomalies={stats.status_counts.get('ANOMALY', 0)} erreurs={stats.status_counts.get('ERROR', 0)}  "
        f"({rate:.1f} sessions/s, ETA {eta_min:.1f} min)",
        end="\r", flush=True,
    )


class Reporter:
    """Affiche la progression soit via curses (tableau de bord ASCII), soit
    en texte simple (\\r) si curses est indisponible/désactivé. Débit limité
    à min_interval secondes pour ne pas ralentir le run sur de petites
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


# ─── Affichage par session (mode texte uniquement) ───────────────────────────

def _print_result(result: dict, verbose: bool) -> None:
    if result["lines"] or verbose:
        cache_tag = " [cache]" if result.get("cached") else ""
        print(f"\n{result['name']}  [{result['status']}] score={result.get('score', 0.0):.1f} "
              f"grade={result.get('grade', '?')}{cache_tag}")
        for line in result["lines"]:
            print(f"  {line}")


# ─── Cœur du traitement (appelé directement, ou via curses.wrapper) ─────────

def _run(stdscr, args) -> tuple[dict, int, dict]:
    run_charuco = not args.skip_charuco
    run_lr_check = not args.skip_lr_check
    run_quality = not args.skip_quality
    checks.ENABLE_VISION = not args.skip_quality_vision  # cf. checks.py — gripper_tracking/label_inversion

    # ── Mode session unique : pas de tableau de bord, juste le résultat ──────
    if args.session:
        session_dir = args.session.resolve()
        result = _process_one(
            str(session_dir), args.apply, run_charuco, args.charuco_sample_fps, force=True,
            run_lr_check=run_lr_check, lr_n_samples=args.lr_n_samples, run_quality=run_quality,
        )
        _print_result(result, verbose=True)
        return {}, (0 if result["status"] != "ERROR" else 1), {"single": True}

    root = None
    if args.list:
        list_path = args.list.resolve()
        try:
            lines = list_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"Impossible de lire la liste '{list_path}': {exc}")
            return {}, 1, {"single": False, "total": 0}
        sessions = sorted({Path(l.strip()).resolve() for l in lines if l.strip()})
        sessions = [s for s in sessions if s.is_dir()]
    else:
        if not args.directory:
            return {}, 1, {"single": False, "total": 0, "no_args": True}
        root = args.directory.resolve()
        sessions = sorted(
            Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
        )

    if not sessions:
        print(f"Aucune session trouvée" + (f" dans {root}" if root else f" dans la liste '{args.list}'"))
        return {}, 0, {"single": False, "total": 0}

    if args.move_clean:
        args.move_clean.mkdir(parents=True, exist_ok=True)
    if args.move_bad:
        args.move_bad.mkdir(parents=True, exist_ok=True)
    mistral_sent_dir = (args.mistral_sent_dir or (root.parent / "session_envoye")) if args.send_mistral else None
    if mistral_sent_dir:
        mistral_sent_dir.mkdir(parents=True, exist_ok=True)

    total = len(sessions)
    label = root.name if root else (args.list.name if args.list else "sessions")

    report_fh = args.report.open("w", encoding="utf-8") if args.report else None
    stats = Stats()
    moved_clean = moved_bad = move_errors = 0
    sent_mistral = sent_mistral_errors = 0
    cached_count = 0
    stop_requested = False
    t0 = time.monotonic()

    reporter = Reporter(stdscr, label, total, t0, min_interval=args.ui_interval)
    reporter.update(stats, force=True)

    def _move_done(fut, dest_label: str):
        """Callback de fin de déplacement : compte le résultat sans jamais
        faire planter la boucle principale si un move échoue (disque plein,
        permissions...)."""
        nonlocal moved_clean, moved_bad, move_errors
        try:
            fut.result()
            if dest_label == "clean":
                moved_clean += 1
            else:
                moved_bad += 1
        except OSError as exc:
            move_errors += 1
            if stdscr is None:
                print(f"\n  [ERREUR déplacement {dest_label}] {exc}", file=sys.stderr)

    def _send_mistral_done(fut, name: str):
        """Callback de fin d'envoi Mistral : ne fait jamais planter la boucle
        principale (réseau down, upload échoué...) — la session reste alors
        sur place et sera retentée au prochain run (cache invalidé puisque non
        déplacée)."""
        nonlocal sent_mistral, sent_mistral_errors
        try:
            ok, msg = fut.result()
        except Exception as exc:  # noqa: BLE001 — défense en profondeur
            ok, msg = False, repr(exc)
        if ok:
            sent_mistral += 1
        else:
            sent_mistral_errors += 1
        if stdscr is None:
            tag = "OK" if ok else "ERREUR"
            print(f"\n  [mistral/{tag}] {name} : {msg}")

    try:
        # Pool de threads dédié aux déplacements : soumis au fil de l'eau pendant
        # que le ProcessPoolExecutor continue d'analyser les sessions suivantes —
        # une session en anomalie part en quarantaine dès qu'elle est détectée,
        # pas seulement à la toute fin du run (visible en direct dans le dossier).
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool, \
             ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as move_pool:
            futures = {
                pool.submit(
                    _process_one, str(s), args.apply, run_charuco,
                    args.charuco_sample_fps, args.force,
                    run_lr_check, args.lr_n_samples, run_quality,
                ): s
                for s in sessions
            }
            for fut in as_completed(futures):
                session_dir = futures[fut]
                result = fut.result()  # _process_one ne lève jamais : pas de try/except requis ici
                stats.add(result)
                if result.get("cached"):
                    cached_count += 1
                if stdscr is None:
                    _print_result(result, args.verbose)
                if report_fh:
                    report_fh.write(json.dumps(result) + "\n")
                    report_fh.flush()

                if result["status"] == "OK":
                    if args.send_mistral and not result.get("mistral_issues"):
                        sf = move_pool.submit(
                            send_session_to_mistral, session_dir, mistral_sent_dir, args.mistral_offline
                        )
                        sf.add_done_callback(lambda f, n=session_dir.name: _send_mistral_done(f, n))
                    elif args.send_mistral:
                        # Structurellement "propre" pour ce pipeline, mais rejetée par les
                        # checks façon SessionsToMistral (cf. validate_mistral_structure) —
                        # pas envoyée, laissée sur place pour inspection.
                        if stdscr is None:
                            print(f"\n  [mistral/SKIP] {session_dir.name} : {result['mistral_issues']}")
                    elif args.move_clean:
                        mv = move_pool.submit(shutil.move, str(session_dir), str(args.move_clean / session_dir.name))
                        mv.add_done_callback(lambda f: _move_done(f, "clean"))
                elif args.move_bad:
                    mv = move_pool.submit(shutil.move, str(session_dir), str(args.move_bad / session_dir.name))
                    mv.add_done_callback(lambda f: _move_done(f, "bad"))

                reporter.update(stats)

                if reporter.quit_requested():
                    stop_requested = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
            # move_pool se ferme ici (context manager) : attend que tous les
            # déplacements en attente se terminent avant de continuer.
    finally:
        if report_fh:
            report_fh.close()

    reporter.update(stats, force=True)

    elapsed = time.monotonic() - t0
    meta = {
        "single": False, "total": total, "elapsed": elapsed, "cached_count": cached_count,
        "moved_clean": moved_clean, "moved_bad": moved_bad, "move_errors": move_errors,
        "sent_mistral": sent_mistral, "sent_mistral_errors": sent_mistral_errors,
        "stopped_early": stop_requested, "apply": args.apply,
        "move_clean_dir": args.move_clean, "move_bad_dir": args.move_bad,
        "send_mistral": args.send_mistral,
    }
    exit_code = 0 if (
        stats.status_counts.get("ANOMALY", 0) == 0
        and stats.status_counts.get("ERROR", 0) == 0
        and sent_mistral_errors == 0
    ) else 1
    return stats, exit_code, meta


def _print_summary(stats, meta: dict, args) -> None:
    if meta.get("no_args"):
        return
    if meta.get("single") or meta.get("total", 0) == 0:
        return

    extra = " (arrêté avant la fin — touche q)" if meta.get("stopped_early") else ""
    elapsed = meta.get("elapsed", 0.0)
    total = meta["total"]

    print(f"\n{'─' * 50}")
    print(f"Sessions analysées : {stats.done}/{total}  (en {elapsed/60:.1f} min, "
          f"{stats.done/max(elapsed,1e-6):.1f} sessions/s){extra}")
    print(f"Depuis le cache    : {meta.get('cached_count', 0)}")
    print(f"Propres            : {stats.status_counts.get('OK', 0)}")
    print(f"Anomalies          : {stats.status_counts.get('ANOMALY', 0)}")
    print(f"Erreurs            : {stats.status_counts.get('ERROR', 0)}")
    print(f"Score moyen        : {stats.avg_score():.1f} / 100")
    if stats.done:
        print("Distribution       : " + "  ".join(
            f"{g}={stats.grade_counts.get(g, 0)} ({stats.grade_counts.get(g, 0) / stats.done * 100:.1f}%)"
            for g in GRADES
        ))
    crit_avgs = stats.criteria_averages()
    if crit_avgs:
        print("Moyennes critères  :")
        for k, v in sorted(crit_avgs.items()):
            print(f"  {k:<24} {v:5.1f}")
    print(f"Problèmes structure (mistral) : {stats.n_mistral_issue}")
    if stats.issue_counts:
        print("Problèmes les plus fréquents  :")
        for name, cnt in stats.top_issues(15):
            print(f"  {name:<28} {cnt}")
    if meta.get("move_clean_dir"):
        print(f"Déplacées (clean)  : {meta.get('moved_clean', 0)}")
    if meta.get("move_bad_dir"):
        print(f"Déplacées (bad)    : {meta.get('moved_bad', 0)}")
    if meta.get("move_errors"):
        print(f"Échecs déplacement : {meta['move_errors']}")
    if meta.get("send_mistral"):
        print(f"Envoyées (Mistral) : {meta.get('sent_mistral', 0)}")
        if meta.get("sent_mistral_errors"):
            print(f"Échecs envoi       : {meta['sent_mistral_errors']}")
    if not meta.get("apply"):
        print("\n(dry-run — relancez avec --apply pour corriger réellement les noms)")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path, help="Répertoire contenant les sessions")
    p.add_argument("--session", type=Path, help="Traiter une seule session (jamais de cache)")
    p.add_argument("--list", type=Path, metavar="FILE",
                   help="Traiter uniquement les sessions listées dans ce fichier "
                        "(un chemin absolu par ligne), au lieu de tout le répertoire. "
                        "Le cache .postcheck.json reste actif. Alternative à 'directory'.")
    p.add_argument("--apply", action="store_true",
                   help="Appliquer réellement les corrections de noms (par défaut : dry-run)")
    p.add_argument("--skip-charuco", action="store_true",
                   help="Sauter l'étape charuco (coûteuse, nécessite opencv)")
    p.add_argument("--charuco-sample-fps", type=float, default=2.0,
                   help="Fréquence (Hz) d'échantillonnage vidéo de detect_charuco_lr (défaut : 2.0)")
    p.add_argument("--skip-lr-check", action="store_true",
                   help="Sauter la vérification left/right (corrélation marqueurs 244/255 ↔ Opening_width, nécessite opencv)")
    p.add_argument("--lr-n-samples", type=int, default=10, metavar="N",
                   help="Nombre de frames échantillonnées pour la vérification left/right (défaut : 10)")
    p.add_argument("--skip-quality", action="store_true",
                   help="Sauter le scoring qualité 0-100/A-F (checks.py) — comportement historique post/ seul")
    p.add_argument("--skip-quality-vision", action="store_true",
                   help="Dans le scoring qualité, sauter les checks vidéo coûteux "
                        "(tracking gripper ArUco/ChArUco, inversion labels) — garde le score "
                        "mais basé uniquement sur les métriques structurelles/flux")
    p.add_argument("--move-clean", type=Path, metavar="DEST",
                   help="Déplacer les sessions propres dans ce répertoire (incompatible avec --send-mistral, "
                        "qui déplace lui-même les sessions envoyées)")
    p.add_argument("--move-bad", type=Path, metavar="DEST",
                   help="Déplacer les sessions en anomalie/erreur dans ce répertoire")
    p.add_argument("--send-mistral", action="store_true",
                   help="Envoie directement à Mistral (via sessions-uploader/SessionsToMistral.py) "
                        "chaque session validée OK par CE pipeline (et sans problème de structure "
                        "façon SessionsToMistral, cf. --skip-quality docstring point 8). Plus strict "
                        "que le cron sessions-uploader habituel, qui ne fait que ce dernier contrôle "
                        "léger. Déplace la session envoyée vers --mistral-sent-dir.")
    p.add_argument("--mistral-sent-dir", type=Path, metavar="DEST",
                   help="Dossier où déplacer les sessions envoyées à Mistral (défaut : "
                        "<parent du dossier de sessions>/session_envoye)")
    p.add_argument("--mistral-offline", action="store_true",
                   help="Avec --send-mistral : n'appelle pas BACKEND_URL (register/mark-sent/mark-send-failed), "
                        "envoie uniquement le zip à Mistral")
    p.add_argument("-j", "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
                   help=f"Processus parallèles (défaut : {_DEFAULT_WORKERS}, tous les cœurs)")
    p.add_argument("--force", action="store_true",
                   help="Ignorer le cache .postcheck.json et tout ré-analyser")
    p.add_argument("--report", type=Path, metavar="JSONL",
                   help="Écrire un rapport JSONL (une ligne par session) en plus de stdout")
    p.add_argument("--ui-interval", type=float, default=0.4,
                   help="Intervalle minimum (s) entre deux rafraîchissements de l'affichage live")
    p.add_argument("--no-ui", action="store_true",
                   help="Désactive le tableau de bord ASCII (curses), force le mode texte simple "
                        "(utile pour cron/Docker/--report/sortie redirigée)")
    p.add_argument("-v", "--verbose", action="store_true", help="Afficher aussi les sessions sans anomalie (mode texte)")
    args = p.parse_args()

    if not args.session and not args.list and not args.directory:
        p.print_help()
        return 1

    if args.send_mistral and args.move_clean:
        p.error("--send-mistral et --move-clean sont incompatibles (--send-mistral déplace déjà "
                "les sessions envoyées vers --mistral-sent-dir)")

    use_ui = (not args.no_ui) and (not args.session) and sys.stdout.isatty()
    if use_ui:
        try:
            import curses
        except ImportError:
            use_ui = False

    if use_ui:
        stats_or_empty, exit_code, meta = curses.wrapper(_run, args)
    else:
        stats_or_empty, exit_code, meta = _run(None, args)

    if not meta.get("single") and not meta.get("no_args"):
        _print_summary(stats_or_empty, meta, args)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
