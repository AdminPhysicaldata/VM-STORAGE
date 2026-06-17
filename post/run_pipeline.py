#!/usr/bin/env python3
"""Pipeline complet de nettoyage/validation d'un répertoire de sessions.

Chaîne, pour chaque session, les contrôles dans cet ordre :

  1. fix_camera_names   — corrige les typos de noms (cameras/ et sensors/)
  2. verify_integrity   — vérifie l'existence EXACTE des fichiers attendus
                           (+ détecte les fichiers présents mais corrompus)
  3. diagnose_shuffle    — détecte les sessions contaminées par un autre device
                           (alignement temporel, durée jsonl + durée vidéo ffprobe)
  4. detect_charuco_lr  — détecte une inversion left/right via les marqueurs
                           ArUco 244/255 + corrélation avec Opening_width

Conçu pour tourner sur des dizaines de milliers de sessions :

  - Parallélisme par PROCESSUS (pas threads) : le décodage vidéo (charuco)
    est CPU-bound, un ProcessPoolExecutor exploite donc tous les cœurs sans
    être bridé par le GIL. Réglable via -j/--workers (défaut : tous les cœurs).
  - Cache persistant par session : un fichier .postcheck.json est écrit dans
    chaque session après son premier passage. Au prochain run, si aucun
    fichier de la session n'a changé (empreinte taille+mtime) ET que les
    paramètres du pipeline sont identiques, la session est court-circuitée
    sans rien redécoder. Sur un dossier de sessions qui grossit en continu
    (le cas réel ici), ça transforme un "tout réanalyser chaque nuit" en
    "n'analyser que les sessions réellement nouvelles". --force ignore le cache.
  - Isolation des erreurs : une session qui plante (vidéo illisible,
    exception inattendue) est rapportée en erreur et n'interrompt jamais le
    reste du lot.
  - Rapport JSONL optionnel (--report) pour audit/diff sans avoir à tout
    réimprimer sur stdout, et progression périodique au lieu d'un print par
    session.

Usage :
    # Rapport complet, aucune modification, tous les cœurs
    python3 run_pipeline.py /media/qbee/T9/sessions/

    # Gros volume : limiter les workers, écrire un rapport JSONL
    python3 run_pipeline.py /media/qbee/T9/sessions/ -j 12 --report report.jsonl

    # Une seule session, verbeux, jamais de cache
    python3 run_pipeline.py --session ../session_20260605_190710

    # Application réelle + tri, en ignorant le cache existant
    python3 run_pipeline.py /media/qbee/T9/sessions/ --apply --force \\
        --move-clean /media/qbee/T9/clean/ --move-bad /media/qbee/T9/quarantine/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fix_camera_names
import verify_integrity
import diagnose_shuffle

_PIPELINE_VERSION = 2
_MARKER_NAME = ".postcheck.json"
_DEFAULT_WORKERS = os.cpu_count() or 4
_PROGRESS_EVERY = 200


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


def _config_key(apply: bool, run_charuco: bool, classify_fps: float, curve_fps: float) -> str:
    # NB : "apply" n'entre pas dans la clé — un résultat "OK" en dry-run reste
    # valide en mode --apply (rien à appliquer). Voir _process_one pour la
    # logique d'invalidation qui dépend du statut ET de apply.
    return f"{_PIPELINE_VERSION}:{run_charuco}:{classify_fps}:{curve_fps}"


def _load_cache(session_dir: Path, fingerprint: str, config_key: str) -> dict | None:
    try:
        data = json.loads((session_dir / _MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fingerprint or data.get("config_key") != config_key:
        return None
    return data


def _save_cache(session_dir: Path, fingerprint: str, config_key: str, status: str, lines: list[str]) -> None:
    marker = session_dir / _MARKER_NAME
    try:
        marker.write_text(json.dumps({
            "fingerprint": fingerprint,
            "config_key": config_key,
            "status": status,
            "lines": lines,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")
    except OSError:
        pass  # un cache qu'on n'arrive pas à écrire n'est pas bloquant


# ─── Traitement d'une session (fonction top-level → picklable pour le pool) ──

def _process_one(
    session_dir_str: str,
    apply: bool,
    run_charuco: bool,
    classify_fps: float,
    curve_fps: float,
    force: bool,
) -> dict:
    """Tout ce qui touche une session, isolé pour tourner dans un worker
    séparé. Ne lève jamais — toute exception est convertie en statut ERROR
    pour ne jamais faire tomber le pool sur une session pourrie."""
    session_dir = Path(session_dir_str)
    name = session_dir.name
    config_key = _config_key(apply, run_charuco, classify_fps, curve_fps)

    try:
        fingerprint = _fingerprint(session_dir)
        cached = None if force else _load_cache(session_dir, fingerprint, config_key)
        # Un cache "OK" est valide quel que soit apply (rien à faire de toute façon).
        # Un cache "ANOMALY"/"ERROR" n'est réutilisable qu'en lecture (apply=False) :
        # si apply=True il faut réellement tenter la correction, pas juste relire
        # un ancien diagnostic jamais appliqué.
        if cached is not None and (cached["status"] == "OK" or not apply):
            return {"name": name, "status": cached["status"], "lines": cached["lines"], "cached": True}

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

        integrity = verify_integrity.check_session(session_dir)
        for subdir, names in sorted(integrity.extra.items()):
            lines.append(f"[en trop]   {subdir}/  {sorted(names)}")
        for subdir, names in sorted(integrity.missing.items()):
            lines.append(f"[manquant]  {subdir}/  {sorted(names)}")
        for subdir, names in sorted(integrity.corrupt.items()):
            lines.append(f"[corrompu]  {subdir}/  {sorted(names)}")
        is_clean = integrity.is_clean

        if (session_dir / "config.json").is_file():
            shuffle_report = diagnose_shuffle.analyze_session(session_dir)
            for finding in shuffle_report.findings:
                lines.append(f"[shuffle/{finding.confidence}] {finding.camera.name} : {', '.join(finding.reasons)}")
            is_clean = is_clean and not shuffle_report.is_contaminated

        if run_charuco and (session_dir / "cameras").is_dir():
            import detect_charuco_lr  # importé seulement si nécessaire (évite la dépendance opencv sinon)
            findings = detect_charuco_lr.analyze_session(session_dir, classify_fps=classify_fps, curve_fps=curve_fps)
            charuco_anomaly = any(
                f.inferred_name and f.inferred_name != f.current_name for f in findings
            ) or sum(1 for f in findings if f.role == "gripper") not in (0, 2) or any(
                f.role == "unreadable" for f in findings
            )
            if charuco_anomaly and apply:
                detect_charuco_lr.apply_fix(session_dir, findings)
                # Le renommage invalide l'empreinte ; on la recalcule pour le cache final.
                fingerprint = _fingerprint(session_dir)
            for f in findings:
                if f.inferred_name and f.inferred_name != f.current_name:
                    lines.append(
                        f"[charuco]   {f.current_name}.mp4 mal nommée → identité probable : {f.inferred_name}"
                    )
                if f.role == "unreadable":
                    lines.append(f"[charuco]   {f.current_name}.mp4 illisible/corrompue")
            n_gripper = sum(1 for f in findings if f.role == "gripper")
            if n_gripper not in (0, 2):
                lines.append(f"[charuco]   {n_gripper} vidéo(s) 'gripper' détectée(s) (attendu 2)")
            is_clean = is_clean and not charuco_anomaly

        status = "OK" if is_clean else "ANOMALY"
        _save_cache(session_dir, fingerprint, config_key, status, lines)
        return {"name": name, "status": status, "lines": lines, "cached": False}

    except Exception as exc:  # noqa: BLE001 — défense en profondeur, jamais de crash du pool
        return {"name": name, "status": "ERROR", "lines": [f"[erreur interne] {exc!r}"], "cached": False}


# ─── Affichage / rapport ─────────────────────────────────────────────────────

def _print_result(result: dict, verbose: bool) -> None:
    if result["lines"] or verbose:
        cache_tag = " [cache]" if result.get("cached") else ""
        print(f"\n{result['name']}  [{result['status']}]{cache_tag}")
        for line in result["lines"]:
            print(f"  {line}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path, help="Répertoire contenant les sessions")
    p.add_argument("--session", type=Path, help="Traiter une seule session (jamais de cache)")
    p.add_argument("--apply", action="store_true",
                   help="Appliquer réellement les corrections de noms (par défaut : dry-run)")
    p.add_argument("--skip-charuco", action="store_true",
                   help="Sauter l'étape charuco (coûteuse, nécessite opencv)")
    p.add_argument("--classify-fps", type=float, default=1.0,
                   help="Fréquence (Hz) de la passe légère head/gripper de detect_charuco_lr (défaut : 1.0)")
    p.add_argument("--curve-fps", type=float, default=5.0,
                   help="Fréquence (Hz) de la passe dense de detect_charuco_lr (défaut : 5.0)")
    p.add_argument("--move-clean", type=Path, metavar="DEST",
                   help="Déplacer les sessions propres dans ce répertoire")
    p.add_argument("--move-bad", type=Path, metavar="DEST",
                   help="Déplacer les sessions en anomalie/erreur dans ce répertoire")
    p.add_argument("-j", "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
                   help=f"Processus parallèles (défaut : {_DEFAULT_WORKERS}, tous les cœurs)")
    p.add_argument("--force", action="store_true",
                   help="Ignorer le cache .postcheck.json et tout ré-analyser")
    p.add_argument("--report", type=Path, metavar="JSONL",
                   help="Écrire un rapport JSONL (une ligne par session) en plus de stdout")
    p.add_argument("-v", "--verbose", action="store_true", help="Afficher aussi les sessions sans anomalie")
    args = p.parse_args()

    run_charuco = not args.skip_charuco

    if args.session:
        session_dir = args.session.resolve()
        result = _process_one(
            str(session_dir), args.apply, run_charuco, args.classify_fps, args.curve_fps, force=True
        )
        _print_result(result, verbose=True)
        return 0

    if not args.directory:
        p.print_help()
        return 1

    root = args.directory.resolve()
    sessions = sorted(
        Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
    )
    if not sessions:
        print(f"Aucune session trouvée dans {root}")
        return 0

    if args.move_clean:
        args.move_clean.mkdir(parents=True, exist_ok=True)
    if args.move_bad:
        args.move_bad.mkdir(parents=True, exist_ok=True)

    total = len(sessions)
    print(f"{total} sessions, {args.workers} workers, charuco={'ON' if run_charuco else 'OFF'}…\n")

    report_fh = args.report.open("w", encoding="utf-8") if args.report else None
    to_move_clean: list[Path] = []
    to_move_bad: list[Path] = []
    counts = {"OK": 0, "ANOMALY": 0, "ERROR": 0}
    cached_count = 0
    done = 0
    t0 = time.time()

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one, str(s), args.apply, run_charuco,
                    args.classify_fps, args.curve_fps, args.force,
                ): s
                for s in sessions
            }
            for fut in as_completed(futures):
                session_dir = futures[fut]
                result = fut.result()  # _process_one ne lève jamais : pas de try/except requis ici
                done += 1
                counts[result["status"]] += 1
                if result.get("cached"):
                    cached_count += 1
                _print_result(result, args.verbose)
                if report_fh:
                    report_fh.write(json.dumps(result) + "\n")
                    report_fh.flush()

                if result["status"] == "OK":
                    if args.move_clean:
                        to_move_clean.append(session_dir)
                elif args.move_bad:
                    to_move_bad.append(session_dir)

                if done % _PROGRESS_EVERY == 0 or done == total:
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (total - done) / rate if rate > 0 else 0
                    print(
                        f"  … {done}/{total}  ok={counts['OK']} anomalies={counts['ANOMALY']} "
                        f"erreurs={counts['ERROR']} cache={cached_count}  "
                        f"({rate:.1f} sessions/s, ETA {eta/60:.1f} min)",
                        end="\r",
                    )
    finally:
        if report_fh:
            report_fh.close()

    print()

    if to_move_clean or to_move_bad:
        print("\nDéplacement des sessions triées…")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = (
                [pool.submit(shutil.move, str(s), str(args.move_clean / s.name)) for s in to_move_clean]
                + [pool.submit(shutil.move, str(s), str(args.move_bad / s.name)) for s in to_move_bad]
            )
            for fut in as_completed(futs):
                fut.result()

    elapsed = time.time() - t0
    print(f"\n{'─' * 50}")
    print(f"Sessions analysées : {total}  (en {elapsed/60:.1f} min, {total/max(elapsed,1e-6):.1f} sessions/s)")
    print(f"Depuis le cache    : {cached_count}")
    print(f"Propres            : {counts['OK']}")
    print(f"Anomalies          : {counts['ANOMALY']}")
    print(f"Erreurs            : {counts['ERROR']}")
    if args.move_clean:
        print(f"Déplacées (clean)  : {len(to_move_clean)}")
    if args.move_bad:
        print(f"Déplacées (bad)    : {len(to_move_bad)}")
    if not args.apply:
        print("\n(dry-run — relancez avec --apply pour corriger réellement les noms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
