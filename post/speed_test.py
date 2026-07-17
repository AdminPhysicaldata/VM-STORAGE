#!/usr/bin/env python3
"""Speed-test du post-pipeline : mesure la vitesse de CHAQUE étape de
run_pipeline.py pour identifier le goulot d'étranglement.

Accepte EXACTEMENT les mêmes options que run_pipeline.py (le parser est
partagé — cf. run_pipeline.build_parser) : copiez-collez votre commande
run_pipeline habituelle en remplaçant juste le nom du script. Les étapes
activées/désactivées et leurs réglages (--skip-charuco, --charuco-sample-fps,
--charuco-one-per-rig-hour, --skip-lr-check, --lr-n-samples, --skip-quality,
--skip-quality-vision, --send-mistral, -j, ...) sont alors chronométrés à
l'identique du vrai run.

Garanties :
  - LECTURE SEULE sur les sessions : --apply est ignoré (dry-run forcé),
    aucun déplacement (--move-*), aucun envoi Mistral réel (--send-mistral ne
    fait que chronométrer la création du zip, dans un dossier temporaire).
  - N'écrit jamais .postcheck.json — le cache du vrai pipeline reste intact.
  - Seuls fichiers créés : les fichiers temporaires de bench (écriture
    disque, zip), supprimés immédiatement après mesure.

Ce qui est mesuré :
  1. Débit disque — lecture séquentielle des sessions (sur des sessions
     HORS échantillon d'analyse pour limiter l'effet du page cache) et
     écriture dans la racine sessions + les destinations --move-clean /
     --move-bad / --mistral-sent-dir si fournies.
  2. Temps de chaque étape du pipeline, session par session, sur un
     échantillon (--sample, défaut 5, réparti sur tout le répertoire) :
     empreinte cache (= coût d'un cache-hit), fix_camera_names,
     verify_camera_sync, verify_integrity, diagnose_shuffle, charuco,
     gripper L/R (passe rapide + escalade full éventuelle), scoring qualité,
     structure mistral, zip.
  3. Projection sur le répertoire complet avec -j N : débit CPU vs débit
     lecture disque vs débit zip → VERDICT sur le goulot d'étranglement,
     avec l'option à ajuster pour le lever.

Usage :
    # même commande que le run nocturne, script différent
    python3 speed_test.py /data/sessions -j 4 --charuco-one-per-rig-hour --send-mistral --no-ui

    # échantillon plus large, rapport JSONL des timings par session
    python3 speed_test.py /data/sessions --sample 12 --report speed_report.jsonl

    # une seule session, détail complet
    python3 speed_test.py --session ../session_20260605_190710 -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as rp  # noqa: E402 — règle aussi sys.path (sessions-uploader) et l'env
import fix_camera_names  # noqa: E402
import verify_camera_sync  # noqa: E402
import verify_integrity  # noqa: E402
import diagnose_shuffle  # noqa: E402
import checks  # noqa: E402
import SessionsToMistral as mistral_uploader  # noqa: E402

_MB = 1024 * 1024
_READ_CHUNK = 8 * _MB
_READ_BENCH_CAP = 2048 * _MB      # lecture séquentielle : au plus 2 Go lus
_WRITE_BENCH_MB = 128             # écriture : 128 Mo par destination testée
_CHARUCO_FRACTION_CAP = 2000      # sessions max inspectées pour la fraction charuco

# (clé, étape correspondante de run_pipeline, libellé affiché)
_STEPS = [
    ("fingerprint", "Empreinte cache (coût d'un cache-hit)"),
    ("fix_names",   "1. fix_camera_names (dry-run)"),
    ("camera_sync", "2. verify_camera_sync"),
    ("integrity",   "3. verify_integrity"),
    ("shuffle",     "4. diagnose_shuffle (ffprobe)"),
    ("charuco",     "5. detect_charuco_lr"),
    ("lr_quick",    "6. gripper L/R (passe rapide)"),
    ("lr_full",     "6b. gripper L/R (escalade full)"),
    ("quality",     "7. scoring qualité (checks.py)"),
    ("structure",   "8. structure mistral"),
    ("zip",         "zip session (--send-mistral)"),
]
_LABELS = dict(_STEPS)

# Étape dominante → option de run_pipeline.py qui permet de la soulager.
_SUGGESTIONS = {
    "fingerprint": "coût incompressible d'un cache-hit — dominé par la latence stat() du système de fichiers",
    "fix_names":   "étape légère (stat + renommage dry-run) — rien à régler",
    "camera_sync": "étape légère (lecture jsonl) — surtout limitée par le disque",
    "integrity":   "lecture/parcours de fichiers — surtout limitée par le disque",
    "shuffle":     "ffprobe + jsonl — surtout limitée par le disque, peu réglable",
    "charuco":     "--charuco-one-per-rig-hour (1 session par poste/heure), "
                   "--charuco-sample-fps plus bas (défaut 2.0), ou --skip-charuco",
    "lr_quick":    "--lr-n-samples plus bas (défaut 10), ou --skip-lr-check",
    "lr_full":     "escalades full-scan (vidéos difficiles) — --skip-lr-check, ou accepter ce coût "
                   "(il ne touche que les sessions inconclusives en passe rapide)",
    "quality":     "--skip-quality-vision (garde le score, sans les checks vidéo coûteux) ou --skip-quality",
    "structure":   "étape légère (stat + json) — rien à régler",
    "zip":         "compression zip — tourne dans le pool de déplacement (max(2, workers//2) threads) ; "
                   "augmenter -j augmente aussi ce pool",
}


def _fmt_s(t: float | None) -> str:
    return "     —" if t is None else f"{t:8.2f} s"


def _fmt_rate(mbps: float | None) -> str:
    return "non mesuré" if mbps is None else f"{mbps:,.1f} Mo/s"


def _session_bytes(session_dir: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(session_dir):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


# ─── Bench disque ─────────────────────────────────────────────────────────────

def read_bench(sessions: list[Path]) -> tuple[float | None, int]:
    """Lecture séquentielle de tous les fichiers des sessions données.
    Retourne (Mo/s, octets lus). NB : si ces fichiers sont déjà dans le page
    cache de l'OS, la valeur est optimiste — d'où le choix de sessions hors
    échantillon d'analyse quand c'est possible."""
    total = 0
    t0 = time.monotonic()
    for session_dir in sessions:
        for root, _dirs, files in os.walk(session_dir):
            for name in files:
                path = os.path.join(root, name)
                try:
                    with open(path, "rb", buffering=0) as fh:
                        while chunk := fh.read(_READ_CHUNK):
                            total += len(chunk)
                            if total >= _READ_BENCH_CAP:
                                break
                except OSError:
                    continue
                if total >= _READ_BENCH_CAP:
                    break
            if total >= _READ_BENCH_CAP:
                break
        if total >= _READ_BENCH_CAP:
            break
    elapsed = time.monotonic() - t0
    if total == 0 or elapsed <= 0:
        return None, 0
    return total / _MB / elapsed, total


def write_bench(dest: Path, size_mb: int = _WRITE_BENCH_MB) -> float | None:
    """Écrit puis supprime un fichier temporaire de size_mb Mo dans dest
    (fsync inclus dans la mesure). Retourne Mo/s, ou None si dest inutilisable."""
    if not dest.is_dir():
        return None
    chunk = os.urandom(_READ_CHUNK)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".speedtest_write_", suffix=".tmp", dir=str(dest))
        t0 = time.monotonic()
        with os.fdopen(fd, "wb") as fh:
            written = 0
            while written < size_mb * _MB:
                fh.write(chunk)
                written += len(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        elapsed = time.monotonic() - t0
        return (written / _MB / elapsed) if elapsed > 0 else None
    except OSError:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─── Chronométrage d'une session, étape par étape ────────────────────────────

def measure_session(session_dir: Path, args, run_charuco: bool, run_lr: bool,
                    run_quality: bool) -> dict:
    """Exécute sur UNE session les mêmes étapes que run_pipeline._process_one,
    dans le même ordre et avec les mêmes réglages, mais en lecture seule
    (dry-run forcé, pas de _save_cache) et en chronométrant chacune."""
    steps: dict[str, float] = {}
    notes: list[str] = []

    def timed(key: str, fn):
        t0 = time.monotonic()
        try:
            return fn()
        finally:
            steps[key] = time.monotonic() - t0
            print(f"      {_LABELS[key]:<42} {_fmt_s(steps[key])}", flush=True)

    timed("fingerprint", lambda: rp._fingerprint(session_dir))

    def _fix_names():
        # dry_run=True TOUJOURS (même si --apply) : le speed-test ne modifie rien,
        # et le coût mesuré (détection) est identique — le renommage lui-même est négligeable.
        fix_camera_names.SessionFixer(
            session_dir, frozenset(fix_camera_names._DEFAULT_EXPECTED), dry_run=True, subdir="cameras"
        ).fix()
        fix_camera_names.SessionFixer(
            session_dir, frozenset({"left", "right"}), dry_run=True, subdir="sensors"
        ).fix()
    timed("fix_names", _fix_names)

    timed("camera_sync", lambda: verify_camera_sync.check_session(session_dir))
    timed("integrity", lambda: verify_integrity.check_session(session_dir))

    if (session_dir / "config.json").is_file():
        timed("shuffle", lambda: diagnose_shuffle.analyze_session(session_dir))

    if run_charuco and (session_dir / "cameras").is_dir():
        import detect_charuco_lr
        timed("charuco", lambda: detect_charuco_lr.analyze_session(
            session_dir, sample_fps=args.charuco_sample_fps))

    if run_lr and (session_dir / "cameras").is_dir() and (session_dir / "sensors").is_dir():
        import detect_gripper_lr_marker_distance as lr_check
        lr_result = timed("lr_quick", lambda: lr_check.quick_analyze_session(
            session_dir, n_samples=args.lr_n_samples))
        verdict = lr_result.verdict if lr_result is not None else None
        if lr_result is not None and verdict.startswith("inconclusive"):
            notes.append("passe rapide L/R inconclusive → escalade full-scan (comme le vrai pipeline)")
            timed("lr_full", lambda: lr_check.analyze_session(session_dir))

    if run_quality:
        def _quality():
            try:
                rp.compute_quality(session_dir)
            except Exception as exc:  # même isolation que le pipeline : on mesure, on ne plante pas
                notes.append(f"scoring qualité en erreur : {exc!r}")
        timed("quality", _quality)

    timed("structure", lambda: rp.validate_mistral_structure(session_dir))

    zip_bytes = 0
    if args.send_mistral:
        def _zip():
            nonlocal zip_bytes
            with tempfile.TemporaryDirectory(prefix="speedtest_zip_") as tmp_dir:
                zip_path = mistral_uploader.zip_session(session_dir, Path(tmp_dir))
                zip_bytes = zip_path.stat().st_size
        timed("zip", _zip)

    return {
        "name": session_dir.name,
        "bytes": _session_bytes(session_dir),
        "zip_bytes": zip_bytes,
        "steps": steps,
        "notes": notes,
    }


# ─── Échantillonnage ─────────────────────────────────────────────────────────

def _spread(items: list, n: int) -> list:
    """n éléments répartis uniformément sur la liste (représentatif d'un
    répertoire qui mélange heures/postes différents)."""
    if n <= 0 or n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def charuco_fraction(sessions: list[Path]) -> tuple[float, str]:
    """Fraction des sessions qui exécuteraient réellement le charuco avec
    --charuco-one-per-rig-hour (représentants de groupe + sessions au
    poste/heure indéterminable). Estimée sur au plus _CHARUCO_FRACTION_CAP
    sessions (lecture d'un config.json par session — coûteux sur NAS)."""
    probe = _spread(sessions, _CHARUCO_FRACTION_CAP)
    groups: set[tuple] = set()
    ungrouped = 0
    for s in probe:
        key = rp._charuco_group_key(s)
        if key is None:
            ungrouped += 1
        else:
            groups.add(key)
    frac = (len(groups) + ungrouped) / len(probe) if probe else 1.0
    note = f"estimée sur {len(probe)} sessions" if len(probe) < len(sessions) else "exacte"
    return frac, note


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> int:
    p = rp.build_parser()
    g = p.add_argument_group(
        "speed-test (options propres à speed_test.py — tout le reste est identique à run_pipeline.py)"
    )
    g.add_argument("--sample", type=int, default=5, metavar="N",
                   help="Nombre de sessions chronométrées, réparties sur le répertoire (défaut : 5 ; 0 = toutes)")
    g.add_argument("--skip-io-bench", action="store_true",
                   help="Sauter les benchs de débit disque (lecture/écriture)")
    args = p.parse_args()

    if not args.session and not args.list and not args.directory:
        p.print_help()
        return 1

    run_charuco = not args.skip_charuco
    run_lr = not args.skip_lr_check
    run_quality = not args.skip_quality
    checks.ENABLE_VISION = not args.skip_quality_vision
    if checks.HAS_CV2:
        checks.cv2.setNumThreads(1)  # même réglage qu'un worker du pipeline (cf. _init_worker)

    print("─" * 74)
    print("SPEED-TEST post-pipeline — LECTURE SEULE")
    print("  --apply ignoré, aucun move, aucun envoi Mistral réel, cache intact")
    if args.apply or args.move_clean or args.move_bad:
        print("  (options --apply/--move-* reçues : seules leurs destinations sont benchées en écriture)")
    print(f"  Étapes actives : charuco={'oui' if run_charuco else 'NON'}"
          f" (fps={args.charuco_sample_fps}, one-per-rig-hour={'oui' if args.charuco_one_per_rig_hour else 'non'})"
          f", lr={'oui' if run_lr else 'NON'} (n={args.lr_n_samples})"
          f", qualité={'oui' if run_quality else 'NON'}"
          f" (vision={'oui' if checks.ENABLE_VISION else 'NON'})"
          f", zip={'oui' if args.send_mistral else 'non'}")
    print(f"  Workers pour la projection : -j {args.workers}")
    print("─" * 74)

    # ── Sélection des sessions : même logique que run_pipeline._run ──────────
    root = None
    if args.session:
        sessions = [args.session.resolve()]
    elif args.list:
        try:
            lines = args.list.resolve().read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"Impossible de lire la liste '{args.list}': {exc}")
            return 1
        sessions = sorted({Path(l.strip()).resolve() for l in lines if l.strip()})
        sessions = [s for s in sessions if s.is_dir()]
    else:
        root = args.directory.resolve()
        t0 = time.monotonic()
        sessions = sorted(
            Path(e.path) for e in os.scandir(root)
            if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
        )
        print(f"\nListing du répertoire : {len(sessions)} sessions en {time.monotonic() - t0:.2f} s")

    if not sessions:
        print("Aucune session trouvée — rien à mesurer.")
        return 1

    sample = _spread(sessions, args.sample if args.sample > 0 else len(sessions))

    # ── Bench disque ─────────────────────────────────────────────────────────
    read_mbps: float | None = None
    write_results: list[tuple[str, float | None]] = []
    if not args.skip_io_bench:
        print("\n[1] Débit disque")
        # Lecture : sur des sessions HORS échantillon d'analyse si possible,
        # pour que le page cache ne fausse pas la mesure.
        sample_set = {str(s) for s in sample}
        others = [s for s in sessions if str(s) not in sample_set]
        read_sessions = _spread(others, len(sample)) if others else sample
        cache_note = "" if others else "  (mêmes sessions que l'analyse : page cache possible, valeur optimiste)"
        read_mbps, read_bytes = read_bench(read_sessions)
        print(f"    Lecture séquentielle sessions : {_fmt_rate(read_mbps)}"
              f"  ({read_bytes / _MB:,.0f} Mo lus){cache_note}")

        write_targets: list[tuple[str, Path | None]] = [("racine sessions", root or sessions[0].parent)]
        if args.move_clean:
            write_targets.append(("--move-clean", args.move_clean))
        if args.move_bad:
            write_targets.append(("--move-bad", args.move_bad))
        if args.mistral_sent_dir:
            write_targets.append(("--mistral-sent-dir", args.mistral_sent_dir))
        for label, dest in write_targets:
            mbps = write_bench(dest.resolve()) if dest else None
            write_results.append((label, mbps))
            print(f"    Écriture ({label:<18}) : {_fmt_rate(mbps)}")

    # ── Chronométrage étape par étape ────────────────────────────────────────
    print(f"\n[2] Temps par étape — {len(sample)} session(s) échantillonnée(s) sur {len(sessions)}"
          f" (1 seul processus, OpenCV 1 thread : comparable à UN worker du pipeline)")
    results: list[dict] = []
    report_fh = args.report.open("w", encoding="utf-8") if args.report else None
    try:
        for i, s in enumerate(sample, 1):
            size_mb = _session_bytes(s) / _MB
            print(f"\n  → {s.name}  ({i}/{len(sample)}, {size_mb:,.0f} Mo)")
            res = measure_session(s, args, run_charuco, run_lr, run_quality)
            for note in res["notes"]:
                print(f"      [note] {note}")
            results.append(res)
            if report_fh:
                report_fh.write(json.dumps(res) + "\n")
                report_fh.flush()
    finally:
        if report_fh:
            report_fh.close()
            print(f"\nTimings par session écrits dans {args.report}")

    # ── Agrégation ───────────────────────────────────────────────────────────
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, _ in _STEPS:
        vals = [r["steps"][key] for r in results if key in r["steps"]]
        if vals:
            means[key] = sum(vals) / len(vals)
            counts[key] = len(vals)

    avg_bytes = sum(r["bytes"] for r in results) / len(results)

    # --charuco-one-per-rig-hour : le charuco ne tourne que sur une fraction
    # des sessions — on pondère son coût moyen dans la projection.
    charuco_frac, charuco_frac_note = 1.0, ""
    if run_charuco and args.charuco_one_per_rig_hour and "charuco" in means and len(sessions) > 1:
        charuco_frac, charuco_frac_note = charuco_fraction(sessions)

    # Le zip tourne dans le move_pool (threads), EN PARALLÈLE de l'analyse
    # (cf. run_pipeline._run) : il a son propre plafond de débit et n'entre
    # pas dans le temps "analyse CPU" par session.
    effective = {k: v for k, v in means.items() if k != "zip"}
    if charuco_frac < 1.0 and "charuco" in effective:
        effective["charuco"] = means["charuco"] * charuco_frac

    total_effective = sum(effective.values())

    print(f"\n[3] Moyenne par session ({len(results)} mesurée(s), {avg_bytes / _MB:,.0f} Mo/session en moyenne)")
    print(f"    {'étape':<44} {'moy/session':>11}   {'part':>6}   sur run complet (-j {args.workers})")
    n_total = len(sessions)
    zip_workers = max(2, args.workers // 2)  # taille du move_pool dans run_pipeline
    for key, label in _STEPS:
        if key not in means:
            continue
        if key == "zip":
            # Pool parallèle à l'analyse : pas de "part" du temps CPU, et le run
            # complet se projette sur les threads du move_pool.
            full_min = means["zip"] * n_total / zip_workers / 60
            print(f"    {label:<44} {_fmt_s(means['zip'])}       —   {full_min:8.1f} min"
                  f"  (pool déplacement, {zip_workers} threads)")
            continue
        eff = effective[key]
        share = eff / total_effective * 100 if total_effective else 0.0
        full_min = eff * n_total / args.workers / 60
        frac_tag = f" ×{charuco_frac:.2f} ({charuco_frac_note})" if key == "charuco" and charuco_frac < 1.0 else ""
        print(f"    {label:<44} {_fmt_s(means[key])}   {share:5.1f}%   {full_min:8.1f} min{frac_tag}")
    print(f"    {'TOTAL analyse (pondéré, hors zip)':<44} {_fmt_s(total_effective)}")

    if args.verbose:
        print("\n    Détail par session :")
        for r in results:
            print(f"      {r['name']}  ({r['bytes'] / _MB:,.0f} Mo)")
            for key, label in _STEPS:
                if key in r["steps"]:
                    print(f"        {label:<42} {_fmt_s(r['steps'][key])}")

    # ── Projection & verdict ─────────────────────────────────────────────────
    print(f"\n[4] Projection sur {n_total} session(s) avec -j {args.workers}")
    caps: list[tuple[str, float, str]] = []  # (nom, sessions/s, clé de suggestion)

    if total_effective > 0:
        thr_cpu = args.workers / total_effective
        top_key = max(effective, key=effective.get)
        caps.append((f"analyse (CPU, étape dominante : {_LABELS[top_key]})", thr_cpu, top_key))
        print(f"    Débit analyse (CPU)   : {thr_cpu:6.2f} sessions/s → {n_total / thr_cpu / 60:8.1f} min")

    if read_mbps:
        thr_disk = read_mbps / (avg_bytes / _MB)
        caps.append(("lecture disque", thr_disk, None))
        print(f"    Débit lecture disque  : {thr_disk:6.2f} sessions/s → {n_total / thr_disk / 60:8.1f} min"
              f"  ({_fmt_rate(read_mbps)} ÷ {avg_bytes / _MB:,.0f} Mo/session)")

    if args.send_mistral and "zip" in means and means["zip"] > 0:
        thr_zip = zip_workers / means["zip"]
        caps.append(("compression zip (--send-mistral)", thr_zip, "zip"))
        avg_zip = sum(r["zip_bytes"] for r in results if r["zip_bytes"]) / max(1, counts.get("zip", 1))
        print(f"    Débit zip ({zip_workers} threads) : {thr_zip:6.2f} sessions/s → {n_total / thr_zip / 60:8.1f} min"
              f"  (zip moyen : {avg_zip / _MB:,.0f} Mo)")
        print(f"    Upload Mistral        : NON MESURÉ (aucun envoi réel) — à {avg_zip / _MB:,.0f} Mo/session,"
              f" comptez (Mo/s de votre lien) ÷ {avg_zip / _MB:,.0f} sessions/s")

    if "fingerprint" in means:
        thr_cache = args.workers / means["fingerprint"]
        print(f"    Nuit 100% cache-hit   : {thr_cache:6.2f} sessions/s → {n_total / thr_cache / 60:8.1f} min"
              f"  (empreinte seule, rien à réanalyser)")

    print("\n" + "═" * 74)
    if not caps:
        print("VERDICT : rien n'a pu être mesuré (aucune étape exécutée ?)")
        return 1

    caps.sort(key=lambda c: c[1])
    name, thr, sugg_key = caps[0]
    print(f"VERDICT — GOULOT D'ÉTRANGLEMENT : {name}")
    print(f"  Débit plafond : {thr:.2f} sessions/s ≈ {n_total / thr / 60:.1f} min pour {n_total} session(s)")
    if len(caps) > 1:
        n2, t2, _ = caps[1]
        margin = (t2 - thr) / thr * 100 if thr else 0
        if margin < 20:
            print(f"  (serré : « {n2} » est à seulement +{margin:.0f}% — quasi co-limitant)")
        else:
            print(f"  Suivant : {n2} ({t2:.2f} sessions/s, +{margin:.0f}%)")
    if sugg_key and sugg_key in _SUGGESTIONS:
        print(f"  Pour le lever : {_SUGGESTIONS[sugg_key]}")
    elif sugg_key is None:
        print("  Pour le lever : le disque limite — stockage plus rapide (SSD/local vs NAS), "
              "ou réduire les octets lus par session (--skip-quality-vision, --charuco-one-per-rig-hour)")
    if args.send_mistral:
        print("  Rappel : l'upload réseau vers Mistral n'est PAS inclus (jamais d'envoi réel en speed-test).")
    print("═" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
