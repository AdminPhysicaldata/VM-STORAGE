#!/usr/bin/env python3
"""
select_complete.py — Sélectionne, parmi les sessions présentes sur disque,
celles qui sont structurellement complètes (le bon nombre de fichiers, taille
non nulle) ET stables (plus aucune écriture récente) : prêtes pour le pipeline
de post-traitement puis l'envoi à Mistral.

Ce filtre est volontairement plus LÉGER que SessionsToMistral.validate_session
(pas de vérification sync_check, ni de cameras/resampled_30hz.jsonl) : ce
dernier fichier est produit par le post-pipeline lui-même (run_pipeline.py),
qui n'a pas encore tourné au moment où ce script s'exécute. On ne vérifie ici
que ce que le rig a dû produire à la fin d'un enregistrement réussi :

  1. result.json présent, result == SUCCESS
  2. config.json lisible
  3. pour chaque caméra déclarée : cameras/<name>.mp4 et .jsonl présents, non vides
  4. pour chaque capteur déclaré : sensors/<name>.jsonl présent, non vide
  5. aucun fichier de la session modifié depuis moins de STABILITY_SECONDS
     (garde-fou contre une session encore en cours de copie SFTP)

Une session qui échoue un de ces points n'est pas une erreur : elle est
simplement reportée au prochain cycle (toujours en cours d'enregistrement/
copie, ou véritablement cassée — dans ce dernier cas SessionsToMistral.py la
détectera et l'écartera définitivement une fois qu'elle aura fini de bouger).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DEFAULT_STABILITY_SECONDS = int(os.environ.get("STABILITY_SECONDS", "300"))


def _latest_mtime(session_dir: Path) -> float:
    latest = session_dir.stat().st_mtime
    for f in session_dir.rglob("*"):
        try:
            latest = max(latest, f.stat().st_mtime)
        except OSError:
            continue
    return latest


def is_session_complete(session_dir: Path, stability_seconds: int = DEFAULT_STABILITY_SECONDS) -> tuple[bool, str]:
    """Retourne (complete, reason). reason est vide si complete=True, sinon
    explique pourquoi la session est reportée."""

    result_path = session_dir / "result.json"
    if not result_path.exists():
        return False, "result.json manquant — enregistrement non terminé"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"result.json illisible : {exc}"
    if str(result.get("result", "")).upper() != "SUCCESS":
        return False, f"result.json non-SUCCESS (valeur : '{result.get('result')}')"

    config_path = session_dir / "config.json"
    if not config_path.exists():
        return False, "config.json manquant"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"config.json illisible : {exc}"

    cam_dir = session_dir / "cameras"
    for cam in config.get("cameras", []):
        name = cam.get("name")
        if not name:
            continue
        mp4 = cam_dir / f"{name}.mp4"
        jsonl = cam_dir / f"{name}.jsonl"
        if not mp4.exists() or mp4.stat().st_size == 0:
            return False, f"cameras/{name}.mp4 manquant ou vide"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            return False, f"cameras/{name}.jsonl manquant ou vide"

    sen_dir = session_dir / "sensors"
    for sen in config.get("sensors", []):
        name = sen.get("name")
        if not name:
            continue
        jsonl = sen_dir / f"{name}.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            return False, f"sensors/{name}.jsonl manquant ou vide"

    age = time.time() - _latest_mtime(session_dir)
    if age < stability_seconds:
        return False, f"modifiée il y a {age:.0f}s (< {stability_seconds}s) — probablement encore en copie"

    return True, ""


def list_complete_sessions(sessions_dir: Path, stability_seconds: int = DEFAULT_STABILITY_SECONDS,
                            verbose: bool = False) -> list[Path]:
    candidates = sorted(
        Path(e.path) for e in os.scandir(sessions_dir)
        if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
    )
    complete: list[Path] = []
    for s in candidates:
        ok, reason = is_session_complete(s, stability_seconds)
        if ok:
            complete.append(s)
        elif verbose:
            print(f"  [report] {s.name} : {reason}", file=sys.stderr, flush=True)
    return complete


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions_dir", type=Path, help="Répertoire contenant les sessions")
    p.add_argument("--stability", type=int, default=DEFAULT_STABILITY_SECONDS, metavar="N",
                   help=f"Délai de stabilité en secondes (défaut : {DEFAULT_STABILITY_SECONDS})")
    p.add_argument("--out", type=Path, metavar="FILE",
                   help="Écrit la liste (un chemin absolu par ligne) dans ce fichier au lieu de stdout")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Affiche aussi, sur stderr, la raison des sessions reportées")
    args = p.parse_args()

    root = args.sessions_dir.resolve()
    if not root.is_dir():
        print(f"Erreur : '{root}' n'est pas un dossier valide", file=sys.stderr)
        return 1

    complete = list_complete_sessions(root, args.stability, args.verbose)

    lines = [str(s) for s in complete]
    if args.out:
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        for line in lines:
            print(line)

    print(f"{len(complete)} session(s) complète(s) sur {sum(1 for _ in root.iterdir() if _.name.startswith('session_'))}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
