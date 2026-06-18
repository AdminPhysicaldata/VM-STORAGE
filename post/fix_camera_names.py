#!/usr/bin/env python3
"""Corrige les noms de caméras erronés dans une session ou un répertoire de sessions.

Applique les corrections partout :
  - Noms de fichiers dans cameras/  (mp4, mkv, avi, mov, jsonl)
  - Contenu de config.json          (champ "name" des caméras)
  - Contenu de analysis.json        (valeurs de chaîne correspondantes)
  - Contenu des .jsonl caméra       (si le nom apparaît dans les données)

Stratégie de résolution du typo → nom correct :
  1. Lowercase exact  : "Left"     → "left"
  2. Strip préfixes   : "fix_head" → "head"
  3. Levenshtein ≤ 2  : "leaft"   → "left"

Usage :
    # Dry-run (voir ce qui sera changé sans toucher aux fichiers)
    python3 fix_camera_names.py /media/qbee/T9/bad_sessions/ --dry-run

    # Correction réelle
    python3 fix_camera_names.py /media/qbee/T9/bad_sessions/

    # Session unique
    python3 fix_camera_names.py --session /media/qbee/T9/bad_sessions/session_20260602_182828

    # Noms attendus personnalisés
    python3 fix_camera_names.py /media/qbee/T9/bad_sessions/ --expected left right head front

    # Parallélisme
    python3 fix_camera_names.py /media/qbee/T9/bad_sessions/ -j 16

    # Sous-dossier sensors/ (pas de "head" côté capteurs)
    python3 fix_camera_names.py /media/qbee/T9/bad_sessions/ --subdir sensors --expected left right
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_DEFAULT_EXPECTED = {"left", "right", "head"}
_RESAMPLED_FILENAME = "resampled_30hz.jsonl"
# resample_report.json n'a aucune référence de nom de caméra (juste des stats) → rien à corriger.
# resampled_30hz.jsonl EN A (clés "frames"."left"/"right"/"head" + chemins "file") mais c'est du
# JSONL (1 objet par ligne), pas un JSON simple : on l'exclut du scan générique de noms ET du
# correcteur JSON générique (_fix_json_file ferait json.loads() sur tout le fichier → erreur),
# et on le corrige avec son propre correcteur dédié _fix_resampled_jsonl() dans fix().
_IGNORE_FILES = {"resample_report.json", _RESAMPLED_FILENAME}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov"}
_STRIP_PREFIXES = ("fix_", "new_", "old_", "temp_", "bad_", "test_")
_DEFAULT_WORKERS = min(16, (os.cpu_count() or 4) * 2)
_PROGRESS_EVERY = 500


# ─── Levenshtein ─────────────────────────────────────────────────────────────

def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for ca in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


# ─── Résolution typo → nom correct ───────────────────────────────────────────

def _sensor_ground_truth(jsonl_path: Path, expected: frozenset[str]) -> str | None:
    """Lit le champ "sensor" de la première ligne d'un sensors/*.jsonl.

    Le firmware Arduino écrit son identité dans chaque ligne ("sensor": "left"/"right"),
    c'est donc une source de vérité plus fiable que le nom de fichier pour ce sous-dossier.
    """
    try:
        with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
        val = json.loads(first).get("sensor")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return val if val in expected else None


def resolve(name: str, expected: frozenset[str]) -> str | None:
    """Retourne le nom correct pour un nom erroné, ou None si indéterminable."""
    if name in expected:
        return None  # déjà correct

    low = name.lower()

    # 1. Lowercase direct
    if low in expected:
        return low

    # 2. Strip préfixes connus
    for prefix in _STRIP_PREFIXES:
        if low.startswith(prefix):
            candidate = low[len(prefix):]
            if candidate in expected:
                return candidate

    # 3. Levenshtein ≤ 2 (après lowercase)
    best, best_d = None, 3
    for exp in expected:
        d = _lev(low, exp)
        if d < best_d:
            best_d, best = d, exp
    if best_d <= 2:
        return best

    return None  # impossible à résoudre automatiquement


# ─── Remplacement récursif dans un objet JSON ────────────────────────────────

def _replace_in_json(obj, renames: dict[str, str]):
    if isinstance(obj, str):
        return renames.get(obj, obj)
    if isinstance(obj, list):
        return [_replace_in_json(x, renames) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_in_json(v, renames) for k, v in obj.items()}
    return obj


# ─── Correction dédiée de cameras/resampled_30hz.jsonl ───────────────────────

_RESAMPLED_FILE_PATH_RE = re.compile(r"(/cameras/)([^/]+)(/)")


def fix_resampled_jsonl(path: Path, renames: dict[str, str], dry_run: bool) -> bool:
    """
    cameras/resampled_30hz.jsonl synchronise les 3 caméras par grille
    temporelle : chaque ligne a une clé "frames"."{nom_caméra}" avec un
    chemin "file" qui encode le nom dans "/cameras/{nom}/frame_xxx.jpg".

    C'est du JSONL (1 objet par ligne), donc le correcteur JSON générique
    (qui fait json.loads() sur tout le fichier) ne peut pas s'en charger —
    d'où cette fonction dédiée, utilisée à la fois par SessionFixer
    (renommages détectés sur les noms de fichiers) et par
    detect_charuco_lr.apply_fix() (renommages head↔gripper). Sans elle, un
    renommage laisse ce fichier pointer vers les ANCIENS noms : toute
    corrélation caméra↔capteur basée sur ses timestamps devient
    silencieusement fausse — une session "corrigée" en apparence mais
    rendue inutilisable par ce fichier resté désynchronisé.

    Corrige aussi au passage le typo connu "rigth" → "right" rencontré sur
    certains rigs, qu'un renommage soit en cours ou non.

    Retourne True si le fichier a été (ou aurait été, en dry-run) modifié.
    """
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False

    changed = False
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        frames = d.get("frames")
        if isinstance(frames, dict):
            new_frames = {}
            for key, info in frames.items():
                norm_key = "right" if key == "rigth" else key
                new_key = renames.get(norm_key, norm_key)
                if new_key != key:
                    changed = True
                if isinstance(info, dict) and isinstance(info.get("file"), str):
                    def _sub(m):
                        seg = "right" if m.group(2) == "rigth" else m.group(2)
                        seg = renames.get(seg, seg)
                        return f"{m.group(1)}{seg}{m.group(3)}"
                    new_file = _RESAMPLED_FILE_PATH_RE.sub(_sub, info["file"])
                    if new_file != info["file"]:
                        info = {**info, "file": new_file}
                        changed = True
                new_frames[new_key] = info
            d["frames"] = new_frames
        new_lines.append(json.dumps(d, ensure_ascii=False))

    if changed and not dry_run:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed


# ─── Correction d'une session ────────────────────────────────────────────────

class SessionFixer:
    def __init__(self, session_dir: Path, expected: frozenset[str], dry_run: bool, subdir: str = "cameras"):
        self.session_dir = session_dir
        self.expected = expected
        self.dry_run = dry_run
        self.subdir = subdir
        self.log: list[str] = []
        self.errors: list[str] = []

    def _record(self, msg: str):
        self.log.append(msg)

    def _error(self, msg: str):
        self.errors.append(msg)

    # ── Construction du mapping ──────────────────────────────────────────────

    def _build_renames(self) -> dict[str, str] | None:
        """
        Renvoie {old_name: new_name} pour toutes les caméras à renommer.
        Renvoie None si un conflit bloque la session.
        """
        cameras_dir = self.session_dir / self.subdir
        if not cameras_dir.is_dir():
            return {}

        # Noms présents (stems)
        present: set[str] = set()
        jsonl_by_stem: dict[str, Path] = {}
        for entry in os.scandir(cameras_dir):
            if not entry.is_file():
                continue
            p = Path(entry.name)
            if p.suffix.lower() in _VIDEO_EXT or p.suffix.lower() == ".jsonl":
                if entry.name not in _IGNORE_FILES:
                    present.add(p.stem)
                    if p.suffix.lower() == ".jsonl":
                        jsonl_by_stem[p.stem] = Path(entry.path)

        renames: dict[str, str] = {}
        for name in present:
            # Pour sensors/, le contenu (champ "sensor" écrit par le firmware) est
            # une vérité-terrain plus fiable que le nom de fichier : même un nom
            # qui "a l'air correct" (ex : right.jsonl) doit être vérifié, car le
            # bug visé ici est précisément l'INTERVERSION de deux noms valides.
            target = None
            if self.subdir == "sensors" and name in jsonl_by_stem:
                target = _sensor_ground_truth(jsonl_by_stem[name], self.expected)
                if target == name:
                    continue  # contenu cohérent avec le nom, rien à faire

            if target is None:
                if name in self.expected:
                    continue  # déjà correct (pas de vérité-terrain contraire)
                target = resolve(name, self.expected)

            if target is None:
                self._error(f"impossible de résoudre '{name}' → attendu {sorted(self.expected)}")
                return None
            renames[name] = target

        if not renames:
            return renames

        # Validation : renames doit former une permutation valide (swaps/cycles
        # autorisés), pas une collision réelle. Une cible déjà occupée par un
        # fichier qui resterait en place serait écrasée → erreur bloquante.
        targets = list(renames.values())
        if len(set(targets)) != len(targets):
            self._error(f"plusieurs fichiers visent le même nom cible : {renames}")
            return None
        for target in targets:
            if target in present and target not in renames:
                self._error(f"conflit : cible '{target}' déjà occupée et non renommée elle-même")
                return None

        return renames

    # ── Renommage des fichiers ───────────────────────────────────────────────

    def _fix_files(self, renames: dict[str, str]):
        """Renomme en 2 phases (via noms temporaires) pour gérer correctement
        les swaps/cycles (ex : left<->right) sans jamais écraser un fichier
        existant pendant le renommage."""
        cameras_dir = self.session_dir / self.subdir
        matched = [
            Path(e.path) for e in os.scandir(cameras_dir)
            if e.is_file() and Path(e.name).stem in renames
        ]
        matched.sort(key=lambda p: p.name)

        if not self.dry_run:
            temp_map: dict[Path, Path] = {}
            for path in matched:
                tmp = path.with_name(path.name + ".namefix_tmp")
                path.rename(tmp)
                temp_map[path] = tmp
            for path, tmp in temp_map.items():
                final = path.with_name(renames[path.stem] + path.suffix)
                tmp.rename(final)

        for path in matched:
            self._record(f"  rename {path.name} → {renames[path.stem]}{path.suffix}")

    # ── Correction d'un fichier JSON ─────────────────────────────────────────

    def _fix_json_file(self, path: Path, renames: dict[str, str]):
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            obj = json.loads(text)
        except (OSError, json.JSONDecodeError):
            return
        new_obj = _replace_in_json(obj, renames)
        if new_obj == obj:
            return
        self._record(f"  update {path.name}")
        if not self.dry_run:
            path.write_text(
                json.dumps(new_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── Correction du contenu d'un .jsonl caméra ─────────────────────────────

    def _fix_jsonl_content(self, path: Path, renames: dict[str, str]):
        """
        Remplace les occurrences des anciens noms dans le contenu d'un JSONL.
        Vérifie d'abord la première ligne pour éviter de lire inutilement.
        """
        if not path.is_file():
            return
        try:
            # Lecture rapide de la première ligne pour décider
            with path.open(encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
            if not any(f'"{old}"' in first for old in renames):
                return  # nom absent → pas besoin de toucher le fichier

            text = path.read_text(encoding="utf-8", errors="replace")
            # Substitution en une seule passe (regex + callback) : un remplacement
            # séquentiel naïf ("left"->"right" puis "right"->"left") corromprait
            # les swaps/cycles en retraduisant ses propres résultats.
            pattern = re.compile("|".join(f'"{re.escape(old)}"' for old in renames))
            new_text = pattern.sub(lambda m: f'"{renames[m.group(0)[1:-1]]}"', text)
            if new_text == text:
                return
            self._record(f"  update content {path.name}")
            if not self.dry_run:
                path.write_text(new_text, encoding="utf-8")
        except OSError:
            pass

    # ── Correction dédiée de cameras/resampled_30hz.jsonl ────────────────────

    def _fix_resampled_jsonl(self, path: Path, renames: dict[str, str]):
        if fix_resampled_jsonl(path, renames, self.dry_run):
            self._record(f"  update content {path.name}")

    # ── Point d'entrée ───────────────────────────────────────────────────────

    def fix(self) -> bool:
        """Retourne True si la session a été (ou aurait été) corrigée."""
        renames = self._build_renames()
        if renames is None:
            return False  # erreur bloquante
        if not renames:
            return False  # rien à faire

        self._record(f"{self.session_dir.name}")
        for old, new in sorted(renames.items()):
            self._record(f"  {old} → {new}")

        # 1. Fichiers caméra
        self._fix_files(renames)

        # 2. config.json
        self._fix_json_file(self.session_dir / "config.json", renames)

        # 3. analysis.json
        self._fix_json_file(self.session_dir / "analysis.json", renames)

        # 4. Autres JSON dans la racine de la session
        for entry in os.scandir(self.session_dir):
            if entry.is_file() and entry.name.endswith(".json"):
                if entry.name not in ("config.json", "analysis.json"):
                    self._fix_json_file(Path(entry.path), renames)

        # 5. Contenu des JSONL caméra (pour les anciens contenus qui auraient les
        #    anciens noms dedans). Ne s'applique JAMAIS à sensors/ : là, le champ
        #    "sensor" du contenu est la vérité-terrain qui a servi à construire
        #    renames — le réécrire reviendrait à corrompre la donnée qui vient
        #    d'être validée (le nom de FICHIER est ce qu'on corrige, pas le contenu).
        cameras_dir = self.session_dir / self.subdir
        if self.subdir != "sensors" and cameras_dir.is_dir():
            for entry in os.scandir(cameras_dir):
                if entry.is_file() and entry.name.endswith(".jsonl"):
                    if entry.name not in _IGNORE_FILES:
                        self._fix_jsonl_content(Path(entry.path), renames)

            # 6. cameras/resampled_30hz.jsonl — exclu du scan générique ci-dessus
            #    (c'est du JSONL, pas un nom de caméra à renommer) mais DOIT être
            #    resynchronisé avec le même mapping, sous peine de rendre la
            #    session inutilisable (timestamps caméra↔capteur désalignés).
            self._fix_resampled_jsonl(cameras_dir / _RESAMPLED_FILENAME, renames)

        return True


# ─── Scan d'un répertoire ────────────────────────────────────────────────────

def _fix_one(session_dir: Path, expected: frozenset[str], dry_run: bool, subdir: str = "cameras") -> SessionFixer:
    fixer = SessionFixer(session_dir, expected, dry_run, subdir)
    fixer.fix()
    return fixer


def scan_and_fix(
    root: Path,
    expected: frozenset[str],
    dry_run: bool,
    workers: int,
    validated: Path | None = None,
    subdir: str = "cameras",
) -> None:
    sessions = sorted(
        Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False)
    )
    if not sessions:
        print(f"Aucun sous-dossier trouvé dans {root}")
        return

    if validated is not None and not dry_run:
        validated.mkdir(parents=True, exist_ok=True)

    total = len(sessions)
    tag = "[DRY-RUN] " if dry_run else ""
    print(f"{tag}{total} sessions, {workers} workers…\n")

    done = fixed = moved = errors = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fix_one, s, expected, dry_run, subdir): s for s in sessions
        }
        for fut in as_completed(futures):
            fixer: SessionFixer = fut.result()
            with lock:
                done += 1
                if fixer.log:
                    fixed += 1
                    for line in fixer.log:
                        print(line)
                    # Déplacement si correction réussie et sans erreur
                    if validated is not None and not fixer.errors and not dry_run:
                        dest = validated / fixer.session_dir.name
                        if dest.exists():
                            print(f"  [SKIP move] {fixer.session_dir.name} déjà dans {validated.name}/")
                        else:
                            import shutil
                            shutil.move(str(fixer.session_dir), str(dest))
                            moved += 1
                            print(f"  → déplacé dans {validated.name}/")
                if fixer.errors:
                    errors += len(fixer.errors)
                    for err in fixer.errors:
                        print(f"  [ERREUR] {fixer.session_dir.name} : {err}", file=sys.stderr)
                if done % _PROGRESS_EVERY == 0:
                    print(f"  … {done}/{total}", end="\r")

    print(f"\n{'─' * 50}")
    print(f"Sessions analysées : {total}")
    print(f"Corrigées          : {fixed}")
    if validated is not None:
        print(f"Déplacées          : {moved}")
    print(f"Erreurs            : {errors}")
    if dry_run:
        print("\n(dry-run : aucun fichier modifié — relancez sans --dry-run pour appliquer)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", nargs="?", type=Path,
                   help="Répertoire contenant les sessions à corriger")
    p.add_argument("--session", type=Path,
                   help="Corriger une seule session")
    p.add_argument("--expected", nargs="+", default=sorted(_DEFAULT_EXPECTED),
                   metavar="NOM",
                   help=f"Noms corrects attendus (défaut : {sorted(_DEFAULT_EXPECTED)})")
    p.add_argument("--dry-run", action="store_true",
                   help="Afficher les corrections sans modifier les fichiers")
    p.add_argument("-v", "--validated", type=Path, metavar="DEST",
                   help="Déplacer les sessions corrigées sans erreur dans ce répertoire")
    p.add_argument("-j", "--jobs", type=int, default=_DEFAULT_WORKERS, metavar="N",
                   help=f"Workers parallèles (défaut : {_DEFAULT_WORKERS})")
    p.add_argument("--subdir", default="cameras", metavar="NOM",
                   help="Sous-dossier de session à corriger (défaut : cameras ; ex : sensors)")
    args = p.parse_args()

    expected = frozenset(args.expected)

    if args.session:
        fixer = SessionFixer(args.session.resolve(), expected, args.dry_run, args.subdir)
        changed = fixer.fix()
        for line in fixer.log:
            print(line)
        for err in fixer.errors:
            print(f"[ERREUR] {err}", file=sys.stderr)
        if not changed and not fixer.errors:
            print("Rien à corriger.")
        if args.dry_run and changed:
            print("\n(dry-run — relancez sans --dry-run pour appliquer)")
        return 0

    if args.directory:
        scan_and_fix(
            args.directory.resolve(), expected, args.dry_run, args.jobs,
            validated=args.validated, subdir=args.subdir,
        )
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
