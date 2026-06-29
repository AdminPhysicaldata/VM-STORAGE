#!/usr/bin/env python3
"""
sync_ssd_to_hdd.py — Redondance SSD → HDD

Pour chaque session présente sur un SSD mais sans copie HDD
(ssd_disk_uuid IS NOT NULL AND hdd_disk_uuid IS NULL), ce script :
  1. Identifie le chemin source sur le SSD (mount_path/session_folder)
  2. Copie vers redondance/session_folder, sur le premier HDD disponible
     avec suffisamment d'espace (le sous-dossier "redondance" isole
     physiquement la copie de secours du reste des données du disque —
     toujours scanné normalement par fs_scanner, juste rangé à part,
     pour qu'un HDD déplacé/fusionné avec un autre reste lisible sans
     confusion entre original et copie)
  3. Met à jour hdd_disk_uuid en base de données

Variables d'environnement :
  POSTGRES_HOST/PORT/DB/USER/PASSWORD
  SYNC_WORKERS      Copies parallèles (défaut : 2)
  MIN_FREE_BYTES    Espace minimum à laisser libre sur le HDD (défaut : 10 GB)

Modes :
  --dry-run    Affiche ce qui serait copié sans rien toucher
  --once       Traite l'intégralité du backlog puis quitte  (défaut)
  --watch      Tourne en boucle, traite les nouvelles sessions
  --interval N Intervalle en secondes pour --watch (défaut : 300)
  --limit N    Nombre maximum de sessions à copier par run
"""

import argparse
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg2

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

SYNC_WORKERS  = int(os.environ.get("SYNC_WORKERS",  "2"))
MIN_FREE_BYTES = int(os.environ.get("MIN_FREE_BYTES", str(10 * 1024 ** 3)))  # 10 GB


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST",     "192.168.1.18"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB",     "robotics"),
        user=os.environ.get("POSTGRES_USER",     "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", "YsLuB46NKoF6WlS3NwUm97vhEtLkjLRQ"),
        connect_timeout=10,
    )


def _free_bytes(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_frsize * st.f_bavail
    except OSError:
        return 0


def _disk_uuid_from_db(conn, disk_type: str) -> dict:
    """Charge les disques de type donné depuis storage_disks (source de vérité)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT disk_uuid, label, mount_path FROM storage_disks WHERE disk_type = %s ORDER BY label",
            (disk_type,),
        )
        return [{"disk_uuid": r[0], "label": r[1], "mount_path": r[2]} for r in cur.fetchall()]


# ── Sessions sans redondance ──────────────────────────────────────────────────

def _fetch_unprotected(conn, limit: int | None) -> list:
    """Sessions sur SSD sans copie HDD — retourne une liste de dicts."""
    with conn.cursor() as cur:
        query = """
            SELECT
                s.session_id,
                s.session_folder,
                s.size_bytes,
                sd.disk_uuid  AS ssd_uuid,
                sd.mount_path AS ssd_mount
            FROM sessions s
            JOIN storage_disks sd ON sd.disk_uuid = s.ssd_disk_uuid
            WHERE s.ssd_disk_uuid IS NOT NULL
              AND s.hdd_disk_uuid IS NULL
              AND s.session_folder IS NOT NULL
            ORDER BY s.started_at ASC NULLS LAST
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        cur.execute(query)
        cols = ["session_id", "session_folder", "size_bytes", "ssd_uuid", "ssd_mount"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── Stats ─────────────────────────────────────────────────────────────────────

def _print_stats(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE ssd_disk_uuid IS NOT NULL)                                   AS on_ssd,
                COUNT(*) FILTER (WHERE ssd_disk_uuid IS NOT NULL AND hdd_disk_uuid IS NOT NULL)     AS redundant,
                COUNT(*) FILTER (WHERE ssd_disk_uuid IS NOT NULL AND hdd_disk_uuid IS NULL)         AS ssd_only
            FROM sessions
        """)
        on_ssd, redundant, ssd_only = cur.fetchone()
    on_ssd   = on_ssd   or 0
    redundant = redundant or 0
    ssd_only  = ssd_only  or 0
    pct = f"{redundant / on_ssd * 100:.1f}%" if on_ssd else "—"
    logger.info("Redondance actuelle : %d/%d sessions (%s) | non protégées : %d",
                redundant, on_ssd, pct, ssd_only)


# ── Copie d'une session ────────────────────────────────────────────────────────

def _copy_session(
    session_id: str,
    session_folder: str,
    size_bytes: int | None,
    ssd_mount: str,
    hdd: dict,
    dry_run: bool,
) -> tuple:
    """
    Copie session_folder de ssd_mount vers hdd["mount_path"].
    Retourne (ok: bool, hdd_uuid: str | None, err: str | None).
    """
    src = Path(ssd_mount) / session_folder
    dst = Path(hdd["mount_path"]) / "redondance" / session_folder

    if not src.exists():
        return False, None, f"source introuvable : {src}"

    if dst.exists():
        logger.debug("%s : destination existe déjà (%s)", session_folder, dst)
        return True, hdd["disk_uuid"], None

    free = _free_bytes(hdd["mount_path"])
    needed = size_bytes or 0
    if free - needed < MIN_FREE_BYTES:
        return False, None, (
            f"espace insuffisant sur {hdd['label']} "
            f"(libre={free // 1024**3}GB, besoin={needed // 1024**3}GB, "
            f"garde={MIN_FREE_BYTES // 1024**3}GB)"
        )

    if dry_run:
        logger.info("[DRY-RUN] %s → %s (%s)", src, dst, _fmt_bytes(needed))
        return True, hdd["disk_uuid"], None

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dst))
        return True, hdd["disk_uuid"], None
    except Exception as exc:
        if dst.exists():
            try:
                shutil.rmtree(str(dst))
            except Exception:
                pass
        return False, None, str(exc)


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "?"
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n} B"


# ── Mise à jour DB ────────────────────────────────────────────────────────────

def _update_hdd_uuid(conn, session_id: str, hdd_uuid: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET hdd_disk_uuid = %s WHERE session_id = %s",
            (hdd_uuid, session_id),
        )
    conn.commit()


# ── Sélection du HDD cible ────────────────────────────────────────────────────

def _pick_hdd(hdds: list, size_bytes: int | None) -> dict | None:
    """Retourne le premier HDD avec suffisamment d'espace libre."""
    needed = size_bytes or 0
    for hdd in hdds:
        if not os.path.isdir(hdd["mount_path"]):
            continue
        free = _free_bytes(hdd["mount_path"])
        if free - needed >= MIN_FREE_BYTES:
            return hdd
    return None


# ── Run principal ─────────────────────────────────────────────────────────────

def run(dry_run: bool, limit: int | None) -> dict:
    """
    Un cycle de synchronisation.
    Retourne {"copied": int, "skipped": int, "errors": int}.
    """
    conn = _pg_connect()
    try:
        hdds = _disk_uuid_from_db(conn, "hdd")
        if not hdds:
            logger.warning("Aucun disque HDD enregistré en base. Vérifiez DISK_MOUNTS et fs_scanner.")
            return {"copied": 0, "skipped": 0, "errors": 0}

        _print_stats(conn)

        sessions = _fetch_unprotected(conn, limit)
        if not sessions:
            logger.info("Toutes les sessions SSD sont déjà redondées sur HDD.")
            return {"copied": 0, "skipped": 0, "errors": 0}

        logger.info("%d session(s) à copier vers HDD", len(sessions))

        copied = skipped = errors = 0

        def _process(s):
            hdd = _pick_hdd(hdds, s["size_bytes"])
            if not hdd:
                return s["session_id"], False, None, "aucun HDD avec espace suffisant"
            ok, hdd_uuid, err = _copy_session(
                s["session_id"], s["session_folder"], s["size_bytes"],
                s["ssd_mount"], hdd, dry_run,
            )
            return s["session_id"], ok, hdd_uuid, err

        with ThreadPoolExecutor(max_workers=SYNC_WORKERS) as pool:
            futures = {pool.submit(_process, s): s for s in sessions}
            for future in as_completed(futures):
                session_id, ok, hdd_uuid, err = future.result()
                if ok:
                    if not dry_run and hdd_uuid:
                        try:
                            _update_hdd_uuid(conn, session_id, hdd_uuid)
                        except Exception as exc:
                            logger.warning("%s : update DB échoué — %s", session_id, exc)
                            errors += 1
                            continue
                    copied += 1
                    logger.info("✓ %s → HDD (%s)", session_id, hdd_uuid or "dry-run")
                elif err and "source introuvable" in err:
                    skipped += 1
                    logger.debug("⊘ %s : %s", session_id, err)
                else:
                    errors += 1
                    logger.warning("✗ %s : %s", session_id, err)

        logger.info(
            "Bilan : %d copiée(s), %d ignorée(s), %d erreur(s)%s",
            copied, skipped, errors, " [DRY-RUN]" if dry_run else "",
        )
        if not dry_run:
            _print_stats(conn)

        return {"copied": copied, "skipped": skipped, "errors": errors}
    finally:
        conn.close()


# ── Migration rétroactive (copies déjà faites à la racine du HDD) ────────────

_RESERVED_DIR_NAMES = {"redondance", "sessions", "sessions_quarantine"}
_MIGRATE_MAX_DEPTH = 6


def _walk_session_dirs(root: str, max_depth: int = _MIGRATE_MAX_DEPTH) -> list:
    """Trouve tous les dossiers session_* sous root, à n'importe quelle
    profondeur (certains HDD rangent leurs sessions dans des dossiers vrac
    comme 'sessions1T0906') — sans descendre dans 'redondance/'. Retourne
    une liste de (parent_dir, folder_name), miroir de fs_scanner._find_sessions."""
    found = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for e in entries:
            if not e.is_dir(follow_symlinks=False):
                continue
            name_lower = e.name.lower()
            if name_lower.startswith("."):
                continue
            if name_lower in _RESERVED_DIR_NAMES:
                continue
            if name_lower.startswith("session"):
                found.append((current, e.name))
                continue
            if depth < max_depth:
                stack.append((e.path, depth + 1))
    return found


def _redundant_folders(conn, hdd_uuid: str) -> set:
    """Noms de session_folder confirmés redondants pour ce HDD (présents
    aussi sur un SSD — voir hdd_disk_uuid/ssd_disk_uuid). Sert de liste
    blanche : on ne déplace JAMAIS un dossier qui n'y figure pas, pour ne
    pas rendre invisible (via _SKIP_DIR_NAMES) la seule copie existante
    d'une session qui n'aurait pas de copie SSD."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT session_folder FROM sessions
               WHERE hdd_disk_uuid = %s AND ssd_disk_uuid IS NOT NULL
                 AND session_folder IS NOT NULL""",
            (hdd_uuid,),
        )
        return {r[0] for r in cur.fetchall()}


def migrate_existing(dry_run: bool) -> dict:
    """
    Déplace les copies déjà faites par d'anciennes versions du script
    (à <hdd_mount>/.../session_folder) vers <hdd_mount>/redondance/session_folder,
    pour les isoler physiquement du reste des données du disque (fs_scanner
    continue de les scanner normalement à leur nouvel emplacement — seul
    le rangement change, pas le suivi en base).
    Ne déplace QUE les dossiers confirmés redondants en base (voir
    _redundant_folders) — une session présente uniquement sur ce HDD
    (pas de ssd_disk_uuid) n'est jamais touchée.
    Retourne {"moved": int, "skipped": int, "errors": int}.
    """
    conn = _pg_connect()
    try:
        hdds = _disk_uuid_from_db(conn, "hdd")
        moved = skipped = errors = 0
        for hdd in hdds:
            mount = hdd["mount_path"]
            if not os.path.isdir(mount):
                continue
            redundant = _redundant_folders(conn, hdd["disk_uuid"])
            logger.info("%s (%s) : %d session(s) confirmée(s) redondante(s) en base",
                        hdd["label"], mount, len(redundant))
            for parent_dir, entry in _walk_session_dirs(mount):
                if entry not in redundant:
                    skipped += 1
                    continue
                src = Path(parent_dir) / entry
                dst = Path(mount) / "redondance" / entry
                if dst.exists():
                    logger.warning("%s : existe déjà dans redondance/, source non déplacée (doublon manuel à vérifier)", entry)
                    skipped += 1
                    continue
                logger.info("%s%s → %s", "[DRY-RUN] " if dry_run else "", src, dst)
                if dry_run:
                    moved += 1
                    continue
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as exc:
                    logger.warning("%s : déplacement échoué — %s", entry, exc)
                    errors += 1
        logger.info("Migration : %d déplacée(s), %d ignorée(s) [pas confirmées redondantes ou conflit], %d erreur(s)%s",
                    moved, skipped, errors, " [DRY-RUN]" if dry_run else "")
        return {"moved": moved, "skipped": skipped, "errors": errors}
    finally:
        conn.close()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Synchronisation SSD → HDD pour la redondance des sessions")
    parser.add_argument("--dry-run",  action="store_true", help="Simulation sans copie ni modification DB")
    parser.add_argument("--watch",    action="store_true", help="Tourne en boucle continue")
    parser.add_argument("--interval", type=int, default=300, metavar="N", help="Intervalle de boucle en secondes (défaut: 300)")
    parser.add_argument("--limit",    type=int, default=None, metavar="N", help="Nombre max de sessions par run")
    parser.add_argument("--migrate-existing", action="store_true",
                         help="Déplace les copies HDD déjà faites à la racine vers redondance/, puis quitte")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] sync_ssd_to_hdd: %(message)s",
    )

    if args.dry_run:
        logger.info("=== MODE DRY-RUN : aucune copie, aucune modification DB ===")

    if args.migrate_existing:
        result = migrate_existing(args.dry_run)
        sys.exit(0 if result["errors"] == 0 else 1)

    if args.watch:
        logger.info("=== Mode --watch | intervalle=%ds ===", args.interval)
        while True:
            try:
                run(args.dry_run, args.limit)
            except Exception:
                logger.exception("Erreur lors du cycle de synchronisation")
            time.sleep(args.interval)
    else:
        try:
            result = run(args.dry_run, args.limit)
            sys.exit(0 if result["errors"] == 0 else 1)
        except Exception:
            logger.exception("Erreur fatale")
            sys.exit(2)


if __name__ == "__main__":
    main()
