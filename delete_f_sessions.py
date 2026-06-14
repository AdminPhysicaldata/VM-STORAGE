#!/usr/bin/env python3
"""
delete_f_sessions.py — Supprime toutes les sessions de grade F pour une date donnée.

Actions :
  1. Requête BDD pour lister les sessions F à la date spécifiée
  2. Affiche la liste + demande confirmation
  3. Supprime les dossiers sur le disque (SESSIONS_DIR)
  4. Supprime les lignes en BDD (sessions + kpi_quality_snapshots recalculé)

Usage :
    python3 delete_f_sessions.py 2026-06-04
    python3 delete_f_sessions.py 2026-06-04 --yes        # pas de confirmation interactive
    python3 delete_f_sessions.py 2026-06-04 --dry-run    # affiche sans supprimer

Variables d'environnement :
    SESSIONS_DIR, POSTGRES_HOST/PORT/DB/USER/PASSWORD
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import psycopg2

SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/data/sessions")


def _pg_connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST",     "192.168.1.18"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB",     "robotics"),
        user=os.environ.get("POSTGRES_USER",     "robotics"),
        password=os.environ.get("POSTGRES_PASSWORD", "YsLuB46NKoF6WlS3NwUm97vhEtLkjLRQ"),
        connect_timeout=10,
    )


def fetch_f_sessions(conn, date: str) -> list[dict]:
    """Récupère toutes les sessions grade F dont started_at correspond à la date."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                session_id,
                COALESCE(session_folder, session_id) AS folder,
                quality_score,
                started_at,
                project_id,
                site_id,
                pipeline_status
            FROM sessions
            WHERE quality_grade = 'F'
              AND started_at::date = %s::date
            ORDER BY started_at
        """, (date,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def folder_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total / 1_048_576


def delete_on_disk(folder: str, sessions_dir: str, dry_run: bool) -> tuple[bool, str]:
    """Supprime le dossier session sur le disque. Retourne (succès, message)."""
    path = Path(sessions_dir) / folder
    if not path.exists():
        return True, f"absent du disque (déjà supprimé ou jamais présent)"
    if dry_run:
        size = folder_size_mb(path)
        return True, f"[DRY-RUN] serait supprimé — {size:.1f} MB"
    try:
        shutil.rmtree(path)
        return True, "supprimé du disque"
    except Exception as exc:
        return False, f"erreur disque : {exc}"


def _get_sessions_rules(conn) -> list[str]:
    """Retourne les noms des rules ON DELETE sur la table sessions."""
    with conn.cursor() as cur:
        # ev_type '4' = DELETE dans pg_rewrite
        cur.execute("""
            SELECT rw.rulename
            FROM pg_rewrite rw
            JOIN pg_class c ON c.oid = rw.ev_class
            WHERE c.relname = 'sessions'
              AND rw.ev_type = '4'
        """)
        return [row[0] for row in cur.fetchall()]


def delete_from_db(conn, session_ids: list[str], dry_run: bool, force: bool = False) -> int:
    """Supprime les sessions en BDD par batch. Retourne le nombre de lignes supprimées."""
    if dry_run or not session_ids:
        return 0

    ids_tuple = tuple(session_ids)

    # ── Diagnostic : vérifier que les IDs existent bien en BDD ───────────────
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id IN %s",
            (ids_tuple,)
        )
        found = cur.fetchone()[0]
    print(f"  Vérification pré-suppression : {found}/{len(session_ids)} session(s) trouvée(s) en BDD")

    if found == 0:
        print("  ERREUR : aucun session_id ne correspond — rien à supprimer.")
        return 0

    # ── Détection des rules ON DELETE ─────────────────────────────────────────
    rules = _get_sessions_rules(conn)
    if rules:
        print(f"  Rule(s) ON DELETE détectée(s) sur sessions : {rules}")
        if not force:
            print("  → Ces rules interceptent le DELETE (soft-delete probable).")
            print("  → Relancez avec --force pour désactiver les rules et supprimer définitivement.")
            return 0
        # Désactivation des rules pour hard-delete
        for rule in rules:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE sessions DISABLE RULE "{rule}"')  # noqa: S608
            conn.commit()
            print(f"  Rule '{rule}' désactivée.")

    # ── Nettoyage des tables enfants (FK sans CASCADE) ─────────────────────────
    child_tables = [
        ("session_treatments", "session_id"),
        ("client_deliveries",  "session_id"),
        ("session_stats",      "session_id"),
    ]
    for table, col in child_tables:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table} WHERE {col} IN %s",  # noqa: S608
                    (ids_tuple,)
                )
                n = cur.rowcount
            conn.commit()
            if n:
                print(f"  {table} : {n} enregistrement(s) supprimé(s)")
        except Exception as exc:
            conn.rollback()
            print(f"  [WARN] {table} : {exc}")

    # ── Suppression des sessions par batch ────────────────────────────────────
    total_deleted = 0
    batch_size = 200

    try:
        for i in range(0, len(session_ids), batch_size):
            batch = tuple(session_ids[i:i + batch_size])
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE session_id IN %s",
                        (batch,)
                    )
                    n = cur.rowcount
                    total_deleted += n
                conn.commit()
                print(f"  Batch {i // batch_size + 1} ({len(batch)} IDs) : {n} supprimée(s)")
            except Exception as exc:
                conn.rollback()
                print(f"  [ERREUR] Batch {i // batch_size + 1} : {exc}")
    finally:
        # Réactivation des rules dans tous les cas
        for rule in rules:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'ALTER TABLE sessions ENABLE RULE "{rule}"')  # noqa: S608
                conn.commit()
                print(f"  Rule '{rule}' réactivée.")
            except Exception as exc:
                print(f"  [WARN] Réactivation rule '{rule}' : {exc}")

    # ── Vérification post-suppression ─────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id IN %s",
            (ids_tuple,)
        )
        remaining = cur.fetchone()[0]
    print(f"  Vérification post-suppression : {remaining} session(s) encore présente(s) en BDD")

    return total_deleted


def recalculate_kpis(conn, dates: set[str]) -> None:
    """Recalcule les snapshots KPI pour les dates affectées."""
    with conn.cursor() as cur:
        for d in dates:
            cur.execute("DELETE FROM kpi_quality_snapshots WHERE snapshot_date = %s::date", (d,))

        cur.execute("""
            INSERT INTO kpi_quality_snapshots (
                project_id, site_id, snapshot_date,
                session_count, scored_count,
                avg_score, min_score, max_score,
                grade_a_count, grade_b_count, grade_c_count,
                grade_d_count, grade_f_count,
                errors_count, warnings_count, updated_at
            )
            SELECT
                COALESCE(project_id,  'unknown'),
                COALESCE(site_id,     'default'),
                COALESCE(started_at::date, CURRENT_DATE),
                COUNT(*),
                COUNT(quality_score),
                ROUND(AVG(quality_score)::numeric, 1),
                MIN(quality_score),
                MAX(quality_score),
                COUNT(*) FILTER (WHERE quality_grade = 'A'),
                COUNT(*) FILTER (WHERE quality_grade = 'B'),
                COUNT(*) FILTER (WHERE quality_grade = 'C'),
                COUNT(*) FILTER (WHERE quality_grade = 'D'),
                COUNT(*) FILTER (WHERE quality_grade = 'F'),
                COALESCE(SUM(
                    jsonb_array_length(
                        COALESCE(metadata->'capture_quality'->'errors', '[]'::jsonb)
                    )
                ), 0),
                COALESCE(SUM(
                    (metadata->'capture_quality'->>'warnings_count')::int
                ), 0),
                now()
            FROM sessions
            WHERE quality_score IS NOT NULL
              AND started_at::date = ANY(%s::date[])
            GROUP BY
                COALESCE(project_id,  'unknown'),
                COALESCE(site_id,     'default'),
                COALESCE(started_at::date, CURRENT_DATE)
            ON CONFLICT (project_id, site_id, snapshot_date) DO UPDATE SET
                session_count  = EXCLUDED.session_count,
                scored_count   = EXCLUDED.scored_count,
                avg_score      = EXCLUDED.avg_score,
                min_score      = EXCLUDED.min_score,
                max_score      = EXCLUDED.max_score,
                grade_a_count  = EXCLUDED.grade_a_count,
                grade_b_count  = EXCLUDED.grade_b_count,
                grade_c_count  = EXCLUDED.grade_c_count,
                grade_d_count  = EXCLUDED.grade_d_count,
                grade_f_count  = EXCLUDED.grade_f_count,
                errors_count   = EXCLUDED.errors_count,
                warnings_count = EXCLUDED.warnings_count,
                updated_at     = now()
        """, (list(dates),))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Supprime les sessions grade F d'une date donnée")
    parser.add_argument("date", help="Date au format YYYY-MM-DD")
    parser.add_argument("--yes",     action="store_true", help="Pas de confirmation interactive")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans supprimer")
    parser.add_argument("--force",   action="store_true",
                        help="Désactive les rules ON DELETE pour suppression définitive")
    parser.add_argument("--sessions-dir", default=SESSIONS_DIR,
                        help=f"Répertoire des sessions (défaut: {SESSIONS_DIR})")
    args = parser.parse_args()

    # Validation date
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"[ERREUR] Format de date invalide : '{args.date}' — attendu YYYY-MM-DD")
        sys.exit(1)

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Recherche sessions grade F du {args.date}...")

    conn = _pg_connect()
    sessions = fetch_f_sessions(conn, args.date)

    if not sessions:
        print(f"Aucune session grade F trouvée pour le {args.date}.")
        conn.close()
        return

    # Affichage
    print(f"\n{len(sessions)} session(s) grade F trouvée(s) :\n")
    total_mb = 0.0
    for s in sessions:
        folder = s["folder"]
        path = Path(args.sessions_dir) / folder
        on_disk = path.exists()
        size = folder_size_mb(path) if on_disk else 0.0
        total_mb += size
        disk_str = f"{size:.1f} MB" if on_disk else "absent du disque"
        print(f"  • {s['session_id']}")
        print(f"    dossier    : {folder} ({disk_str})")
        print(f"    score      : {s['quality_score']} | started_at : {s['started_at']}")
        print(f"    projet     : {s['project_id']} | site : {s['site_id']}")
        print(f"    statut     : {s['pipeline_status']}")
        print()

    print(f"Total disque estimé : {total_mb:.1f} MB")

    if args.dry_run:
        print("\n[DRY-RUN] Aucune suppression effectuée.")
        conn.close()
        return

    if not args.yes:
        answer = input(f"\nSupprimer ces {len(sessions)} session(s) ? [oui/non] : ").strip().lower()
        if answer not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            conn.close()
            return

    # Suppression
    print()
    ok_disk = 0
    fail_disk = 0
    deleted_ids = []
    affected_dates = set()

    for s in sessions:
        success, msg = delete_on_disk(s["folder"], args.sessions_dir, dry_run=False)
        if success:
            ok_disk += 1
        else:
            fail_disk += 1
            print(f"  [WARN] {s['folder']} : {msg}")
        deleted_ids.append(s["session_id"])
        if s["started_at"]:
            affected_dates.add(str(s["started_at"].date()))

    # Suppression BDD
    deleted_db = delete_from_db(conn, deleted_ids, dry_run=False, force=args.force)

    # Recalcul KPI
    if affected_dates:
        print("Recalcul des KPI...")
        recalculate_kpis(conn, affected_dates)

    conn.close()

    print(f"\n=== Terminé ===")
    print(f"  Dossiers supprimés  : {ok_disk}")
    print(f"  Erreurs disque      : {fail_disk}")
    print(f"  Lignes BDD supprim. : {deleted_db}")
    print(f"  KPI recalculés pour : {', '.join(sorted(affected_dates)) or '—'}")


if __name__ == "__main__":
    main()
