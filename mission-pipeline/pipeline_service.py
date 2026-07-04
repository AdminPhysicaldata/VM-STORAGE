#!/usr/bin/env python3
"""
pipeline_service.py — Service continu à 3 étages, chacun avec sa propre file
et son propre pool de workers, pour maximiser le débit de bout en bout :

  [récupération]  discovery_queue   <- N threads qui scannent SESSIONS_DIR et
                                        vérifient la complétude (bon nombre de
                                        fichiers, session stable) en parallèle
        |
        v
  [traitement]     ready_queue      <- M process workers qui font tourner le
                                        post-pipeline (renommage, sync, shuffle,
                                        left/right — post/run_pipeline.py)
        |
        v
  [envoi]          (aucune file en sortie) <- K threads qui revalident,
                                        zippent et envoient à Mistral, puis
                                        déplacent la session vers
                                        SESSIONS_ENVOYE_DIR (visible en SFTP)

Les trois étages tournent EN PERMANENCE et EN PARALLÈLE (contrairement à un
pipeline batch qui ferait "tout découvrir, puis tout traiter, puis tout
envoyer" séquentiellement à chaque cycle) : une session peut être en cours
d'envoi tandis que la suivante est encore dans le post-pipeline et qu'une
troisième vient d'être découverte. Chaque file est bornée (maxsize) : si un
étage est saturé, l'étage amont bloque naturellement sur queue.put() — c'est
la régulation de débit (backpressure), pas besoin de plafond de volume séparé.

Réutilise sans dupliquer :
  - select_complete.is_session_complete  (critère "complète + stable")
  - run_pipeline._process_one             (post-pipeline, déjà parallélisable
                                            et picklable pour ProcessPoolExecutor)
  - SessionsToMistral.*                   (validation finale, zip, upload,
                                            suivi backend, fichier de dodge)

Variables d'environnement : voir mission-pipeline/Dockerfile / docker-compose.yml.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as pipeline
import select_complete
import SessionsToMistral as mistral

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mission-pipeline").info


# ─── Configuration ────────────────────────────────────────────────────────────

SESSIONS_DIR  = Path(os.environ.get("SESSIONS_DIR", "/data/sessions"))
SENT_DIR      = Path(os.environ.get("SESSIONS_ENVOYE_DIR", str(SESSIONS_DIR.parent / "session_envoye")))
QUARANTINE_DIR_STR = os.environ.get("QUARANTINE_DIR", "")
QUARANTINE_DIR = Path(QUARANTINE_DIR_STR) if QUARANTINE_DIR_STR else None

# Étage 1 — récupération (découverte + vérification de complétude)
DISCOVERY_WORKERS  = int(os.environ.get("DISCOVERY_WORKERS", "4"))
DISCOVERY_INTERVAL = int(os.environ.get("DISCOVERY_INTERVAL", "10"))
STABILITY_SECONDS  = int(os.environ.get("STABILITY_SECONDS", "300"))

# Étage 2 — traitement (post-pipeline : renommage/sync/shuffle/left-right)
PIPELINE_WORKERS    = int(os.environ.get("PIPELINE_WORKERS", os.environ.get("WORKERS", "4")))
APPLY                = os.environ.get("APPLY", "1") == "1"
RUN_CHARUCO          = os.environ.get("SKIP_CHARUCO", "0") != "1"
CHARUCO_SAMPLE_FPS   = float(os.environ.get("CHARUCO_SAMPLE_FPS", "2.0"))
RUN_LR_CHECK         = os.environ.get("SKIP_LR_CHECK", "0") != "1"
LR_N_SAMPLES         = int(os.environ.get("LR_N_SAMPLES", "10"))

# Étage 3 — envoi (revalidation + zip + upload Mistral)
UPLOAD_WORKERS      = int(os.environ.get("UPLOAD_WORKERS", "12"))
OFFLINE_UPLOAD       = os.environ.get("OFFLINE_UPLOAD", "0") == "1"
BACKEND_BATCH_SIZE   = int(os.environ.get("BACKEND_BATCH_SIZE", "50"))
BACKEND_FLUSH_SECONDS = int(os.environ.get("BACKEND_FLUSH_SECONDS", "20"))

STATS_INTERVAL = int(os.environ.get("STATS_INTERVAL", "60"))

# Fichier de log consultable (état courant + historique), distinct des logs
# stdout (éphémères / soumis à la rotation docker logs). Une ligne par
# intervalle STATS_INTERVAL.
_LOG_FILE_STR = os.environ.get("LOG_FILE", "")
LOG_FILE = Path(_LOG_FILE_STR) if _LOG_FILE_STR else None

# Tailles de file bornées : régulation de débit entre étages (un étage saturé
# fait simplement attendre l'étage amont sur .put(), sans plafond de volume
# global à gérer séparément).
DISCOVERY_QUEUE_MAXSIZE = max(4, PIPELINE_WORKERS * 3)
READY_QUEUE_MAXSIZE     = max(4, UPLOAD_WORKERS * 3)

mistral.OFFLINE = OFFLINE_UPLOAD


# ─── État partagé ─────────────────────────────────────────────────────────────

discovered_queue: "queue.Queue[Path]" = queue.Queue(maxsize=DISCOVERY_QUEUE_MAXSIZE)
ready_queue: "queue.Queue[Path]"      = queue.Queue(maxsize=READY_QUEUE_MAXSIZE)

in_flight: set[str] = set()
in_flight_lock = threading.Lock()

dodge = mistral.load_dodge(SESSIONS_DIR)
dodge_lock = threading.Lock()

stats = {"discovered": 0, "pipeline_ok": 0, "pipeline_anomaly": 0, "pipeline_error": 0,
         "uploaded": 0, "upload_failed": 0, "skipped": 0}
stats_lock = threading.Lock()


def _incr(key: str) -> None:
    with stats_lock:
        stats[key] += 1


def _mark(name: str) -> None:
    with in_flight_lock:
        in_flight.discard(name)


# ─── Lot d'appels backend — accumulateur avec flush par taille OU par temps ──

class BackendBatcher:
    """Regroupe les appels register-bulk/mark-sent-bulk/mark-send-failed-bulk
    pour limiter le nombre de requêtes HTTP, sans bloquer le flux d'envoi sur
    un lot jamais complété (flush périodique en plus du flush par taille)."""

    def __init__(self, batch_size: int, flush_seconds: int):
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def add(self, item: dict) -> None:
        with self._lock:
            self._buf.append(item)
            if len(self._buf) < self._batch_size:
                return
            batch = self._buf
            self._buf = []
        self._flush(batch)

    def _flush_loop(self) -> None:
        while not self._stop.wait(self._flush_seconds):
            with self._lock:
                if not self._buf:
                    continue
                batch = self._buf
                self._buf = []
            self._flush(batch)

    def _flush(self, batch: list[dict]) -> None:
        if not batch or mistral.OFFLINE:
            return
        register_items = [
            {"folder_name": e["name"], "analysis": e["analysis"], "config": e["config"],
             "mission": e["mission"], "size_bytes": e["size"]}
            for e in batch if e["analysis"] is not None
        ]
        session_ids = mistral.db_register_sessions_bulk(register_items)

        delivered, failed_refs = [], []
        for e in batch:
            ref = session_ids.get(e["name"]) or e["name"]
            if e["status"] == "ok":
                delivered.append({"session_ref": ref, "size_bytes": e["zip_size"], "duration_seconds": e["duration"]})
            else:
                failed_refs.append(ref)

        mistral.api_mark_sent_bulk(delivered)
        mistral.api_mark_send_failed_bulk(failed_refs)
        log(f"[backend] lot de {len(batch)} session(s) synchronisée(s) "
            f"({len(register_items)} register, {len(delivered)} sent, {len(failed_refs)} failed)")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        with self._lock:
            batch, self._buf = self._buf, []
        self._flush(batch)


backend_batcher = BackendBatcher(BACKEND_BATCH_SIZE, BACKEND_FLUSH_SECONDS)


# ─── Étage 1 — récupération ───────────────────────────────────────────────────

def discovery_loop() -> None:
    while True:
        try:
            candidates = [
                Path(e.path) for e in os.scandir(SESSIONS_DIR)
                if e.is_dir(follow_symlinks=False) and e.name.startswith("session_")
            ]
        except OSError as exc:
            log(f"[récupération] erreur de listing de {SESSIONS_DIR} : {exc}")
            time.sleep(DISCOVERY_INTERVAL)
            continue

        with dodge_lock:
            already_done = {e["name"] for e in dodge["sessions"]} | {e["name"] for e in dodge["skipped"]}
        with in_flight_lock:
            pending_names = set(in_flight)

        todo = [c for c in candidates if c.name not in already_done and c.name not in pending_names]

        if todo:
            # Vérification de complétude répartie sur plusieurs workers (I/O-bound :
            # stat() sur tous les fichiers attendus de chaque session candidate) —
            # c'est ce parallélisme qui constitue la "file de récupération".
            with ThreadPoolExecutor(max_workers=DISCOVERY_WORKERS) as scan_pool:
                futs = {scan_pool.submit(select_complete.is_session_complete, s, STABILITY_SECONDS): s for s in todo}
                for fut in as_completed(futs):
                    s = futs[fut]
                    try:
                        complete, reason = fut.result()
                    except Exception as exc:
                        complete, reason = False, f"erreur de vérification : {exc}"
                    if complete:
                        with in_flight_lock:
                            in_flight.add(s.name)
                        discovered_queue.put(s)  # bloque si l'étage 2 est saturé (backpressure)
                        _incr("discovered")
                        log(f"[récupération] {s.name} → file de traitement")
                    else:
                        logging.getLogger("mission-pipeline").debug("[récupération] %s reporté : %s", s.name, reason)

        time.sleep(DISCOVERY_INTERVAL)


# ─── Étage 2 — traitement (post-pipeline) ────────────────────────────────────

def _handle_pipeline_result(session_dir: Path, result: dict) -> None:
    name = session_dir.name
    status = result.get("status")
    for line in result.get("lines", []):
        log(f"  [{name}] {line}")

    if status == "OK":
        _incr("pipeline_ok")
        ready_queue.put(session_dir)  # bloque si l'étage 3 est saturé (backpressure)
        return

    _incr("pipeline_anomaly" if status == "ANOMALY" else "pipeline_error")
    if QUARANTINE_DIR:
        try:
            QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(session_dir), str(QUARANTINE_DIR / name))
            log(f"[traitement] {name} [{status}] → quarantaine")
        except OSError as exc:
            log(f"[traitement] {name} [{status}] — échec déplacement quarantaine : {exc}")
    else:
        log(f"[traitement] {name} [{status}] — laissé en place (pas de QUARANTINE_DIR)")
    _mark(name)


def pipeline_stage_loop(process_pool: ProcessPoolExecutor) -> None:
    pending: dict = {}
    max_pending = PIPELINE_WORKERS * 2

    while True:
        while len(pending) < max_pending:
            try:
                session_dir = discovered_queue.get(timeout=1)
            except queue.Empty:
                break
            fut = process_pool.submit(
                pipeline._process_one, str(session_dir), APPLY, RUN_CHARUCO,
                CHARUCO_SAMPLE_FPS, True, RUN_LR_CHECK, LR_N_SAMPLES,
            )
            pending[fut] = session_dir

        if not pending:
            time.sleep(0.5)
            continue

        done, _ = wait(list(pending.keys()), timeout=1, return_when=FIRST_COMPLETED)
        for fut in done:
            session_dir = pending.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:  # le worker n'est pas censé lever, mais on ne fait jamais confiance
                result = {"name": session_dir.name, "status": "ERROR", "lines": [f"[crash worker] {exc!r}"]}
            _handle_pipeline_result(session_dir, result)


# ─── Étage 3 — envoi (revalidation + zip + upload Mistral) ───────────────────

def _upload_job(session_dir: Path) -> dict:
    """Tourne dans un thread du pool d'envoi : revalide (le post-pipeline a pu
    modifier des fichiers depuis la découverte), zippe et envoie à Mistral."""
    name = session_dir.name

    issues = mistral.validate_session(session_dir)
    if issues:
        return {"session_dir": session_dir, "outcome": "invalid", "detail": issues}

    errors = mistral.read_analysis_errors(session_dir)
    if errors:
        return {"session_dir": session_dir, "outcome": "rejected", "detail": errors}

    is_empty, reason, size = mistral.analyze_session(session_dir)
    if is_empty:
        return {"session_dir": session_dir, "outcome": "empty", "detail": reason}

    # Déjà livrée à Mistral depuis un autre disque (copie SSD/HDD dupliquée) :
    # jamais ré-uploadée. delivered_folders_cached = snapshot BDD rafraîchi
    # périodiquement (service continu → un snapshot figé deviendrait périmé),
    # lookup O(1) par session.
    if not mistral.OFFLINE and name in mistral.delivered_folders_cached(mistral.MISTRAL_CLIENT_ID):
        return {"session_dir": session_dir, "outcome": "already_sent",
                "detail": "déjà livrée à Mistral depuis un autre disque"}

    analysis, config, mission = mistral.read_session_metadata(session_dir, dry_run=False)
    duration = mistral.read_duration(session_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = mistral.zip_session(session_dir, Path(tmp_dir))
        zip_size = zip_path.stat().st_size
        log(f"[envoi] {name} — archive {mistral.format_size(zip_size)}")
        ok = mistral.upload_zip_to_mistral(str(zip_path))

    return {
        "session_dir": session_dir, "outcome": "ok" if ok else "failed",
        "size": size, "zip_size": zip_size, "duration": duration,
        "analysis": analysis, "config": config, "mission": mission,
    }


def _handle_upload_result(res: dict) -> None:
    session_dir, name = res["session_dir"], res["session_dir"].name
    outcome = res["outcome"]

    if outcome in ("invalid", "rejected", "empty"):
        with dodge_lock:
            mistral.mark_skipped(SESSIONS_DIR, dodge, name, outcome, res["detail"])
        log(f"[envoi] {name} écarté définitivement ({outcome}) : {res['detail']}")
        _incr("skipped")
        _mark(name)
        return

    if outcome == "already_sent":
        with dodge_lock:
            mistral.mark_skipped(SESSIONS_DIR, dodge, name, "already_sent_elsewhere", res["detail"])
        SENT_DIR.mkdir(parents=True, exist_ok=True)
        mistral.move_session_to_sent(session_dir, SENT_DIR)
        log(f"[envoi] {name} déjà livrée à Mistral (autre disque) — déplacée sans ré-upload")
        _incr("skipped")
        _mark(name)
        return

    backend_batcher.add({
        "name": name, "analysis": res["analysis"], "config": res["config"], "mission": res["mission"],
        "size": res["size"], "duration": res["duration"], "status": outcome, "zip_size": res["zip_size"],
    })

    if outcome == "ok":
        with dodge_lock:
            mistral.mark_uploaded(SESSIONS_DIR, dodge, name, res["zip_size"], res["duration"])
        SENT_DIR.mkdir(parents=True, exist_ok=True)
        mistral.move_session_to_sent(session_dir, SENT_DIR)
        log(f"[envoi] {name} envoyée et déplacée → {SENT_DIR} "
            f"({mistral.format_size(res['zip_size'])}, {mistral.format_duration(res['duration'])})")
        _incr("uploaded")
    else:
        log(f"[envoi] {name} échec d'upload — reportée au prochain cycle de découverte")
        _incr("upload_failed")

    # En cas d'échec réseau, la session n'est ni dodge-marquée ni déplacée :
    # elle reste "complète" sur disque et sera redécouverte au prochain cycle.
    _mark(name)


def upload_stage_loop(upload_pool: ThreadPoolExecutor) -> None:
    pending: dict = {}
    max_pending = UPLOAD_WORKERS * 2

    while True:
        while len(pending) < max_pending:
            try:
                session_dir = ready_queue.get(timeout=1)
            except queue.Empty:
                break
            fut = upload_pool.submit(_upload_job, session_dir)
            pending[fut] = session_dir

        if not pending:
            time.sleep(0.5)
            continue

        done, _ = wait(list(pending.keys()), timeout=1, return_when=FIRST_COMPLETED)
        for fut in done:
            session_dir = pending.pop(fut)
            try:
                res = fut.result()
            except Exception as exc:
                log(f"[envoi] {session_dir.name} — erreur interne non rattrapée : {exc!r}")
                _mark(session_dir.name)
                continue
            _handle_upload_result(res)


# ─── Stats périodiques + fichier de log consultable ──────────────────────────

_LOG_FILE_HEADER = (
    "# horodatage | sessions_en_queue (traitement/envoi) | "
    "sessions_traitees_depuis_demarrage (ok/anomalies/erreurs) | "
    "sessions_envoyees_total (cumul persistant) | heures_envoyees_total (cumul persistant)"
)


def _append_log_file(line: str) -> None:
    if not LOG_FILE:
        return
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        log(f"[stats] impossible d'écrire dans {LOG_FILE} : {exc}")


def stats_loop() -> None:
    while True:
        time.sleep(STATS_INTERVAL)
        with stats_lock:
            snapshot = dict(stats)
        with in_flight_lock:
            n_in_flight = len(in_flight)
        # Sessions/heures envoyées : cumul persistant (fichier de dodge), pas
        # seulement depuis le démarrage de ce process — c'est le chiffre qui
        # compte pour le suivi métier et il survit aux redémarrages.
        with dodge_lock:
            total_sent = len(dodge["sessions"])
            total_hours = sum(e["duration_seconds"] for e in dodge["sessions"]) / 3600

        n_queue_traitement = discovered_queue.qsize()
        n_queue_envoi = ready_queue.qsize()
        n_traitees = snapshot["pipeline_ok"] + snapshot["pipeline_anomaly"] + snapshot["pipeline_error"]

        line = (
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
            f"sessions_en_queue={n_queue_traitement + n_queue_envoi} "
            f"(traitement={n_queue_traitement}, envoi={n_queue_envoi}) | "
            f"sessions_traitees={n_traitees} "
            f"(ok={snapshot['pipeline_ok']}, anomalies={snapshot['pipeline_anomaly']}, erreurs={snapshot['pipeline_error']}) | "
            f"sessions_envoyees={total_sent} | "
            f"heures_envoyees={total_hours:.1f}h"
        )
        log(f"[stats] {line}  (en_cours={n_in_flight}, échecs_envoi={snapshot['upload_failed']}, écartées={snapshot['skipped']})")
        _append_log_file(line)


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if QUARANTINE_DIR:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
            _append_log_file(_LOG_FILE_HEADER)

    log(f"mission-pipeline démarré — SESSIONS_DIR={SESSIONS_DIR}  SENT_DIR={SENT_DIR}  "
        f"QUARANTINE_DIR={QUARANTINE_DIR or '(désactivé)'}  LOG_FILE={LOG_FILE or '(désactivé)'}")
    log(f"étages : récupération={DISCOVERY_WORKERS} threads (intervalle {DISCOVERY_INTERVAL}s)  "
        f"traitement={PIPELINE_WORKERS} process (apply={APPLY})  envoi={UPLOAD_WORKERS} threads "
        f"(offline={OFFLINE_UPLOAD})")

    with ProcessPoolExecutor(max_workers=PIPELINE_WORKERS) as process_pool, \
         ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as upload_pool:

        threads = [
            threading.Thread(target=discovery_loop, name="discovery", daemon=True),
            threading.Thread(target=pipeline_stage_loop, args=(process_pool,), name="pipeline-stage", daemon=True),
            threading.Thread(target=upload_stage_loop, args=(upload_pool,), name="upload-stage", daemon=True),
            threading.Thread(target=stats_loop, name="stats", daemon=True),
        ]
        for t in threads:
            t.start()

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            log("arrêt demandé...")
        finally:
            backend_batcher.stop()


if __name__ == "__main__":
    main()
