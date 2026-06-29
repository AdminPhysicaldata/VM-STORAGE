#!/bin/bash
# Boucle "cron-like" pour run_pipeline.py : attend l'heure cible (RUN_AT,
# heure locale du conteneur — régler TZ dans docker-compose.yml/.env), lance
# un passage complet, puis se rendort jusqu'au lendemain. Pas de vrai daemon
# cron : plus simple à superviser/logger dans un conteneur à usage unique, et
# c'est le même style que fs-scanner/sync-ssd-hdd (boucle + restart: always).
set -u

SESSIONS_DIR="${SESSIONS_DIR:-/data/sessions}"
QUARANTINE_DIR="${QUARANTINE_DIR:-}"          # vide = ne rien déplacer
REPORTS_DIR="${REPORTS_DIR:-/data/reports}"
WORKERS="${WORKERS:-4}"
RUN_AT="${RUN_AT:-03:00}"                      # HH:MM, heure locale du conteneur (TZ)
APPLY="${APPLY:-0}"                            # 0 = dry-run (recommandé), 1 = --apply réel
SKIP_CHARUCO="${SKIP_CHARUCO:-0}"
SKIP_LR_CHECK="${SKIP_LR_CHECK:-0}"
SKIP_QUALITY="${SKIP_QUALITY:-0}"               # 1 = désactive le scoring qualité 0-100/A-F (checks.py)
SKIP_QUALITY_VISION="${SKIP_QUALITY_VISION:-0}" # 1 = scoring qualité sans les checks vidéo coûteux
KEEP_REPORTS="${KEEP_REPORTS:-30}"             # nb de rapports JSONL conservés
SEND_MISTRAL="${SEND_MISTRAL:-0}"              # 1 = envoie à Mistral les sessions validées OK (--send-mistral)
MISTRAL_SENT_DIR="${MISTRAL_SENT_DIR:-}"       # vide = défaut SessionsToMistral (<parent sessions>/session_envoye)
MISTRAL_OFFLINE="${MISTRAL_OFFLINE:-0}"        # 1 = pas d'appel BACKEND_URL lors de l'envoi Mistral
LOCK_FILE=/tmp/post_pipeline.lock
RUN_NOW="${RUN_NOW:-0}"                        # 1 = lance un passage immédiat au démarrage (debug)

mkdir -p "$REPORTS_DIR"

log() { echo "[$(date -Iseconds)] $*"; }

run_once() {
    local ts args report
    ts=$(date +%Y%m%d_%H%M%S)
    report="$REPORTS_DIR/report-$ts.jsonl"
    args=(-j "$WORKERS" --report "$report" --no-ui)
    [ "$APPLY" = "1" ] && args+=(--apply)
    [ "$SKIP_CHARUCO" = "1" ] && args+=(--skip-charuco)
    [ "$SKIP_LR_CHECK" = "1" ] && args+=(--skip-lr-check)
    [ "$SKIP_QUALITY" = "1" ] && args+=(--skip-quality)
    [ "$SKIP_QUALITY_VISION" = "1" ] && args+=(--skip-quality-vision)
    [ -n "$QUARANTINE_DIR" ] && args+=(--move-bad "$QUARANTINE_DIR")
    if [ "$SEND_MISTRAL" = "1" ]; then
        args+=(--send-mistral)
        [ -n "$MISTRAL_SENT_DIR" ] && args+=(--mistral-sent-dir "$MISTRAL_SENT_DIR")
        [ "$MISTRAL_OFFLINE" = "1" ] && args+=(--mistral-offline)
    fi

    log "démarrage run_pipeline.py sur $SESSIONS_DIR ${args[*]}"
    # -E 99 : code de sortie dédié si le verrou est déjà pris, pour ne jamais
    # le confondre avec un vrai code d'erreur de run_pipeline.py (ex: 1).
    flock -n -E 99 "$LOCK_FILE" python3 -u run_pipeline.py "$SESSIONS_DIR" "${args[@]}"
    status=$?
    if [ "$status" -eq 99 ]; then
        log "run précédent encore en cours (lock pris) — cycle sauté"
    elif [ "$status" -ne 0 ]; then
        log "run_pipeline.py a terminé avec le code $status (voir $report)"
    else
        log "run terminé, rapport → $report"
    fi

    # Rotation : ne garder que les KEEP_REPORTS rapports les plus récents.
    ls -1t "$REPORTS_DIR"/report-*.jsonl 2>/dev/null | tail -n +"$((KEEP_REPORTS + 1))" | xargs -r rm -f
}

seconds_until_next_run() {
    python3 -c "
import datetime
hh, mm = '$RUN_AT'.split(':')
now = datetime.datetime.now()
target = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
if target <= now:
    target += datetime.timedelta(days=1)
print(int((target - now).total_seconds()))
"
}

log "post-pipeline démarré — prochain run à $RUN_AT (TZ=${TZ:-UTC}), mode=$([ "$APPLY" = "1" ] && echo APPLY || echo DRY-RUN)"

if [ "$RUN_NOW" = "1" ]; then
    log "RUN_NOW=1 — passage immédiat avant la boucle"
    run_once
fi

while true; do
    sleep_s=$(seconds_until_next_run)
    log "prochain passage dans ${sleep_s}s"
    sleep "$sleep_s"
    run_once
done
