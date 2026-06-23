FROM python:3.13-slim-bookworm

# Image autonome, à builder/lancer sur une machine SÉPARÉE de vm-storage :
# elle monte les sessions du serveur en SFTP (sshfs) au lieu de volumes
# Docker locaux. Toutes les variables d'environnement sont hardcodées
# ci-dessous : aucun -e ni .env nécessaire pour lancer le conteneur.

# ffmpeg       → ffprobe (durée vidéo dans diagnose_shuffle.py, via run_pipeline.py)
# sshfs        → montage des dossiers "sessions"/"sessions_envoyees" exposés par
#                le serveur vm-storage (service "sftp" de son docker-compose)
# openssh-client → ssh-keyscan etc. (dépendance de sshfs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg sshfs openssh-client tzdata \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# opencv-contrib-python-headless + numpy : run_pipeline.py (charuco/lr-check)
# requests : SessionsToMistral.py (API backend + upload Mistral)
RUN pip install --no-cache-dir opencv-contrib-python-headless numpy requests

WORKDIR /app

# Réutilise les scripts existants du post-pipeline et de l'uploader Mistral
# tels quels — pipeline_service.py ne fait qu'orchestrer les trois étages
# (récupération/traitement/envoi) en appelant leurs fonctions, il ne duplique
# aucune logique de validation/correction/upload.
COPY post/*.py .
COPY sessions-uploader/SessionsToMistral.py .
COPY mission-pipeline/select_complete.py .
COPY mission-pipeline/pipeline_service.py .
COPY mission-pipeline/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV TZ=UTC

# ── Accès SFTP au serveur vm-storage (192.168.1.20) ──────────────────────────
# Identifiants identiques à vm-storage/.env (SFTP_USER/SFTP_PASS) : à changer
# ici ET là-bas si jamais ils sont rotés.
ENV SFTP_HOST=192.168.1.20
ENV SFTP_PORT=2222
ENV SFTP_USER=exoria
ENV SFTP_PASS=TIjBvo6uelQ2aYaXbuo1mQQxxZd7GAMo

# ── pipeline_service.py — chemins (montés en SFTP par entrypoint.sh) ─────────
ENV SESSIONS_DIR=/data/sessions
ENV SESSIONS_ENVOYE_DIR=/data/session_envoye
# La quarantaine (sessions_quarantine_data) n'est pas exposée en SFTP par
# vm-storage : QUARANTINE_DIR vide = les sessions en anomalie restent en
# place (visibles, non déplacées) plutôt que de tenter un déplacement
# impossible depuis cette machine distante.
ENV QUARANTINE_DIR=
ENV LOG_FILE=/data/logs/mission_pipeline.log

# ── Étage 1 — récupération ───────────────────────────────────────────────────
ENV DISCOVERY_WORKERS=4
ENV DISCOVERY_INTERVAL=10
ENV STABILITY_SECONDS=300

# ── Étage 2 — traitement (post-pipeline) ─────────────────────────────────────
ENV PIPELINE_WORKERS=4
ENV APPLY=1
ENV SKIP_CHARUCO=0
ENV SKIP_LR_CHECK=0

# ── Étage 3 — envoi (revalidation + zip + upload Mistral) ───────────────────
ENV UPLOAD_WORKERS=12
ENV OFFLINE_UPLOAD=0
ENV BACKEND_BATCH_SIZE=50
ENV BACKEND_FLUSH_SECONDS=20
ENV STATS_INTERVAL=60
ENV DELIVERY_CLIENT_ID=mistral
ENV DELIVERY_CLIENT_NAME="Mistral AI"
ENV BACKEND_URL=http://192.168.1.15:5000/api
ENV INTERNAL_API_TOKEN=b1445bb1f44ca2ae6dab11a11cbd62b41805e02726b49c4f9e87199011176970

ENTRYPOINT ["./entrypoint.sh"]
