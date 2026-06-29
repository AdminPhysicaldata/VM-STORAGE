FROM python:3.13-slim-bookworm

# ffmpeg → fournit ffprobe (durée vidéo dans diagnose_shuffle.py)
# util-linux → fournit flock (verrou anti-double-exécution dans l'entrypoint)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg util-linux tzdata \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# opencv-contrib-python-headless : pas de matplotlib volontairement, --plot
# n'est utilisé qu'en debug manuel, jamais dans la boucle automatique.
# pandas : requis par checks.py (scoring qualité fusionné depuis treatment-worker).
# requests : requis par SessionsToMistral.py (import direct, --send-mistral).
RUN pip install --no-cache-dir opencv-contrib-python-headless numpy pandas requests

WORKDIR /app
COPY post/*.py .
COPY sessions-uploader/SessionsToMistral.py .
COPY post/nightly_entrypoint.sh /usr/local/bin/nightly_entrypoint.sh
RUN chmod +x /usr/local/bin/nightly_entrypoint.sh

ENV TZ=UTC

ENTRYPOINT ["/usr/local/bin/nightly_entrypoint.sh"]
