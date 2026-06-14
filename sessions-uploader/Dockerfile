FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir requests psycopg2-binary kafka-python

WORKDIR /app
COPY SessionsToMistral.py .
COPY SessionsToMail.py .
COPY SessionsWatcher.py .
COPY StorageHeartbeat.py .

COPY cronjobs/crontab /etc/cron.d/sessions-uploader
RUN chmod 0644 /etc/cron.d/sessions-uploader

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
