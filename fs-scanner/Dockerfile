FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psycopg2-binary requests

WORKDIR /app
COPY fs_scanner.py .
COPY sync_ssd_to_hdd.py .

ENTRYPOINT ["python", "-u", "fs_scanner.py"]
