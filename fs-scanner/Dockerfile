FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psycopg2-binary

WORKDIR /app
COPY fs_scanner.py .

ENTRYPOINT ["python", "-u", "fs_scanner.py"]
