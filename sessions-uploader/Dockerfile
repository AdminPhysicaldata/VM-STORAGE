FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir requests psycopg2-binary

COPY SessionsToMistral.py .

CMD ["python", "SessionsToMistral.py", "/data/sessions"]
