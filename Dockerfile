FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# En el hosting free (CPU compartida) "tiny" evita timeouts largos.
# Localmente seguís pudiendo usar "base" o "small" con WHISPER_MODEL.
ENV WHISPER_MODEL=tiny
ENV PORT=10000

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 300 app:app"]
