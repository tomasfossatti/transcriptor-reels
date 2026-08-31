FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# argostranslate se instala sin sus deps (torch/stanza/sacremoses no se usan
# aca: traducimos con ctranslate2 directo, ver app.py) para que la imagen
# sea liviana y no se quede sin RAM en hosts free-tier.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps argostranslate==1.9.6

COPY . .

# En el hosting free (CPU compartida) "tiny" evita timeouts largos.
# Localmente seguís pudiendo usar "base" o "small" con WHISPER_MODEL.
ENV WHISPER_MODEL=tiny
ENV PORT=10000

# Evita que MKL/OpenMP/OpenBLAS reserven buffers por cada CPU del host
# fisico (no la cuota real del contenedor), lo que dispara la RAM usada.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 1 --timeout 300 app:app"]
