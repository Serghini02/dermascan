FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libespeak1 espeak wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Instalar Piper TTS vía pip
RUN pip install --no-cache-dir piper-tts && \
    mkdir -p /app/tts_models && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -O /app/tts_models/en.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -O /app/tts_models/en.onnx.json && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx -O /app/tts_models/es.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json -O /app/tts_models/es.onnx.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn eventlet

COPY . .

# Crear carpetas necesarias
RUN mkdir -p models database data

EXPOSE 3333

CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
