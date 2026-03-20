FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn eventlet

COPY . .

# Crear carpetas necesarias
RUN mkdir -p models database data

EXPOSE 3333

CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
