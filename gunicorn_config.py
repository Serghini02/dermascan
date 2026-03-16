"""Gunicorn config para producción con WebSocket."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1                    # 1 worker para WebSocket (eventlet)
worker_class = "eventlet"      # Necesario para Flask-SocketIO
timeout = 120
accesslog = "-"
errorlog = "-"
loglevel = "info"
