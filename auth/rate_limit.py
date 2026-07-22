"""
Limitador de intentos de login en memoria (sin dependencias externas).
No apto para despliegues multi-proceso, mitiga fuerza bruta básica.
"""
import threading
import time

from config import LOGIN_MAX_ATTEMPTS, LOGIN_LOCKOUT_MINUTES

_lock = threading.Lock()
_attempts = {}  # key -> [timestamps de intentos fallidos]


def _prune(key, window_seconds):
    now = time.time()
    _attempts[key] = [t for t in _attempts.get(key, []) if now - t < window_seconds]


def is_locked_out(key):
    window = LOGIN_LOCKOUT_MINUTES * 60
    with _lock:
        _prune(key, window)
        return len(_attempts.get(key, [])) >= LOGIN_MAX_ATTEMPTS


def register_failure(key):
    window = LOGIN_LOCKOUT_MINUTES * 60
    with _lock:
        _prune(key, window)
        _attempts.setdefault(key, []).append(time.time())


def reset(key):
    with _lock:
        _attempts.pop(key, None)
