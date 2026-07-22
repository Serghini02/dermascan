"""
Emisión y verificación de JWT (access + refresh) para EpidermAI.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from config import (
    JWT_SECRET, JWT_ALGORITHM, JWT_ACCESS_EXPIRES_MIN, JWT_REFRESH_EXPIRES_DAYS
)


def _now():
    return datetime.now(timezone.utc)


def create_access_token(user):
    payload = {
        "sub": user["id"],
        "name": user["name"],
        "email": user["email"],
        "type": "access",
        "iat": _now(),
        "exp": _now() + timedelta(minutes=JWT_ACCESS_EXPIRES_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id):
    """Devuelve (token, jti, expires_at) — el jti se guarda hasheado en BD."""
    jti = secrets.token_urlsafe(32)
    expires_at = _now() + timedelta(days=JWT_REFRESH_EXPIRES_DAYS)
    payload = {
        "sub": user_id,
        "jti": jti,
        "type": "refresh",
        "iat": _now(),
        "exp": expires_at,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, jti, expires_at


def hash_token_id(jti):
    return hashlib.sha256(jti.encode("utf-8")).hexdigest()


def decode_token(token, expected_type=None):
    """Devuelve el payload o None si el token es inválido/expirado/tipo incorrecto."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    if expected_type and payload.get("type") != expected_type:
        return None
    return payload
