"""
Decoradores de autenticación/CSRF para las rutas de Flask.
"""
from functools import wraps

from flask import request, jsonify, redirect, g

from auth.jwt_utils import decode_token


def _get_current_user():
    token = request.cookies.get("access_token")
    if not token:
        return None
    payload = decode_token(token, expected_type="access")
    if not payload:
        return None
    return {"id": payload["sub"], "name": payload.get("name"), "email": payload.get("email")}


def login_required(view_fn):
    """Protege rutas /api/*: responde 401 JSON si no hay sesión válida."""
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        user = _get_current_user()
        if not user:
            return jsonify({"error": "No autenticado"}), 401
        g.user = user
        return view_fn(*args, **kwargs)
    return wrapped


def login_required_page(view_fn):
    """Protege rutas de página: redirige a /login si no hay sesión válida."""
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        user = _get_current_user()
        if not user:
            return redirect("/login")
        g.user = user
        return view_fn(*args, **kwargs)
    return wrapped


def csrf_protect(view_fn):
    """Compara el header X-CSRF-Token con la cookie csrf_token (double-submit)."""
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        cookie_token = request.cookies.get("csrf_token")
        header_token = request.headers.get("X-CSRF-Token")
        if not cookie_token or not header_token or cookie_token != header_token:
            return jsonify({"error": "CSRF token inválido o ausente"}), 403
        return view_fn(*args, **kwargs)
    return wrapped
