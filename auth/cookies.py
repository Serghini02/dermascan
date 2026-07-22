"""
Helpers para emitir y limpiar las cookies de sesión (access, refresh, csrf).
"""
import secrets

from config import JWT_ACCESS_EXPIRES_MIN, JWT_REFRESH_EXPIRES_DAYS


def set_auth_cookies(response, access_token, refresh_token, is_production):
    csrf_token = secrets.token_urlsafe(32)

    response.set_cookie(
        "access_token", access_token,
        httponly=True, samesite="Lax", secure=is_production,
        max_age=JWT_ACCESS_EXPIRES_MIN * 60, path="/",
    )
    response.set_cookie(
        "refresh_token", refresh_token,
        httponly=True, samesite="Strict", secure=is_production,
        max_age=JWT_REFRESH_EXPIRES_DAYS * 24 * 3600, path="/api/auth",
    )
    response.set_cookie(
        "csrf_token", csrf_token,
        httponly=False, samesite="Lax", secure=is_production,
        max_age=JWT_ACCESS_EXPIRES_MIN * 60, path="/",
    )
    return response


def clear_auth_cookies(response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/api/auth")
    response.delete_cookie("csrf_token", path="/")
    return response
