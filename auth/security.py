"""
Utilidades de seguridad — hashing de contraseñas y validación de entrada.
"""
import re
from werkzeug.security import generate_password_hash, check_password_hash

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password):
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)


def verify_password(password, password_hash):
    return check_password_hash(password_hash, password)


def validate_email(email):
    if not email or len(email) > 254 or not EMAIL_RE.match(email):
        return "Introduce un email válido."
    return None


def validate_password(password):
    if not password or len(password) < 8:
        return "La contraseña debe tener al menos 8 caracteres."
    if not re.search(r"[a-z]", password):
        return "La contraseña debe incluir al menos una minúscula."
    if not re.search(r"[A-Z]", password):
        return "La contraseña debe incluir al menos una mayúscula."
    if not re.search(r"\d", password):
        return "La contraseña debe incluir al menos un número."
    return None


def validate_name(name):
    if not name or not name.strip() or len(name.strip()) > 80:
        return "Introduce un nombre válido."
    return None
