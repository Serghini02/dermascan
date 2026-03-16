"""
Corrección de texto con expresiones regulares.
Normaliza transcripciones de voz y textos médicos.
"""

import re


# Reglas de corrección ortográfica (errores comunes en dictado médico)
SPELLING_RULES = [
    # Terminología médica
    (r'\bmelanona\b', 'melanoma'),
    (r'\bmelonoma\b', 'melanoma'),
    (r'\bcarcinona\b', 'carcinoma'),
    (r'\bqueratocis\b', 'queratosis'),
    (r'\bqueratosys\b', 'queratosis'),
    (r'\bdermatofivroma\b', 'dermatofibroma'),
    (r'\bdermatofiboma\b', 'dermatofibroma'),
    (r'\basimetria\b', 'asimetría'),
    (r'\bdiametro\b', 'diámetro'),
    (r'\bdiagnositco\b', 'diagnóstico'),
    (r'\bdiagnostico\b', 'diagnóstico'),
    (r'\blesion\b', 'lesión'),
    (r'\bevolucion\b', 'evolución'),

    # Síntomas
    (r'\bdulor\b', 'dolor'),
    (r'\bpicason\b', 'picazón'),
    (r'\besccosor\b', 'escozor'),
    (r'\binflamcion\b', 'inflamación'),
    (r'\binflamación\b', 'inflamación'),
    (r'\bhinchason\b', 'hinchazón'),
    (r'\bsangrao\b', 'sangrado'),
    (r'\bmolestya\b', 'molestia'),
    (r'\bcomeson\b', 'comezón'),
    (r'\bcomezon\b', 'comezón'),

    # Partes del cuerpo
    (r'\bespalda\b', 'espalda'),
    (r'\bpecho\b', 'pecho'),

    # Verbos comunes
    (r'\bduele\b', 'duele'),
    (r'\bduelemucho\b', 'duele mucho'),
    (r'\bpicamucho\b', 'pica mucho'),
    (r'\bsangra\b', 'sangra'),
    (r'\bcambiao\b', 'cambiado'),
    (r'\bcrecio\b', 'creció'),
    (r'\bcrecido\b', 'crecido'),

    # Errores de transcripción de voz
    (r'\bsi\b', 'sí'),  # Afirmación con tilde
    (r'\bno se\b', 'no sé'),
    (r'\bmas\b', 'más'),
    (r'\btambien\b', 'también'),
    (r'\bultimamente\b', 'últimamente'),
    (r'\bultimo\b', 'último'),

    # Números dictados
    (r'\bdos\s+semanas?\b', '2 semanas'),
    (r'\btres\s+meses?\b', '3 meses'),
    (r'\bcuatro\s+meses?\b', '4 meses'),
    (r'\bcinco\s+años?\b', '5 años'),
    (r'\bseis\s+meses?\b', '6 meses'),
    (r'\bun\s+año\b', '1 año'),
    (r'\bun\s+mes\b', '1 mes'),
    (r'\buna\s+semana\b', '1 semana'),
]

# Muletillas a eliminar
FILLER_PATTERNS = [
    r'\b(pues|eh|mm+|emm+|bueno|vale|a\s+ver|aver|osea|o\s+sea)\b',
    r'\b(este|esto|eso)\b(?=\s*,)',
    r'\.{2,}',
]


def correct_text(text):
    """
    Corrige un texto médico con reglas regex.

    Args:
        text: String original

    Returns:
        dict con texto corregido, lista de correcciones y estadísticas
    """
    if not text or not text.strip():
        return {"corrected": "", "corrections": [], "total_corrections": 0}

    corrected = text
    corrections = []

    # 1. Limpiar muletillas
    for pattern in FILLER_PATTERNS:
        matches = re.finditer(pattern, corrected, re.IGNORECASE)
        for m in matches:
            corrections.append({
                "type": "muletilla",
                "original": m.group(),
                "corrected": "",
                "position": m.start(),
            })
        corrected = re.sub(pattern, '', corrected, flags=re.IGNORECASE)

    # 2. Normalizar espacios múltiples
    before = corrected
    corrected = re.sub(r'\s+', ' ', corrected).strip()
    if before != corrected:
        corrections.append({
            "type": "espacios",
            "original": "espacios múltiples",
            "corrected": "espacio simple",
        })

    # 3. Aplicar reglas de ortografía
    for pattern, replacement in SPELLING_RULES:
        matches = list(re.finditer(pattern, corrected, re.IGNORECASE))
        if matches:
            for m in matches:
                if m.group().lower() != replacement.lower():
                    corrections.append({
                        "type": "ortografía",
                        "original": m.group(),
                        "corrected": replacement,
                        "position": m.start(),
                    })
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    # 4. Capitalizar primera letra de oraciones
    corrected = re.sub(r'(?:^|[.!?]\s+)(\w)', lambda m: m.group().upper(), corrected)

    # 5. Normalizar puntuación
    corrected = re.sub(r'\s+([.,;:!?])', r'\1', corrected)
    corrected = re.sub(r'([.,;:!?])(?=[^\s])', r'\1 ', corrected)

    return {
        "corrected": corrected.strip(),
        "corrections": corrections,
        "total_corrections": len(corrections),
        "original": text,
    }
