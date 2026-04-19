"""
Extractor de síntomas de texto del paciente usando regex y NLP.
Procesa respuestas del paciente sobre dolor, picor, cambios, etc.
"""

import re


# Patrones regex para cada síntoma
SYMPTOM_PATTERNS = {
    "dolor": {
        "positive": [
            r'\b(sí|si)\b.*\b(du[eé]le|dolor|molest[ia]a?)\b',
            r'\b(du[eé]le|dolor|molest[ia]a?|escuece|arde)\b',
            r'\b(me\s+duele|tengo\s+dolor|siento\s+dolor)\b',
            r'\b(bastante|mucho|algo)\s+(dolor|molestia)\b',
        ],
        "negative": [
            r'\b(no)\b.*\b(du[eé]le|dolor|molest)\b',
            r'\b(no\s+me\s+duele)\b',
            r'\b(nada\s+de\s+dolor)\b',
            r'\b(sin\s+dolor)\b',
        ],
    },
    "picor": {
        "positive": [
            r'\b(sí|si)\b.*\b(pica|pic[ao]r|escozor|comez[oó]n)\b',
            r'\b(pica|pic[ao]r|escozor|comez[oó]n|ras[ck]a)\b',
            r'\b(me\s+pica|tengo\s+picor)\b',
            r'\b(a\s+veces\s+pica)\b',
        ],
        "negative": [
            r'\b(no)\b.*\b(pica|picor|escozor)\b',
            r'\b(no\s+me\s+pica)\b',
            r'\b(sin\s+picor)\b',
        ],
    },
    "tamaño": {
        "positive": [
            r'\b(sí|si)\b.*\b(crec|cambi|grande)\b',
            r'\b(ha\s+crecido|está\s+creciendo|más\s+grande)\b',
            r'\b(cambi[oóa]do?\s+de\s+tamaño)\b',
            r'\b(aumenta(do)?|crece|creci[oó]|crecido|agrand)\b',
            r'\b(era\s+más\s+pequeñ[oa])\b',
            r'\b(ha\s+aumentado|ha\s+crecido)\b',
        ],
        "negative": [
            r'\b(no)\b.*\b(crec|cambi|grande)\b',
            r'\b(mismo\s+tamaño|no\s+ha\s+cambiado)\b',
            r'\b(igual\s+que\s+siempre)\b',
        ],
    },
    "sangrado": {
        "positive": [
            r'\b(sí|si)\b.*\b(sangr[aoe]|sangr[oó]|hemorrag)\b',
            r'\b(sangr[ae]|sangr[oó]|ha\s+sangrado|sangrado)\b',
            r'\b(sale\s+sangre|echó\s+sangre)\b',
            r'\b(a\s+veces\s+sangra)\b',
            r'\b(veces?\s+sangr)\b',
        ],
        "negative": [
            r'\b(no)\b.*\b(sangr[aoe]|sangrado)\b',
            r'\b(nunca\s+ha\s+sangrado)\b',
            r'\b(sin\s+sangrado)\b',
        ],
    },
    "color": {
        "positive": [
            r'\b(sí|si)\b.*\b(cambi[oóa]|color|oscurec|oscuro)\b',
            r'\b(cambi[oóa](do?)?\s+de\s+color|cambi[oóa]\s+de\s+color)\b',
            r'\b(más\s+oscuro|más\s+negro|más\s+rojo|más\s+claro|era\s+más\s+claro)\b',
            r'colores?\s+diferent',
            r'\b(multicolor|irregular)\b',
            r'\b(oscureci(do|ó)|se\s+ha\s+oscurecido)\b',
            r'\b(antes\s+era\s+\w+\s+ahora\s+es)\b',
            r'\b(partes?\s+rojiz|manchas?\s+dentro)\b',
        ],
        "negative": [
            r'\b(no)\b.*\b(cambi|color)\b',
            r'\b(mismo\s+color|color\s+igual|el\s+mismo|igual(?:\s+de\s+color)?)\b',
            r'\b(color\s+es\s+el\s+mismo|sigue\s+igual|color\s+uniforme)\b',
        ],
    },
    "duracion": {
        "positive": [
            r'(\d+)\s*(días?|semanas?|meses?|años?)',
            r'\b(hace\s+(poco|mucho|tiempo|unos|bastante|algún))\b',
            r'\b(recientemente|reciente|nuevo|últimamente)\b',
            r'\b(desde\s+hace)\s+\w+',
            r'\b(siempre|toda\s+la\s+vida|de\s+nacimiento)\b',
            r'\b(hace\s+\w+\s+tiempo)\b',
            r'estas?\s+(semanas?|días?|meses?)',
        ],
        "negative": [],
    },
}


from .symptom_model import SymptomClassifier

# Instancia global del clasificador
classifier = SymptomClassifier()
classifier.load()

def extract_symptoms(text):
    """
    Extrae síntomas del texto del paciente.
    Usa el modelo entrenado de ML si está disponible, con fallback a regex.
    """
    text_lower = text.lower().strip()
    results = {}

    # Intentar predicción con el modelo ML
    ml_predictions = None
    if classifier.is_trained:
        ml_predictions = classifier.predict(text_lower)

    for symptom, patterns in SYMPTOM_PATTERNS.items():
        detected = False
        is_positive = None
        duration_value = None

        # Si es un síntoma que el modelo ML maneja, lo usamos
        if ml_predictions and symptom in ml_predictions:
            is_positive = ml_predictions[symptom]
            detected = is_positive
        else:
            # Fallback a REGEX (o para duración que no está en el clasificador base)
            # Comprobar patrones negativos primero
            for pattern in patterns.get("negative", []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    is_positive = False
                    break

            # Si no hay negativo, buscar positivo
            if is_positive is None:
                for pattern in patterns.get("positive", []):
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        is_positive = True
                        detected = True
                        break

        # Caso especial para duración (siempre requiere regex para extraer valores)
        if symptom == "duracion":
            dur_match = re.search(r'(\d+)\s*(días?|semanas?|meses?|años?)', text_lower)
            if dur_match:
                duration_value = {
                    "value": int(dur_match.group(1)),
                    "unit": dur_match.group(2),
                }
                detected = True
                is_positive = True
            elif is_positive is None:
                # Buscar palabras clave generales de tiempo si no hay números
                time_keywords = [r'hace\s+mucho', r'tiempo', r'siempre', r'reciente', r'nuevo']
                for kw in time_keywords:
                    if re.search(kw, text_lower):
                        is_positive = True
                        detected = True
                        break

        results[symptom] = {
            "detected": detected,
            "positive": is_positive,
            "duration": duration_value,
        }

    return results


def train_symptom_extractor():
    """Entrena (o re-entrena) el modelo de síntomas."""
    classifier.train()
    return True


def symptoms_to_vector(symptoms):
    """
    Convierte los síntomas extraídos a un vector numérico para el DRL.

    Returns:
        list de 6 valores (0.0 o 1.0): [dolor, picor, tamaño, sangrado, color, duracion]
    """
    keys = ["dolor", "picor", "tamaño", "sangrado", "color", "duracion"]
    vector = []
    for key in keys:
        s = symptoms.get(key, {})
        if s.get("positive") is True:
            vector.append(1.0)
        elif s.get("positive") is False:
            vector.append(0.0)
        else:
            vector.append(-1.0)  # No preguntado aún
    return vector


def get_symptom_summary(symptoms):
    """Genera un resumen textual de los síntomas detectados."""
    summary_parts = []
    labels = {
        "dolor": "Dolor",
        "picor": "Picor/escozor",
        "tamaño": "Cambio de tamaño",
        "sangrado": "Sangrado",
        "color": "Cambio de color",
        "duracion": "Duración",
    }

    for key, label in labels.items():
        s = symptoms.get(key, {})
        if s.get("positive") is True:
            text = f"✅ {label}: Sí"
            if s.get("duration"):
                d = s["duration"]
                text += f" ({d['value']} {d['unit']})"
            summary_parts.append(text)
        elif s.get("positive") is False:
            summary_parts.append(f"❌ {label}: No")
        else:
            summary_parts.append(f"❓ {label}: No evaluado")

    return summary_parts
