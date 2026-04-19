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

def extract_symptoms(text, context_symptom=None):
    """
    Extrae síntomas del texto del paciente.
    Usa el modelo entrenado de ML si está disponible, con fallback a regex.
    """
    text_lower = text.lower().strip()
    results = {}

    # Normalización agresiva de acentos
    import unicodedata
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    text_clean = strip_accents(text_lower)
    # Eliminar puntuación
    import string
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation)).strip()

    generic_negatives = ["no", "nada", "tampoco", "nada de nada", "que va", "nop", "no nada"]
    generic_positives = ["si", "claro", "por supuesto", "un poco", "si un poco", "algo", "mucho", "bastante"]
    
    is_generic_neg = text_clean in generic_negatives
    is_generic_pos = text_clean in generic_positives

    if (is_generic_neg or is_generic_pos) and context_symptom:
        results[context_symptom] = {
            "detected": True,
            "positive": is_generic_pos
        }
        return results

    # Intentar predicción con el modelo ML
    ml_predictions = None
    if classifier.is_trained:
        ml_predictions = classifier.predict(text_lower)

    for symptom, patterns in SYMPTOM_PATTERNS.items():
        detected = False
        is_positive = None
        
        # 1. Comprobar si el síntoma es el objetivo de la pregunta actual
        is_context = (symptom == context_symptom)

        # 2. Intentar predicción con ML SOLO si el modelo está entrenado
        if ml_predictions and symptom in ml_predictions:
            # Solo consideramos el ML si predice POSITIVO o si es el síntoma del contexto
            if ml_predictions[symptom] is True:
                is_positive = True
                detected = True
            elif is_context:
                is_positive = False
                detected = True

        # 3. Fallback/Refuerzo con REGEX (Mayor prioridad si hay match explícito)
        # Comprobar patrones negativos
        for pattern in patterns.get("negative", []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_positive = False
                detected = True
                break

        # Buscar positivo
        if is_positive is not None and is_positive is False and not is_context:
             # Si ya es negativo por ML pero no es contexto, lo ignoramos a menos que regex diga positivo
             pass
        
        # Siempre buscar positivo por regex
        for pattern in patterns.get("positive", []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_positive = True
                detected = True
                break

        # Caso especial para duración
        if symptom == "duracion":
            dur_match = re.search(r'(\d+)\s*(días?|semanas?|meses?|años?)', text_lower)
            if dur_match:
                results[symptom] = {
                    "detected": True,
                    "positive": True,
                    "duration": {"value": int(dur_match.group(1)), "unit": dur_match.group(2)}
                }
                continue # Ya procesado
            elif is_positive is None:
                for kw in [r'hace\s+mucho', r'tiempo', r'siempre', r'reciente', r'nuevo']:
                    if re.search(kw, text_lower):
                        is_positive = True
                        detected = True
                        break

        # Solo añadir al resultado si fue detectado en este input
        if detected:
            results[symptom] = {
                "detected": True,
                "positive": is_positive,
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
