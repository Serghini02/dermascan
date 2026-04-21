"""
Extractor de síntomas de texto del paciente usando regex y NLP.
Procesa respuestas del paciente sobre dolor, picor, cambios, etc.
"""

import re


# Patrones regex para cada síntoma
SYMPTOM_PATTERNS = {
    "dolor": {
        "positive": [
            r'(sí|si).*(dol|duel|escoz|molest|quem|ard|pinch|punz|latid|puls)',
            r'(duele|dolor|molestia|molesta|escozor|escuece|ardor|arde|pinchazo|punzada|quemaz[oó]n|quema|latido|puls[aá]til|sensibilidad|sensible)',
        ],
        "negative": [
            r'\bno\b.*(dol|duel|molest|experimento)',
            r'(sin\s+dolor|ni\s+rastro\s+de\s+dolor|nada\s+de\s+dolor|no\s+hay\s+dolor|cero\s+molestias|no\s+experimento\s+molestia|para\s+nada)',
        ],
    },
    "picor": {
        "positive": [
            r'(sí|si).*(pic|comez|rasc|hormig|un\s+poquito)',
            r'(picor|pica|comez[oó]n|rascar|rasc[oó]|picaz[oó]n|picado|hormigueo|un\s+poquito)',
        ],
        "negative": [
            r'\bno\b.*(pic|rasc|comez|noto)',
            r'(sin\s+picor|no\s+noto\s+picor|no\s+me\s+rasco|sin\s+comez[oó]n|ni\s+me\s+pica|ni\s+pica|no\s+tengo\s+comez[oó]n|sin\s+rastro)',
        ],
    },
    "tamaño": {
        "positive": [
            r'(sí|si).*(crec|cambi|grande|aument|expand|extend|abult|duplic|noto)',
            r'(ha\s+crecido|está\s+creciendo|más\s+grande|aumenta(do)?|crece|creci[oó]|crecido|agrand|expand|extend|abultado|engrosado|duplicado|mancha|forma|crecio)',
        ],
        "negative": [
            r'\bno\b.*(crec|cambi|grande|variado|noto)',
            r'(mismo\s+tama[ñn]o|no\s+ha\s+cambiado|no\s+ha\s+variado|igual\s+que\s+siempre|no\s+ha\s+crecido|id[eé]ntico|no\s+noto\s+cambio|sin\s+variaci[oó]n)',
        ],
    },
    "sangrado": {
        "positive": [
            r'(sí|si).*(sangr|hemorrag|manch|supur|costr)',
            r'(sangr[aeó]|sangre|ha\s+sangrado|sangrado|hemorragia|manch[aó]|supura|costra)',
        ],
        "negative": [
            r'(no|ning[uú]n).*(sangr|hemorrag|noto|episodio)',
            r'(nunca\s+ha\s+sangrado|sin\s+sangrado|no\s+suelta\s+sangre|completamente\s+seco|ni\s+una\s+sola\s+vez|jam[aá]s|ning[uú]n\s+episodio)',
        ],
    },
    "color": {
        "positive": [
            r'(sí|si).*(cambi|color|oscurec|oscuro|negro|rojo|marr|blanc|azul|rojiz|bicolor|manch)',
            r'(oscureci(do|ó)|más\s+oscuro|más\s+negro|más\s+rojo|más\s+rojizo|más\s+claro|tonalidad|pigmentaci[oó]n|bicolor|manchas?\s+dentro|borde\s+azulado|zonas\s+blancas)',
        ],
        "negative": [
            r'\bno\b.*(cambi|color|noto|mutado|variaci)',
            r'(mismo\s+color|color\s+igual|uniforme|no\s+ha\s+mutado|mantiene\s+su\s+tono|estabilidad|id[eé]ntico|sin\s+variaci[oó]n\s+crom[aá]tica|tonalidad\s+sigue\s+siendo)',
        ],
    },
    "duracion": {
        "positive": [
            r'(\d+|quince|un|una|dos|tres|varios?|un\s+par|unos?|unas?)\s*(días?|semanas?|meses?|años?|d[eé]cada|verano)',
            r'(hace\s+(poco|mucho|quince|tiempo|unos|bastante|algún|varios?|memoria))',
            r'(recientemente|reciente|nuevo|últimamente|apareci[oó]|not[eé]|sali[oó]|memoria|lleva\s+conmigo|año|semanas)',
            r'(desde\s+hace|de\s+toda\s+la\s+vida|de\s+nacimiento|desde\s+que\s+nac[ií])',
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

    generic_negatives = ["no", "nada", "tampoco", "nada de nada", "que va", "nop", "no nada", "jamas", "nunca"]
    generic_positives = ["si", "claro", "por supuesto", "un poco", "un poquito", "si un poco", "algo", "mucho", "bastante"]
    
    is_generic_neg = text_clean in generic_negatives or any(kw in text_clean.split() for kw in ["no", "jamas", "nunca", "tampoco", "nada"])
    is_generic_pos = text_clean in generic_positives or any(kw in text_clean.split() for kw in ["si", "claro", "bastante", "mucho", "poquito"])

    if (is_generic_neg or is_generic_pos) and context_symptom:
        results[context_symptom] = {
            "detected": True,
            "positive": True if (is_generic_pos and not is_generic_neg) else False
        }
        if len(text_clean.split()) <= 3:
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
        neg_matched = False
        for pattern in patterns.get("negative", []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_positive = False
                detected = True
                neg_matched = True
                break

        # Buscar positivo SOLO si no hubo match negativo explícito
        if not neg_matched:
            for pattern in patterns.get("positive", []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    is_positive = True
                    detected = True
                    break

        # Caso especial para duración
        if symptom == "duracion":
            dur_match = re.search(r'(\d+|unos?|unas?)\s*(días?|semanas?|meses?|años?)', text_lower)
            if dur_match:
                val_str = dur_match.group(1)
                val_int = int(val_str) if val_str.isdigit() else None
                results[symptom] = {
                    "detected": True,
                    "positive": True,
                    "duration": {"value": val_int or val_str, "unit": dur_match.group(2)}
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
