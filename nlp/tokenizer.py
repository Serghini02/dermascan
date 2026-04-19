"""
Tokenizador personalizado para español médico/dermatológico.
Incluye tokenización por palabras, limpieza, y vocabulario médico.
"""

import re
from collections import Counter


# Vocabulario médico dermatológico
MEDICAL_VOCAB = {
    # Síntomas
    "dolor", "picor", "escozor", "ardor", "molestia", "inflamación",
    "hinchazón", "enrojecimiento", "sangrado", "supuración", "comezón",
    # Tipos de lesión
    "lunar", "nevo", "melanoma", "carcinoma", "queratosis", "lesión",
    "mancha", "verruga", "peca", "dermatofibroma", "nevus",
    # Características
    "asimetría", "borde", "color", "diámetro", "evolución",
    "irregular", "oscuro", "claro", "marrón", "negro", "rojo", "azul",
    # Partes del cuerpo
    "brazo", "pierna", "espalda", "pecho", "cara", "cuello",
    "hombro", "abdomen", "muslo", "mano", "pie", "cabeza", "torso",
    # Acciones
    "crecer", "cambiar", "sangrar", "doler", "picar", "molestar", "pica", "duele", "duele un poco", "molesta",
    "duele", "sangra", "crecía", "puntiagudo", "áspero", "pica mucho",
    # Temporalidad
    "semana", "mes", "año", "día", "recientemente", "siempre",
    "nuevo", "antiguo", "reciente",
}

# Stopwords en español
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "con", "por", "para", "que", "se",
    "es", "y", "o", "a", "no", "si", "me", "mi", "yo", "te",
    "lo", "le", "su", "nos", "muy", "más", "pero", "como",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "ha", "he", "hay", "han", "tiene", "tengo", "tiene",
    "pues", "eh", "bueno", "vale", "mira", "ver", "vamos",
}


class MedicalTokenizer:
    """Tokenizador para texto médico en español."""

    def __init__(self):
        self.vocab = MEDICAL_VOCAB
        self.stopwords = STOPWORDS

    def tokenize(self, text):
        """
        Tokeniza texto en español médico.

        Args:
            text: String de texto

        Returns:
            dict con tokens, token_details, medical_terms, stats
        """
        if not text or not text.strip():
            return {"tokens": [], "token_details": [], "medical_terms": [],
                    "stats": {"total": 0, "unique": 0, "medical": 0}}

        # 1. Normalización
        normalized = self._normalize(text)

        # 2. Tokenización por palabras
        raw_tokens = self._word_tokenize(normalized)

        # 3. Análisis de cada token
        token_details = []
        clean_tokens = []
        medical_terms = []

        for i, token in enumerate(raw_tokens):
            detail = {
                "token": token,
                "position": i,
                "is_stopword": token in self.stopwords,
                "is_medical": token in self.vocab,
                "is_number": token.isdigit(),
                "is_punctuation": bool(re.match(r'^[^\w\s]+$', token)),
                "length": len(token),
            }

            # Lematización básica
            detail["lemma"] = self._basic_lemma(token)

            token_details.append(detail)

            if not detail["is_stopword"] and not detail["is_punctuation"]:
                clean_tokens.append(token)

            if detail["is_medical"]:
                medical_terms.append(token)

        # 4. Estadísticas
        token_counts = Counter(clean_tokens)

        return {
            "original": text,
            "normalized": normalized,
            "tokens": clean_tokens,
            "all_tokens": raw_tokens,
            "token_details": token_details,
            "medical_terms": list(set(medical_terms)),
            "token_frequencies": dict(token_counts.most_common(20)),
            "stats": {
                "total_tokens": len(raw_tokens),
                "unique_tokens": len(set(raw_tokens)),
                "clean_tokens": len(clean_tokens),
                "medical_terms": len(set(medical_terms)),
                "stopwords_removed": sum(1 for d in token_details if d["is_stopword"]),
            },
        }

    def _normalize(self, text):
        """Normaliza el texto: minúsculas, limpieza básica."""
        text = text.lower().strip()
        # Eliminar caracteres especiales excepto acentos
        text = re.sub(r'[^\w\sáéíóúñü.,;:!?¿¡-]', ' ', text)
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        return text

    def _word_tokenize(self, text):
        """Tokenización por palabras, manteniendo puntuación separada."""
        # Separar puntuación
        text = re.sub(r'([.,;:!?¿¡])', r' \1 ', text)
        tokens = text.split()
        return [t.strip() for t in tokens if t.strip()]

    def _basic_lemma(self, word):
        """Lematización básica por reglas (sin librería externa)."""
        if len(word) <= 3:
            return word

        # Plurales
        if word.endswith("es") and len(word) > 4:
            if word[:-2] in self.vocab:
                return word[:-2]
        if word.endswith("s") and len(word) > 3:
            if word[:-1] in self.vocab:
                return word[:-1]

        # Verbos comunes (gerundios, participios)
        if word.endswith("ando"):
            return word[:-4] + "ar"
        if word.endswith("iendo"):
            return word[:-5] + "er"
        if word.endswith("ado"):
            return word[:-3] + "ar"
        if word.endswith("ido"):
            return word[:-3] + "ir"

        return word


# Singleton
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = MedicalTokenizer()
    return _tokenizer
