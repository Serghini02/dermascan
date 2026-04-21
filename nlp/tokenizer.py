"""
Custom tokenizer for clinical dermatological text in English.
Includes word tokenization, cleaning, and medical vocabulary.
"""

import re
from collections import Counter


# Dermatological medical vocabulary
MEDICAL_VOCAB = {
    # Symptoms
    "pain", "itching", "itchy", "sting", "stinging", "burn", "burning", 
    "discomfort", "inflammation", "swelling", "redness", "bleeding", 
    "oozing", "scab", "scratch", "itch", "hurts", "hurt",
    # Lesion types
    "mole", "nevus", "melanoma", "carcinoma", "keratosis", "lesion",
    "spot", "wart", "freckle", "dermatofibroma", "birthmark",
    # Features
    "asymmetry", "border", "color", "diameter", "evolution",
    "irregular", "dark", "clear", "brown", "black", "red", "blue",
    # Body parts
    "arm", "leg", "back", "chest", "face", "neck",
    "shoulder", "abdomen", "thigh", "hand", "foot", "head", "torso",
    # Actions
    "grow", "change", "bleed", "hurt", "itch", "growing", "changed", "bled", "hurts",
    "sharp", "rough", "smooth", "growing",
    # Chronology
    "week", "month", "year", "day", "recently", "always",
    "new", "old", "recent", "ago", "years", "weeks", "months", "days",
}

# English stop words
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my", 
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", 
    "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "would", 
    "could", "should", "ought", "i'm", "you're", "he's", "she's", "it's", 
    "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", 
    "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", 
    "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", 
    "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", 
    "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's", 
    "that's", "who's", "what's", "here's", "there's", "when's", "where's", 
    "why's", "how's", "d", "ll", "m", "re", "ve", "rd",
}


class MedicalTokenizer:
    """Tokenizer for English clinical text."""

    def __init__(self):
        self.vocab = MEDICAL_VOCAB
        self.stopwords = STOPWORDS

    def tokenize(self, text):
        """
        Tokenizes English medical text.

        Args:
            text: Input string
        Returns:
            dict with tokens, token_details, medical_terms, stats
        """
        if not text or not text.strip():
            return {"tokens": [], "token_details": [], "medical_terms": [],
                    "stats": {"total": 0, "unique": 0, "medical": 0}}

        # 1. Normalization
        normalized = self._normalize(text)

        # 2. Word Tokenization
        raw_tokens = self._word_tokenize(normalized)

        # 3. Analyze each token
        token_details = []
        clean_tokens = []
        medical_terms = []

        for i, token in enumerate(raw_tokens):
            detail = {
                "token": token,
                "position": i,
                "is_stopword": token in self.stopwords or token.replace("'", "") in self.stopwords,
                "is_medical": token in self.vocab or token.replace("'", "") in self.vocab,
                "is_number": token.isdigit(),
                "is_punctuation": bool(re.match(r'^[^\w\s]+$', token)),
                "length": len(token),
            }

            # Basic Lemmatization
            detail["lemma"] = self._basic_lemma(token)

            token_details.append(detail)

            if not detail["is_stopword"] and not detail["is_punctuation"]:
                clean_tokens.append(token)

            if detail["is_medical"]:
                medical_terms.append(token)

        # 4. Statistics
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
        """Normalizes text: lowercase, basic cleaning."""
        text = text.lower().strip()
        # Clean special chars but keep essential punctuation for tokenization
        text = re.sub(r'[^\w\s\'.,;:!?¿¡-]', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def _word_tokenize(self, text):
        """Word tokenization, keeping punctuation separate."""
        # Separate punctuation
        text = re.sub(r'([.,;:!?¿¡])', r' \1 ', text)
        tokens = text.split()
        return [t.strip() for t in tokens if t.strip()]

    def _basic_lemma(self, word):
        """Basic rule-based lemmatization for English."""
        if len(word) <= 3:
            return word

        # Plural
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            if word[:-1] in self.vocab:
                return word[:-1]
        
        # Gerund
        if word.endswith("ing") and len(word) > 5:
            if word[:-3] in self.vocab:
                return word[:-3]

        return word


# Singleton
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = MedicalTokenizer()
    return _tokenizer
