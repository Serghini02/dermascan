"""
Symptom extractor from patient text using regex and NLP.
Processes patient responses about pain, itching, changes, etc.
"""

import re
import unicodedata
import string


# Regex patterns for each symptom (English)
SYMPTOM_PATTERNS = {
    "pain": {
        "positive": [
            r'(yes|yeah|hurt|pain|sting|burn|stab|throb)',
            r'(si|duele|dolor|molestia|pica|escozor|pinchazo|quema|punzada|sensible|sensibilidad)',
            r'(hurts|pain|discomfort|stings|stinging|burns|burning|stabbing|stabs|throbbing|throb|sensitive|sensitivity)',
        ],
        "negative": [
            r'(no|none|n\'?t|not|don\'?t|doesn\'?t)\s*(hurt|pain|discomfort|feel|any|had)',
            r'(no\s+pain|without\s+pain|not\s+hurting|no\s+discomfort|at\s+all|zero\s+discomfort|don\'?t\s+feel|don\'?t\s+have)',
            r'(no\s+me\s+duele|sin\s+dolor|nada\s+de\s+dolor|no\s+molesta|ninguna\s+molestia|no\s+siento\s+nada)',
        ],
    },
    "itching": {
        "positive": [
            r'(yes|yeah|itch|scratch|tingle|little\s+bit)',
            r'(si|pica|picor|escozor|rascar|ganas\s+de\s+rascar|hormigueo|quemazon)',
            r'(itching|itchy|itch|scratch|scratching|tingling|tingle|little\s+bit)',
        ],
        "negative": [
            r'(no|none|n\'?t|not|don\'?t|doesn\'?t)\s*(itch|scratch|notice|any)',
            r'(no\s+itching|not\s+itchy|don\'?t\s+scratch|doesn\'?t\s+itch|without\s+itching|neither\s+itchy|no\s+trace)',
            r'(no\s+pica|no\s+siento\s+picor|no\s+me\s+rasco|sin\s+picor|nada\s+de\s+picor)',
        ],
    },
    "size": {
        "positive": [
            r'(yes|yeah|grow|change|large|big|increase|expand|extend|bulky|double|notice)',
            r'(si|crecido|crece|mas\s+grande|aumentado|cambio|cambiado|evolucionado|extendido)',
            r'(has\s+grown|is\s+growing|larger|bigger|increase(d)?|grows|grew|growth|expanded|extended|bulky|thickened|doubled|spot|shape|growing)',
        ],
        "negative": [
            r'(no|none|n\'?t|hasn\'?t)\s*(grow|change|big|varied|notice|increase)',
            r'(same\s+size|hasn\'?t\s+changed|hasn\'?t\s+varied|same\s+as\s+always|hasn\'?t\s+grown|identical|no\s+change|without\s+variation)',
            r'(igual|mismo\s+tamaño|no\s+ha\s+crecido|no\s+ha\s+cambiado|no\s+ha\s+variado|esta\s+igual)',
        ],
    },
    "bleeding": {
        "positive": [
            r'(yes|yeah|bleed|blood|stain|ooze|scab)',
            r'(si|sangra|sangrado|sangre|costra|herida|supura|supuracion)',
            r'(bleeds|blood|has\s+bled|bleeding|hemorrhage|stain(ed)?|oozes|oozing|scab)',
        ],
        "negative": [
            r'(no|none|never|hasn\'?t)\s*(bleed|blood|notice|episode|bleeding)',
            r'(never\s+bled|without\s+bleeding|no\s+blood|completely\s+dry|not\s+once|never|any\s+episode)',
            r'(no\s+sangra|nunca\s+ha\s+sangrado|no\s+sale\s+sangre|seco|sin\s+sangre)',
        ],
    },
    "color": {
        "positive": [
            r'(yes|yeah|change|color|dark|black|red|brown|white|blue|reddish|bicolor|spot)',
            r'(si|cambio|color|oscuro|negro|rojo|marron|blanco|azul|rojizo|mancha)',
            r'(darkened|darker|blacker|redder|reddish|lighter|tonality|pigmentation|bicolor|spots?\s+inside|bluish\s+edge|white\s+areas)',
        ],
        "negative": [
            r'(no|none|hasn\'?t|n\'?t)\s*(change|color|notice|mutated|variation)',
            r'(same\s+color|color\s+equal|uniform|hasn\'t\s+mutated|maintains\s+its\s+tone|stability|identical|without\s+chromatic\s+variation|tonality\s+remains)',
            r'(mismo\s+color|no\s+ha\s+cambiado|igual\s+color|sin\s+cambio\s+de\s+color)',
        ],
    },
    "duration": {
        "positive": [
            r'(\d+|fifteen|one|a|two|three|several?|a\s+couple|some|pocos?|varios?|muchos?)',
            r'(ago|since|for\s+|desde|hace|llevo|tiempo|años?|meses|semanas|dias)',
            r'(recently|recent|new|lately|appeared|noticed|came\s+out|memory|been\s+with\s+me|year|weeks)',
            r'(recien|nuevo|aparecido|notado|salido|memoria|toda\s+la\s+vida|nacimiento)',
        ],
        "negative": [],
    },
}


from .symptom_model import SymptomClassifier

# Global classifier instance
classifier = SymptomClassifier()
classifier.load()

def extract_symptoms(text, context_symptom=None):
    """
    Extract symptoms from patient text.
    Uses trained ML model if available, with regex fallback.
    """
    text_lower = text.lower().strip()
    results = {}

    # Strip accents
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    text_clean = strip_accents(text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation)).strip()

    # English generics
    generic_negatives = ["no", "none", "neither", "nothing", "not at all", "nope", "never", "hardly", "no thanks"]
    generic_positives = ["yes", "yeah", "of course", "a little", "bit", "somewhat", "much", "quite", "yes a bit", "yep"]
    
    # Check if the text is a simple generic answer
    is_generic_neg = text_clean in generic_negatives or text_clean == "no"
    is_generic_pos = text_clean in generic_positives or text_clean == "yes"

    if (is_generic_neg or is_generic_pos) and context_symptom:
        results[context_symptom] = {
            "detected": True,
            "positive": True if is_generic_pos else False
        }
        # If it's a very short response, we don't need to check other symptoms
        if len(text_clean.split()) <= 2:
            return results

    # Try ML prediction
    ml_predictions = None
    if classifier.is_trained:
        ml_predictions = classifier.predict(text_lower)

    for symptom, patterns in SYMPTOM_PATTERNS.items():
        detected = False
        is_positive = None
        
        is_context = (symptom == context_symptom)

        # 2. ML Prediction
        if ml_predictions and symptom in ml_predictions:
            if ml_predictions[symptom] is True:
                is_positive = True
                detected = True
            elif is_context:
                is_positive = False
                detected = True

        # 3. Regex Fallback (Higher priority for explicit matches)
        neg_matched = False
        for pattern in patterns.get("negative", []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_positive = False
                detected = True
                neg_matched = True
                break

        if not neg_matched:
            for pattern in patterns.get("positive", []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    is_positive = True
                    detected = True
                    break

        # Duration special case
        if symptom == "duration":
            dur_match = re.search(r'(\d+|some?|a?)\s*(days?|weeks?|months?|years?)', text_lower)
            if dur_match:
                val_str = dur_match.group(1)
                val_int = int(val_str) if val_str.isdigit() else None
                results[symptom] = {
                    "detected": True,
                    "positive": True,
                    "duration": {"value": val_int or val_str, "unit": dur_match.group(2)}
                }
                continue 
            elif is_positive is None:
                for kw in [r'long\s+ago', r'time', r'lifetime', r'recent', r'new']:
                    if re.search(kw, text_lower):
                        is_positive = True
                        detected = True
                        break

        if detected:
            results[symptom] = {
                "detected": True,
                "positive": is_positive,
            }

    return results


def train_symptom_extractor():
    """Trains (re-trains) the symptom model."""
    classifier.train()
    return True


def symptoms_to_vector(symptoms):
    """Converts symptoms to vector for DRL."""
    keys = ["pain", "itching", "size", "bleeding", "color", "duration"]
    vector = []
    for key in keys:
        s = symptoms.get(key, {})
        if s.get("positive") is True:
            vector.append(1.0)
        elif s.get("positive") is False:
            vector.append(0.0)
        else:
            vector.append(-1.0) 
    return vector


def get_symptom_summary(symptoms):
    """Generates text summary of detected symptoms."""
    summary_parts = []
    labels = {
        "pain": "Pain",
        "itching": "Itching/stinging",
        "size": "Size change",
        "bleeding": "Bleeding",
        "color": "Color change",
        "duration": "Duration",
    }

    for key, label in labels.items():
        s = symptoms.get(key, {})
        if s.get("positive") is True:
            text = f"✅ {label}: Yes"
            if s.get("duration"):
                d = s["duration"]
                text += f" ({d['value']} {d['unit']})"
            summary_parts.append(text)
        elif s.get("positive") is False:
            summary_parts.append(f"❌ {label}: No")
        else:
            summary_parts.append(f"❓ {label}: Not evaluated")

    return summary_parts
