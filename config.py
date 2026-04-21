"""
Configuración centralizada — DermaScan
"""
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database", "dermascan.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# HAM10000 — se resuelve dinámicamente
HAM10000_DIR = None  # Se setea en runtime al detectar la ubicación

# =============================================================================
# CNN — Clasificación de lesiones
# =============================================================================
CNN_CONFIG = {
    "image_size": 240,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "epochs": 25,
    "num_classes": 7,
    "model_path": os.path.join(MODELS_DIR, "skin_cnn.pth"),
}

# Diagnosis Classes
DIAGNOSIS_CLASSES = {
    0: {"code": "akiec", "name": "Actinic keratosis", "risk": "pre-malignant"},
    1: {"code": "bcc",   "name": "Basal cell carcinoma", "risk": "malignant"},
    2: {"code": "bkl",   "name": "Benign keratosis-like lesions", "risk": "benign"},
    3: {"code": "df",    "name": "Dermatofibroma", "risk": "benign"},
    4: {"code": "mel",   "name": "Melanoma", "risk": "malignant"},
    5: {"code": "nv",    "name": "Melanocytic nevi", "risk": "benign"},
    6: {"code": "vasc",  "name": "Vascular lesions", "risk": "benign"},
}

CLASS_TO_IDX = {v["code"]: k for k, v in DIAGNOSIS_CLASSES.items()}

RISK_LEVELS = {
    "benign": {"level": 1, "color": "#10b981", "label": "Low"},
    "pre-malignant": {"level": 2, "color": "#f97316", "label": "Medium"},
    "malignant": {"level": 3, "color": "#ef4444", "label": "High"},
}

# =============================================================================
# NLP — Symptoms
# =============================================================================
SYMPTOM_QUESTIONS = {
    0: {"id": "pain",     "text": "Do you feel any pain or discomfort in the lesion?"},
    1: {"id": "itching",  "text": "Is it itchy or do you feel a burning sensation?"},
    2: {"id": "size",     "text": "Has it changed in size recently?"},
    3: {"id": "bleeding", "text": "Has it ever bled?"},
    4: {"id": "color",    "text": "Has it changed color?"},
    5: {"id": "duration", "text": "How long have you had it or noticed these changes?"},
}

# =============================================================================
# DRL — Deep Reinforcement Learning
# =============================================================================
DRL_CONFIG = {
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_size": 10000,
    "target_update": 100,
    "hidden_size": 128,
    "max_episodes": 500,
    "state_size": 14,   # 7 CNN probs + 6 symptoms + 1 num_questions_asked
    "action_size": 8,   # 6 questions + diagnose + request another photo
}

ACTION_NAMES = {
    0: "ASK_PAIN",
    1: "ASK_ITCH",
    2: "ASK_SIZE",
    3: "ASK_BLEEDING",
    4: "ASK_COLOR",
    5: "ASK_DURATION",
    6: "DIAGNOSE",
    7: "REQUEST_NEW_PHOTO",
}

# =============================================================================
# FLASK
# =============================================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
