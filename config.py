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
    "image_size": 224,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 15,
    "num_classes": 7,
    "model_path": os.path.join(MODELS_DIR, "skin_cnn.pth"),
}

# Clases HAM10000
DIAGNOSIS_CLASSES = {
    0: {"code": "akiec", "name": "Queratosis actínica", "risk": "pre-maligno"},
    1: {"code": "bcc",   "name": "Carcinoma basocelular", "risk": "maligno"},
    2: {"code": "bkl",   "name": "Queratosis benigna", "risk": "benigno"},
    3: {"code": "df",    "name": "Dermatofibroma", "risk": "benigno"},
    4: {"code": "mel",   "name": "Melanoma", "risk": "maligno"},
    5: {"code": "nv",    "name": "Nevo melanocítico", "risk": "benigno"},
    6: {"code": "vasc",  "name": "Lesión vascular", "risk": "benigno"},
}

CLASS_TO_IDX = {v["code"]: k for k, v in DIAGNOSIS_CLASSES.items()}

RISK_LEVELS = {
    "benigno": {"level": 1, "color": "#10b981", "label": "Bajo"},
    "pre-maligno": {"level": 2, "color": "#f97316", "label": "Medio"},
    "maligno": {"level": 3, "color": "#ef4444", "label": "Alto"},
}

# =============================================================================
# NLP — Síntomas
# =============================================================================
SYMPTOM_QUESTIONS = {
    0: {"id": "dolor",   "text": "¿Sientes dolor o molestia en el lunar?"},
    1: {"id": "picor",   "text": "¿Te pica o sientes escozor en la zona?"},
    2: {"id": "tamaño",  "text": "¿Ha cambiado de tamaño recientemente?"},
    3: {"id": "sangrado","text": "¿Ha sangrado alguna vez?"},
    4: {"id": "color",   "text": "¿Ha cambiado de color?"},
    5: {"id": "duracion","text": "¿Desde cuándo lo tienes o has notado cambios?"},
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
    "action_size": 8,   # 6 preguntas + diagnosticar + pedir otra foto
}

ACTION_NAMES = {
    0: "PREGUNTAR_DOLOR",
    1: "PREGUNTAR_PICOR",
    2: "PREGUNTAR_TAMAÑO",
    3: "PREGUNTAR_SANGRADO",
    4: "PREGUNTAR_COLOR",
    5: "PREGUNTAR_DURACIÓN",
    6: "DIAGNOSTICAR",
    7: "PEDIR_OTRA_FOTO",
}

# =============================================================================
# FLASK
# =============================================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
