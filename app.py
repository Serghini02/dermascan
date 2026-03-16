"""
Servidor Flask — DermaScan: Escáner Inteligente de Lunares
Integra CNN (PyTorch), NLP (tokenización + voz), y DRL (DQN).
"""

import os
import sys
import json
import threading
import base64

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, DATABASE_PATH,
    DIAGNOSIS_CLASSES, RISK_LEVELS, SYMPTOM_QUESTIONS, ACTION_NAMES
)
from database.db_manager import DatabaseManager
from vision.cnn_model import SkinClassifier
from vision.skin_analyzer import analyze_mole
from nlp.tokenizer import get_tokenizer
from nlp.symptom_extractor import extract_symptoms, symptoms_to_vector, get_symptom_summary
from nlp.regex_corrector import correct_text
from rl.dqn_agent import DQNAgent
from rl.train import train_agent, evaluate_agent

# =============================================================================
# INIT
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dermascan-dev-2026')
IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet' if IS_PRODUCTION else None)

db = DatabaseManager(DATABASE_PATH)
classifier = SkinClassifier()
classifier.load_model()
dqn_agent = DQNAgent()
dqn_agent.load()
tokenizer = get_tokenizer()

# Estado de la sesión de diagnóstico actual
current_session = {
    "active": False,
    "cnn_result": None,
    "symptoms": {},
    "questions_asked": [],
    "abcde_scores": None,
}

# Detectar y cargar HAM10000
def init_ham10000():
    """Busca HAM10000 y carga metadata en la BD."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"[App] HAM10000 encontrado en: {path}")

        # Buscar CSV
        for root, _, files in os.walk(path):
            for f in files:
                if "metadata" in f.lower() and f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    db.load_ham10000_metadata(csv_path)
                    return path
    except Exception as e:
        print(f"[App] HAM10000 no disponible: {e}")
    return None

ham10000_path = init_ham10000()

# =============================================================================
# RUTAS — PÁGINAS
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


# =============================================================================
# RUTAS — VISIÓN / ESCÁNER
# =============================================================================

@app.route('/api/scan', methods=['POST'])
def scan_image():
    """Escanea una imagen de un lunar."""
    data = request.json
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "No se recibió imagen"}), 400

    try:
        # Decodificar imagen base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. Clasificación CNN
        if classifier.loaded:
            cnn_result = classifier.predict(img_pil)
        else:
            cnn_result = classifier.predict_demo()

        # 2. Análisis ABCDE
        abcde = analyze_mole(img_bgr)

        # Actualizar sesión
        current_session["active"] = True
        current_session["cnn_result"] = cnn_result
        current_session["abcde_scores"] = abcde
        current_session["symptoms"] = {}
        current_session["questions_asked"] = []

        # 3. Consultar al DRL qué pregunta hacer primero
        state = _build_state()
        drl_prediction = dqn_agent.predict(state)

        return jsonify({
            "cnn": cnn_result,
            "abcde": abcde,
            "next_action": drl_prediction,
            "next_question": _get_question_text(drl_prediction["action"]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/scan/demo', methods=['POST'])
def scan_demo():
    """Escaneo demo sin imagen real."""
    cnn_result = classifier.predict_demo()

    # ABCDE simulado
    abcde = {
        "asymmetry": {"score": round(np.random.uniform(0, 0.8), 2), "detail": "Simulado"},
        "border": {"score": round(np.random.uniform(0, 0.7), 2), "detail": "Simulado"},
        "color": {"score": round(np.random.uniform(0, 0.6), 2), "detail": "Simulado"},
        "diameter": {"score": round(np.random.uniform(0, 0.5), 2), "detail": "Simulado"},
        "evolution": {"score": round(np.random.uniform(0, 0.4), 2), "detail": "Simulado"},
        "total_score": round(np.random.uniform(0, 7), 1),
        "risk": np.random.choice(["bajo", "medio", "alto"], p=[0.5, 0.35, 0.15]),
    }

    current_session["active"] = True
    current_session["cnn_result"] = cnn_result
    current_session["abcde_scores"] = abcde
    current_session["symptoms"] = {}
    current_session["questions_asked"] = []

    state = _build_state()
    drl_pred = dqn_agent.predict(state)

    return jsonify({
        "cnn": cnn_result,
        "abcde": abcde,
        "next_action": drl_pred,
        "next_question": _get_question_text(drl_pred["action"]),
    })


# =============================================================================
# RUTAS — NLP / VOZ
# =============================================================================

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Procesa texto transcrito del paciente."""
    data = request.json
    text = data.get("text", "")
    question_id = data.get("question_id", "")

    if not text.strip():
        return jsonify({"error": "Texto vacío"}), 400

    # 1. Corrección regex
    correction = correct_text(text)

    # 2. Tokenización
    token_result = tokenizer.tokenize(correction["corrected"])

    # 3. Extracción de síntomas
    symptoms = extract_symptoms(correction["corrected"])

    # 4. Actualizar sesión
    current_session["symptoms"].update(symptoms)
    if question_id:
        current_session["questions_asked"].append(question_id)

    # 5. Construir estado y preguntar al DRL qué hacer después
    state = _build_state()
    drl_pred = dqn_agent.predict(state)

    # ¿Es diagnóstico final?
    is_final = drl_pred["action"] == 6
    diagnosis = None
    if is_final:
        diagnosis = _finalize_diagnosis()

    return jsonify({
        "correction": correction,
        "tokens": token_result,
        "symptoms": symptoms,
        "symptom_summary": get_symptom_summary(current_session["symptoms"]),
        "next_action": drl_pred,
        "next_question": _get_question_text(drl_pred["action"]),
        "is_final": is_final,
        "diagnosis": diagnosis,
    })


@app.route('/api/nlp/tokenize', methods=['POST'])
def nlp_tokenize():
    """Tokeniza texto puro."""
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Texto vacío"}), 400
    result = tokenizer.tokenize(text)
    return jsonify(result)


@app.route('/api/nlp/correct', methods=['POST'])
def nlp_correct():
    """Corrige texto con regex."""
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Texto vacío"}), 400
    result = correct_text(text)
    return jsonify(result)


# =============================================================================
# RUTAS — DRL
# =============================================================================

@app.route('/api/rl/train', methods=['POST'])
def rl_train():
    """Inicia entrenamiento del DRL."""
    data = request.json or {}
    episodes = data.get("episodes", 300)

    def cb(episode, total_episodes, reward, avg_reward, loss, epsilon, progress):
        socketio.emit('training_progress', {
            'episode': episode, 'total_episodes': total_episodes,
            'reward': round(reward, 2), 'avg_reward': round(avg_reward, 2),
            'loss': round(loss, 6), 'epsilon': round(epsilon, 4),
            'progress': progress,
        })

    def run():
        results = train_agent(max_episodes=episodes, callback=cb, db_path=DATABASE_PATH)
        dqn_agent.load()
        socketio.emit('training_complete', results)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "training_started", "episodes": episodes})


@app.route('/api/rl/evaluate', methods=['GET'])
def rl_evaluate():
    return jsonify(evaluate_agent(num_episodes=20, db_path=DATABASE_PATH))


@app.route('/api/rl/status', methods=['GET'])
def rl_status():
    model_exists = os.path.exists("models/dqn_dermascan.pth")
    status = {
        "model_loaded": model_exists,
        "epsilon": dqn_agent.epsilon,
        "steps_done": dqn_agent.steps_done,
        "episodes_trained": len(dqn_agent.episode_rewards),
    }
    results_path = os.path.join("models", "drl_training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            status["last_training"] = json.load(f)
    return jsonify(status)


# =============================================================================
# RUTAS — HISTORIAL
# =============================================================================

@app.route('/api/history', methods=['GET'])
def get_history():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(db.get_consultations(limit=limit))


@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    return jsonify(db.get_ham10000_stats())


# =============================================================================
# HELPERS
# =============================================================================

def _build_state():
    """Construye el vector de estado para el DRL."""
    cnn_probs = [0.0] * 7
    if current_session.get("cnn_result"):
        cnn_probs = current_session["cnn_result"].get("prob_vector", [0.0] * 7)

    symptom_vec = symptoms_to_vector(current_session.get("symptoms", {}))
    num_questions = len(current_session.get("questions_asked", [])) / 6.0

    state = cnn_probs + symptom_vec + [num_questions]
    return np.array(state, dtype=np.float32)


def _get_question_text(action):
    """Obtiene el texto de la pregunta para una acción."""
    if action <= 5:
        return SYMPTOM_QUESTIONS[action]["text"]
    elif action == 6:
        return "El sistema está listo para emitir un diagnóstico."
    elif action == 7:
        return "Se recomienda tomar otra fotografía del lunar."
    return ""


def _finalize_diagnosis():
    """Genera el diagnóstico final con toda la info acumulada."""
    cnn = current_session.get("cnn_result", {})
    abcde = current_session.get("abcde_scores", {})
    symptoms = current_session.get("symptoms", {})

    # Guardar en BD
    db.add_consultation(
        cnn_diagnosis=cnn.get("diagnosis_code"),
        cnn_confidence=cnn.get("confidence", 0),
        cnn_probabilities=cnn.get("probabilities"),
        symptoms=symptoms,
        abcde_scores=abcde,
        drl_diagnosis=cnn.get("diagnosis_code"),
        risk_level=cnn.get("risk_level"),
        questions_asked=len(current_session.get("questions_asked", [])),
        final_recommendation=_generate_recommendation(cnn, abcde, symptoms),
    )

    return {
        "diagnosis": cnn.get("diagnosis_name", "No determinado"),
        "diagnosis_code": cnn.get("diagnosis_code"),
        "risk_level": cnn.get("risk_level", "desconocido"),
        "risk_label": cnn.get("risk_label", "—"),
        "confidence": cnn.get("confidence", 0),
        "abcde_total": abcde.get("total_score", 0),
        "recommendation": _generate_recommendation(cnn, abcde, symptoms),
        "symptom_summary": get_symptom_summary(symptoms),
    }


def _generate_recommendation(cnn, abcde, symptoms):
    """Genera recomendación basada en todos los datos."""
    risk = cnn.get("risk_level", "benigno")
    abcde_score = abcde.get("total_score", 0)
    positive_symptoms = sum(
        1 for s in symptoms.values()
        if isinstance(s, dict) and s.get("positive") is True
    )

    if risk == "maligno" or abcde_score > 7:
        return "⚠️ URGENTE: Consulte a un dermatólogo inmediatamente. Se han detectado signos que requieren evaluación profesional urgente."
    elif risk == "pre-maligno" or abcde_score > 5 or positive_symptoms >= 3:
        return "⚡ RECOMENDADO: Agende una cita con un dermatólogo para evaluación. Se han encontrado algunos indicadores que merecen atención profesional."
    elif abcde_score > 3 or positive_symptoms >= 2:
        return "📋 SEGUIMIENTO: Monitorice el lunar y tome fotos periódicas. Si nota cambios, consulte a un dermatólogo."
    else:
        return "✅ BAJO RIESGO: El lunar no presenta signos preocupantes. Continúe con autoexámenes regulares."


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Modo demo sin HAM10000')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🔬 DermaScan — Escáner Inteligente de Lunares")
    print(f"  URL: http://localhost:{FLASK_PORT}")
    print(f"  Modo: {'DEMO' if args.demo else 'PRODUCCIÓN'}")
    print(f"  HAM10000: {'✅ Cargado' if ham10000_path else '❌ No disponible'}")
    print(f"  CNN: {'✅ Cargada' if classifier.loaded else '⚠️ Demo mode'}")
    print(f"{'='*60}\n")

    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT,
                 debug=FLASK_DEBUG, allow_unsafe_werkzeug=True)
