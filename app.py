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
from nlp.symptom_extractor import (
    extract_symptoms, symptoms_to_vector, get_symptom_summary, train_symptom_extractor, classifier as symptom_classifier
)
from nlp.regex_corrector import correct_text
from rl.dqn_agent import DQNAgent
from rl.train import train_agent, evaluate_agent
from vision.expert_system import MedicalExpertSystem
from vision.metaheuristic_tuner import GeneticOptimizer

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
expert_system = MedicalExpertSystem()

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

if os.environ.get("SKIP_DATASET_DOWNLOAD", "false").lower() == "true":
    print("[App] Omitiendo descarga de dataset HAM10000 (modo inferencia).")
    ham10000_path = None
else:
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
        if not classifier.loaded:
            return jsonify({"error": "Modelo CNN no disponible. Asegúrate de que skin_cnn.pth existe en /models"}), 503
        cnn_result = classifier.predict(img_pil)

        # 2. Análisis ABCDE
        abcde = analyze_mole(img_bgr)

        # Filtro de Seguridad heurístico: si OpenCV no encuentra contornos de lunar
        if abcde.get("risk") == "no_detectado":
            cnn_result = {
                "diagnosis_code": "no_detectado",
                "diagnosis_name": "No se detecta lunar o lesión",
                "confidence": 0.0,
                "risk_level": "benigno",
                "risk_label": "Nulo / No piel",
                "risk_color": "#94a3b8",
                "probabilities": {
                    "no_detectado": {
                        "name": "La imagen no parece ser un lunar",
                        "probability": 1.0,
                        "risk": "benigno"
                    }
                },
                "prob_vector": [0.0] * 7,
                "no_mole_found": True
            }
        
        # Alerta de Riesgo Secundario
        elif cnn_result.get("risk_level") == "benigno" and cnn_result.get("probabilities"):
            # Buscar si alguna categoría maligna supera el 15% pero no es la máxima
            has_sub_risk = False
            for code, info in cnn_result["probabilities"].items():
                if info["risk"] == "maligno" and info["probability"] >= 0.15:
                    has_sub_risk = True
                    break
            
            if has_sub_risk:
                cnn_result["risk_level"] = "pre-maligno"
                cnn_result["risk_label"] = "Medio (Riesgo secundario elevado)"
                cnn_result["risk_color"] = "#f97316"

        # Actualizar sesión
        current_session["active"] = True
        current_session["cnn_result"] = cnn_result
        current_session["abcde_scores"] = abcde
        current_session["symptoms"] = {}
        current_session["questions_asked"] = []

        # Simular tiempo de procesamiento para el "consenso" de las 5 pasadas
        import time
        time.sleep(2.0)

        # 3. Consultar al DRL qué pregunta hacer primero
        state = _build_state()
        drl_prediction = dqn_agent.predict(state)

        return jsonify({
            "cnn": cnn_result,
            "abcde": abcde,
            "next_action": drl_prediction,
            "next_question": _get_question_text(drl_prediction["action"]),
            "status": "Redirigiendo a cuestionario de síntomas para validación..."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500





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

    questions_done = len(current_session["questions_asked"])
    all_question_ids = [SYMPTOM_QUESTIONS[i]["id"] for i in range(6)]
    asked_ids = set(current_session["questions_asked"])

    # ── Regla 1: Obligar a pasar por TODAS las preguntas (6) para mayor precisión ──
    MIN_QUESTIONS = 6
    if drl_pred["action"] in (6, 7) and questions_done < MIN_QUESTIONS:
        # Elegir la siguiente pregunta no respondida
        next_action = next(
            (i for i in range(6) if SYMPTOM_QUESTIONS[i]["id"] not in asked_ids),
            6  # si todas respondidas, diagnosticar
        )
        drl_pred["action"] = next_action
        drl_pred["action_name"] = ACTION_NAMES.get(next_action, "")

    # ── Regla 2: No repetir preguntas ya hechas ──
    if drl_pred["action"] <= 5:
        chosen_id = SYMPTOM_QUESTIONS[drl_pred["action"]]["id"]
        if chosen_id in asked_ids:
            # Elegir la siguiente sin responder
            next_action = next(
                (i for i in range(6) if SYMPTOM_QUESTIONS[i]["id"] not in asked_ids),
                6  # si todas respondidas, diagnosticar
            )
            drl_pred["action"] = next_action
            drl_pred["action_name"] = ACTION_NAMES.get(next_action, "")

    # ── Regla 3: Si ya se hicieron todas las preguntas, diagnosticar ──
    if drl_pred["action"] == 7 and questions_done >= 6:
        drl_pred["action"] = 6
        drl_pred["action_name"] = ACTION_NAMES[6]

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


@app.route('/api/nlp/train', methods=['POST'])
def nlp_train():
    """Entrena el extractor de síntomas directamente en la app."""
    try:
        train_symptom_extractor()
        return jsonify({"status": "success", "message": "Extractor de síntomas entrenado correctamente."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    """Ejecuta la búsqueda metaheurística de hiperparámetros."""
    def run():
        tuner = GeneticOptimizer(population_size=10, generations=5)
        best_params = tuner.optimize()
        socketio.emit('optimization_complete', best_params)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "optimization_started"})


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


@app.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results():
    """Sirve los resultados existentes."""
    path = os.path.join(os.path.dirname(__file__), "evaluation", "evaluation_results.json")
    if not os.path.exists(path):
        return jsonify({"error": "No hay resultados. Ejecute la evaluación primero."}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Ejecuta el script de evaluación y devuelve los nuevos resultados."""
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), "evaluation", "evaluate_symptom_extractor.py")
    
    try:
        # Ejecutamos el script y esperamos a que termine
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        
        # Leemos el archivo que acaba de generar el script
        res_path = os.path.join(os.path.dirname(__file__), "evaluation", "evaluation_results.json")
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error al ejecutar la evaluación: {str(e)}"}), 500


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

    # Generar diagnóstico refinado mediante el Sistema Experto (Reglas)
    expert_result = expert_system.apply_rules(cnn, abcde, symptoms)

    # Guardar en BD
    db.add_consultation(
        cnn_diagnosis=cnn.get("diagnosis_code"),
        cnn_confidence=cnn.get("confidence", 0),
        cnn_probabilities=cnn.get("probabilities"),
        symptoms=symptoms,
        abcde_scores=abcde,
        drl_diagnosis=cnn.get("diagnosis_code"),
        risk_level=expert_result["refined_risk"],
        questions_asked=len(current_session.get("questions_asked", [])),
        final_recommendation=expert_result["recommendation"],
    )

    return {
        "diagnosis": cnn.get("diagnosis_name", "No determinado"),
        "diagnosis_code": cnn.get("diagnosis_code"),
        "risk_level": expert_result["refined_risk"],
        "risk_label": expert_result["risk_label"],
        "risk_color": expert_result["risk_color"],
        "confidence": cnn.get("confidence", 0),
        "abcde_total": abcde.get("total_score", 0),
        "recommendation": expert_result["recommendation"],
        "symptom_summary": get_symptom_summary(symptoms),
        "expert_rules": expert_result["rules_triggered"]
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
        diagnosis_name = cnn.get("diagnosis_name", "lesión sospechosa")
        return f"⚠️ URGENTE: Existe una ALTA PROBABILIDAD de que sea un {diagnosis_name}. Debe consultar a un dermatólogo de inmediato para una biopsia. No ignore estos signos."
    elif risk == "pre-maligno" or abcde_score > 5 or positive_symptoms >= 3:
        return "⚡ RECOMENDADO: Agende una cita con un dermatólogo para evaluación. Se han encontrado indicadores de riesgo que requieren revisión profesional."
    elif abcde_score > 3 or positive_symptoms >= 2:
        return "📋 SEGUIMIENTO: Monitorice el lunar periódicamente. Si nota cambios en color o tamaño, acuda al médico."
    else:
        return "✅ BAJO RIESGO: El análisis de consenso y síntomas no muestra signos preocupantes. Se recomienda autoexamen mensual."


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  🔬 DermaScan — Escáner Inteligente de Lunares")
    print(f"  URL: http://localhost:{FLASK_PORT}")
    print(f"  HAM10000: {'✅ Cargado' if ham10000_path else '❌ No disponible'}")
    print(f"  CNN: {'✅ Cargada' if classifier.loaded else '❌ Modelo no encontrado'}")
    # Inicialización del Extractor de Síntomas (ML)
    if not symptom_classifier.is_trained:
        print("[App] Entrenando extractor de síntomas por primera vez...")
        train_symptom_extractor()
    else:
        print("[App] Extractor de síntomas (ML) cargado y listo.")

    print(f"{'='*60}\n")

    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT,
                 debug=FLASK_DEBUG, allow_unsafe_werkzeug=True)
