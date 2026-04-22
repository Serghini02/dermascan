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

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
import tempfile
import subprocess
import platform
import asyncio
import edge_tts

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

# Almacén de sesiones por Socket ID (multi-usuario)
sessions_store = {}

def get_session(sid):
    if sid not in sessions_store:
        sessions_store[sid] = {
            "active": False,
            "cnn_result": None,
            "symptoms": {},
            "questions_asked": [],
            "abcde_scores": None,
            "image_data": None,
        }
    return sessions_store[sid]

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

def background_heavy_analysis(session_id, img_pil, img_bgr, image_b64):
    """Tarea de fondo aislada por sala de sesión."""
    try:
        def progress_cb(percent):
            # Notificar solo al dispositivo que inició la sesión
            socketio.emit('scan_progress', {'progress': percent}, to=session_id)
            
        # 1. CNN Refinada
        cnn_result = classifier.predict(img_pil, n_passes=300, callback=progress_cb)
        
        # 2. ABCDE Refinado
        abcde = analyze_mole(img_bgr, n_passes=300, callback=progress_cb)
        
        # Actualizar sesión específica
        sess = get_session(session_id)
        sess["cnn_result"] = cnn_result
        sess["abcde_scores"] = abcde
        
        # Notificar fin mediante WebSockets a la sala privada
        socketio.emit('scan_complete', {
            'cnn': cnn_result,
            'abcde': abcde,
            'symptom_summary': get_symptom_summary(sess.get("symptoms", {}))
        }, to=session_id)
        print(f"[App] Análisis completado para sesión: {session_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        socketio.emit('scan_error', {'error': str(e)}, to=session_id)


@socketio.on('join_session')
def handle_join(data):
    """Une al socket a una sala privada basada en su ID de dispositivo."""
    session_id = data.get('session_id')
    if session_id:
        import flask
        from flask_socketio import join_room
        join_room(session_id)
        print(f"[WS] Socket {request.sid} se unió a la sala: {session_id}")

@app.route('/api/scan', methods=['POST'])
def scan_image():
    """Escaneo rápido inicial + Lanzamiento de análisis pesado en hilos."""
    data = request.json
    image_b64 = data.get("image", "")
    session_id = data.get("session_id") # Identificador persistente
    
    if not image_b64:
        return jsonify({"error": "No se recibió imagen"}), 400
    if not session_id:
        return jsonify({"error": "No se recibió Session ID"}), 400

    try:
        # Decodificar imagen base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # EVALUACIÓN RÁPIDA (1 pasada) para decidir la primera pregunta de inmediato
        cnn_fast = classifier.predict(img_pil, n_passes=1)
        abcde_fast = analyze_mole(img_bgr, n_passes=1)

        # Filtro de Seguridad rápido
        if abcde_fast.get("risk") == "no_detectado":
             return jsonify({
                 "error": "No se detecta un lunar claro. Por favor, encuadre la lesión en el círculo y asegure buena iluminación."
             }), 422

        # Inicializar sesión aislada
        sess = get_session(session_id)
        sess["active"] = True
        sess["cnn_result"] = cnn_fast
        sess["abcde_scores"] = abcde_fast
        sess["symptoms"] = {}
        sess["questions_asked"] = []
        sess["image_data"] = image_b64

        # Lanzar proceso pesado
        threading.Thread(target=background_heavy_analysis, args=(session_id, img_pil, img_bgr, image_b64)).start()

        # Consultar DRL para la PRIMERA PREGUNTA
        # El DRL se ejecuta de forma síncrona para la primera respuesta
        state = _build_state(session_id)
        drl_prediction = dqn_agent.predict(state)
        # Aplicar reglas de flujo para forzar que la primera acción sea una pregunta
        drl_prediction = apply_flow_rules(drl_prediction, [])
        
        return jsonify({
            "status": "in_progress",
            "cnn": cnn_fast,
            "abcde": abcde_fast,
            "next_action": drl_prediction,
            "next_question": _get_question_text(drl_prediction["action"]),
            "symptom_summary": get_symptom_summary({}),
            "message": "Analyzing image in detail..."
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
    try:
        data = request.json
        text = data.get("text", "")
        question_id = data.get("question_id", "")
        session_id = data.get("session_id") # Identificador persistente
        
        if not text.strip():
            return jsonify({"error": "Texto vacío"}), 400
        if not session_id:
            return jsonify({"error": "No se recibió Session ID"}), 400

        sess = get_session(session_id)
        
        # 1. Corrección regex
        correction = correct_text(text)

        # 2. Tokenización
        token_result = tokenizer.tokenize(correction["corrected"])

        # 3. Extracción de síntomas
        symptoms = extract_symptoms(correction["corrected"], context_symptom=question_id)

        # 4. Actualizar sesión específica
        sess["symptoms"].update(symptoms)
        if question_id:
            sess["questions_asked"].append(question_id)

        # 5. Construir estado y preguntar al DRL qué hacer después
        state = _build_state(session_id)
        drl_pred = dqn_agent.predict(state)

        # Aplicar reglas de flujo (Obligar preguntas, evitar repetidas)
        drl_pred = apply_flow_rules(drl_pred, sess["questions_asked"])

        # ¿Es diagnóstico final?
        is_final = drl_pred["action"] == 6
        diagnosis = None
        if is_final:
            diagnosis = _finalize_diagnosis(session_id)

        return jsonify({
            "correction": correction,
            "tokens": token_result,
            "symptoms": symptoms,
            "symptom_summary": get_symptom_summary(sess["symptoms"]),
            "next_action": drl_pred,
            "next_question": _get_question_text(drl_pred["action"]) if not is_final else "",
            "is_final": is_final,
            "diagnosis": diagnosis,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR /api/voice/process] question_id={data.get('question_id','?')} text={data.get('text','?')[:80]}: {e}")
        return jsonify({"error": str(e)}), 500



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
        return jsonify({"status": "success", "message": "Symptom extractor trained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Caché para evitar regenerar el mismo audio
TTS_CACHE = {}

@app.route('/api/tts')
def text_to_speech():
    """Genera audio usando Edge-TTS con caché y procesamiento en memoria."""
    text = request.args.get("text", "").strip()
    lang = request.args.get("lang", "en")
    if not text:
        return "No text provided", 400
    
    # Crear una clave para la caché
    cache_key = f"{lang}:{text}"
    if cache_key in TTS_CACHE:
        return send_file(BytesIO(TTS_CACHE[cache_key]), mimetype='audio/mpeg')
    
    # Seleccionar voz neuronal
    voice = "en-US-GuyNeural" if lang.startswith("en") else "es-ES-AlvaroNeural"
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = []
        
        async def _get_data():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.append(chunk["data"])
        
        asyncio.run(_get_data())
        
        # Unir los trozos de audio
        final_audio = b"".join(audio_data)
        
        # Guardar en caché (limitar tamaño si es necesario, aquí simple)
        if len(TTS_CACHE) > 100: # Evitar consumo excesivo de RAM
            TTS_CACHE.clear()
        TTS_CACHE[cache_key] = final_audio
        
        return send_file(BytesIO(final_audio), mimetype='audio/mpeg')
    except Exception as e:
        return str(e), 500


@app.route('/api/nlp/status', methods=['GET'])
def nlp_status():
    """Estado del extractor de síntomas ML."""
    return jsonify({
        "is_trained": symptom_classifier.is_trained,
        "model_exists": os.path.exists(symptom_classifier.model_path),
        "num_symptoms": len(symptom_classifier.symptoms) if symptom_classifier.is_trained else 0,
    })


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
    return jsonify(evaluate_agent(num_episodes=50, db_path=DATABASE_PATH))


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
    
    # Añadir configuración técnica para el dashboard
    status["config"] = {
        "lr": dqn_agent.config.get("learning_rate"),
        "gamma": dqn_agent.config.get("gamma"),
        "batch_size": dqn_agent.config.get("batch_size"),
        "epsilon_decay": dqn_agent.config.get("epsilon_decay")
    }
    return jsonify(status)


# =============================================================================
# RUTAS — HISTORIAL
# =============================================================================

@app.route('/api/history', methods=['GET'])
def get_history():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(db.get_consultations(limit=limit))


@app.route('/api/history/<int:consultation_id>', methods=['DELETE'])
def delete_history(consultation_id):
    """Elimina una consulta del historial."""
    success = db.delete_consultation(consultation_id)
    if success:
        return jsonify({"status": "deleted", "id": consultation_id})
    return jsonify({"error": "Consultation not found"}), 404


@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    return jsonify(db.get_ham10000_stats())


@app.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results():
    """Sirve los resultados existentes."""
    path = os.path.join(os.path.dirname(__file__), "evaluation", "evaluation_results.json")
    if not os.path.exists(path):
        return jsonify({"error": "No results. Run the evaluation first."}), 404
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

def apply_flow_rules(drl_pred, asked_ids_list):
    """
    Asegura que el flujo de preguntas sea coherente.
    - Obliga a preguntar un mínimo de síntomas.
    - Evita repeticiones.
    """
    asked_ids = set(asked_ids_list)
    questions_done = len(asked_ids)
    MIN_QUESTIONS = 6 # Forzar valoración completa de los 6 síntomas principales

    curr_action = drl_pred["action"]
    symptom_ids = ["pain", "itching", "size", "bleeding", "color", "duration"]
    
    # Regla 1: Si intenta diagnosticar pero faltan preguntas, forzar pregunta
    if curr_action in (6, 7) and questions_done < MIN_QUESTIONS:
        next_action_idx = next(
            (i for i in range(6) if symptom_ids[i] not in asked_ids),
            6
        )
        drl_pred["action"] = next_action_idx
        drl_pred["action_name"] = ACTION_NAMES.get(next_action_idx, "")
    
    # Regla 2: Evitar repetir preguntas
    elif curr_action <= 5:
        chosen_id = symptom_ids[curr_action]
        if chosen_id in asked_ids:
            next_action_idx = next(
                (i for i in range(6) if symptom_ids[i] not in asked_ids),
                6
            )
            drl_pred["action"] = next_action_idx
            drl_pred["action_name"] = ACTION_NAMES.get(next_action_idx, "")

    return drl_pred


def _build_state(sid):
    """Construye el vector de estado para el DRL."""
    sess = get_session(sid)
    cnn_probs = [0.0] * 7
    if sess.get("cnn_result"):
        cnn_probs = sess["cnn_result"].get("prob_vector", [0.0] * 7)

    symptom_vec = symptoms_to_vector(sess.get("symptoms", {}))
    num_questions = len(sess.get("questions_asked", [])) / 6.0

    state = cnn_probs + symptom_vec + [num_questions]
    return np.array(state, dtype=np.float32)


def _get_question_text(action):
    """Obtains the question text for a given action."""
    if action <= 5:
        return SYMPTOM_QUESTIONS[action]["text"]
    elif action == 6:
        return "The system is ready to provide a diagnosis."
    elif action == 7:
        return "It is recommended to take another photograph of the lesion."
    return ""


def _finalize_diagnosis(sid):
    """Genera el diagnóstico final con toda la info acumulada."""
    sess = get_session(sid)
    cnn = sess.get("cnn_result", {})
    abcde = sess.get("abcde_scores", {})
    symptoms = sess.get("symptoms", {})

    # Generar diagnóstico refinado mediante el Sistema Experto (Reglas)
    expert_result = expert_system.apply_rules(cnn, abcde, symptoms)

    # Guardar en BD (incluyendo la imagen de la sesión)
    db.add_consultation(
        cnn_diagnosis=cnn.get("diagnosis_code"),
        image_data=sess.get("image_data"),
        cnn_confidence=cnn.get("confidence", 0),
        cnn_probabilities=cnn.get("probabilities"),
        symptoms=symptoms,
        abcde_scores=abcde,
        drl_diagnosis=cnn.get("diagnosis_code"),
        risk_level=expert_result["refined_risk"],
        questions_asked=len(sess.get("questions_asked", [])),
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
