"""
Entorno Gymnasium para diagnóstico dermatológico secuencial.
El agente decide qué preguntas hacer al paciente y cuándo emitir diagnóstico.
Los datos provienen del dataset real HAM10000.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DRL_CONFIG, DIAGNOSIS_CLASSES, CLASS_TO_IDX, SYMPTOM_QUESTIONS


# Probabilidades de síntomas por tipo de lesión (basadas en literatura médica)
SYMPTOM_PROBABILITIES = {
    "mel": {"dolor": 0.3, "picor": 0.4, "tamaño": 0.8, "sangrado": 0.5, "color": 0.85, "duracion": 0.7},
    "bcc": {"dolor": 0.2, "picor": 0.3, "tamaño": 0.6, "sangrado": 0.4, "color": 0.5, "duracion": 0.5},
    "akiec": {"dolor": 0.15, "picor": 0.35, "tamaño": 0.3, "sangrado": 0.2, "color": 0.3, "duracion": 0.4},
    "bkl": {"dolor": 0.05, "picor": 0.15, "tamaño": 0.2, "sangrado": 0.05, "color": 0.2, "duracion": 0.3},
    "df": {"dolor": 0.1, "picor": 0.1, "tamaño": 0.1, "sangrado": 0.05, "color": 0.1, "duracion": 0.2},
    "nv": {"dolor": 0.02, "picor": 0.05, "tamaño": 0.05, "sangrado": 0.02, "color": 0.05, "duracion": 0.1},
    "vasc": {"dolor": 0.1, "picor": 0.1, "tamaño": 0.15, "sangrado": 0.3, "color": 0.15, "duracion": 0.2},
}

SYMPTOM_KEYS = ["dolor", "picor", "tamaño", "sangrado", "color", "duracion"]


class DermaDiagnosisEnv(gym.Env):
    """
    Entorno de diagnóstico dermatológico para DRL.

    Estado (14 features):
        [0:7]  — Probabilidades CNN (7 clases)
        [7:13] — Síntomas: -1=no preguntado, 0=negativo, 1=positivo
        [13]   — Nº preguntas realizadas / max_preguntas

    Acciones (8):
        0-5: Preguntar por un síntoma específico
        6:   Emitir diagnóstico (fin episodio)
        7:   Pedir otra foto (obtener nueva predicción CNN)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, patient_data=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(DRL_CONFIG["state_size"],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(DRL_CONFIG["action_size"])

        self.patient_data = patient_data or []
        self.current_patient = None
        self.current_idx = 0

        # Estado del episodio
        self.cnn_probs = np.zeros(7, dtype=np.float32)
        self.symptoms = np.full(6, -1.0, dtype=np.float32)  # -1 = no preguntado
        self.questions_asked = 0
        self.max_questions = 6
        self.already_asked = set()

    def _get_observation(self):
        return np.concatenate([
            self.cnn_probs,
            self.symptoms,
            [self.questions_asked / self.max_questions]
        ]).astype(np.float32)

    def _generate_cnn_probs(self, true_dx):
        """Simula output de CNN para un diagnóstico real."""
        idx = CLASS_TO_IDX.get(true_dx, 5)
        # Distribución con pico en la clase correcta (simula CNN imperfecta)
        alpha = np.ones(7) * 0.3
        alpha[idx] = 3.0  # Más probabilidad en la clase correcta
        probs = np.random.dirichlet(alpha)
        return probs.astype(np.float32)

    def _generate_symptom(self, true_dx, symptom_key):
        """Genera síntoma basado en probabilidades reales de la literatura."""
        prob = SYMPTOM_PROBABILITIES.get(true_dx, {}).get(symptom_key, 0.1)
        return 1.0 if random.random() < prob else 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.patient_data:
            self.current_idx = random.randint(0, len(self.patient_data) - 1)
            self.current_patient = self.patient_data[self.current_idx]
        else:
            # Paciente sintético
            dx_codes = list(CLASS_TO_IDX.keys())
            self.current_patient = {
                "dx": random.choice(dx_codes),
                "age": random.uniform(20, 80),
                "sex": random.choice(["male", "female"]),
            }

        # Simular output de CNN
        self.cnn_probs = self._generate_cnn_probs(self.current_patient["dx"])

        # Reset síntomas
        self.symptoms = np.full(6, -1.0, dtype=np.float32)
        self.questions_asked = 0
        self.already_asked = set()

        return self._get_observation(), {}

    def step(self, action):
        true_dx = self.current_patient["dx"]
        is_malignant = true_dx in ("mel", "bcc", "akiec")
        reward = 0.0
        terminated = False
        info = {"action": action, "true_dx": true_dx}

        if action <= 5:  # PREGUNTAR SÍNTOMA
            symptom_key = SYMPTOM_KEYS[action]

            if action in self.already_asked:
                # Penalizar preguntas repetidas
                reward = -0.5
                info["result"] = "PREGUNTA_REPETIDA"
            else:
                # Simular respuesta del paciente
                answer = self._generate_symptom(true_dx, symptom_key)
                self.symptoms[action] = answer
                self.already_asked.add(action)
                self.questions_asked += 1

                # Recompensa por preguntar algo relevante
                if is_malignant and answer == 1.0:
                    reward = 0.3  # Pregunta reveladora para caso maligno
                else:
                    reward = -0.1  # Coste por cada pregunta

                info["result"] = "PREGUNTA_CONTESTADA"
                info["symptom"] = symptom_key
                info["answer"] = answer

        elif action == 6:  # DIAGNOSTICAR
            # El diagnóstico se basa en la probabilidad más alta
            predicted_idx = int(np.argmax(self.cnn_probs))
            predicted_dx = DIAGNOSIS_CLASSES[predicted_idx]["code"]
            predicted_risk = DIAGNOSIS_CLASSES[predicted_idx]["risk"]
            true_risk = next(d["risk"] for d in DIAGNOSIS_CLASSES.values() if d["code"] == true_dx)

            # Ajustar predicción con síntomas
            symptom_boost = 0
            asked_symptoms = [i for i in range(6) if self.symptoms[i] >= 0]
            positive_symptoms = sum(1 for i in asked_symptoms if self.symptoms[i] == 1.0)

            if positive_symptoms >= 3 and not is_malignant:
                symptom_boost = -1  # Falsa alarma probable
            elif positive_symptoms >= 2 and is_malignant:
                symptom_boost = 2   # Síntomas confirman malignidad

            # Recompensa por diagnóstico
            if predicted_dx == true_dx:
                reward = 5.0 + symptom_boost  # Diagnóstico exacto
                info["result"] = "DIAGNOSTICO_CORRECTO"
            elif predicted_risk == true_risk:
                reward = 2.0 + symptom_boost  # Riesgo correcto al menos
                info["result"] = "RIESGO_CORRECTO"
            elif is_malignant and predicted_risk == "benigno":
                reward = -8.0  # Error grave: maligno clasificado como benigno
                info["result"] = "ERROR_GRAVE_FALSO_NEGATIVO"
            elif not is_malignant and predicted_risk == "maligno":
                reward = -2.0  # Error: benigno clasificado como maligno (menos grave)
                info["result"] = "ERROR_FALSO_POSITIVO"
            else:
                reward = -1.0
                info["result"] = "DIAGNOSTICO_INCORRECTO"

            # Bonus por eficiencia (pocas preguntas)
            if self.questions_asked <= 3:
                reward += 0.5

            info["predicted_dx"] = predicted_dx
            terminated = True

        elif action == 7:  # PEDIR OTRA FOTO
            if self.questions_asked == 0:
                # Primera vez, puede ser útil
                self.cnn_probs = self._generate_cnn_probs(true_dx)
                reward = -0.2
                info["result"] = "NUEVA_FOTO"
            else:
                reward = -0.5  # Costoso pedir otra foto
                info["result"] = "FOTO_EXTRA"

        truncated = self.questions_asked >= self.max_questions and not terminated

        return self._get_observation(), reward, terminated, truncated, info
