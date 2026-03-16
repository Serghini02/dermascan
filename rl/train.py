"""
Entrenamiento del agente DQN para diagnóstico dermatológico.
Usa TODOS los datos reales de HAM10000 — iteración completa por la BD.
"""

import sys
import os
import json
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import DermaDiagnosisEnv
from rl.dqn_agent import DQNAgent
from database.db_manager import DatabaseManager
from config import DRL_CONFIG, DATABASE_PATH, MODELS_DIR, CLASS_TO_IDX


def load_patient_data(db_path=None):
    """Carga TODOS los datos de pacientes desde la BD (HAM10000 metadata)."""
    db = DatabaseManager(db_path or DATABASE_PATH)
    data = db.get_ham10000_data()

    if not data:
        print("[Train DRL] No hay datos HAM10000. Generando sintéticos...")
        data = []
        for _ in range(2000):
            dx = random.choice(list(CLASS_TO_IDX.keys()))
            data.append({
                "dx": dx,
                "age": random.uniform(20, 80),
                "sex": random.choice(["male", "female"]),
                "localization": random.choice([
                    "back", "trunk", "upper extremity", "lower extremity",
                    "face", "scalp", "abdomen", "chest",
                ]),
            })
        print(f"[Train DRL] Generados {len(data)} pacientes sintéticos.")
        return data

    print(f"[Train DRL] Cargados {len(data)} pacientes reales de HAM10000.")
    return data


def train_agent(max_episodes=None, callback=None, db_path=None):
    """
    Entrena el agente DQN de forma exhaustiva.
    
    El entrenamiento recorre TODOS los pacientes de HAM10000 en múltiples épocas.
    En cada episodio:
      1. Se selecciona un paciente real de la BD
      2. Se simulan las probabilidades CNN basadas en su diagnóstico real
      3. El agente decide qué preguntas hacer (max 6 síntomas)
      4. Al diagnosticar, se recompensa/penaliza según acierto
      5. Se ejecutan múltiples pasos de aprendizaje por episodio
    """
    # Cargar TODA la base de datos de pacientes
    patient_data = load_patient_data(db_path)
    total_patients = len(patient_data)

    # Calcular número de episodios: como mínimo recorrer TODA la BD varias veces
    min_episodes = total_patients * 3  # 3 épocas completas sobre la BD
    max_episodes = max(max_episodes or DRL_CONFIG["max_episodes"], min_episodes)

    env = DermaDiagnosisEnv(patient_data=patient_data)
    agent = DQNAgent()
    
    # Intentar cargar modelo previo para continuar entrenamiento
    agent.load()

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO DQN — Diagnóstico Dermatológico")
    print(f"  Pacientes en BD: {total_patients}")
    print(f"  Episodios totales: {max_episodes}")
    print(f"  Épocas sobre BD: {max_episodes / total_patients:.1f}x")
    print(f"  Steps de aprendizaje por episodio: 4")
    print(f"  Device: {agent.device}")
    print(f"{'='*60}\n")

    rewards_history = []
    losses_history = []
    best_reward = float("-inf")
    best_avg = float("-inf")

    # Crear índice secuencial para asegurar que vemos TODOS los pacientes
    patient_order = list(range(total_patients))

    for episode in range(max_episodes):
        # Rotar por todos los pacientes secuencialmente (como épocas reales)
        if episode % total_patients == 0:
            random.shuffle(patient_order)
            epoch_num = episode // total_patients + 1
            if epoch_num <= 10 or epoch_num % 5 == 0:
                print(f"\n  📦 Época {epoch_num} — barajando {total_patients} pacientes...")

        # Seleccionar paciente de la BD (secuencial, no aleatorio)
        patient_idx = patient_order[episode % total_patients]
        env.current_patient = patient_data[patient_idx]

        # Reset con el paciente específico
        state, _ = env.reset()
        # Forzar el paciente seleccionado (reset puede haberlo cambiado)
        env.current_patient = patient_data[patient_idx]
        env.cnn_probs = env._generate_cnn_probs(env.current_patient["dx"])
        state = env._get_observation()

        episode_reward = 0
        episode_losses = []
        steps = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)

            # Múltiples pasos de aprendizaje por step (más aprendizaje)
            for _ in range(4):
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        rewards_history.append(episode_reward)
        losses_history.append(avg_loss)
        agent.episode_rewards.append(episode_reward)

        # Guardar mejor modelo por reward promedio (más estable)
        if len(rewards_history) >= 50:
            current_avg = np.mean(rewards_history[-50:])
            if current_avg > best_avg:
                best_avg = current_avg
                agent.save("models/dqn_dermascan_best.pth")

        if episode_reward > best_reward:
            best_reward = episode_reward

        # Callback para la UI
        if callback:
            callback(
                episode=episode, total_episodes=max_episodes,
                reward=episode_reward,
                avg_reward=np.mean(rewards_history[-50:]) if rewards_history else 0,
                loss=avg_loss, epsilon=agent.epsilon,
                progress=round((episode + 1) / max_episodes * 100, 1),
            )

        # Log cada 100 episodios o cada nueva época
        if (episode + 1) % 100 == 0 or (episode + 1) == max_episodes:
            avg = np.mean(rewards_history[-100:])
            patients_seen = min(episode + 1, total_patients)
            epochs_done = (episode + 1) / total_patients
            print(f"  Ep {episode+1:5d}/{max_episodes} | "
                  f"Reward: {episode_reward:7.1f} | Avg(100): {avg:7.1f} | "
                  f"Loss: {avg_loss:.5f} | ε: {agent.epsilon:.4f} | "
                  f"Épocas: {epochs_done:.1f}")

        # Checkpoint intermedio cada 500 episodios
        if (episode + 1) % 500 == 0:
            agent.save("models/dqn_dermascan.pth")
            print(f"  💾 Checkpoint guardado (ep {episode+1})")

    # Guardar modelo final
    agent.save("models/dqn_dermascan.pth")

    results = {
        "total_episodes": max_episodes,
        "total_patients_in_db": total_patients,
        "epochs_over_db": round(max_episodes / total_patients, 1),
        "best_reward": float(best_reward),
        "best_avg_reward_50": float(best_avg),
        "final_avg_reward": float(np.mean(rewards_history[-100:])),
        "final_epsilon": agent.epsilon,
        "rewards_history": [float(r) for r in rewards_history],
        "losses_history": [float(l) for l in losses_history],
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "drl_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Completado.")
    print(f"     Pacientes procesados: {total_patients}")
    print(f"     Épocas completas: {max_episodes / total_patients:.1f}x")
    print(f"     Mejor reward promedio: {best_avg:.1f}")
    print(f"     Mejor reward individual: {best_reward:.1f}\n")

    return results


def evaluate_agent(num_episodes=100, db_path=None):
    """
    Evalúa el agente sin exploración sobre datos reales.
    Por defecto 100 episodios para una evaluación fiable.
    """
    patient_data = load_patient_data(db_path)
    env = DermaDiagnosisEnv(patient_data=patient_data)
    agent = DQNAgent()
    if not agent.load("models/dqn_dermascan_best.pth"):
        if not agent.load():
            return {"error": "No hay modelo entrenado"}

    results = {
        "correct_diagnoses": 0, "correct_risk": 0, "total": 0,
        "false_negatives": 0, "false_positives": 0,
        "avg_questions": 0, "rewards": [],
        "per_class": {},
    }
    total_questions = 0

    # Evaluar con pacientes reales
    eval_patients = random.sample(patient_data, min(num_episodes, len(patient_data)))

    for patient in eval_patients:
        env.current_patient = patient
        state, _ = env.reset()
        env.current_patient = patient
        env.cnn_probs = env._generate_cnn_probs(patient["dx"])
        state = env._get_observation()

        ep_reward = 0
        dx = patient["dx"]

        while True:
            pred = agent.predict(state)
            next_state, reward, term, trunc, info = env.step(pred["action"])
            ep_reward += reward

            if info.get("result") == "DIAGNOSTICO_CORRECTO":
                results["correct_diagnoses"] += 1
            elif info.get("result") == "RIESGO_CORRECTO":
                results["correct_risk"] += 1
            elif info.get("result") == "ERROR_GRAVE_FALSO_NEGATIVO":
                results["false_negatives"] += 1
            elif info.get("result") == "ERROR_FALSO_POSITIVO":
                results["false_positives"] += 1

            state = next_state
            if term or trunc:
                results["total"] += 1
                total_questions += env.questions_asked

                # Estadísticas por clase
                if dx not in results["per_class"]:
                    results["per_class"][dx] = {"total": 0, "correct": 0}
                results["per_class"][dx]["total"] += 1
                if info.get("result") in ("DIAGNOSTICO_CORRECTO", "RIESGO_CORRECTO"):
                    results["per_class"][dx]["correct"] += 1
                break

        results["rewards"].append(ep_reward)

    total = max(results["total"], 1)
    results["accuracy"] = round(results["correct_diagnoses"] / total * 100, 1)
    results["risk_accuracy"] = round(
        (results["correct_diagnoses"] + results["correct_risk"]) / total * 100, 1)
    results["avg_questions"] = round(total_questions / total, 1)
    results["avg_reward"] = round(float(np.mean(results["rewards"])), 2)

    # Accuracy por clase
    for dx, cls_data in results["per_class"].items():
        cls_data["accuracy"] = round(cls_data["correct"] / max(cls_data["total"], 1) * 100, 1)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodios de entrenamiento (default: 3x total pacientes)")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.evaluate:
        r = evaluate_agent(num_episodes=100)
        print(f"\n📊 Accuracy exacta: {r['accuracy']}%")
        print(f"📊 Accuracy riesgo: {r['risk_accuracy']}%")
        print(f"📊 Avg preguntas: {r['avg_questions']}")
        print(f"📊 Falsos negativos: {r['false_negatives']}")
        print(f"\n📊 Accuracy por clase:")
        for dx, d in r.get("per_class", {}).items():
            print(f"   {dx}: {d['accuracy']}% ({d['correct']}/{d['total']})")
    else:
        train_agent(max_episodes=args.episodes)
