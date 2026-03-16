"""
Agente DQN (Deep Q-Network) con PyTorch para diagnóstico dermatológico.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DRL_CONFIG, ACTION_NAMES


class DQNetwork(nn.Module):
    """Red neuronal profunda para estimar Q-values."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Agente DQN completo para diagnóstico dermatológico."""

    def __init__(self, state_size=None, action_size=None, config=None):
        self.config = config or DRL_CONFIG
        self.state_size = state_size or self.config["state_size"]
        self.action_size = action_size or self.config["action_size"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Dispositivo: {self.device}")

        # Redes Q-network y Target network
        self.q_network = DQNetwork(
            self.state_size, self.action_size, self.config["hidden_size"]
        ).to(self.device)

        self.target_network = DQNetwork(
            self.state_size, self.action_size, self.config["hidden_size"]
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizador
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config["learning_rate"]
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss para estabilidad

        # Replay buffer
        self.memory = ReplayBuffer(self.config["buffer_size"])

        # Exploración epsilon-greedy
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_min = self.config["epsilon_end"]
        self.epsilon_decay = self.config["epsilon_decay"]

        # Métricas
        self.steps_done = 0
        self.training_losses = []
        self.episode_rewards = []

    def select_action(self, state, training=True):
        """Selecciona acción con epsilon-greedy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Paso de aprendizaje con experience replay."""
        if len(self.memory) < self.config["batch_size"]:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config["batch_size"]
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-values actuales
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config["gamma"] * next_q

        # Loss y backprop
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)

        # Actualizar target network
        self.steps_done += 1
        if self.steps_done % self.config["target_update"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath="models/dqn_dermascan.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "losses": self.training_losses[-1000:],
            "rewards": self.episode_rewards,
        }, filepath)
        print(f"[DQN] Modelo guardado en {filepath}")

    def load(self, filepath="models/dqn_dermascan.pth"):
        if not os.path.exists(filepath):
            print(f"[DQN] No se encontró modelo en {filepath}")
            return False
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.training_losses = checkpoint.get("losses", [])
        self.episode_rewards = checkpoint.get("rewards", [])
        print(f"[DQN] Modelo cargado (epsilon={self.epsilon:.4f})")
        return True

    def predict(self, state):
        """Predice la mejor acción sin exploración."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).cpu().numpy()[0]

        action = int(np.argmax(q_values))
        return {
            "action": action,
            "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
            "q_values": {
                ACTION_NAMES[i]: round(float(q_values[i]), 4)
                for i in range(len(q_values))
            },
            "confidence": round(float(np.max(q_values) - np.mean(q_values)), 4),
        }
