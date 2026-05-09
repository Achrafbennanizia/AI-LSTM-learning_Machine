"""
GuitarAI — Deep Q-Learning Architektur
Behandelt die Übungsempfehlung als Reinforcement-Learning-Problem.

Konzept:
  State  → Leo's aktueller Skill-Zustand (Feature-Vektor)
  Action → Welches Katalogelement empfehlen (15 Aktionen)
  Reward → Verbesserung der Fehlerraten in der nächsten Session

Warum RL für dieses Problem?
  - Klassisches Supervised Learning braucht "richtige Antworten" als Labels.
    Beim echten Gerät weiß man nie sicher, welche Übung für Leo optimal ist.
  - RL lernt durch Versuch & Irrtum: schlechte Empfehlung → negativer Reward,
    gute Empfehlung → positiver Reward → Policy verbessert sich automatisch.
  - Natürlich inkrementell: jede neue Session liefert automatisch einen
    Reward-Signal ohne manuelle Annotation.

Komponenten:
  QNetwork     — Approximiert Q(state, action) für alle 15 Aktionen
  ReplayBuffer — Speichert (s, a, r, s') für stabiles Training (= Experience Replay)
  DQNAgent     — Epsilon-Greedy Policy + Training Loop
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ── Q-Netzwerk ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Approximiert Q(state, action) für alle Aktionen gleichzeitig.
    Output-Dim = Anzahl Katalog-Einträge (15).
    """

    def __init__(self, state_dim=17, n_actions=15, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """state: (batch, state_dim) → Q-Werte: (batch, n_actions)"""
        return self.net(state)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Experience Replay Buffer: (state, action, reward, next_state, done).

    Doppelter Zweck:
    1) Stabilisiert DQN-Training (bricht zeitliche Korrelationen auf)
    2) Dient gleichzeitig als Experience-Replay für Catastrophic Forgetting —
       alte Erfahrungen bleiben im Buffer und werden regelmäßig neu gesampelt.
    """

    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states)),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(np.array(next_states)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """Double DQN mit Epsilon-Greedy Exploration und Target Network.

    Double DQN (Verbesserung gegenüber vanilla DQN):
      - Online-Netz wählt die beste Aktion
      - Target-Netz bewertet diese Aktion
      → Reduziert Überschätzung der Q-Werte erheblich

    Epsilon-Greedy:
      - Mit Wahrscheinlichkeit ε: zufällige Empfehlung (Exploration)
      - Mit Wahrscheinlichkeit 1-ε: beste bekannte Empfehlung (Exploitation)
      - ε sinkt über die Zeit: erst viel erkunden, dann exploit
    """

    def __init__(
        self,
        state_dim=17,
        n_actions=15,
        hidden_dims=(128, 64),
        dropout=0.2,
        lr=1e-3,
        gamma=0.95,            # Discount-Faktor: wie viel zählt zukünftiger Reward
        epsilon_start=1.0,     # Anfangs viel erkunden
        epsilon_end=0.05,      # Mindest-Exploration
        epsilon_decay=0.995,   # Wie schnell sinkt ε
        buffer_capacity=10_000,
        batch_size=64,
        target_update_freq=10, # Alle N Schritte: Target-Netz synchronisieren
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.steps = 0

        # Online-Netz (wird trainiert) + Target-Netz (stabil, wird periodisch kopiert)
        self.online_net = QNetwork(state_dim, n_actions, hidden_dims, dropout).to(device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dims, dropout).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── Aktion wählen ─────────────────────────────────────────────────────────
    def select_action(self, state):
        """Epsilon-Greedy: erkunden oder besten Q-Wert nehmen."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.online_net(s).argmax(dim=1).item()

    # ── Reward berechnen ─────────────────────────────────────────────────────
    @staticmethod
    def compute_reward(prev_errors, next_errors, weights=None):
        """Reward = gewichtete Verbesserung der Fehlerraten.

        Positiver Reward wenn Fehler sinken, negativer wenn sie steigen.
        Gewichtung erlaubt z.B. Griff-Fehler stärker zu gewichten als Timing.

        Args:
            prev_errors: np.array (5,) — Fehlerraten der vorherigen Session
            next_errors: np.array (5,) — Fehlerraten der neuen Session
            weights: np.array (5,) — Gewichtung pro Fehlerklasse
        """
        if weights is None:
            weights = np.array([1.0, 1.0, 1.0, 1.0, 0.8])  # e_griff..e_muting
        delta = prev_errors - next_errors   # positiv = Verbesserung
        reward = float(np.dot(weights, delta))
        # Clipping verhindert extreme Rewards die Training destabilisieren
        return float(np.clip(reward, -2.0, 2.0))

    # ── Einen Lernschritt ─────────────────────────────────────────────────────
    def learn(self):
        """Einen Gradient-Schritt auf einem gesampelten Minibatch."""
        if len(self.buffer) < self.batch_size:
            return None   # Zu wenig Daten → überspringen

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Double DQN: Online-Netz wählt Aktion, Target-Netz bewertet sie
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            target_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            y = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, y)   # Huber Loss: robuster als MSE

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Epsilon-Decay: weniger erkunden mit der Zeit
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Target-Netz periodisch synchronisieren
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def push(self, state, action, reward, next_state, done=False):
        """Erfahrung in den Buffer legen."""
        self.buffer.push(state, action, reward, next_state, done)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.online_net.parameters())
