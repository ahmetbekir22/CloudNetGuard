"""
CloudNetGuard — SDN politika motoru.
Kural tabanlı karar verici + opsiyonel basit Q-table RL.
"""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict

from actions import Action, ActionResult, HONEYPOT_IP

log = logging.getLogger(__name__)

POLICY_MODE = os.environ.get("POLICY", "rules")   # rules | rl


# ---------------------------------------------------------------------------
# Kural tabanlı politika
# ---------------------------------------------------------------------------

class RulePolicy:
    """
    Eşik değerlerine ve tahmin edilen tehdit tipine göre aksiyon seç.
    """

    def decide(self, anomaly_score: float, predicted_type: str) -> ActionResult:
        score = anomaly_score

        if score > 0.9 and predicted_type == "ddos":
            return ActionResult(
                action=Action.BLOCK,
                reason=f"DDoS saldırısı yüksek güven ile tespit edildi (skor={score:.2f})",
            )
        if score > 0.85 and predicted_type == "tunnel":
            return ActionResult(
                action=Action.REDIRECT,
                reason=f"DNS tünelleme tespit edildi (skor={score:.2f})",
                target=HONEYPOT_IP,
            )
        if score > 0.8 and predicted_type == "flux":
            return ActionResult(
                action=Action.BLOCK,
                reason=f"Domain flux tespit edildi (skor={score:.2f})",
            )
        if score > 0.6:
            return ActionResult(
                action=Action.MIRROR,
                reason=f"Şüpheli trafik izleniyor (skor={score:.2f})",
            )
        return ActionResult(
            action=Action.ALLOW,
            reason=f"Normal trafik (skor={score:.2f})",
        )


# ---------------------------------------------------------------------------
# Q-table RL politikası (basit, offline eğitimli)
# ---------------------------------------------------------------------------

class QLearningPolicy:
    """
    State: (anomaly_type_idx, score_bin)
    Actions: BLOCK(0), REDIRECT(1), MIRROR(2), ALLOW(3)
    """

    ACTIONS = [Action.BLOCK, Action.REDIRECT, Action.MIRROR, Action.ALLOW]
    TYPE_IDX = {"normal": 0, "tunnel": 1, "ddos": 2, "flux": 3}
    SCORE_BINS = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.05) -> None:
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        n_states = len(self.TYPE_IDX) * (len(self.SCORE_BINS) - 1)
        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * 4)
        self._pretrain()

    def _state(self, predicted_type: str, score: float) -> tuple:
        type_idx = self.TYPE_IDX.get(predicted_type, 0)
        bin_idx  = 0
        for i in range(len(self.SCORE_BINS) - 1):
            if self.SCORE_BINS[i] <= score < self.SCORE_BINS[i + 1]:
                bin_idx = i
                break
        return (type_idx, bin_idx)

    def _pretrain(self) -> None:
        """Q-tabloyu heuristic değerlerle önceden doldur."""
        rewards = {
            # (type_idx, bin_idx): {action_idx: reward}
            (2, 4): {0: 1.0},   # ddos yüksek → BLOCK
            (1, 3): {1: 1.0},   # tunnel orta-yüksek → REDIRECT
            (3, 2): {2: 0.8},   # flux orta → MIRROR
            (0, 0): {3: 1.0},   # normal düşük → ALLOW
        }
        for state, action_rewards in rewards.items():
            for a_idx, r in action_rewards.items():
                self.q_table[state][a_idx] = r

    def decide(self, anomaly_score: float, predicted_type: str) -> ActionResult:
        state = self._state(predicted_type, anomaly_score)
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = int(max(range(4), key=lambda i: self.q_table[state][i]))
        action = self.ACTIONS[action_idx]
        return ActionResult(
            action=action,
            reason=f"RL kararı (state={state}, q={self.q_table[state][action_idx]:.2f})",
            target=HONEYPOT_IP if action == Action.REDIRECT else "",
        )

    def update(self, state, action_idx: int, reward: float, next_state) -> None:
        """Online Q-update (opsiyonel feedback döngüsü için)."""
        q_sa = self.q_table[state][action_idx]
        max_q_next = max(self.q_table[next_state])
        self.q_table[state][action_idx] = q_sa + self.alpha * (
            reward + self.gamma * max_q_next - q_sa
        )


# ---------------------------------------------------------------------------
# Politika fabrika
# ---------------------------------------------------------------------------

def get_policy():
    if POLICY_MODE == "rl":
        log.info("RL politikası kullanılıyor")
        return QLearningPolicy()
    log.info("Kural tabanlı politika kullanılıyor")
    return RulePolicy()
