"""
CloudNetGuard — LSTM tabanlı zaman serisi anomali tespit modeli (PyTorch).
Sliding window üzerinde DDoS burst ve domain flux pattern'lerini yakalar.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAnomalyDetector(nn.Module):
    """
    Binary LSTM sınıflandırıcı.
    Input:  (batch, seq_len, input_dim)  — N zaman adımı feature penceresi
    Output: (batch, 1)                  — anomali olasılığı
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]          # son zaman adımı
        return self.classifier(last_hidden)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Döndürür: (olasılık, binary_label)
        """
        with torch.no_grad():
            prob = self(x)
            label = (prob >= threshold).float()
        return prob, label


def build_lstm(input_dim: int = 12, seq_len: int = 20) -> LSTMAnomalyDetector:
    return LSTMAnomalyDetector(input_dim=input_dim)
