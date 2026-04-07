"""
CloudNetGuard — Autoencoder anomali tespit modeli (PyTorch).
12 feature → encode → decode → reconstruction error → anomali skoru.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Simetrik Autoencoder.
    Encoder:  12 → 8 → 4
    Decoder:  4  → 8 → 12
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 4) -> None:
        super().__init__()
        hidden = input_dim * 2 // 3 + latent_dim  # ≈ 8

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction error per sample."""
        with torch.no_grad():
            x_hat = self(x)
            return ((x - x_hat) ** 2).mean(dim=1)

    def anomaly_score(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Reconstruction error'u 0-1 arasına normalize et."""
        err = self.reconstruction_error(x)
        # Yumuşak sigmoid normalizasyonu
        score = torch.sigmoid((err - threshold) * 30.0)
        return score


def build_autoencoder(input_dim: int = 12) -> Autoencoder:
    return Autoencoder(input_dim=input_dim)
