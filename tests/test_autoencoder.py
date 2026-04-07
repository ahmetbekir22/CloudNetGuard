"""Unit tests — ai-engine/models/autoencoder.py"""

import sys
import pytest
import torch

sys.path.insert(0, "ai-engine")
from models.autoencoder import Autoencoder, build_autoencoder


def test_forward_shape():
    model = build_autoencoder(input_dim=12)
    x = torch.rand(8, 12)
    out = model(x)
    assert out.shape == (8, 12)


def test_output_range():
    """Decoder Sigmoid → çıktı [0,1] aralığında olmalı."""
    model = build_autoencoder()
    x = torch.rand(32, 12)
    out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_reconstruction_error_shape():
    model = build_autoencoder()
    x = torch.rand(16, 12)
    err = model.reconstruction_error(x)
    assert err.shape == (16,)


def test_reconstruction_error_nonnegative():
    model = build_autoencoder()
    x = torch.rand(10, 12)
    err = model.reconstruction_error(x)
    assert (err >= 0).all()


def test_anomaly_score_range():
    """Sigmoid çıktısı → skor (0,1) aralığında."""
    model = build_autoencoder()
    x = torch.rand(20, 12)
    scores = model.anomaly_score(x, threshold=0.05)
    assert scores.min().item() > 0.0
    assert scores.max().item() < 1.0


def test_anomaly_score_monotone():
    """Yüksek reconstruction error → yüksek anomaly score."""
    model = build_autoencoder()
    # Normal veri: düşük hata beklenir
    x_normal = torch.zeros(1, 12)
    # Aşırı outlier: yüksek hata beklenir
    x_outlier = torch.ones(1, 12) * 10.0
    threshold = 0.05
    s_normal  = model.anomaly_score(x_normal, threshold).item()
    s_outlier = model.anomaly_score(x_outlier, threshold).item()
    assert s_outlier > s_normal


def test_single_sample():
    model = build_autoencoder()
    x = torch.rand(1, 12)
    out = model(x)
    assert out.shape == (1, 12)


def test_gradient_flows():
    """Backward pass hatasız geçmeli."""
    model = build_autoencoder()
    x = torch.rand(4, 12)
    out = model(x)
    loss = ((out - x) ** 2).mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
