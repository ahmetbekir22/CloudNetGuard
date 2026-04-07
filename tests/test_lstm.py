"""Unit tests — ai-engine/models/lstm.py"""

import sys
import pytest
import torch

sys.path.insert(0, "ai-engine")
from models.lstm import LSTMAnomalyDetector, build_lstm


def test_forward_shape():
    model = build_lstm(input_dim=12, seq_len=20)
    x = torch.rand(8, 20, 12)
    out = model(x)
    assert out.shape == (8, 1)


def test_output_range():
    """Sigmoid → çıktı (0,1) aralığında olmalı."""
    model = build_lstm()
    x = torch.rand(16, 20, 12)
    out = model(x)
    assert out.min().item() > 0.0
    assert out.max().item() < 1.0


def test_single_sample():
    model = build_lstm()
    x = torch.rand(1, 20, 12)
    out = model(x)
    assert out.shape == (1, 1)


def test_predict_returns_tuple():
    model = build_lstm()
    x = torch.rand(4, 20, 12)
    prob, label = model.predict(x)
    assert prob.shape == (4, 1)
    assert label.shape == (4, 1)


def test_predict_binary_labels():
    model = build_lstm()
    x = torch.rand(10, 20, 12)
    _, labels = model.predict(x, threshold=0.5)
    unique_vals = labels.unique().tolist()
    assert all(v in (0.0, 1.0) for v in unique_vals)


def test_gradient_flows():
    model = build_lstm()
    x = torch.rand(4, 20, 12)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_different_seq_lengths():
    """Farklı sequence uzunlukları çalışmalı."""
    for seq_len in (5, 10, 30):
        model = build_lstm()
        x = torch.rand(2, seq_len, 12)
        out = model(x)
        assert out.shape == (2, 1)


def test_batch_independence():
    """Her batch örneği bağımsız — tek örnek vs toplu tahmin aynı sonuç vermeli."""
    model = build_lstm()
    model.eval()
    x = torch.rand(3, 20, 12)
    with torch.no_grad():
        batch_out = model(x)
        single_outs = torch.cat([model(x[i:i+1]) for i in range(3)])
    assert torch.allclose(batch_out, single_outs, atol=1e-5)
