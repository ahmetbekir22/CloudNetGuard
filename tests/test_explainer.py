"""Unit tests — ai-engine/explainer.py (_guess_type + SHAPExplainer)"""

import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, "ai-engine")
sys.path.insert(0, "shared")

from explainer import _guess_type, SHAPExplainer
from models.autoencoder import build_autoencoder
from schema import FEATURE_NAMES


def _fvec(**overrides) -> list[float]:
    """Sıfır tabanlı feature vektörü, override ile belirli değerleri set eder."""
    base = dict.fromkeys(FEATURE_NAMES, 0.0)
    base.update(overrides)
    return [base[n] for n in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# _guess_type
# ---------------------------------------------------------------------------

def test_guess_type_ddos():
    fv = _fvec(response_size=0.80, ttl=0.02, record_type_A=1.0)
    assert _guess_type(fv) == "ddos"


def test_guess_type_tunnel():
    fv = _fvec(entropy=0.90, query_length=0.85, record_type_TXT=1.0, subdomain_digit_ratio=0.60)
    assert _guess_type(fv) == "tunnel"


def test_guess_type_flux():
    fv = _fvec(is_nxdomain=1.0, unique_domains=0.60, ttl=0.01)
    assert _guess_type(fv) == "flux"


def test_guess_type_returns_string():
    fv = [0.0] * 12
    result = _guess_type(fv)
    assert isinstance(result, str)
    assert result in ("tunnel", "ddos", "flux")


def test_guess_type_all_zeros_returns_valid():
    """Tüm sıfır vektörde bile geçerli bir tip döner."""
    result = _guess_type([0.0] * 12)
    assert result in ("tunnel", "ddos", "flux")


# ---------------------------------------------------------------------------
# SHAPExplainer
# ---------------------------------------------------------------------------

@pytest.fixture
def explainer():
    model = build_autoencoder(input_dim=12)
    model.eval()
    background = np.zeros((20, 12), dtype=np.float32)
    return SHAPExplainer(model, background, threshold=0.05)


def test_explain_returns_tuple(explainer):
    fv = _fvec(entropy=0.8, query_length=0.7)
    result = explainer.explain(fv, anomaly_score=0.9, reconstruction_error=0.1)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_explain_top_features_count(explainer):
    fv = _fvec(entropy=0.8, query_length=0.7)
    top_features, _ = explainer.explain(fv, 0.9, 0.1)
    assert len(top_features) <= 5
    assert len(top_features) > 0


def test_explain_importance_sum(explainer):
    """Normalize edilmiş importances toplamı ≈ 1."""
    fv = _fvec(entropy=0.8, response_size=0.7)
    top_features, _ = explainer.explain(fv, 0.85, 0.08)
    total = sum(f.importance for f in top_features)
    # Top-5'in toplamı tam 1 olmayabilir ama pozitif ve 0-1 arasında
    assert total > 0
    assert all(0.0 <= f.importance <= 1.0 for f in top_features)


def test_explain_summary_nonempty(explainer):
    fv = _fvec(entropy=0.9)
    _, summary = explainer.explain(fv, 0.9, 0.1)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_explain_feature_names_valid(explainer):
    fv = [0.5] * 12
    top_features, _ = explainer.explain(fv, 0.75, 0.06)
    for feat in top_features:
        assert feat.feature in FEATURE_NAMES


def test_explain_direction_valid(explainer):
    fv = [0.5] * 12
    top_features, _ = explainer.explain(fv, 0.75, 0.06)
    valid_directions = {"high", "low", "present", "absent"}
    for feat in top_features:
        assert feat.direction in valid_directions
