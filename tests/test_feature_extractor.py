"""Unit tests — dns-collector/feature_extractor.py"""

import sys
import math
import pytest

sys.path.insert(0, "dns-collector")
from synthetic import RawDNSPacket
from feature_extractor import (
    extract_features,
    _shannon_entropy,
    _clamp_normalize,
    _WindowTracker,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _packet(**kwargs) -> RawDNSPacket:
    defaults = dict(
        timestamp=1000.0,
        src_ip="10.0.0.1",
        dst_ip="8.8.8.8",
        query="example.com",
        query_type="A",
        ttl=300,
        response_size=512,
        is_nxdomain=False,
        label="normal",
    )
    defaults.update(kwargs)
    return RawDNSPacket(**defaults)


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------

def test_entropy_empty_string():
    assert _shannon_entropy("") == 0.0


def test_entropy_single_char():
    assert _shannon_entropy("aaa") == pytest.approx(0.0)


def test_entropy_two_equal():
    e = _shannon_entropy("ab")
    assert e == pytest.approx(1.0)


def test_entropy_positive():
    assert _shannon_entropy("randomstring") > 0.0


def test_entropy_high_randomness():
    # Rastgele DNS tünelleme stringi → yüksek entropi
    s = "a1b2c3d4e5f6g7h8i9j0"
    assert _shannon_entropy(s) > 3.0


# ---------------------------------------------------------------------------
# _clamp_normalize
# ---------------------------------------------------------------------------

def test_clamp_below():
    assert _clamp_normalize(-5.0, 0.0, 10.0) == 0.0


def test_clamp_above():
    assert _clamp_normalize(20.0, 0.0, 10.0) == 1.0


def test_clamp_mid():
    assert _clamp_normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)


def test_clamp_same_bounds():
    assert _clamp_normalize(3.0, 5.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# _WindowTracker
# ---------------------------------------------------------------------------

def test_tracker_query_rate_single():
    t = _WindowTracker(window_seconds=5.0)
    t.update("10.0.0.1", "a.com", now=1000.0)
    rate = t.query_rate("10.0.0.1")
    assert rate == pytest.approx(1 / 5.0)


def test_tracker_query_rate_multiple():
    t = _WindowTracker(window_seconds=5.0)
    for i in range(10):
        t.update("10.0.0.1", "a.com", now=1000.0 + i * 0.1)
    rate = t.query_rate("10.0.0.1")
    assert rate == pytest.approx(10 / 5.0)


def test_tracker_eviction():
    t = _WindowTracker(window_seconds=5.0)
    t.update("10.0.0.1", "a.com", now=1000.0)
    # 6 saniye sonra eski girdi evict edilmeli
    t.update("10.0.0.1", "b.com", now=1006.0)
    rate = t.query_rate("10.0.0.1")
    assert rate == pytest.approx(1 / 5.0)


def test_tracker_unique_domains():
    t = _WindowTracker(window_seconds=5.0)
    for domain in ["a.com", "b.com", "a.com", "c.com"]:
        t.update("10.0.0.1", domain, now=1000.0)
    assert t.unique_domains() == 3


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

def test_feature_vector_length():
    tracker = _WindowTracker()
    pkt = _packet()
    features = extract_features(pkt, tracker=tracker)
    assert len(features) == 12


def test_feature_vector_range():
    tracker = _WindowTracker()
    pkt = _packet()
    features = extract_features(pkt, tracker=tracker)
    for val in features:
        assert 0.0 <= val <= 1.0, f"Feature out of range: {val}"


def test_is_nxdomain_flag():
    tracker = _WindowTracker()
    pkt = _packet(is_nxdomain=True)
    features = extract_features(pkt, tracker=tracker)
    idx = FEATURE_NAMES.index("is_nxdomain")
    assert features[idx] == pytest.approx(1.0)


def test_record_type_txt():
    tracker = _WindowTracker()
    pkt = _packet(query_type="TXT")
    features = extract_features(pkt, tracker=tracker)
    idx_txt = FEATURE_NAMES.index("record_type_TXT")
    idx_a   = FEATURE_NAMES.index("record_type_A")
    assert features[idx_txt] == pytest.approx(1.0)
    assert features[idx_a]   == pytest.approx(0.0)


def test_high_entropy_tunnel_query():
    """Uzun rastgele subdomain → yüksek normalized entropy."""
    tracker = _WindowTracker()
    pkt = _packet(query="a1b2c3d4e5f6g7h8.evil.com", query_type="TXT", timestamp=1000.0)
    features = extract_features(pkt, tracker=tracker)
    idx = FEATURE_NAMES.index("entropy")
    assert features[idx] > 0.3


def test_long_query_normalized():
    tracker = _WindowTracker()
    pkt = _packet(query="x" * 150 + ".com")
    features = extract_features(pkt, tracker=tracker)
    idx = FEATURE_NAMES.index("query_length")
    assert features[idx] > 0.5


def test_query_rate_increases_with_volume():
    tracker = _WindowTracker()
    # 50 sorgu aynı IP'den — yüksek rate beklenir
    base_time = 1000.0
    for i in range(50):
        pkt = _packet(src_ip="10.0.0.1", timestamp=base_time + i * 0.05)
        extract_features(pkt, tracker=tracker)
    last_pkt = _packet(src_ip="10.0.0.1", timestamp=base_time + 50 * 0.05)
    features = extract_features(last_pkt, tracker=tracker)
    idx = FEATURE_NAMES.index("query_rate")
    assert features[idx] > 0.0
