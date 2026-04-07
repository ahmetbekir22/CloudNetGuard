"""Unit tests — shared/schema.py"""

import json
import sys
import pytest

sys.path.insert(0, "shared")
from schema import (
    AnomalyRecord,
    DNSFeatureRecord,
    FeatureExplanation,
    SDNAction,
    FEATURE_NAMES,
    ANOMALY_TYPES,
)


# ---------------------------------------------------------------------------
# FEATURE_NAMES
# ---------------------------------------------------------------------------

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 12


def test_feature_names_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_anomaly_types():
    assert set(ANOMALY_TYPES) == {"normal", "tunnel", "ddos", "flux"}


# ---------------------------------------------------------------------------
# DNSFeatureRecord round-trip
# ---------------------------------------------------------------------------

def _make_dns_record(**kwargs) -> DNSFeatureRecord:
    defaults = dict(
        timestamp="2026-04-07T10:00:00+00:00",
        src_ip="10.0.0.1",
        dst_ip="8.8.8.8",
        query="example.com",
        query_type="A",
        features=[0.1] * 12,
        label="normal",
    )
    defaults.update(kwargs)
    return DNSFeatureRecord(**defaults)


def test_dns_record_round_trip():
    rec = _make_dns_record(features=[round(i * 0.08, 2) for i in range(12)])
    data = rec.to_redis()
    restored = DNSFeatureRecord.from_redis(data)
    assert restored.src_ip == rec.src_ip
    assert restored.features == pytest.approx(rec.features, abs=1e-5)
    assert restored.label == "normal"


def test_dns_record_redis_values_are_strings():
    data = _make_dns_record().to_redis()
    assert all(isinstance(v, str) for v in data.values())


def test_dns_record_features_json():
    rec = _make_dns_record(features=[0.5] * 12)
    data = rec.to_redis()
    parsed = json.loads(data["features"])
    assert parsed == [0.5] * 12


# ---------------------------------------------------------------------------
# AnomalyRecord round-trip
# ---------------------------------------------------------------------------

def _make_anomaly(**kwargs) -> AnomalyRecord:
    feat = FeatureExplanation(
        feature="entropy", importance=0.42, value=0.78, direction="high"
    )
    defaults = dict(
        timestamp="2026-04-07T10:00:00+00:00",
        src_ip="10.0.0.2",
        query="suspicious.evil.com",
        anomaly_score=0.92,
        is_anomaly=True,
        predicted_type="tunnel",
        reconstruction_error=0.045,
        top_features=[feat],
        summary="DNS tünelleme tespit edildi",
        feature_vector=[0.1 * i for i in range(12)],
    )
    defaults.update(kwargs)
    return AnomalyRecord(**defaults)


def test_anomaly_round_trip():
    rec = _make_anomaly()
    data = rec.to_redis()
    restored = AnomalyRecord.from_redis(data)

    assert restored.src_ip == rec.src_ip
    assert restored.is_anomaly is True
    assert restored.predicted_type == "tunnel"
    assert restored.anomaly_score == pytest.approx(0.92)
    assert len(restored.top_features) == 1
    assert restored.top_features[0].feature == "entropy"
    assert restored.top_features[0].importance == pytest.approx(0.42)
    assert restored.feature_vector == pytest.approx(rec.feature_vector, abs=1e-5)


def test_anomaly_false_round_trip():
    rec = _make_anomaly(is_anomaly=False, predicted_type="normal", top_features=[])
    data = rec.to_redis()
    restored = AnomalyRecord.from_redis(data)
    assert restored.is_anomaly is False
    assert restored.top_features == []


def test_anomaly_empty_feature_vector():
    rec = _make_anomaly(feature_vector=[])
    data = rec.to_redis()
    restored = AnomalyRecord.from_redis(data)
    assert restored.feature_vector == []


# ---------------------------------------------------------------------------
# SDNAction round-trip
# ---------------------------------------------------------------------------

def test_sdn_action_round_trip():
    action = SDNAction(
        timestamp="2026-04-07T10:00:01+00:00",
        src_ip="10.0.0.3",
        query="flood.attack.com",
        anomaly_score=0.88,
        predicted_type="ddos",
        action="BLOCK",
        reason="DDoS eşiği aşıldı",
    )
    data = action.to_redis()
    restored = SDNAction.from_redis(data)

    assert restored.action == "BLOCK"
    assert restored.predicted_type == "ddos"
    assert restored.anomaly_score == pytest.approx(0.88)
    assert restored.policy_version == "rules-v1"


def test_sdn_action_redis_strings():
    action = SDNAction(
        timestamp="2026-04-07T10:00:00+00:00",
        src_ip="1.2.3.4",
        query="x.com",
        anomaly_score=0.5,
        predicted_type="flux",
        action="MIRROR",
        reason="izleme",
    )
    data = action.to_redis()
    assert all(isinstance(v, str) for v in data.values())
