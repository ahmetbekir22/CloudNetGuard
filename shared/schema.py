"""
CloudNetGuard — Ortak veri şemaları.
Tüm servisler bu modülü kullanır.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DNSFeatureRecord:
    """dns-collector → ai-engine (dns:features stream)."""

    timestamp: str
    src_ip: str
    dst_ip: str
    query: str
    query_type: str                     # A, AAAA, TXT, MX, CNAME, NS
    features: list[float]               # 12-element normalized feature vector
    label: str = "normal"               # normal | tunnel | ddos | flux

    def to_redis(self) -> dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "query": self.query,
            "query_type": self.query_type,
            "features": json.dumps(self.features),
            "label": self.label,
        }

    @classmethod
    def from_redis(cls, data: dict) -> "DNSFeatureRecord":
        return cls(
            timestamp=data["timestamp"],
            src_ip=data["src_ip"],
            dst_ip=data["dst_ip"],
            query=data["query"],
            query_type=data["query_type"],
            features=json.loads(data["features"]),
            label=data.get("label", "normal"),
        )


@dataclass
class FeatureExplanation:
    """Tek bir feature'ın XAI açıklaması."""

    feature: str
    importance: float
    value: float
    direction: str          # "high" | "low" | "present" | "absent"


@dataclass
class AnomalyRecord:
    """ai-engine → sdn-sim / dashboard (dns:anomalies stream)."""

    timestamp: str
    src_ip: str
    query: str
    anomaly_score: float
    is_anomaly: bool
    predicted_type: str                         # normal | tunnel | ddos | flux
    reconstruction_error: float
    top_features: list[FeatureExplanation] = field(default_factory=list)
    summary: str = ""

    def to_redis(self) -> dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "query": self.query,
            "anomaly_score": str(self.anomaly_score),
            "is_anomaly": str(self.is_anomaly),
            "predicted_type": self.predicted_type,
            "reconstruction_error": str(self.reconstruction_error),
            "top_features": json.dumps([asdict(f) for f in self.top_features]),
            "summary": self.summary,
        }

    @classmethod
    def from_redis(cls, data: dict) -> "AnomalyRecord":
        raw_features = json.loads(data.get("top_features", "[]"))
        top_features = [FeatureExplanation(**f) for f in raw_features]
        return cls(
            timestamp=data["timestamp"],
            src_ip=data["src_ip"],
            query=data["query"],
            anomaly_score=float(data["anomaly_score"]),
            is_anomaly=data["is_anomaly"] == "True",
            predicted_type=data["predicted_type"],
            reconstruction_error=float(data["reconstruction_error"]),
            top_features=top_features,
            summary=data.get("summary", ""),
        )


@dataclass
class SDNAction:
    """sdn-sim → dashboard (sdn:actions stream)."""

    timestamp: str
    src_ip: str
    query: str
    anomaly_score: float
    predicted_type: str
    action: str             # BLOCK | REDIRECT | MIRROR | ALLOW
    reason: str
    policy_version: str = "rules-v1"

    def to_redis(self) -> dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "query": self.query,
            "anomaly_score": str(self.anomaly_score),
            "predicted_type": self.predicted_type,
            "action": self.action,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_redis(cls, data: dict) -> "SDNAction":
        return cls(
            timestamp=data["timestamp"],
            src_ip=data["src_ip"],
            query=data["query"],
            anomaly_score=float(data["anomaly_score"]),
            predicted_type=data["predicted_type"],
            action=data["action"],
            reason=data.get("reason", ""),
            policy_version=data.get("policy_version", "rules-v1"),
        )


# Feature isim listesi — sıra önemli (model giriş vektörüyle eşleşmeli)
FEATURE_NAMES: list[str] = [
    "query_length",
    "entropy",
    "subdomain_count",
    "ttl",
    "query_rate",
    "record_type_A",
    "record_type_TXT",
    "record_type_MX",
    "response_size",
    "unique_domains",
    "is_nxdomain",
    "subdomain_digit_ratio",
]

ANOMALY_TYPES: list[str] = ["normal", "tunnel", "ddos", "flux"]
