"""
CloudNetGuard — Demo için hazır veri Redis'e yükle.
Canlı servisler olmadan dashboard'u test etmek için kullanılır.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

from feature_extractor import extract_features
from synthetic import SyntheticGenerator
from schema import AnomalyRecord, FeatureExplanation, SDNAction

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


def seed(n_normal: int = 200, n_anomaly: int = 100) -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    log.info("Redis bağlandı")

    gen = SyntheticGenerator(anomaly_ratio=0.0, seed=123)
    normal_pkts = gen.generate_batch(n_normal)

    gen_a = SyntheticGenerator(anomaly_ratio=1.0, seed=456)
    anomaly_pkts = gen_a.generate_batch(n_anomaly)

    all_pkts = normal_pkts + anomaly_pkts

    for pkt in all_pkts:
        features = extract_features(pkt)
        is_anomaly = pkt.label != "normal"
        score = 0.85 + 0.10 * (hash(pkt.query) % 100) / 100 if is_anomaly else 0.1

        top_features = []
        if is_anomaly:
            top_features = [
                FeatureExplanation("entropy", 0.45, features[1], "high"),
                FeatureExplanation("query_length", 0.32, features[0], "high"),
            ]

        rec = AnomalyRecord(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            src_ip=pkt.src_ip,
            query=pkt.query,
            anomaly_score=round(score, 4),
            is_anomaly=is_anomaly,
            predicted_type=pkt.label if is_anomaly else "normal",
            reconstruction_error=round(score * 0.05, 6),
            top_features=top_features,
            summary=f"Seed verisi — {pkt.label}" if is_anomaly else "",
        )
        r.xadd("dns:anomalies", rec.to_redis(), maxlen=5000)

        if is_anomaly:
            action = "BLOCK" if pkt.label == "ddos" else "REDIRECT" if pkt.label == "tunnel" else "MIRROR"
            sdn = SDNAction(
                timestamp=rec.timestamp,
                src_ip=pkt.src_ip,
                query=pkt.query,
                anomaly_score=rec.anomaly_score,
                predicted_type=pkt.label,
                action=action,
                reason=f"Seed: {pkt.label} tespit edildi",
            )
            r.xadd("sdn:actions", sdn.to_redis(), maxlen=2000)

        time.sleep(0.002)

    log.info("Seed tamamlandı: %d normal + %d anomali yüklendi", n_normal, n_anomaly)


if __name__ == "__main__":
    seed()
