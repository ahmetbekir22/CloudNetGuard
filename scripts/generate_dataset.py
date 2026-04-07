"""
CloudNetGuard — Büyük sentetik dataset üretici.
CSV formatında eğitim verisi oluşturur.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))

from feature_extractor import extract_features, FEATURE_NAMES
from synthetic import SyntheticGenerator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate(
    output_path: str,
    n_samples: int = 100_000,
    anomaly_ratio: float = 0.20,
) -> None:
    gen = SyntheticGenerator(anomaly_ratio=anomaly_ratio, seed=42)
    packets = gen.generate_batch(n_samples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header = FEATURE_NAMES + ["label"]
    label_map = {"normal": 0, "tunnel": 1, "ddos": 1, "flux": 1}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, pkt in enumerate(packets):
            feats = extract_features(pkt)
            label = label_map.get(pkt.label, 0)
            writer.writerow([f"{v:.6f}" for v in feats] + [pkt.label])
            if (i + 1) % 10_000 == 0:
                log.info("Üretilen: %d / %d", i + 1, n_samples)

    log.info("Dataset kaydedildi: %s  (%d satır)", output_path, n_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudNetGuard dataset üreticisi")
    parser.add_argument("--output", default="data/train.csv", help="Çıktı CSV yolu")
    parser.add_argument("--n",      type=int, default=100_000, help="Örnek sayısı")
    parser.add_argument("--anomaly-ratio", type=float, default=0.20)
    args = parser.parse_args()
    generate(args.output, args.n, args.anomaly_ratio)


if __name__ == "__main__":
    main()
