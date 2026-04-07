"""
CloudNetGuard — Model eğitim pipeline'ı.
Sentetik veya CIC-Bell-DNS-2021 gerçek verisiyle eğitim yapar.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-engine"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudNetGuard model eğitim pipeline'ı")

    # Veri kaynağı
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--cic-dir",  help="CIC-Bell-DNS-2021 CSV klasörü")
    src.add_argument("--csv",      default="data/train.csv", help="Sentetik CSV yolu")

    parser.add_argument("--generate",    action="store_true", help="Sentetik dataset üret")
    parser.add_argument("--n-samples",   type=int, default=50_000)
    parser.add_argument("--model",       choices=["autoencoder", "lstm", "both"], default="autoencoder")
    parser.add_argument("--epochs-ae",   type=int, default=50)
    parser.add_argument("--epochs-lstm", type=int, default=30)
    parser.add_argument("--max-benign",  type=int, default=50_000, help="CIC: max normal satır")
    parser.add_argument("--max-anomaly", type=int, default=5_000,  help="CIC: max anomali satır/tip")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Veri yükleme
    # ------------------------------------------------------------------ #
    if args.cic_dir:
        log.info("CIC-Bell-DNS-2021 verisi yükleniyor: %s", args.cic_dir)
        from load_cic_dns import load_arrays
        X, y = load_arrays(args.cic_dir, args.max_benign, args.max_anomaly)
    else:
        if args.generate or not os.path.exists(args.csv):
            log.info("Sentetik dataset üretiliyor: %s", args.csv)
            from generate_dataset import generate
            generate(args.csv, args.n_samples)
        from trainer import load_csv
        X, y = load_csv(args.csv)

    X_normal = X[y == 0]
    log.info("Toplam: %d | Normal: %d | Anomali: %d",
             len(X), len(X_normal), int((y == 1).sum()))

    # ------------------------------------------------------------------ #
    # Eğitim
    # ------------------------------------------------------------------ #
    from trainer import (
        train_autoencoder, train_lstm,
        save_autoencoder, save_lstm,
    )

    if args.model in ("autoencoder", "both"):
        log.info("Autoencoder eğitiliyor (%d epoch)...", args.epochs_ae)
        ae_model, threshold = train_autoencoder(X_normal, epochs=args.epochs_ae)
        save_autoencoder(ae_model, threshold)

    if args.model in ("lstm", "both"):
        log.info("LSTM eğitiliyor (%d epoch)...", args.epochs_lstm)
        lstm_model = train_lstm(X, y, epochs=args.epochs_lstm)
        save_lstm(lstm_model)

    log.info("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
