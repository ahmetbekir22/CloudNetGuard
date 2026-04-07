"""
CloudNetGuard — Model eğitim pipeline'ı.
Dataset üret → eğit → pretrained/ klasörüne kaydet.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-engine"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudNetGuard model eğitim pipeline'ı")
    parser.add_argument("--csv",        default="data/train.csv")
    parser.add_argument("--model",      choices=["autoencoder", "lstm", "both"], default="both")
    parser.add_argument("--epochs-ae",  type=int, default=50)
    parser.add_argument("--epochs-lstm",type=int, default=30)
    parser.add_argument("--generate",   action="store_true", help="Önce dataset üret")
    parser.add_argument("--n-samples",  type=int, default=100_000)
    args = parser.parse_args()

    if args.generate or not os.path.exists(args.csv):
        log.info("Dataset üretiliyor: %s", args.csv)
        from generate_dataset import generate
        generate(args.csv, args.n_samples)

    log.info("Model eğitimi başlıyor...")
    from trainer import load_csv, train_autoencoder, train_lstm, save_autoencoder, save_lstm
    import numpy as np

    X, y = load_csv(args.csv)
    X_normal = X[y == 0]
    log.info("Toplam: %d örnek, Normal: %d, Anomali: %d", len(X), len(X_normal), (y == 1).sum())

    if args.model in ("autoencoder", "both"):
        ae_model, threshold = train_autoencoder(X_normal, epochs=args.epochs_ae)
        save_autoencoder(ae_model, threshold)

    if args.model in ("lstm", "both"):
        lstm_model = train_lstm(X, y, epochs=args.epochs_lstm)
        save_lstm(lstm_model)

    log.info("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
