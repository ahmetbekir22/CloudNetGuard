"""
CloudNetGuard — Model eğitim scripti (offline).
CSV dataset üzerinde Autoencoder ve LSTM eğitir, pretrained/ klasörüne kaydeder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/app/shared")
sys.path.insert(0, os.path.dirname(__file__))

from models.autoencoder import build_autoencoder
from models.lstm import build_lstm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
os.makedirs(PRETRAINED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    CSV formatı: feature_0,...,feature_11,label
    label: normal=0, tunnel=1, ddos=1, flux=1
    """
    import csv
    rows, labels = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # başlık satırı
        for row in reader:
            features = list(map(float, row[:12]))
            label_str = row[12].strip() if len(row) > 12 else "normal"
            label = 0 if label_str == "normal" else 1
            rows.append(features)
            labels.append(label)
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Autoencoder eğitimi
# ---------------------------------------------------------------------------

def train_autoencoder(
    X_normal: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple:
    model = build_autoencoder(input_dim=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor = torch.tensor(X_normal)
    loader = DataLoader(TensorDataset(tensor, tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, _ in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            log.info("Epoch %d/%d  loss=%.6f", epoch, epochs, total_loss / len(loader))

    # Anomali eşiği: normal verinin %95 reconstruction error percentile'ı
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(tensor).numpy()
    threshold = float(np.percentile(errors, 95))
    log.info("Autoencoder eşiği (p95): %.6f", threshold)
    return model, threshold


# ---------------------------------------------------------------------------
# LSTM eğitimi
# ---------------------------------------------------------------------------

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 20,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> object:
    model = build_lstm(input_dim=12, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Sliding window
    seqs, labels = [], []
    for i in range(len(X) - seq_len):
        seqs.append(X[i : i + seq_len])
        labels.append(y[i + seq_len - 1])

    X_t = torch.tensor(np.array(seqs))
    y_t = torch.tensor(np.array(labels)).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            log.info("LSTM Epoch %d/%d  loss=%.6f", epoch, epochs, total_loss / len(loader))

    return model


# ---------------------------------------------------------------------------
# Kaydetme / yükleme
# ---------------------------------------------------------------------------

def save_autoencoder(model, threshold: float) -> None:
    path = os.path.join(PRETRAINED_DIR, "autoencoder.pt")
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, path)
    log.info("Autoencoder kaydedildi: %s", path)


def save_lstm(model) -> None:
    path = os.path.join(PRETRAINED_DIR, "lstm.pt")
    torch.save(model.state_dict(), path)
    log.info("LSTM kaydedildi: %s", path)


def load_autoencoder(input_dim: int = 12):
    path = os.path.join(PRETRAINED_DIR, "autoencoder.pt")
    checkpoint = torch.load(path, map_location="cpu")
    model = build_autoencoder(input_dim=input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["threshold"]


def load_lstm(input_dim: int = 12):
    path = os.path.join(PRETRAINED_DIR, "lstm.pt")
    model = build_lstm(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CloudNetGuard model eğitici")
    parser.add_argument("--csv", required=True, help="Eğitim CSV dosyası")
    parser.add_argument("--model", choices=["autoencoder", "lstm", "both"], default="both")
    parser.add_argument("--epochs-ae", type=int, default=50)
    parser.add_argument("--epochs-lstm", type=int, default=30)
    args = parser.parse_args()

    X, y = load_csv(args.csv)
    X_normal = X[y == 0]

    if args.model in ("autoencoder", "both"):
        log.info("Autoencoder eğitimi başlıyor (%d normal örnek)...", len(X_normal))
        ae_model, threshold = train_autoencoder(X_normal, epochs=args.epochs_ae)
        save_autoencoder(ae_model, threshold)

    if args.model in ("lstm", "both"):
        log.info("LSTM eğitimi başlıyor (%d toplam örnek)...", len(X))
        lstm_model = train_lstm(X, y, epochs=args.epochs_lstm)
        save_lstm(lstm_model)

    log.info("Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
