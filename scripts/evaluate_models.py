"""
CloudNetGuard — Model değerlendirme scripti.
Eğitim verisinden bağımsız yeni sentetik test seti üretir,
Autoencoder + LSTM + Ensemble skorlarını raporlar.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-engine"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from feature_extractor import extract_features, FEATURE_NAMES, _WindowTracker
from synthetic import SyntheticGenerator
from trainer import load_autoencoder, load_lstm

# ---------------------------------------------------------------------------
# Test verisi üret (farklı seed)
# ---------------------------------------------------------------------------

def generate_test_data(n: int = 10_000, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    gen = SyntheticGenerator(anomaly_ratio=0.20, seed=seed)
    packets = gen.generate_batch(n, rate=100.0)
    tracker = _WindowTracker(window_seconds=5.0)

    X, y = [], []
    for pkt in packets:
        feats = extract_features(pkt, tracker=tracker)
        X.append(feats)
        y.append(0 if pkt.label == "normal" else 1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrik hesaplama
# ---------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def print_metrics(name: str, m: dict) -> None:
    log.info(
        f"\n{'─'*40}\n{name}\n{'─'*40}\n"
        f"  Accuracy:  {m['acc']:.4f}\n"
        f"  Precision: {m['prec']:.4f}\n"
        f"  Recall:    {m['rec']:.4f}\n"
        f"  F1:        {m['f1']:.4f}\n"
        f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}"
    )


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Test verisi üretiliyor (seed=99, n=10000)...")
    X, y = generate_test_data(n=10_000, seed=99)
    log.info("Dağılım — Normal: %d | Anomali: %d", (y==0).sum(), (y==1).sum())

    # ── Autoencoder ──────────────────────────────────────────────────────────
    ae_model, ae_threshold = load_autoencoder()
    t = torch.tensor(X)
    with torch.no_grad():
        ae_scores = ae_model.anomaly_score(t, ae_threshold).numpy()
    ae_preds = (ae_scores >= 0.5).astype(int)
    print_metrics("Autoencoder", metrics(y, ae_preds))

    # ── LSTM ─────────────────────────────────────────────────────────────────
    try:
        lstm_model = load_lstm()
        seq_len = 20
        seqs, y_seq = [], []
        for i in range(len(X) - seq_len):
            seqs.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        X_seq = torch.tensor(np.array(seqs))
        y_seq = np.array(y_seq)
        with torch.no_grad():
            lstm_probs = lstm_model(X_seq).squeeze().numpy()
        lstm_preds = (lstm_probs >= 0.5).astype(int)
        print_metrics("LSTM", metrics(y_seq, lstm_preds))

        # ── Ensemble ─────────────────────────────────────────────────────────
        ae_scores_seq = ae_scores[seq_len:]   # LSTM ile hizala
        ens_scores = 0.2 * ae_scores_seq + 0.8 * lstm_probs
        ens_preds  = (ens_scores >= 0.5).astype(int)
        print_metrics("Ensemble (0.6 AE + 0.4 LSTM)", metrics(y_seq, ens_preds))

    except FileNotFoundError:
        log.warning("lstm.pt bulunamadı — sadece autoencoder değerlendirildi.")

    log.info("\nDeğerlendirme tamamlandı.")


if __name__ == "__main__":
    main()
