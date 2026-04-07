"""
CloudNetGuard — AI Engine ana servisi.
Redis dns:features stream'ini tüketir, anomali tespiti yapar,
XAI açıklaması üretir ve dns:anomalies stream'ine yazar.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import redis
import torch

sys.path.insert(0, "/app/shared")
sys.path.insert(0, os.path.dirname(__file__))

from schema import AnomalyRecord, DNSFeatureRecord, FEATURE_NAMES
from models.autoencoder import build_autoencoder
from models.lstm import build_lstm
from trainer import load_autoencoder, load_lstm
from explainer import SHAPExplainer

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","service":"ai-engine","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

REDIS_HOST         = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT         = int(os.environ.get("REDIS_PORT", "6379"))
MODEL_TYPE         = os.environ.get("MODEL_TYPE", "autoencoder")   # autoencoder | lstm | ensemble
ANOMALY_THRESHOLD  = float(os.environ.get("ANOMALY_THRESHOLD", "0.85"))
IN_STREAM          = "dns:features"
OUT_STREAM         = "dns:anomalies"
CONSUMER_GROUP     = "ai-engine-group"
CONSUMER_NAME      = "ai-engine-1"
BATCH_SIZE         = int(os.environ.get("BATCH_SIZE", "32"))
PRETRAINED_DIR     = os.path.join(os.path.dirname(__file__), "pretrained")
SEQ_LEN            = 20


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

def connect_redis(retries: int = 10, delay: float = 2.0) -> redis.Redis:
    for attempt in range(1, retries + 1):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            log.info("Redis bağlantısı kuruldu")
            return r
        except redis.ConnectionError as exc:
            log.warning("Redis bağlanamadı (%d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError("Redis bağlantısı kurulamadı.")


def ensure_consumer_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(IN_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group oluşturuldu: %s", CONSUMER_GROUP)
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            log.info("Consumer group zaten mevcut: %s", CONSUMER_GROUP)
        else:
            raise


# ---------------------------------------------------------------------------
# Model yükleme
# ---------------------------------------------------------------------------

def load_models() -> tuple:
    """Eğitilmiş modelleri yükle. Yoksa dummy model oluştur."""
    ae_path   = os.path.join(PRETRAINED_DIR, "autoencoder.pt")
    lstm_path = os.path.join(PRETRAINED_DIR, "lstm.pt")

    if os.path.exists(ae_path):
        ae_model, ae_threshold = load_autoencoder()
        log.info("Autoencoder yüklendi (threshold=%.4f)", ae_threshold)
    else:
        log.warning("Pretrained autoencoder bulunamadı — random init + otomatik kalibrasyon.")
        ae_model = build_autoencoder(12)
        ae_model.eval()
        # Kalibrasyon: random normal örnekler üzerinde p50 reconstruction error'ı threshold al.
        # Bu sayede ~%50 anomali oranı ile demo çalışır; gerçek eğitimde p95 kullanılır.
        with torch.no_grad():
            calib = torch.rand(500, 12)
            errors = ae_model.reconstruction_error(calib).numpy()
        ae_threshold = float(np.percentile(errors, 50))
        log.info("Kalibrasyon tamamlandı, threshold=%.4f (p50)", ae_threshold)

    lstm_model = None
    if MODEL_TYPE in ("lstm", "ensemble") and os.path.exists(lstm_path):
        lstm_model = load_lstm()
        log.info("LSTM yüklendi")

    return ae_model, ae_threshold, lstm_model


def build_explainer(ae_model, ae_threshold: float) -> SHAPExplainer:
    """Arka plan verisi olarak sıfır vektörleri kullan."""
    background = np.zeros((50, 12), dtype=np.float32)
    return SHAPExplainer(ae_model, background, ae_threshold)


# ---------------------------------------------------------------------------
# Tahmin
# ---------------------------------------------------------------------------

_lstm_buffer: list[list[float]] = []


def predict(
    feature_vector: list[float],
    ae_model,
    ae_threshold: float,
    lstm_model,
) -> tuple[float, bool, float, str]:
    """
    Döndürür: (anomaly_score, is_anomaly, reconstruction_error, predicted_type)
    """
    t = torch.tensor([feature_vector], dtype=torch.float32)

    with torch.no_grad():
        recon_err = float(ae_model.reconstruction_error(t).item())
        ae_score  = float(ae_model.anomaly_score(t, ae_threshold).item())

    final_score = ae_score

    if lstm_model is not None and MODEL_TYPE in ("lstm", "ensemble"):
        _lstm_buffer.append(feature_vector)
        if len(_lstm_buffer) >= SEQ_LEN:
            seq = torch.tensor([_lstm_buffer[-SEQ_LEN:]], dtype=torch.float32)
            with torch.no_grad():
                lstm_prob = float(lstm_model(seq).item())
            if MODEL_TYPE == "ensemble":
                final_score = 0.6 * ae_score + 0.4 * lstm_prob
            else:
                final_score = lstm_prob
            _lstm_buffer.pop(0)

    is_anomaly = final_score >= 0.5
    predicted_type = "normal"
    if is_anomaly:
        from explainer import _guess_type
        predicted_type = _guess_type(feature_vector)

    return final_score, is_anomaly, recon_err, predicted_type


# ---------------------------------------------------------------------------
# İşlem döngüsü
# ---------------------------------------------------------------------------

def process_messages(
    r: redis.Redis,
    ae_model,
    ae_threshold: float,
    lstm_model,
    explainer: SHAPExplainer,
) -> None:
    anomaly_count = 0
    total_count   = 0

    while True:
        try:
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {IN_STREAM: ">"},
                count=BATCH_SIZE,
                block=1000,
            )
        except redis.ResponseError as e:
            log.error("XREADGROUP hatası: %s", e)
            time.sleep(1)
            continue

        if not messages:
            continue

        for _stream, entries in messages:
            for msg_id, data in entries:
                try:
                    record = DNSFeatureRecord.from_redis(data)
                    features = record.features

                    score, is_anomaly, recon_err, pred_type = predict(
                        features, ae_model, ae_threshold, lstm_model
                    )

                    top_features, summary = [], ""
                    if is_anomaly:
                        top_features, summary = explainer.explain(features, score, recon_err)

                    anomaly_rec = AnomalyRecord(
                        timestamp=record.timestamp,
                        src_ip=record.src_ip,
                        query=record.query,
                        anomaly_score=round(score, 4),
                        is_anomaly=is_anomaly,
                        predicted_type=pred_type,
                        reconstruction_error=round(recon_err, 6),
                        top_features=top_features,
                        summary=summary,
                    )
                    r.xadd(OUT_STREAM, anomaly_rec.to_redis(), maxlen=20_000, approximate=True)
                    r.xack(IN_STREAM, CONSUMER_GROUP, msg_id)

                    total_count += 1
                    if is_anomaly:
                        anomaly_count += 1
                    if total_count % 200 == 0:
                        ratio = anomaly_count / total_count * 100
                        log.info("İşlenen: %d | Anomali: %d (%.1f%%)", total_count, anomaly_count, ratio)

                except Exception as exc:
                    log.exception("Mesaj işlenemedi (id=%s): %s", msg_id, exc)
                    r.xack(IN_STREAM, CONSUMER_GROUP, msg_id)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    r = connect_redis()
    ensure_consumer_group(r)
    ae_model, ae_threshold, lstm_model = load_models()
    explainer = build_explainer(ae_model, ae_threshold)
    log.info("AI Engine hazır. Model: %s", MODEL_TYPE)
    process_messages(r, ae_model, ae_threshold, lstm_model, explainer)


if __name__ == "__main__":
    main()
