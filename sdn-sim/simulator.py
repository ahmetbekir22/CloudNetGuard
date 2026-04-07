"""
CloudNetGuard — SDN simülatör ana servisi.
Flask API + Redis dns:anomalies tüketicisi.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import redis
from flask import Flask, jsonify, request

sys.path.insert(0, "/app/shared")
sys.path.insert(0, os.path.dirname(__file__))

from schema import AnomalyRecord, SDNAction
from actions import Action
from policy import get_policy

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","service":"sdn-sim","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

REDIS_HOST     = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.environ.get("REDIS_PORT", "6379"))
IN_STREAM      = "dns:anomalies"
OUT_STREAM     = "sdn:actions"
CONSUMER_GROUP = "sdn-sim-group"
CONSUMER_NAME  = "sdn-sim-1"
POLICY_VERSION = os.environ.get("POLICY", "rules") + "-v1"
PORT           = int(os.environ.get("PORT", "5001"))

# ---------------------------------------------------------------------------
# Uygulama durumu
# ---------------------------------------------------------------------------

app = Flask(__name__)
_action_log: deque[dict] = deque(maxlen=500)
_policy = get_policy()
_r: redis.Redis | None = None


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
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


# ---------------------------------------------------------------------------
# Tüketici döngüsü (ayrı thread)
# ---------------------------------------------------------------------------

def consumer_loop(r: redis.Redis) -> None:
    ensure_consumer_group(r)
    log.info("SDN simülatör tüketici döngüsü başladı")
    while True:
        try:
            messages = r.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {IN_STREAM: ">"}, count=10, block=1000,
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
                    anomaly = AnomalyRecord.from_redis(data)
                    result  = _policy.decide(anomaly.anomaly_score, anomaly.predicted_type)

                    sdn_action = SDNAction(
                        timestamp=anomaly.timestamp,
                        src_ip=anomaly.src_ip,
                        query=anomaly.query,
                        anomaly_score=anomaly.anomaly_score,
                        predicted_type=anomaly.predicted_type,
                        action=result.action.value,
                        reason=result.reason,
                        policy_version=POLICY_VERSION,
                    )
                    r.xadd(OUT_STREAM, sdn_action.to_redis(), maxlen=10_000, approximate=True)
                    r.xack(IN_STREAM, CONSUMER_GROUP, msg_id)

                    entry = {
                        "timestamp":      sdn_action.timestamp,
                        "src_ip":         sdn_action.src_ip,
                        "query":          sdn_action.query,
                        "anomaly_score":  sdn_action.anomaly_score,
                        "predicted_type": sdn_action.predicted_type,
                        "action":         sdn_action.action,
                        "reason":         sdn_action.reason,
                    }
                    _action_log.appendleft(entry)

                    if result.action != Action.ALLOW:
                        log.info("[%s] %s → %s", result.action.value, anomaly.src_ip, anomaly.query)

                except Exception as exc:
                    log.exception("Mesaj işlenemedi: %s", exc)
                    r.xack(IN_STREAM, CONSUMER_GROUP, msg_id)


# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

@app.route("/decide", methods=["POST"])
def decide():
    """Anomali verisini al, karar ver (REST endpoint)."""
    body = request.get_json(force=True, silent=True) or {}
    score = float(body.get("anomaly_score", 0.0))
    ptype = str(body.get("predicted_type", "normal"))
    result = _policy.decide(score, ptype)
    return jsonify({
        "action": result.action.value,
        "reason": result.reason,
        "target": result.target,
    })


@app.route("/actions", methods=["GET"])
def get_actions():
    """Son N aksiyon logunu getir."""
    n = min(int(request.args.get("n", 50)), 500)
    return jsonify(list(_action_log)[:n])


@app.route("/policy", methods=["GET"])
def get_policy_info():
    """Aktif politika bilgisini döndür."""
    return jsonify({
        "version":     POLICY_VERSION,
        "type":        os.environ.get("POLICY", "rules"),
        "thresholds": {
            "block_ddos":    0.9,
            "redirect_tunnel": 0.85,
            "block_flux":    0.8,
            "mirror":        0.6,
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    global _r
    _r = connect_redis()
    t = threading.Thread(target=consumer_loop, args=(_r,), daemon=True)
    t.start()
    log.info("SDN simülatör Flask API :%d adresinde başlatılıyor", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
