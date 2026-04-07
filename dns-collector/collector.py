"""
CloudNetGuard — DNS Collector ana servisi.
Sentetik / pcap / canlı trafikten DNS paketleri toplar,
feature vektörlerine dönüştürür ve Redis Streams'e yazar.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import redis

sys.path.insert(0, "/app/shared")

from feature_extractor import extract_features
from synthetic import SyntheticGenerator

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","service":"dns-collector","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

REDIS_HOST   = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.environ.get("REDIS_PORT", "6379"))
MODE         = os.environ.get("MODE", "synthetic")   # synthetic | pcap | live
RATE         = float(os.environ.get("RATE", "100"))
ANOMALY_RATIO = float(os.environ.get("ANOMALY_RATIO", "0.15"))
STREAM_NAME  = "dns:features"
PCAP_PATH    = os.environ.get("PCAP_PATH", "/app/data/traffic.pcap")

# ---------------------------------------------------------------------------
# Redis bağlantısı
# ---------------------------------------------------------------------------

def connect_redis(retries: int = 10, delay: float = 2.0) -> redis.Redis:
    for attempt in range(1, retries + 1):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            log.info("Redis bağlantısı kuruldu (%s:%s)", REDIS_HOST, REDIS_PORT)
            return r
        except redis.ConnectionError as exc:
            log.warning("Redis bağlanamadı (deneme %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError("Redis'e bağlanılamadı, çıkılıyor.")


# ---------------------------------------------------------------------------
# Yayıncı
# ---------------------------------------------------------------------------

def publish_packet(r: redis.Redis, packet, features: list[float]) -> None:
    ts = datetime.fromtimestamp(packet.timestamp, tz=timezone.utc).isoformat()
    record = {
        "timestamp":  ts,
        "src_ip":     packet.src_ip,
        "dst_ip":     packet.dst_ip,
        "query":      packet.query,
        "query_type": packet.query_type,
        "features":   json.dumps(features),
        "label":      packet.label,
    }
    r.xadd(STREAM_NAME, record, maxlen=50_000, approximate=True)


# ---------------------------------------------------------------------------
# Modlar
# ---------------------------------------------------------------------------

def run_synthetic(r: redis.Redis) -> None:
    log.info("Sentetik mod başlatıldı (rate=%.0f/sn, anomaly_ratio=%.2f)", RATE, ANOMALY_RATIO)
    gen = SyntheticGenerator(rate=RATE, anomaly_ratio=ANOMALY_RATIO)
    count = 0
    for packet in gen.generate():
        features = extract_features(packet)
        publish_packet(r, packet, features)
        count += 1
        if count % 500 == 0:
            log.info("Yayınlanan paket: %d", count)


def run_pcap(r: redis.Redis) -> None:
    """scapy ile pcap dosyasından paket oku."""
    try:
        from scapy.all import DNS, DNSQR, IP, rdpcap  # type: ignore
    except ImportError:
        log.error("scapy kurulu değil. pip install scapy")
        return

    log.info("Pcap modu: %s", PCAP_PATH)
    try:
        packets = rdpcap(PCAP_PATH)
    except FileNotFoundError:
        log.error("Pcap dosyası bulunamadı: %s", PCAP_PATH)
        return

    from synthetic import RawDNSPacket

    count = 0
    for pkt in packets:
        if not (pkt.haslayer(DNS) and pkt.haslayer(DNSQR)):
            continue
        dns = pkt[DNS]
        ip  = pkt[IP] if pkt.haslayer(IP) else None
        raw = RawDNSPacket(
            timestamp=float(pkt.time),
            src_ip=ip.src if ip else "0.0.0.0",
            dst_ip=ip.dst if ip else "0.0.0.0",
            query=pkt[DNSQR].qname.decode().rstrip("."),
            query_type=_qtype_str(pkt[DNSQR].qtype),
            ttl=dns.an.ttl if dns.an else 300,
            response_size=len(pkt),
            is_nxdomain=(dns.rcode == 3),
            label="normal",
        )
        features = extract_features(raw)
        publish_packet(r, raw, features)
        count += 1
        if count % 1000 == 0:
            log.info("Pcap paketi işlendi: %d", count)
    log.info("Pcap tamamlandı, toplam: %d paket", count)


def _qtype_str(qtype: int) -> str:
    mapping = {1: "A", 28: "AAAA", 16: "TXT", 15: "MX", 5: "CNAME", 2: "NS"}
    return mapping.get(qtype, "A")


def run_live(r: redis.Redis) -> None:
    """scapy ile canlı trafik yakala (root gerektirir)."""
    try:
        from scapy.all import DNS, DNSQR, IP, sniff  # type: ignore
    except ImportError:
        log.error("scapy kurulu değil.")
        return

    from synthetic import RawDNSPacket

    log.info("Canlı yakalama modu başlatıldı (udp port 53)")

    def process(pkt) -> None:
        if not (pkt.haslayer(DNS) and pkt.haslayer(DNSQR)):
            return
        dns = pkt[DNS]
        ip  = pkt[IP] if pkt.haslayer(IP) else None
        raw = RawDNSPacket(
            timestamp=time.time(),
            src_ip=ip.src if ip else "0.0.0.0",
            dst_ip=ip.dst if ip else "0.0.0.0",
            query=pkt[DNSQR].qname.decode().rstrip("."),
            query_type=_qtype_str(pkt[DNSQR].qtype),
            ttl=dns.an.ttl if dns.an else 300,
            response_size=len(pkt),
            is_nxdomain=(dns.rcode == 3),
            label="normal",
        )
        features = extract_features(raw)
        publish_packet(r, raw, features)

    sniff(filter="udp port 53", prn=process, store=False)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    r = connect_redis()
    mode_map = {
        "synthetic": run_synthetic,
        "pcap":      run_pcap,
        "live":      run_live,
    }
    runner = mode_map.get(MODE)
    if runner is None:
        log.error("Geçersiz mod: %s (synthetic | pcap | live)", MODE)
        sys.exit(1)
    runner(r)


if __name__ == "__main__":
    main()
