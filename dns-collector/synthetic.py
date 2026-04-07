"""
CloudNetGuard — Sentetik DNS trafik üreticisi.
Normal trafik + enjekte edilmiş anomaliler (tunnel, ddos, flux) üretir.
"""

from __future__ import annotations

import math
import random
import string
import time
from dataclasses import dataclass
from typing import Generator

# ---------------------------------------------------------------------------
# Sabit listeler
# ---------------------------------------------------------------------------

POPULAR_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "twitter.com",
    "instagram.com", "linkedin.com", "microsoft.com", "apple.com", "netflix.com",
    "reddit.com", "github.com", "stackoverflow.com", "wikipedia.org", "cloudflare.com",
    "akamai.com", "fastly.com", "cdn77.com", "yahoo.com", "bing.com",
]

QUERY_TYPES = ["A", "AAAA", "TXT", "MX", "CNAME", "NS"]
NORMAL_QUERY_TYPES_WEIGHTS = [0.60, 0.15, 0.05, 0.08, 0.10, 0.02]

TLD_LIST = [".com", ".net", ".org", ".io", ".co", ".info"]

PRIVATE_SUBNETS = [
    ("192.168.1.", 254),
    ("10.0.0.", 254),
    ("172.16.0.", 100),
]


def _random_ip() -> str:
    subnet, count = random.choice(PRIVATE_SUBNETS)
    return subnet + str(random.randint(1, count))


def _shannon_entropy(s: str) -> float:
    """Verilen string'in Shannon entropisini hesapla."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


# ---------------------------------------------------------------------------
# Paket yapısı (raw, feature extraction öncesi)
# ---------------------------------------------------------------------------

@dataclass
class RawDNSPacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    query: str
    query_type: str
    ttl: int
    response_size: int
    is_nxdomain: bool
    label: str              # normal | tunnel | ddos | flux


# ---------------------------------------------------------------------------
# Üretici fonksiyonlar
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """
    Gerçekçi sentetik DNS trafik üreticisi.
    Rate kontrolü, anomali enjeksiyonu ve label üretimi sağlar.
    """

    def __init__(
        self,
        rate: float = 100.0,          # paket/sn
        anomaly_ratio: float = 0.15,  # anomali oranı
        seed: int | None = None,
    ) -> None:
        self.rate = rate
        self.anomaly_ratio = anomaly_ratio
        self._query_counts: dict[str, list[float]] = {}  # ip → zaman damgaları
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------ #
    #   Normal trafik
    # ------------------------------------------------------------------ #

    def _normal_packet(self) -> RawDNSPacket:
        domain = random.choice(POPULAR_DOMAINS)
        subdomain = random.choice(["www", "mail", "cdn", "api", "static", ""])
        query = f"{subdomain}.{domain}" if subdomain else domain
        qtype = random.choices(QUERY_TYPES, weights=NORMAL_QUERY_TYPES_WEIGHTS)[0]
        return RawDNSPacket(
            timestamp=time.time(),
            src_ip=_random_ip(),
            dst_ip="8.8.8.8",
            query=query,
            query_type=qtype,
            ttl=random.randint(60, 3600),
            response_size=random.randint(60, 512),
            is_nxdomain=False,
            label="normal",
        )

    # ------------------------------------------------------------------ #
    #   DNS Tunneling
    # ------------------------------------------------------------------ #

    def _tunnel_packet(self) -> RawDNSPacket:
        """Base64/hex benzeri uzun subdomain — yüksek entropi."""
        payload_len = random.randint(30, 60)
        payload = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=payload_len)
        )
        # Birden fazla etiket oluştur
        chunks = [payload[i : i + 20] for i in range(0, len(payload), 20)]
        evil_domain = random.choice(["evil.com", "c2server.net", "exfil.io"])
        query = ".".join(chunks) + "." + evil_domain
        return RawDNSPacket(
            timestamp=time.time(),
            src_ip=_random_ip(),
            dst_ip="8.8.8.8",
            query=query,
            query_type=random.choices(["TXT", "A", "CNAME"], weights=[0.6, 0.2, 0.2])[0],
            ttl=random.randint(10, 60),
            response_size=random.randint(200, 512),
            is_nxdomain=random.random() < 0.1,
            label="tunnel",
        )

    # ------------------------------------------------------------------ #
    #   DDoS burst
    # ------------------------------------------------------------------ #

    def _ddos_packet(self) -> RawDNSPacket:
        """Aynı hedefe yüksek hızda küçük sorgular."""
        target = random.choice(POPULAR_DOMAINS)
        # Spoofed kaynak IP'ler
        src_ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        return RawDNSPacket(
            timestamp=time.time(),
            src_ip=src_ip,
            dst_ip="8.8.8.8",
            query=target,
            query_type="A",
            ttl=random.randint(1, 30),
            response_size=random.randint(512, 4096),
            is_nxdomain=False,
            label="ddos",
        )

    # ------------------------------------------------------------------ #
    #   Domain flux (DGA benzeri)
    # ------------------------------------------------------------------ #

    def _flux_packet(self) -> RawDNSPacket:
        """DGA benzeri rastgele domain — düşük TTL, yüksek NXDOMAIN."""
        length = random.randint(6, 12)
        name = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
        tld = random.choice(TLD_LIST)
        query = name + tld
        return RawDNSPacket(
            timestamp=time.time(),
            src_ip=_random_ip(),
            dst_ip="8.8.8.8",
            query=query,
            query_type=random.choice(["A", "AAAA"]),
            ttl=random.randint(1, 30),
            response_size=random.randint(60, 150),
            is_nxdomain=random.random() < 0.4,
            label="flux",
        )

    # ------------------------------------------------------------------ #
    #   Ana üretici
    # ------------------------------------------------------------------ #

    def generate(self) -> Generator[RawDNSPacket, None, None]:
        """
        Sonsuz paket akışı üretir.
        rate ve anomaly_ratio çevre değişkenlerine göre ayarlanır.
        """
        interval = 1.0 / max(self.rate, 1)
        anomaly_generators = [
            self._tunnel_packet,
            self._ddos_packet,
            self._flux_packet,
        ]
        while True:
            if random.random() < self.anomaly_ratio:
                func = random.choice(anomaly_generators)
                yield func()
            else:
                yield self._normal_packet()
            time.sleep(interval)

    def generate_batch(self, n: int) -> list[RawDNSPacket]:
        """n adet paket üretir (rate sleep olmadan — eğitim verisi için)."""
        packets: list[RawDNSPacket] = []
        anomaly_generators = [
            self._tunnel_packet,
            self._ddos_packet,
            self._flux_packet,
        ]
        for _ in range(n):
            if random.random() < self.anomaly_ratio:
                func = random.choice(anomaly_generators)
                packets.append(func())
            else:
                packets.append(self._normal_packet())
        return packets
