"""
CloudNetGuard — DNS paket feature extraction modülü.
RawDNSPacket → 12 boyutlu normalize feature vektörü.
"""

from __future__ import annotations

import math
import time
from collections import deque

from synthetic import RawDNSPacket

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "query_length",
    "entropy",
    "subdomain_count",
    "ttl",
    "query_rate",
    "record_type_A",
    "record_type_TXT",
    "record_type_MX",
    "response_size",
    "unique_domains",
    "is_nxdomain",
    "subdomain_digit_ratio",
]

# Min-max normalizasyon sınırları (domain bilgisine göre ayarlandı)
_NORM_BOUNDS: dict[str, tuple[float, float]] = {
    "query_length":          (5.0,   200.0),
    "entropy":               (0.0,   5.0),
    "subdomain_count":       (1.0,   20.0),
    "ttl":                   (0.0,   3600.0),
    "query_rate":            (0.0,   200.0),
    "record_type_A":         (0.0,   1.0),
    "record_type_TXT":       (0.0,   1.0),
    "record_type_MX":        (0.0,   1.0),
    "response_size":         (40.0,  4096.0),
    "unique_domains":        (1.0,   500.0),
    "is_nxdomain":           (0.0,   1.0),
    "subdomain_digit_ratio": (0.0,   1.0),
}


def _clamp_normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


# ---------------------------------------------------------------------------
# Durum bilgisi gerektiren feature'lar için pencere takibi
# ---------------------------------------------------------------------------

class _WindowTracker:
    """Son T saniyelik penceredeki istatistikleri tutar."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._window = window_seconds
        self._ip_times: dict[str, deque[float]] = {}
        self._domain_window: deque[tuple[float, str]] = deque()

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._window
        # Domain penceresi temizle
        while self._domain_window and self._domain_window[0][0] < cutoff:
            self._domain_window.popleft()

    def update(self, src_ip: str, query: str, now: float) -> None:
        # IP sorgu zamanları
        if src_ip not in self._ip_times:
            self._ip_times[src_ip] = deque()
        self._ip_times[src_ip].append(now)
        cutoff = now - self._window
        while self._ip_times[src_ip] and self._ip_times[src_ip][0] < cutoff:
            self._ip_times[src_ip].popleft()
        # Domain penceresi
        self._domain_window.append((now, query))
        self._evict_old(now)

    def query_rate(self, src_ip: str) -> float:
        """Son penceredeki sorgu/sn."""
        times = self._ip_times.get(src_ip, deque())
        return len(times) / self._window

    def unique_domains(self) -> int:
        """Son penceredeki benzersiz domain sayısı."""
        return len({d for _, d in self._domain_window})


# Modül düzeyinde paylaşılan tracker
_tracker = _WindowTracker(window_seconds=5.0)


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def extract_features(packet: RawDNSPacket, tracker: "_WindowTracker | None" = None) -> list[float]:
    """
    RawDNSPacket'ten 12 boyutlu normalize feature vektörü çıkar.
    Döndürülen liste FEATURE_NAMES sırasını takip eder.
    tracker: özel tracker (batch/eğitim için); None ise global runtime tracker kullanılır.
    """
    now = packet.timestamp  # packet'in kendi zaman damgasını kullan
    t = tracker if tracker is not None else _tracker
    t.update(packet.src_ip, packet.query, now)

    # ---- ham değerler ----
    parts = packet.query.split(".")
    subdomain_parts = parts[:-2] if len(parts) > 2 else []
    subdomain_str = ".".join(subdomain_parts)

    query_length       = float(len(packet.query))
    entropy            = _shannon_entropy(subdomain_str) if subdomain_str else _shannon_entropy(packet.query)
    subdomain_count    = float(len(parts))
    ttl                = float(packet.ttl)
    query_rate         = t.query_rate(packet.src_ip)
    record_type_a      = 1.0 if packet.query_type == "A" else 0.0
    record_type_txt    = 1.0 if packet.query_type == "TXT" else 0.0
    record_type_mx     = 1.0 if packet.query_type == "MX" else 0.0
    response_size      = float(packet.response_size)
    unique_domains     = float(t.unique_domains())
    is_nxdomain        = 1.0 if packet.is_nxdomain else 0.0

    digit_ratio = 0.0
    if subdomain_str:
        digits = sum(c.isdigit() for c in subdomain_str)
        digit_ratio = digits / len(subdomain_str)

    raw = {
        "query_length":          query_length,
        "entropy":               entropy,
        "subdomain_count":       subdomain_count,
        "ttl":                   ttl,
        "query_rate":            query_rate,
        "record_type_A":         record_type_a,
        "record_type_TXT":       record_type_txt,
        "record_type_MX":        record_type_mx,
        "response_size":         response_size,
        "unique_domains":        unique_domains,
        "is_nxdomain":           is_nxdomain,
        "subdomain_digit_ratio": digit_ratio,
    }

    return [
        _clamp_normalize(raw[name], *_NORM_BOUNDS[name])
        for name in FEATURE_NAMES
    ]
