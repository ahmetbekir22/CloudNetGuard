"""
CloudNetGuard — CIC-Bell-DNS-2021 dataset adapter.
Domain reputation CSV'lerini modelimizin 12 feature vektörüne dönüştürür.

Mevcut sütunlar:   TTL, len, entropy, numeric_percentage, subdomain, tld
Eksik sütunlar:    query_rate, response_size, record_type_*, unique_domains,
                   is_nxdomain → label'a göre makul varsayılanlarla doldurulur.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Iterator

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dns-collector"))
from feature_extractor import FEATURE_NAMES, _clamp_normalize, _NORM_BOUNDS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sütun eşleme
# ---------------------------------------------------------------------------

# CSV sütun → bizim feature adı ve normalize aralığı
_COL_MAP = {
    "len":                ("query_length",          5.0,   200.0),
    "entropy":            ("entropy",               0.0,   5.0),
    "TTL":                ("ttl",                   0.0,   3600.0),
    "numeric_percentage": ("subdomain_digit_ratio", 0.0,   1.0),
}

# Label kategorilerine göre eksik feature varsayılanları
# normal: düşük query_rate, A tipi sorgular, küçük response
# anomali: yüksek query_rate, TXT ağırlıklı, büyük response
_DEFAULTS: dict[str, dict[str, float]] = {
    "normal": {
        "query_rate":     0.05,   # düşük
        "record_type_A":  0.85,
        "record_type_TXT":0.03,
        "record_type_MX": 0.05,
        "response_size":  0.08,   # küçük
        "unique_domains": 0.06,
        "is_nxdomain":    0.01,
        "subdomain_count":0.12,
    },
    "malware": {
        "query_rate":     0.45,
        "record_type_A":  0.40,
        "record_type_TXT":0.45,   # tünel şüphesi
        "record_type_MX": 0.05,
        "response_size":  0.55,
        "unique_domains": 0.60,
        "is_nxdomain":    0.30,
        "subdomain_count":0.55,
    },
    "phishing": {
        "query_rate":     0.20,
        "record_type_A":  0.70,
        "record_type_TXT":0.10,
        "record_type_MX": 0.10,
        "response_size":  0.20,
        "unique_domains": 0.40,
        "is_nxdomain":    0.15,
        "subdomain_count":0.30,
    },
    "spam": {
        "query_rate":     0.35,
        "record_type_A":  0.50,
        "record_type_TXT":0.20,
        "record_type_MX": 0.25,
        "response_size":  0.30,
        "unique_domains": 0.50,
        "is_nxdomain":    0.20,
        "subdomain_count":0.25,
    },
}

# label → binary (trainer için)
LABEL_MAP = {
    "normal":   "normal",
    "malware":  "tunnel",
    "phishing": "flux",
    "spam":     "ddos",
}


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if (v == v) else default   # NaN kontrolü
    except (ValueError, TypeError):
        return default


def _row_to_features(row: dict, category: str) -> list[float]:
    """Tek bir CSV satırını 12 feature vektörüne çevir."""
    defaults = _DEFAULTS.get(category, _DEFAULTS["normal"])
    feat: dict[str, float] = dict(defaults)  # eksik feature'lar için başlangıç

    # CSV'den gelen sütunları override et
    for col, (fname, lo, hi) in _COL_MAP.items():
        if col in row:
            raw = _safe_float(row[col], default=(lo + hi) / 2)
            feat[fname] = _clamp_normalize(raw, lo, hi)

    # subdomain sütunu varsa subdomain_count'u override et
    if "subdomain" in row and row["subdomain"].strip() not in ("", "nan", "0"):
        parts = row["subdomain"].strip().split(".")
        raw_count = len(parts) + 2   # subdomain + sld + tld
        feat["subdomain_count"] = _clamp_normalize(float(raw_count), 1.0, 20.0)

    return [feat[name] for name in FEATURE_NAMES]


def _bracket_split(line: str) -> list[str]:
    """
    Parantez/köşeli parantez içindeki virgülleri yok sayan CSV splitter.
    ['a', 'b'], defaultdict(...), [('x', 1), ('y', 2)] gibi alanları tek parça tutar.
    """
    import re
    fields: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in line.rstrip("\n"):
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    fields.append("".join(current).strip())
    return fields


def _merge_extra_fields(fields: list[str], expected: int) -> list[str]:
    """
    Bracket-split sonrası hâlâ fazla alan varsa (Domain_Age, Registrar gibi
    bracket dışı virgüller) pattern tabanlı birleştirme yapar.
    """
    import re
    result = list(fields)
    i = 0
    while len(result) > expected and i < len(result) - 1:
        # "8256 days" → bir sonraki "11:53:49..." ile birleştir
        if re.match(r"^\d+ days$", result[i]):
            result[i] = result[i] + ", " + result.pop(i + 1)
            continue
        # "MarkMonitor" → sonrası "Inc." / "LLC" / "Ltd" ise birleştir
        if i > 0 and re.match(r"^(Inc|LLC|Ltd|Co|Corp|GmbH|B\.V)\.?$", result[i], re.I):
            result[i - 1] = result[i - 1] + ", " + result.pop(i)
            i = max(0, i - 1)
            continue
        i += 1
    return result


def _parse_csv_robust(path: str) -> Iterator[dict]:
    """
    CIC CSV'lerindeki çoklu gömülü virgül sorununu çözen parser.
    1) Bracket-aware split: list/dict literal içindeki virgülleri atlar.
    2) Kalan fazla alanları pattern ile birleştirir (Domain_Age, Registrar).
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return

    header = next(csv.reader([lines[0]]))
    # Duplike sütun adlarını (ör. iki "Country") tekil yap
    seen: dict[str, int] = {}
    clean_header: list[str] = []
    for col in header:
        if col in seen:
            seen[col] += 1
            clean_header.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            clean_header.append(col)
    n_cols = len(clean_header)

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        fields = _bracket_split(line)

        if len(fields) != n_cols:
            fields = _merge_extra_fields(fields, n_cols)

        # Hâlâ eşleşmiyorsa satırı atla
        if len(fields) != n_cols:
            continue

        yield dict(zip(clean_header, fields))


def iter_dataset(
    csv_dir: str,
    max_benign: int = 50_000,
    max_anomaly: int = 5_000,
) -> Iterator[tuple[list[float], str]]:
    """
    Yields: (feature_vector, label_str)
    label_str: "normal" | "tunnel" | "flux" | "ddos"
    """
    files = {
        "normal":   ("CSV_benign.csv",   max_benign),
        "malware":  ("CSV_malware.csv",  max_anomaly),
        "phishing": ("CSV_phishing.csv", max_anomaly),
        "spam":     ("CSV_spam.csv",     max_anomaly),
    }

    for category, (fname, limit) in files.items():
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            log.warning("Dosya bulunamadı, atlandı: %s", path)
            continue

        label = LABEL_MAP[category]
        count = 0
        for row in _parse_csv_robust(path):
            if count >= limit:
                break
            features = _row_to_features(row, category)
            yield features, label
            count += 1
        log.info("Yüklendi: %s → %d satır (%s)", fname, count, label)


def load_arrays(
    csv_dir: str,
    max_benign: int = 50_000,
    max_anomaly: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Döndürür: (X, y)
    X: (N, 12) float32
    y: (N,)    float32  — 0=normal, 1=anomali
    """
    X_list, y_list = [], []
    label_to_int = {"normal": 0, "tunnel": 1, "flux": 1, "ddos": 1}

    for features, label in iter_dataset(csv_dir, max_benign, max_anomaly):
        X_list.append(features)
        y_list.append(label_to_int.get(label, 1))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info("Toplam: %d örnek | Normal: %d | Anomali: %d",
             len(y), int((y == 0).sum()), int((y == 1).sum()))
    return X, y


if __name__ == "__main__":
    # Hızlı test
    csv_dir = "/Users/ahmetbekir/Downloads/CSVs"
    X, y = load_arrays(csv_dir, max_benign=1000, max_anomaly=200)
    print("X shape:", X.shape)
    print("İlk örnek:", X[0])
    print("Label dağılımı — normal:", (y == 0).sum(), "anomali:", (y == 1).sum())
