"""
CloudNetGuard Dashboard — Flask + Plotly Dash ana uygulaması.
Gerçek zamanlı DNS anomali izleme arayüzü.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque, defaultdict

import redis
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from flask import Flask

sys.path.insert(0, "/app/shared")

from schema import AnomalyRecord, SDNAction

from layouts import anomalies as anomalies_layout
from layouts import overview as overview_layout
from layouts import xai as xai_layout

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","service":"dashboard","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

REDIS_HOST  = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.environ.get("REDIS_PORT", "6379"))
PORT        = int(os.environ.get("PORT", "8050"))

ANOMALY_STREAM  = "dns:anomalies"
ACTIONS_STREAM  = "sdn:actions"
CONSUMER_GROUP  = "dashboard-group"
CONSUMER_NAME   = "dashboard-1"

# ---------------------------------------------------------------------------
# Durum deposu (bellek içi, son N kayıt)
# ---------------------------------------------------------------------------

MAX_RECORDS = 2000

_anomalies:  deque[dict] = deque(maxlen=MAX_RECORDS)
_actions:    deque[dict] = deque(maxlen=MAX_RECORDS)
_traffic_ts: deque[str]  = deque(maxlen=300)
_traffic_n:  deque[int]  = deque(maxlen=300)   # normal sayısı
_traffic_a:  deque[int]  = deque(maxlen=300)   # anomali sayısı

_last_anomaly_id  = "$"
_last_action_id   = "$"


# ---------------------------------------------------------------------------
# Redis bağlantısı
# ---------------------------------------------------------------------------

def connect_redis() -> redis.Redis | None:
    for attempt in range(1, 6):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            log.info("Redis bağlantısı kuruldu")
            return r
        except redis.ConnectionError:
            log.warning("Redis bağlanamadı (%d/5), bekleniyor...", attempt)
            time.sleep(2)
    log.error("Redis bağlantısı kurulamadı — demo verisi kullanılacak.")
    return None


_redis: redis.Redis | None = connect_redis()


def _ensure_groups() -> None:
    if _redis is None:
        return
    for stream in (ANOMALY_STREAM, ACTIONS_STREAM):
        try:
            _redis.xgroup_create(stream, CONSUMER_GROUP, id="$", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise


_ensure_groups()


# ---------------------------------------------------------------------------
# Veri çekme
# ---------------------------------------------------------------------------

def fetch_new_data() -> None:
    global _last_anomaly_id, _last_action_id
    if _redis is None:
        return
    try:
        msgs = _redis.xread({ANOMALY_STREAM: _last_anomaly_id}, count=100, block=0)
        if msgs:
            for _s, entries in msgs:
                for msg_id, data in entries:
                    try:
                        rec = AnomalyRecord.from_redis(data)
                        _anomalies.appendleft({
                            "timestamp":      rec.timestamp,
                            "src_ip":         rec.src_ip,
                            "query":          rec.query,
                            "anomaly_score":  rec.anomaly_score,
                            "is_anomaly":     rec.is_anomaly,
                            "predicted_type": rec.predicted_type,
                            "top_features":   [
                                {"feature": f.feature, "importance": f.importance,
                                 "value": f.value, "direction": f.direction}
                                for f in rec.top_features
                            ],
                            "summary":        rec.summary,
                            "feature_vector": rec.feature_vector,
                        })
                    except Exception as exc:
                        log.warning("Anomali parse hatası: %s", exc)
                _last_anomaly_id = entries[-1][0]

        action_msgs = _redis.xread({ACTIONS_STREAM: _last_action_id}, count=100, block=0)
        if action_msgs:
            for _s, entries in action_msgs:
                for msg_id, data in entries:
                    try:
                        act = SDNAction.from_redis(data)
                        _actions.appendleft({
                            "timestamp":      act.timestamp,
                            "src_ip":         act.src_ip,
                            "query":          act.query,
                            "anomaly_score":  act.anomaly_score,
                            "predicted_type": act.predicted_type,
                            "action":         act.action,
                            "reason":         act.reason,
                        })
                    except Exception as exc:
                        log.warning("Aksiyon parse hatası: %s", exc)
                _last_action_id = entries[-1][0]
    except Exception as exc:
        log.error("Redis okuma hatası: %s", exc)


# ---------------------------------------------------------------------------
# Dash uygulaması
# ---------------------------------------------------------------------------

server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[],
    suppress_callback_exceptions=True,
    title="CloudNetGuard",
)

app.layout = html.Div([
    # Nav bar
    html.Nav([
        html.A("CloudNet", className="navbar-brand", href="/"),
        html.Span("Guard", className="navbar-brand"),
        html.Div([
            dcc.Link("Genel Bakış", href="/",          className="nav-link"),
            dcc.Link("Anomaliler",  href="/anomalies", className="nav-link"),
            dcc.Link("XAI",         href="/xai",       className="nav-link"),
        ], className="nav-links"),
    ], className="navbar"),

    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
], style={"minHeight": "100vh", "width": "100%", "backgroundColor": "#ffffff"})


# ---- Sayfa yönlendirme ----

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname: str):
    if pathname == "/anomalies":
        return anomalies_layout.layout()
    if pathname == "/xai":
        return xai_layout.layout()
    return overview_layout.layout()


# ---- Overview callbacks ----

@app.callback(
    Output("metric-total",      "children"),
    Output("metric-anomalies",  "children"),
    Output("metric-ratio",      "children"),
    Output("metric-actions",    "children"),
    Output("traffic-timeline",  "figure"),
    Output("anomaly-gauge",     "figure"),
    Output("threat-pie",        "figure"),
    Output("recent-anomalies-list", "children"),
    Input("interval-overview",  "n_intervals"),
)
def update_overview(_):
    fetch_new_data()

    total_q    = len(_anomalies)
    anomaly_q  = sum(1 for a in _anomalies if a["is_anomaly"])
    ratio      = anomaly_q / total_q if total_q else 0.0
    action_cnt = len(_actions)

    # Trafik zaman serisi — 1 saniyelik dilimler halinde say
    import datetime as _dt
    from collections import OrderedDict
    bins: dict[str, list[int]] = OrderedDict()
    for r in reversed(list(_anomalies)[:600]):
        try:
            sec = r["timestamp"][:19]   # "2026-04-07T10:41:43"
            if sec not in bins:
                bins[sec] = [0, 0]
            if r["is_anomaly"]:
                bins[sec][1] += 1
            else:
                bins[sec][0] += 1
        except Exception:
            pass
    last_bins = list(bins.items())[-40:]   # son 40 saniye
    ts     = [b[0] for b in last_bins]
    normal = [b[1][0] for b in last_bins]
    anom   = [b[1][1] for b in last_bins]

    traffic_fig = overview_layout.build_traffic_figure(ts, normal, anom)
    gauge_fig   = overview_layout.build_gauge_figure(ratio)

    type_counts: dict[str, int] = defaultdict(int)
    for a in list(_anomalies)[:200]:
        if a["is_anomaly"]:
            type_counts[a["predicted_type"]] += 1
    pie_fig = overview_layout.build_pie_figure(dict(type_counts))

    # Son 10 anomali listesi
    recent_items = []
    shown = 0
    for a in list(_anomalies):
        if not a["is_anomaly"]:
            continue
        if shown >= 8:
            break
        ptype = a["predicted_type"]
        q = a["query"]
        q_display = q if len(q) <= 38 else q[:35] + "…"
        recent_items.append(html.Div([
            html.Span(className=f"anomaly-dot"),
            html.Div([
                html.Div(a["src_ip"], className="ip"),
                html.Div(q_display, className="q"),
            ], className="info"),
            html.Span(ptype.upper(), className=f"type-badge {ptype}"),
            html.Span(f"{a['anomaly_score']:.2f}", className="score-badge"),
        ], className=f"anomaly-item {ptype}"))
        shown += 1

    return (
        str(total_q),
        str(anomaly_q),
        f"{ratio*100:.1f}%",
        str(action_cnt),
        traffic_fig,
        gauge_fig,
        pie_fig,
        recent_items,
    )


def _type_color(ptype: str) -> str:
    return {"tunnel": "#cba6f7", "ddos": "#f38ba8", "flux": "#fab387"}.get(ptype, "#a6e3a1")


# ---- Anomalies callbacks ----

@app.callback(
    Output("anomaly-scatter", "figure"),
    Output("anomaly-table",   "data"),
    Input("interval-anomalies", "n_intervals"),
    Input("filter-type",        "value"),
    Input("filter-score",       "value"),
    prevent_initial_call=False,
)
def update_anomalies(_, filter_type, min_score):
    fetch_new_data()
    records = [a for a in list(_anomalies) if a["is_anomaly"]]

    if filter_type and filter_type != "all":
        records = [r for r in records if r["predicted_type"] == filter_type]
    if min_score is not None:
        records = [r for r in records if r["anomaly_score"] >= min_score]

    scatter = anomalies_layout.build_scatter_figure(records)

    # Aksiyon bilgisini anomali listesiyle eşleştir (basit: src_ip ile)
    action_map = {a["src_ip"]: a["action"] for a in list(_actions)[:500]}
    table_data = [
        {
            "timestamp":      r["timestamp"],
            "src_ip":         r["src_ip"],
            "query":          r["query"][:50],
            "anomaly_score":  round(r["anomaly_score"], 3),
            "predicted_type": r["predicted_type"],
            "action":         action_map.get(r["src_ip"], "—"),
        }
        for r in records[:200]
    ]
    return scatter, table_data


# ---- XAI callbacks ----

@app.callback(
    Output("xai-anomaly-selector", "options"),
    Input("interval-xai", "n_intervals"),
)
def update_xai_options(_):
    # top_features boş olsa bile tüm anomalileri göster
    anomaly_list = [a for a in list(_anomalies)[:100] if a["is_anomaly"]]
    return [
        {
            "label": f"{a['timestamp'][11:19]}  {a['src_ip']}  [{a['predicted_type'].upper()}]  {a['anomaly_score']:.2f}",
            "value": str(i),
        }
        for i, a in enumerate(anomaly_list)
    ]


@app.callback(
    Output("xai-bar-chart",         "figure"),
    Output("xai-waterfall",         "figure"),
    Output("xai-summary-box",       "children"),
    Output("xai-feature-comparison","figure"),
    Input("xai-anomaly-selector",   "value"),
    prevent_initial_call=True,
)
def update_xai_detail(selected_idx):
    if selected_idx is None:
        return no_update, no_update, no_update, no_update

    anomaly_list = [a for a in list(_anomalies)[:100] if a["is_anomaly"]]
    idx = int(selected_idx)
    if idx >= len(anomaly_list):
        return no_update, no_update, no_update, no_update

    a = anomaly_list[idx]

    # Semantik olarak doğru feature sırası + gerçek değerler
    top_features = _derive_importance(a)
    summary      = a.get("summary") or _auto_summary(a)

    bar_fig   = xai_layout.build_bar_figure(top_features)
    wfall_fig = xai_layout.build_waterfall_figure(top_features)
    comp_fig  = xai_layout.build_comparison_figure(top_features)

    return bar_fig, wfall_fig, summary, comp_fig


def _derive_importance(a: dict) -> list[dict]:
    """
    Anomali tipine göre semantik feature önceliklendirmesi.
    Feature'ın normal aralığından sapması kadar önemlidir.
    Sıfır değer sorunu: deviation-based importance + minimum floor.
    """
    from schema import FEATURE_NAMES
    ptype = a.get("predicted_type", "tunnel")
    fvec  = a.get("feature_vector", [])
    fdict = dict(zip(FEATURE_NAMES, fvec)) if fvec else {}

    # Normal aralık ortalamaları (build_comparison_figure ile tutarlı)
    _normal_mid = {
        "query_length":          0.175,
        "entropy":               0.300,
        "subdomain_count":       0.150,
        "ttl":                   0.550,
        "query_rate":            0.050,
        "record_type_A":         0.750,
        "record_type_TXT":       0.050,
        "record_type_MX":        0.050,
        "response_size":         0.125,
        "unique_domains":        0.175,
        "is_nxdomain":           0.025,
        "subdomain_digit_ratio": 0.075,
    }

    # (feature, base_weight, direction, low_is_bad)
    rules = {
        "tunnel": [
            ("entropy",               3.0, "high",    False),
            ("query_length",          2.5, "high",    False),
            ("record_type_TXT",       2.0, "present", False),
            ("subdomain_digit_ratio", 1.5, "high",    False),
            ("subdomain_count",       1.0, "high",    False),
        ],
        "ddos": [
            ("response_size",   3.0, "high",    False),
            ("ttl",             2.5, "low",     True),
            ("query_rate",      2.0, "high",    False),
            ("record_type_A",   1.5, "present", False),
            ("subdomain_count", 1.0, "high",    False),
        ],
        "flux": [
            ("is_nxdomain",   3.0, "present", False),
            ("unique_domains",2.5, "high",    False),
            ("ttl",           2.0, "low",     True),
            ("entropy",       1.5, "high",    False),
            ("query_length",  1.0, "high",    False),
        ],
    }.get(ptype, [
        ("entropy",      2.0, "high", False),
        ("query_length", 1.5, "high", False),
        ("query_rate",   1.0, "high", False),
    ])

    results = []
    for feat, weight, direction, low_is_bad in rules:
        val = fdict.get(feat, 0.0)
        mid = _normal_mid.get(feat, 0.3)

        # Deviation from normal midpoint — yüksek sapma = yüksek önem
        if low_is_bad:
            # Düşük değer anomalik: normal_mid'den ne kadar düşük?
            deviation = max(0.0, mid - val)
        else:
            # Yüksek değer anomalik: normal_mid'den ne kadar yüksek?
            deviation = max(0.0, val - mid)

        # Minimum floor: her feature en az %10 * weight kadar önem taşır
        importance = weight * max(deviation, 0.1)

        results.append({
            "feature":    feat,
            "importance": importance,
            "value":      round(val, 4),
            "direction":  direction,
        })

    # Normalize et — toplamı 1'e getir
    total = sum(r["importance"] for r in results) or 1.0
    for r in results:
        r["importance"] = round(r["importance"] / total, 4)

    # Öneme göre sırala (yüksekten düşüğe)
    return sorted(results, key=lambda x: x["importance"], reverse=True)


def _auto_summary(a: dict) -> str:
    ptype = a.get("predicted_type", "tunnel")
    score = a.get("anomaly_score", 0)
    msgs = {
        "tunnel": f"Yüksek entropi ve uzun sorgu uzunluğu DNS tünelleme göstergesi (skor: {score:.2f})",
        "ddos":   f"Yüksek yanıt boyutu ve düşük TTL DDoS saldırısı göstergesi (skor: {score:.2f})",
        "flux":   f"NXDOMAIN oranı ve benzersiz domain sayısı domain flux göstergesi (skor: {score:.2f})",
    }
    return msgs.get(ptype, f"Anomali tespit edildi (skor: {score:.2f})")


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Dashboard başlatılıyor: http://0.0.0.0:%d", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
