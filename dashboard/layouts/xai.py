"""CloudNetGuard Dashboard — XAI açıklama sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#6b6b80", family="Inter, system-ui, sans-serif", size=11),
    margin=dict(t=10, b=36, l=160, r=32),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e1e2a", zeroline=False, tickfont=dict(size=10)),
)


def layout() -> html.Div:
    return html.Div([
        html.H2("XAI Açıklamaları", className="page-title"),
        html.P("Bir anomali seçin — modelin neden bu kararı verdiğini görün.",
               style={"color": "#6b6b80", "marginBottom": "20px", "fontSize": "13px"}),

        dcc.Dropdown(
            id="xai-anomaly-selector",
            placeholder="Anomali seçin...",
            style={"maxWidth": "620px", "marginBottom": "20px"},
        ),

        html.Div([
            html.Div([
                html.H3("Feature Önem Skorları"),
                dcc.Graph(id="xai-bar-chart", config={"displayModeBar": False},
                          style={"height": "260px"}),
            ], className="chart-card"),
            html.Div([
                html.H3("Kümülatif Etki"),
                dcc.Graph(id="xai-waterfall", config={"displayModeBar": False},
                          style={"height": "260px"}),
            ], className="chart-card"),
        ], className="charts-row"),

        html.Div([
            html.H3("Açıklama"),
            html.Div(id="xai-summary-box", className="summary-box",
                     children="Yukarıdan bir anomali seçin."),
        ], className="chart-card"),

        html.Div([
            html.H3("Feature Değerleri / Normal Aralık"),
            dcc.Graph(id="xai-feature-comparison", config={"displayModeBar": False},
                      style={"height": "260px"}),
        ], className="chart-card"),

        dcc.Interval(id="interval-xai", interval=5000, n_intervals=0),
    ], className="page-content")


def build_bar_figure(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return _empty()
    names  = [f["feature"] for f in top_features]
    values = [f["importance"] for f in top_features]
    colors = ["#ef4444" if f["direction"] in ("high", "present") else "#22c55e"
              for f in top_features]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#6b6b80"),
    ))
    fig.update_layout(**_PLOT, xaxis_title="Önem")
    return fig


def build_waterfall_figure(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return _empty()
    measures = ["relative"] * len(top_features) + ["total"]
    x_vals   = [f["feature"] for f in top_features] + ["Toplam"]
    y_vals   = [f["importance"] * (1 if f["direction"] in ("high","present") else -1)
                for f in top_features] + [None]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x_vals, y=y_vals,
        connector=dict(line=dict(color="#1e1e2a", width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#6366f1")),
    ))
    fig.update_layout(
        **_PLOT,
        margin=dict(t=10, b=60, l=32, r=16),
    )
    return fig


def build_comparison_figure(
    top_features: list[dict],
    normal_ranges: dict | None = None,
) -> go.Figure:
    if not top_features:
        return _empty()
    if normal_ranges is None:
        normal_ranges = {
            "query_length":          (0.05, 0.30),
            "entropy":               (0.10, 0.50),
            "subdomain_count":       (0.05, 0.25),
            "ttl":                   (0.20, 0.90),
            "query_rate":            (0.00, 0.10),
            "record_type_A":         (0.50, 1.00),
            "record_type_TXT":       (0.00, 0.10),
            "record_type_MX":        (0.00, 0.10),
            "response_size":         (0.05, 0.20),
            "unique_domains":        (0.05, 0.30),
            "is_nxdomain":           (0.00, 0.05),
            "subdomain_digit_ratio": (0.00, 0.15),
        }
    names   = [f["feature"] for f in top_features]
    actuals = [f["value"]   for f in top_features]
    lo      = [normal_ranges.get(n, (0, 0.5))[0] for n in names]
    hi      = [normal_ranges.get(n, (0, 0.5))[1] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Aktüel", x=names, y=actuals,
                         marker=dict(color="#6366f1", line=dict(width=0)),
                         opacity=0.85))
    fig.add_trace(go.Scatter(name="Normal min", x=names, y=lo, mode="markers",
                             marker=dict(symbol="line-ew-open", size=14,
                                         color="#22c55e", line=dict(width=2))))
    fig.add_trace(go.Scatter(name="Normal max", x=names, y=hi, mode="markers",
                             marker=dict(symbol="line-ew-open", size=14,
                                         color="#f59e0b", line=dict(width=2))))
    fig.update_layout(
        **_PLOT,
        margin=dict(t=10, b=60, l=48, r=16),
        yaxis_range=[0, 1],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        barmode="overlay",
    )
    return fig


def _empty() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#6b6b80"))
    return fig
