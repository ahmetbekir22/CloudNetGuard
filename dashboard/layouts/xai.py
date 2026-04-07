"""CloudNetGuard Dashboard — XAI açıklama sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#888888", family="Inter, -apple-system, sans-serif", size=11),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e4e4e8",
                    font=dict(color="#111111", size=12, family="Inter")),
)


def layout() -> html.Div:
    return html.Div([
        html.H2("XAI Açıklamaları", className="page-title"),
        html.P("Bir anomali seçin — modelin kararını açıklayan feature analizi.",
               style={"color": "#888888", "marginBottom": "18px", "fontSize": "13px"}),

        dcc.Dropdown(
            id="xai-anomaly-selector",
            placeholder="Anomali seçin…",
            style={"maxWidth": "580px", "marginBottom": "18px"},
        ),

        html.Div([
            html.Div([
                html.H3("Feature Önem Skorları"),
                dcc.Graph(id="xai-bar-chart", config={"displayModeBar": False},
                          style={"height": "250px"}),
            ], className="chart-card"),
            html.Div([
                html.H3("Kümülatif Etki"),
                dcc.Graph(id="xai-waterfall", config={"displayModeBar": False},
                          style={"height": "250px"}),
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
                      style={"height": "250px"}),
        ], className="chart-card"),

        dcc.Interval(id="interval-xai", interval=5000, n_intervals=0),
    ], className="page-content")


def build_bar_figure(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return _empty()
    names  = [f["feature"] for f in top_features]
    values = [f["importance"] for f in top_features]
    colors = ["#e53935" if f["direction"] in ("high", "present") else "#22c55e"
              for f in top_features]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#888888"),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOT,
        margin=dict(t=4, b=28, l=150, r=56),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#aaaaaa")),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#444444")),
        xaxis_title="Önem",
        height=250,
    )
    return fig


def build_waterfall_figure(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return _empty()
    # Tüm importance pozitif — anomali kararına katkıyı göster
    measures = ["relative"] * len(top_features) + ["total"]
    x_vals   = [f["feature"] for f in top_features] + ["Toplam"]
    y_vals   = [abs(f["importance"]) for f in top_features] + [None]
    colors   = ["#e53935" if f["direction"] in ("high", "present") else "#4f46e5"
                for f in top_features]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x_vals, y=y_vals,
        connector=dict(line=dict(color="#e4e4e8", width=1)),
        increasing=dict(marker=dict(color="#e53935", line=dict(width=0))),
        decreasing=dict(marker=dict(color="#4f46e5", line=dict(width=0))),
        totals=dict(marker=dict(color="#4f46e5", line=dict(width=0))),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOT,
        margin=dict(t=4, b=56, l=44, r=12),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#888888"),
                   tickangle=-30),
        yaxis=dict(gridcolor="#f0f0f2", zeroline=False, tickfont=dict(size=10, color="#aaaaaa")),
        height=250,
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
    fig.add_trace(go.Bar(
        name="Aktüel", x=names, y=actuals,
        marker=dict(color="#4f46e5", opacity=0.8, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.3f}<extra>Aktüel</extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Normal alt", x=names, y=lo, mode="markers",
        marker=dict(symbol="line-ew-open", size=12,
                    color="#22c55e", line=dict(width=2)),
        hovertemplate="%{x}: %{y:.3f}<extra>Normal alt</extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Normal üst", x=names, y=hi, mode="markers",
        marker=dict(symbol="line-ew-open", size=12,
                    color="#d97706", line=dict(width=2)),
        hovertemplate="%{x}: %{y:.3f}<extra>Normal üst</extra>",
    ))
    fig.update_layout(
        **_PLOT,
        margin=dict(t=4, b=56, l=44, r=12),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#888888"),
                   tickangle=-25),
        yaxis=dict(gridcolor="#f0f0f2", zeroline=False, tickfont=dict(size=10, color="#aaaaaa"),
                   range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    orientation="h", x=1, xanchor="right", y=1.12),
        barmode="overlay",
        height=250,
    )
    return fig


def _empty() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#888888"))
    return fig
