"""CloudNetGuard Dashboard — Overview sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#888888", family="Inter, -apple-system, sans-serif", size=11),
    margin=dict(t=4, b=32, l=44, r=12),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#aaaaaa"),
               linecolor="#e4e4e8", tickcolor="#e4e4e8"),
    yaxis=dict(gridcolor="#f0f0f2", zeroline=False, tickfont=dict(size=10, color="#aaaaaa")),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e4e4e8",
                    font=dict(color="#111111", size=12, family="Inter")),
)


def layout() -> html.Div:
    return html.Div([
        html.H2("Genel Bakış", className="page-title"),

        html.Div([
            _metric("metric-total",     "Toplam Sorgu",   "0"),
            _metric("metric-anomalies", "Anomali",        "0"),
            _metric("metric-ratio",     "Anomali Oranı",  "0%"),
            _metric("metric-actions",   "SDN Aksiyonu",   "0"),
        ], className="metrics-row"),

        html.Div([
            html.Div([
                html.H3("DNS Trafik Akışı"),
                dcc.Graph(id="traffic-timeline", config={"displayModeBar": False},
                          style={"height": "190px"}),
            ], className="chart-card wide"),
            html.Div([
                html.H3("Anomali Oranı"),
                dcc.Graph(id="anomaly-gauge", config={"displayModeBar": False},
                          style={"height": "190px"}),
            ], className="chart-card narrow"),
        ], className="charts-row"),

        html.Div([
            html.Div([
                html.H3("Tehdit Dağılımı"),
                dcc.Graph(id="threat-pie", config={"displayModeBar": False},
                          style={"height": "210px"}),
            ], className="chart-card"),
            html.Div([
                html.H3("Son Anomaliler"),
                html.Div(id="recent-anomalies-list"),
            ], className="chart-card"),
        ], className="charts-row"),

        dcc.Interval(id="interval-overview", interval=2000, n_intervals=0),
    ], className="page-content")


def _metric(id_: str, label: str, value: str) -> html.Div:
    return html.Div([html.H4(label), html.H2(value, id=id_)], className="metric-card")


def build_traffic_figure(timestamps, normal_counts, anomaly_counts) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=normal_counts, name="Normal",
        mode="lines", line=dict(color="#22c55e", width=1.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.05)",
        hovertemplate="%{y}<extra>Normal</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=anomaly_counts, name="Anomali",
        mode="lines", line=dict(color="#e53935", width=1.5),
        fill="tozeroy", fillcolor="rgba(229,57,53,0.06)",
        hovertemplate="%{y}<extra>Anomali</extra>",
    ))
    fig.update_layout(
        **_PLOT,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    x=1, xanchor="right", y=1.15,
                    font=dict(size=11), itemsizing="constant"),
        hovermode="x unified",
    )
    return fig


def build_gauge_figure(anomaly_ratio: float) -> go.Figure:
    pct = round(anomaly_ratio * 100, 1)
    color = "#22c55e" if pct < 15 else "#d97706" if pct < 40 else "#e53935"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color="#111111", size=30, family="Inter"),
                    valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0,
                      tickfont=dict(size=9, color="#bbbbbb"),
                      tickvals=[0, 50, 100]),
            bar=dict(color=color, thickness=0.55),
            bgcolor="#f7f7f8",
            bordercolor="#e4e4e8",
            borderwidth=1,
            steps=[dict(range=[0, 100], color="#f7f7f8")],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#888888"),
        margin=dict(t=16, b=16, l=24, r=24),
        height=190,
    )
    return fig


def build_pie_figure(type_counts: dict) -> go.Figure:
    if not type_counts:
        type_counts = {"—": 1}
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = {"tunnel": "#4f46e5", "ddos": "#e53935", "flux": "#d97706", "—": "#e4e4e8"}
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[colors.get(l, "#888888") for l in labels],
                    line=dict(color="#ffffff", width=2)),
        hole=0.68,
        textfont=dict(size=11, color="#111111"),
        hovertemplate="%{label}: %{value}<extra></extra>",
        showlegend=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888888", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), x=1, xanchor="right"),
        margin=dict(t=4, b=4, l=4, r=80),
        height=210,
    )
    return fig
