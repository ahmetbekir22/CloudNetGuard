"""CloudNetGuard Dashboard — Overview sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#6b6b80", family="Inter, system-ui, sans-serif", size=11),
    margin=dict(t=10, b=36, l=48, r=16),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e1e2a", zeroline=False, tickfont=dict(size=10)),
)


def layout() -> html.Div:
    return html.Div([
        html.H2("Genel Bakış", className="page-title"),

        html.Div([
            _metric("metric-total",     "Toplam Sorgu",    "0"),
            _metric("metric-anomalies", "Anomali",         "0"),
            _metric("metric-ratio",     "Anomali Oranı",   "0%"),
            _metric("metric-actions",   "SDN Aksiyonu",    "0"),
        ], className="metrics-row"),

        html.Div([
            html.Div([
                html.H3("DNS Trafik Akışı"),
                dcc.Graph(id="traffic-timeline", config={"displayModeBar": False},
                          style={"height": "200px"}),
            ], className="chart-card wide"),
            html.Div([
                html.H3("Anomali Oranı"),
                dcc.Graph(id="anomaly-gauge", config={"displayModeBar": False},
                          style={"height": "200px"}),
            ], className="chart-card narrow"),
        ], className="charts-row"),

        html.Div([
            html.Div([
                html.H3("Tehdit Dağılımı"),
                dcc.Graph(id="threat-pie", config={"displayModeBar": False},
                          style={"height": "220px"}),
            ], className="chart-card"),
            html.Div([
                html.H3("Son Anomaliler"),
                html.Div(id="recent-anomalies-list"),
            ], className="chart-card"),
        ], className="charts-row"),

        dcc.Interval(id="interval-overview", interval=2000, n_intervals=0),
    ], className="page-content")


def _metric(id_: str, label: str, value: str) -> html.Div:
    return html.Div([
        html.H4(label),
        html.H2(value, id=id_),
    ], className="metric-card")


def build_traffic_figure(timestamps, normal_counts, anomaly_counts) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=normal_counts, name="Normal",
        mode="lines",
        line=dict(color="#22c55e", width=1.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=anomaly_counts, name="Anomali",
        mode="lines",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
    ))
    fig.update_layout(
        **_PLOT,
        legend=dict(bgcolor="rgba(0,0,0,0)", x=1, xanchor="right", y=1,
                    font=dict(size=10), orientation="h"),
        hovermode="x unified",
    )
    return fig


def build_gauge_figure(anomaly_ratio: float) -> go.Figure:
    pct = round(anomaly_ratio * 100, 1)
    color = "#22c55e" if pct < 20 else "#f59e0b" if pct < 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color="#e8e8f0", size=28, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor="rgba(0,0,0,0)",
                      tickfont=dict(size=9, color="#6b6b80")),
            bar=dict(color=color, thickness=0.6),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 100], color="#1e1e2a"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b6b80", family="Inter"),
        margin=dict(t=20, b=20, l=20, r=20),
        height=200,
    )
    return fig


def build_pie_figure(type_counts: dict) -> go.Figure:
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = {"tunnel": "#6366f1", "ddos": "#ef4444", "flux": "#f59e0b", "normal": "#22c55e"}
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[colors.get(l, "#6b6b80") for l in labels],
                    line=dict(color="#0a0a0f", width=2)),
        hole=0.65,
        textfont=dict(size=11, color="#e8e8f0"),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b6b80", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(t=8, b=8, l=8, r=8),
        height=220,
        showlegend=True,
    )
    return fig
