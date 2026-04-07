"""CloudNetGuard Dashboard — Anomali detay sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#6b6b80", family="Inter, system-ui, sans-serif", size=11),
    margin=dict(t=10, b=36, l=48, r=16),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e1e2a", zeroline=False, tickfont=dict(size=10)),
)

_TABLE_STYLE = dict(
    style_table={"overflowX": "auto", "borderRadius": "8px", "overflow": "hidden"},
    style_header={
        "backgroundColor": "#111118",
        "color": "#6b6b80",
        "fontWeight": "500",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.06em",
        "border": "none",
        "borderBottom": "1px solid #1e1e2a",
        "padding": "10px 14px",
    },
    style_cell={
        "backgroundColor": "#0a0a0f",
        "color": "#e8e8f0",
        "border": "none",
        "borderBottom": "1px solid #1e1e2a",
        "padding": "10px 14px",
        "fontSize": "12px",
        "fontFamily": "Inter, system-ui, sans-serif",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#0d0d14"},
        {"if": {"filter_query": '{predicted_type} = "ddos"'},   "color": "#ef4444"},
        {"if": {"filter_query": '{predicted_type} = "tunnel"'}, "color": "#6366f1"},
        {"if": {"filter_query": '{predicted_type} = "flux"'},   "color": "#f59e0b"},
    ],
)


def layout() -> html.Div:
    return html.Div([
        html.H2("Anomali Detayları", className="page-title"),

        html.Div([
            html.Div([
                html.Label("Tehdit Tipi"),
                dcc.Dropdown(
                    id="filter-type",
                    options=[
                        {"label": "Tümü",    "value": "all"},
                        {"label": "Tünel",   "value": "tunnel"},
                        {"label": "DDoS",    "value": "ddos"},
                        {"label": "Flux",    "value": "flux"},
                    ],
                    value="all",
                    clearable=False,
                    style={"width": "180px"},
                ),
            ]),
            html.Div([
                html.Label("Min Skor"),
                dcc.Slider(id="filter-score", min=0, max=1, step=0.05, value=0.5,
                           marks={0: "0", 0.5: "0.5", 1: "1"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "minWidth": "200px", "maxWidth": "360px"}),
        ], className="filters-row"),

        html.Div([
            html.H3("Anomali Zaman Serisi"),
            dcc.Graph(id="anomaly-scatter", config={"displayModeBar": False},
                      style={"height": "260px"}),
        ], className="chart-card"),

        html.Div([
            html.H3("Anomali Kayıtları"),
            dash_table.DataTable(
                id="anomaly-table",
                columns=[
                    {"name": "Zaman",     "id": "timestamp"},
                    {"name": "Kaynak IP", "id": "src_ip"},
                    {"name": "Sorgu",     "id": "query"},
                    {"name": "Skor",      "id": "anomaly_score"},
                    {"name": "Tip",       "id": "predicted_type"},
                    {"name": "Aksiyon",   "id": "action"},
                ],
                page_size=15,
                sort_action="native",
                filter_action="native",
                **_TABLE_STYLE,
            ),
        ], className="chart-card"),

        dcc.Interval(id="interval-anomalies", interval=3000, n_intervals=0),
    ], className="page-content")


def build_scatter_figure(records: list[dict]) -> go.Figure:
    colors = {"tunnel": "#6366f1", "ddos": "#ef4444", "flux": "#f59e0b"}
    fig = go.Figure()
    for atype, color in colors.items():
        subset = [r for r in records if r.get("predicted_type") == atype]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[r["timestamp"] for r in subset],
            y=[r["anomaly_score"] for r in subset],
            mode="markers",
            name=atype.upper(),
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(width=0)),
            customdata=[[r["src_ip"], r["query"]] for r in subset],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Skor: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOT,
        yaxis_range=[0, 1],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                    x=1, xanchor="right", y=1.1),
        hovermode="closest",
    )
    return fig
