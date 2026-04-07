"""CloudNetGuard Dashboard — Anomali detay sayfası."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html

_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#888888", family="Inter, -apple-system, sans-serif", size=11),
    margin=dict(t=4, b=32, l=44, r=12),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#aaaaaa"),
               linecolor="#e4e4e8"),
    yaxis=dict(gridcolor="#f0f0f2", zeroline=False, tickfont=dict(size=10, color="#aaaaaa")),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e4e4e8",
                    font=dict(color="#111111", size=12, family="Inter")),
)

_TABLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": "#f7f7f8",
        "color": "#888888",
        "fontWeight": "500",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.06em",
        "border": "none",
        "borderBottom": "1px solid #e4e4e8",
        "padding": "10px 14px",
        "fontFamily": "Inter, sans-serif",
    },
    style_cell={
        "backgroundColor": "#ffffff",
        "color": "#111111",
        "border": "none",
        "borderBottom": "1px solid #f0f0f2",
        "padding": "9px 14px",
        "fontSize": "12px",
        "fontFamily": "Inter, sans-serif",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
        {"if": {"filter_query": '{predicted_type} = "ddos"'},
         "color": "#e53935"},
        {"if": {"filter_query": '{predicted_type} = "tunnel"'},
         "color": "#4f46e5"},
        {"if": {"filter_query": '{predicted_type} = "flux"'},
         "color": "#d97706"},
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
                        {"label": "Tümü",   "value": "all"},
                        {"label": "Tünel",  "value": "tunnel"},
                        {"label": "DDoS",   "value": "ddos"},
                        {"label": "Flux",   "value": "flux"},
                    ],
                    value="all",
                    clearable=False,
                    style={"width": "160px"},
                ),
            ]),
            html.Div([
                html.Label("Min Anomali Skoru"),
                dcc.Slider(id="filter-score", min=0, max=1, step=0.05, value=0.5,
                           marks={0: "0", 0.5: "0.5", 1: "1"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "minWidth": "200px", "maxWidth": "340px"}),
        ], className="filters-row"),

        html.Div([
            html.H3("Zaman Serisi"),
            dcc.Graph(id="anomaly-scatter", config={"displayModeBar": False},
                      style={"height": "240px"}),
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
                **_TABLE,
            ),
        ], className="chart-card"),

        dcc.Interval(id="interval-anomalies", interval=3000, n_intervals=0),
    ], className="page-content")


def build_scatter_figure(records: list[dict]) -> go.Figure:
    palette = {"tunnel": "#4f46e5", "ddos": "#e53935", "flux": "#d97706"}
    fig = go.Figure()
    for atype, color in palette.items():
        subset = [r for r in records if r.get("predicted_type") == atype]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[r["timestamp"] for r in subset],
            y=[r["anomaly_score"] for r in subset],
            mode="markers",
            name=atype.upper(),
            marker=dict(color=color, size=6, opacity=0.7, line=dict(width=0)),
            customdata=[[r["src_ip"], r["query"]] for r in subset],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Skor: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOT,
        yaxis_range=[0, 1],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h",
                    x=1, xanchor="right", y=1.12),
        hovermode="closest",
    )
    return fig
