# app.py
import os
import pathlib
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

from flask import Flask, Response
from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

# -----------------------------------------------------------------------------
# Configuration (via environment variables)
# -----------------------------------------------------------------------------
BQ_PROJECT   = os.getenv("BQ_PROJECT", "")
BQ_VIEW      = os.getenv("BQ_VIEW", "")  # e.g. market_data.market_dashboard_view_tickers
BQ_LOCATION  = os.getenv("BQ_LOCATION", "EU")
CREDS_PATH   = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

TIMEZONE = "Europe/Berlin"  # used in queries & timestamps
DEFAULT_LOOKBACK = 10

# -----------------------------------------------------------------------------
# Flask + Dash init
# -----------------------------------------------------------------------------
server = Flask(__name__)

# --- DIAG ENDPOINT ------------------------------------------------------------
@server.route("/diag")
def diag():
    """Small diagnostics page to verify BQ connectivity from the running service."""
    lines = []
    lines.append(f"GOOGLE_APPLICATION_CREDENTIALS={CREDS_PATH}")
    lines.append(f"BQ_PROJECT={BQ_PROJECT}")
    lines.append(f"BQ_VIEW={BQ_VIEW}")
    lines.append(f"BQ_LOCATION={BQ_LOCATION}")

    try:
        client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)
        sql = f"""
        SELECT COUNT(*) c
        FROM `{BQ_PROJECT}.{BQ_VIEW}`
        WHERE date >= DATE_SUB(CURRENT_DATE("{TIMEZONE}"), INTERVAL {DEFAULT_LOOKBACK} DAY)
        """
        df = client.query(sql).result().to_dataframe()
        lines.append(f"Query OK. Rows last {DEFAULT_LOOKBACK}d = {int(df['c'][0])}")
        return Response("\n".join(lines), mimetype="text/plain")
    except Exception:
        lines.append("ERROR:\n" + traceback.format_exc())
        return Response("\n".join(lines), mimetype="text/plain", status=500)
# -----------------------------------------------------------------------------

external_stylesheets = [dbc.themes.COSMO]  # nice clean theme
app: Dash = Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.title = "Market Growth Monitor"

# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
def _bq_enabled() -> bool:
    return bool(BQ_PROJECT and BQ_VIEW and os.path.exists(CREDS_PATH))

def query_bigquery(lookback_days: int) -> pd.DataFrame:
    """
    Query BigQuery view for the last `lookback_days` of data.
    Expected columns in the view:
      date (DATE), sector (STRING), ticker (STRING),
      price_return_7d (FLOAT), sentiment_score (FLOAT)
    """
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    sql = f"""
    SELECT
      date,
      sector,
      ticker,
      price_return_7d,
      sentiment_score
    FROM `{BQ_PROJECT}.{BQ_VIEW}`
    WHERE date >= DATE_SUB(CURRENT_DATE("{TIMEZONE}"), INTERVAL {lookback_days} DAY)
    """
    df = client.query(sql).result().to_dataframe()
    return df

def load_csv_fallback() -> pd.DataFrame:
    """
    Optional fallback: if you have CSVs mounted in ./data/*.csv with
    the same schema, load and concat them. Otherwise return empty df.
    """
    data_dir = pathlib.Path("./data")
    files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f, parse_dates=["date"])
            dfs.append(tmp)
        except Exception:
            pass
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        # ensure dtypes
        if df["date"].dtype != "datetime64[ns]":
            df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=["date", "sector", "ticker", "price_return_7d", "sentiment_score"])

def get_data(lookback_days: int) -> tuple[pd.DataFrame, str, str]:
    """
    Try BigQuery; if it fails, use CSV fallback.
    Returns (df, source_label, last_update_str)
    """
    try:
        if _bq_enabled():
            df = query_bigquery(lookback_days)
            src = "BigQuery ✓"
        else:
            df = load_csv_fallback()
            src = "CSV Fallback ⚠️"
    except (DefaultCredentialsError, Exception):
        df = load_csv_fallback()
        src = "CSV Fallback ⚠️"

    # normalize types
    if not df.empty:
        # if date is string, parse to date
        if df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        last_dt = max(df["date"])
        last_update = pd.to_datetime(str(last_dt)).strftime("%Y-%m-%d")
    else:
        last_update = "Keine Daten"

    return df, src, last_update

# -----------------------------------------------------------------------------
# Ranking Logic
# -----------------------------------------------------------------------------
def compute_topn(df: pd.DataFrame, sector: str, lookback_days: int, topn: int) -> pd.DataFrame:
    """
    Build an 'investment ranking' table for a sector (or 'All'):
      - aggregate last `lookback_days` by ticker
      - compute normalized score of price_return_7d and sentiment_score
      - final_score = 0.6 * return_norm + 0.4 * sentiment_norm
    """
    if df.empty:
        return df

    # Filter by sector if needed
    if sector != "All":
        df = df[df["sector"] == sector]

    if df.empty:
        return df

    # limit last lookback_days already done in query,
    # but if CSV fallback had more, filter again:
    cutoff = pd.to_datetime(datetime.now().date()) - pd.Timedelta(days=lookback_days)
    df_f = df.copy()
    df_f["date_dt"] = pd.to_datetime(df_f["date"])
    df_f = df_f[df_f["date_dt"] >= cutoff]

    # Aggregate by ticker
    agg = df_f.groupby(["ticker", "sector"], as_index=False).agg(
        avg_return_7d = ("price_return_7d", "mean"),
        avg_sentiment  = ("sentiment_score", "mean"),
        last_date      = ("date", "max"),
        n_days         = ("date", "count"),
    )

    # Min-max normalization (robust for bounded 0..1 sentiment)
    def safe_minmax(s: pd.Series) -> pd.Series:
        if s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    agg["return_norm"]    = safe_minmax(agg["avg_return_7d"])
    agg["sentiment_norm"] = safe_minmax(agg["avg_sentiment"])

    # Weighted final score
    agg["final_score"] = 0.6 * agg["return_norm"] + 0.4 * agg["sentiment_norm"]

    # Sort & TopN
    agg = agg.sort_values(["final_score", "avg_return_7d"], ascending=False).head(topn)

    # Nicely formatted columns
    agg = agg[[
        "ticker", "sector", "final_score", "avg_return_7d", "avg_sentiment", "n_days", "last_date"
    ]]
    agg = agg.rename(columns={
        "ticker": "Ticker",
        "sector": "Sektor",
        "final_score": "Score",
        "avg_return_7d": "Ø Return 7d",
        "avg_sentiment": "Ø Sentiment",
        "n_days": "Tage im Fenster",
        "last_date": "Letztes Datum",
    })
    agg["Score"]        = agg["Score"].round(3)
    agg["Ø Return 7d"]  = (agg["Ø Return 7d"] * 100.0).round(2)  # show as %
    agg["Ø Sentiment"]  = agg["Ø Sentiment"].round(3)
    return agg

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
def make_layout():
    return dbc.Container(
        [
            html.H3("Market Growth Monitor", className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="source-banner"), width=4),
                    dbc.Col(html.Div(id="last-update-banner"), width=4),
                    dbc.Col(
                        dbc.Button("Daten neu laden", id="reload", color="primary", className="mt-1"),
                        width=4, className="text-end"
                    ),
                ],
                className="my-2"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Sektor auswählen"),
                            dcc.Dropdown(id="sector-select", options=[{"label":"All","value":"All"}],
                                         value="All", clearable=False),
                        ], md=4
                    ),
                    dbc.Col(
                        [
                            html.Label("Top-N Firmen"),
                            dcc.Dropdown(
                                id="topn-select",
                                options=[{"label":str(n), "value":n} for n in [5,10,15,20]],
                                value=10, clearable=False
                            ),
                        ], md=4
                    ),
                    dbc.Col(
                        [
                            html.Label("Lookback (Tage)"),
                            dcc.Dropdown(
                                id="lookback-select",
                                options=[{"label":str(n), "value":n} for n in [5,7,10,14,21,30]],
                                value=10, clearable=False
                            ),
                        ], md=4
                    ),
                ],
                className="mb-3"
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6(id="chart1-title"),
                            dcc.Graph(id="sentiment-line", config={"displayModeBar": False}),
                        ],
                        md=6
                    ),
                    dbc.Col(
                        [
                            html.H6(id="chart2-title"),
                            dcc.Graph(id="return-bar", config={"displayModeBar": False}),
                        ],
                        md=6
                    ),
                ]
            ),

            html.Hr(),
            html.H5("Top-N Investment Ranking"),
            dash_table.DataTable(
                id="ranking-table",
                columns=[{"name": c, "id": c} for c in
                         ["Ticker", "Sektor", "Score", "Ø Return 7d", "Ø Sentiment", "Tage im Fenster", "Letztes Datum"]],
                page_size=10,
                sort_action="native",
                style_table={"overflowX":"auto"},
                style_cell={"padding":"8px", "fontFamily":"Inter, system-ui, -apple-system, Segoe UI, Roboto"},
                style_header={"fontWeight":"600"},
            ),

            html.Div(id="hidden-status", style={"display":"none"}),  # for cache status/messages
        ],
        fluid=True
    )

app.layout = make_layout()

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("source-banner", "children"),
    Output("last-update-banner", "children"),
    Output("sector-select", "options"),
    Output("sentiment-line", "figure"),
    Output("return-bar", "figure"),
    Output("ranking-table", "data"),
    Output("chart1-title", "children"),
    Output("chart2-title", "children"),
    Input("reload", "n_clicks"),
    Input("sector-select", "value"),
    Input("topn-select", "value"),
    Input("lookback-select", "value"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, sector_value, topn_value, lookback_value):
    lookback = lookback_value or DEFAULT_LOOKBACK
    df, source, last_update = get_data(lookback)

    # Sector options
    sectors = sorted(df["sector"].dropna().unique().tolist()) if not df.empty else []
    sector_options = [{"label":"All", "value":"All"}] + [{"label":s, "value":s} for s in sectors]
    if not sector_value or (sector_value not in ["All"] + sectors):
        sector_value = "All"

    # Titles
    chart1_title = f"Sentiment Score – {sector_value}"
    chart2_title = f"7-Tage-Return – {sector_value}"

    # Filter for charts
    df_plot = df.copy()
    if sector_value != "All":
        df_plot = df_plot[df_plot["sector"] == sector_value]

    # Build Sentiment line (average per day)
    if not df_plot.empty:
        ts_sent = df_plot.groupby("date", as_index=False)["sentiment_score"].mean()
        fig_sent = px.line(ts_sent, x="date", y="sentiment_score")
        fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    else:
        fig_sent = px.line(pd.DataFrame({"date":[], "sentiment_score":[]}), x="date", y="sentiment_score")
        fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    # Build Return bar (average per day)
    if not df_plot.empty:
        ts_ret = df_plot.groupby("date", as_index=False)["price_return_7d"].mean()
        ts_ret["price_return_7d_pct"] = ts_ret["price_return_7d"] * 100.0
        fig_ret = px.bar(ts_ret, x="date", y="price_return_7d_pct")
        fig_ret.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Return 7d (%)")
    else:
        fig_ret = px.bar(pd.DataFrame({"date":[], "price_return_7d_pct":[]}), x="date", y="price_return_7d_pct")
        fig_ret.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Return 7d (%)")

    # Ranking
    ranking_df = compute_topn(df, sector_value, lookback, topn_value or 10)
    ranking_data = ranking_df.to_dict("records") if not ranking_df.empty else []

    # Banners
    source_banner = html.Div(
        [
            html.Span("Datenquelle: "),
            html.Strong(source),
        ]
    )
    last_update_banner = html.Div(
        [
            html.Span("Letztes Update: "),
            html.Strong(last_update),
        ]
    )

    return (source_banner, last_update_banner, sector_options,
            fig_sent, fig_ret, ranking_data, chart1_title, chart2_title)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind to PORT if defined (Render sets $PORT)
    port = int(os.environ.get("PORT", 10000))
   app.run(host="0.0.0.0", port=port, debug=False)



