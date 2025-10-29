# app.py — BigQuery ohne pyarrow (kein Storage API), CSV-Fallback, Statuszeile

import os
import pandas as pd
from flask import Flask
from flask_basicauth import BasicAuth
from google.cloud import bigquery
from google.oauth2 import service_account
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# ---- Konfig (Env überschreibt Defaults) ----
LOGIN_USER = os.environ.get("LOGIN_USER", "Allen")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "Chester01!")

BQ_PROJECT  = os.environ.get("BQ_PROJECT", "market-growth-monitor")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "EU")
SERVICE_KEY_FILENAME = os.environ.get("SERVICE_KEY_FILENAME", "market-growth-monitor-c20a3876d9c9.json")

BQ_SQL = os.environ.get(
    "BQ_SQL",
    """
    SELECT sector, date, price_return_7d, sentiment_score
    FROM `market-growth-monitor.market_data.market_dashboard_view`
    ORDER BY date DESC
    LIMIT 2000
    """
)

CSV_FALLBACK = os.environ.get("CSV_FALLBACK", "daily_metrics.csv")

# ---- Flask + BasicAuth ----
server = Flask(__name__)
server.config["BASIC_AUTH_USERNAME"] = LOGIN_USER
server.config["BASIC_AUTH_PASSWORD"] = LOGIN_PASS
server.config["BASIC_AUTH_FORCE"] = True
basic_auth = BasicAuth(server)

# ---- Dash ----
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Growth Monitor"

# ---- Daten laden ----
DATA_SOURCE = "Unbekannt"
LAST_UPDATE = "Keine Daten"
LAST_ERROR  = None
SERVICE_ACCOUNT_EMAIL = "unbekannt"

def _abs_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)

def load_data() -> pd.DataFrame:
    global DATA_SOURCE, LAST_UPDATE, LAST_ERROR, SERVICE_ACCOUNT_EMAIL

    # 1) BigQuery (ohne Storage API, damit kein pyarrow nötig ist)
    try:
        key_path = _abs_path(SERVICE_KEY_FILENAME)
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Secret File nicht gefunden: {key_path}")

        creds = service_account.Credentials.from_service_account_file(key_path)
        SERVICE_ACCOUNT_EMAIL = creds.service_account_email
        client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION, credentials=creds)

        job = client.query(BQ_SQL)                 # startet Query
        rows = job.result()                        # wartet auf Ende (RowIterator)
        df_bq = rows.to_dataframe()                # KEIN bqstorage_client -> kein pyarrow nötig

        if df_bq.empty:
            raise ValueError("BigQuery: View/Query lieferte 0 Zeilen.")

        if "date" in df_bq.columns:
            df_bq["date"] = pd.to_datetime(df_bq["date"], errors="coerce")
        for col in ("price_return_7d", "sentiment_score"):
            if col in df_bq.columns:
                df_bq[col] = pd.to_numeric(df_bq[col], errors="coerce")

        DATA_SOURCE = "BigQuery ✅"
        LAST_UPDATE = df_bq["date"].max().strftime("%Y-%m-%d") if "date" in df_bq.columns else "Unbekannt"
        LAST_ERROR = None
        return df_bq

    except Exception as e:
        # 2) CSV Fallback
        DATA_SOURCE = "CSV Fallback ⚓"
        LAST_ERROR = str(e)
        try:
            csv_path = _abs_path(CSV_FALLBACK)
            if os.path.exists(csv_path):
                df_csv = pd.read_csv(csv_path, parse_dates=["date"])
                if not df_csv.empty:
                    LAST_UPDATE = df_csv["date"].max().strftime("%Y-%m-%d")
                    for col in ("sector", "price_return_7d", "sentiment_score"):
                        if col not in df_csv.columns:
                            df_csv[col] = pd.NA
                    return df_csv
        except Exception as e_csv:
            LAST_ERROR += f" | CSV-Fehler: {e_csv}"

        LAST_UPDATE = "Keine Daten"
        return pd.DataFrame(columns=["sector","date","price_return_7d","sentiment_score"])

# Initiale Daten
df = load_data()
sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns and not df.empty else ["Tech"]

# ---- Layout ----
app.layout = dbc.Container(fluid=True, children=[
    html.H2("📈 Market Growth Monitor"),
    html.Div(id="data-status", className="mb-3", style={"fontWeight": "bold"}),

    dbc.Row([
        dbc.Col([
            html.Label("Sektor auswählen"),
            dcc.Dropdown(
                id="sector",
                options=[{"label": s, "value": s} for s in sectors],
                value=sectors[0],
                clearable=False
            )
        ], width=4),
        dbc.Col(dbc.Button("🔄 Daten neu laden", id="btn-refresh", color="primary"), width=2),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="sentiment-chart"), width=6),
        dbc.Col(dcc.Graph(id="return-chart"), width=6)
    ]),
])

# ---- Callback ----
@app.callback(
    Output("sentiment-chart", "figure"),
    Output("return-chart", "figure"),
    Output("data-status", "children"),
    Input("sector", "value"),
    Input("btn-refresh", "n_clicks"),
)
def update_dashboard(selected_sector, _n):
    global df
    df = load_data()

    status = f"📊 Datenquelle: {DATA_SOURCE} | 🕒 Letztes Update: {LAST_UPDATE} | 👤 SA: {SERVICE_ACCOUNT_EMAIL}"
    if LAST_ERROR:
        status += f" | ⚠️ Fehler: {LAST_ERROR}"

    if df.empty or "sector" not in df.columns:
        empty = px.scatter(title="Keine Daten verfügbar")
        return empty, empty, status

    sel = df[df["sector"] == selected_sector].copy() if selected_sector else df.copy()
    if "date" in sel.columns:
        sel = sel.sort_values("date")

    fig_sent = px.line(sel, x="date", y="sentiment_score", title=f"Sentiment Score – {selected_sector}")
    fig_ret  = px.bar(sel, x="date", y="price_return_7d", title=f"7-Tage-Return – {selected_sector}")

    return fig_sent, fig_ret, status

# ---- Start ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

    app.run(host="0.0.0.0", port=port)

