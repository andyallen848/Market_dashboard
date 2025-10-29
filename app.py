# app.py — Market Growth Monitor (Dash + BigQuery, EU-Region, CSV-Fallback)

import os
from datetime import datetime
import pandas as pd

from flask import Flask
from flask_basicauth import BasicAuth
from google.cloud import bigquery

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


# -----------------------------------------------------------------------------
# Konfiguration (über Env Vars anpassbar)
# -----------------------------------------------------------------------------
# Auth
LOGIN_USER = os.environ.get("LOGIN_USER", "Allen")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "Chester01!")

# BigQuery
BQ_PROJECT  = os.environ.get("BQ_PROJECT", "market-growth-monitor")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "EU")  # EU ist häufig richtig
# Service-Account-JSON wird bei Render als Secret File im Projektroot abgelegt:
SERVICE_KEY_FILENAME = os.environ.get(
    "SERVICE_KEY_FILENAME", "market-growth-monitor-c20a3876d9c9.json"
)

# SQL-Quelle: View/Table mit genau den Spalten
#   sector STRING, date DATE, price_return_7d FLOAT64, sentiment_score FLOAT64
BQ_SQL = os.environ.get(
    "BQ_SQL",
    """
    SELECT sector, date, price_return_7d, sentiment_score
    FROM `market-growth-monitor.market_data.market_dashboard_view`
    ORDER BY date DESC
    LIMIT 2000
    """
)

# CSV-Fallback-Datei (liegt im selben Ordner wie app.py)
CSV_FALLBACK = os.environ.get("CSV_FALLBACK", "daily_metrics.csv")


# -----------------------------------------------------------------------------
# Flask + Basic Auth
# -----------------------------------------------------------------------------
server = Flask(__name__)
server.config["BASIC_AUTH_USERNAME"] = LOGIN_USER
server.config["BASIC_AUTH_PASSWORD"] = LOGIN_PASS
server.config["BASIC_AUTH_FORCE"] = True
basic_auth = BasicAuth(server)


# -----------------------------------------------------------------------------
# Dash App
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Growth Monitor"


# -----------------------------------------------------------------------------
# Daten laden (BigQuery zuerst, sonst CSV, sonst Dummy)
# -----------------------------------------------------------------------------
DATA_SOURCE = "Unbekannt"
LAST_UPDATE = "Keine Daten"
LAST_ERROR  = None

def _abs_path(filename: str) -> str:
    """Hilfsfunktion: absoluter Pfad relativ zu dieser Datei."""
    return os.path.join(os.path.dirname(__file__), filename)

def load_data() -> pd.DataFrame:
    """Versuche BigQuery, dann CSV, sonst Dummy. Hält globale Statusvariablen aktuell."""
    global DATA_SOURCE, LAST_UPDATE, LAST_ERROR

    # 1) BigQuery
    try:
        key_path = _abs_path(SERVICE_KEY_FILENAME)
        if os.path.exists(key_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

            client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)
            df_bq = client.query(BQ_SQL).to_dataframe()

            if not df_bq.empty:
                # Typen bereinigen
                if "date" in df_bq.columns:
                    df_bq["date"] = pd.to_datetime(df_bq["date"])
                for col in ("price_return_7d", "sentiment_score"):
                    if col in df_bq.columns:
                        df_bq[col] = pd.to_numeric(df_bq[col], errors="coerce")

                DATA_SOURCE = "BigQuery ✅"
                LAST_UPDATE = (
                    df_bq["date"].max().strftime("%Y-%m-%d") if "date" in df_bq.columns else "Unbekannt"
                )
                LAST_ERROR = None
                return df_bq
            else:
                LAST_ERROR = "BigQuery: View/Query lieferte 0 Zeilen."
                raise ValueError(LAST_ERROR)
        else:
            LAST_ERROR = f"Secret File nicht gefunden: {key_path}"
            raise FileNotFoundError(LAST_ERROR)

    except Exception as e:
        # 2) CSV-Fallback
        DATA_SOURCE = "CSV Fallback ⚓"
        LAST_ERROR = str(e)
        csv_path = _abs_path(CSV_FALLBACK)
        try:
            if os.path.exists(csv_path):
                df_csv = pd.read_csv(csv_path, parse_dates=["date"])
                if not df_csv.empty:
                    LAST_UPDATE = df_csv["date"].max().strftime("%Y-%m-%d")
                    # Sicherstellen, dass Spalten existieren
                    for col in ("sector", "price_return_7d", "sentiment_score"):
                        if col not in df_csv.columns:
                            df_csv[col] = pd.NA
                    return df_csv
        except Exception as e_csv:
            LAST_ERROR += f" | CSV-Fehler: {e_csv}"

        # 3) Dummy — minimal lauffähig
        LAST_UPDATE = "Keine Daten"
        return pd.DataFrame(columns=["sector", "date", "price_return_7d", "sentiment_score"])


# Initiales Laden
df = load_data()
sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns and not df.empty else ["Tech"]


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
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
        dbc.Col([
            dbc.Button("🔄 Daten neu laden", id="btn-refresh", color="primary")
        ], width=2),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="sentiment-chart"), width=6),
        dbc.Col(dcc.Graph(id="return-chart"), width=6)
    ]),
])


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("sentiment-chart", "figure"),
    Output("return-chart", "figure"),
    Output("data-status", "children"),
    Input("sector", "value"),
    Input("btn-refresh", "n_clicks")
)
def update_dashboard(selected_sector, _n):
    global df
    df = load_data()

    # Statuszeile
    status = f"📊 Datenquelle: {DATA_SOURCE} | 🕒 Letztes Update: {LAST_UPDATE}"
    if LAST_ERROR:
        status += f" | ⚠️ Fehler: {LAST_ERROR}"

    # Falls leer, leere Plots zurückgeben
    if df.empty or "sector" not in df.columns:
        empty = px.scatter(title="Keine Daten verfügbar")
        return empty, empty, status

    # Auswahl
    sel = df[df["sector"] == selected_sector].copy() if selected_sector else df.copy()
    if "date" in sel.columns:
        sel = sel.sort_values("date")

    # Charts
    fig_sent = px.line(sel, x="date", y="sentiment_score", title=f"Sentiment Score – {selected_sector}")
    fig_ret  = px.bar(sel, x="date", y="price_return_7d", title=f"7-Tage-Return – {selected_sector}")

    return fig_sent, fig_ret, status


# -----------------------------------------------------------------------------
# Start (Render)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

