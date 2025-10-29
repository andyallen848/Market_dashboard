# app.py — Market Growth Monitor (BigQuery EU, dynamischer Sektor-Dropdown, CSV-Fallback)

import os
import pandas as pd

from flask import Flask
from flask_basicauth import BasicAuth
from google.cloud import bigquery
from google.oauth2 import service_account

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px


# -----------------------------
# Konfiguration (per Env überschreibbar)
# -----------------------------
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
    LIMIT 5000
    """
)

CSV_FALLBACK = os.environ.get("CSV_FALLBACK", "daily_metrics.csv")


# -----------------------------
# Flask + BasicAuth
# -----------------------------
server = Flask(__name__)
server.config["BASIC_AUTH_USERNAME"] = LOGIN_USER
server.config["BASIC_AUTH_PASSWORD"] = LOGIN_PASS
server.config["BASIC_AUTH_FORCE"] = True
basic_auth = BasicAuth(server)


# -----------------------------
# Dash
# -----------------------------
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Growth Monitor"


# -----------------------------
# Datenladen (BigQuery ohne pyarrow, CSV-Fallback)
# -----------------------------
DATA_SOURCE = "Unbekannt"
LAST_UPDATE = "Keine Daten"
LAST_ERROR  = None
SERVICE_ACCOUNT_EMAIL = "unbekannt"
df = pd.DataFrame(columns=["sector", "date", "price_return_7d", "sentiment_score"])

def _abs_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)

def load_data() -> pd.DataFrame:
    """Lädt Daten aus BigQuery; fällt auf CSV zurück. Setzt Status-Variablen."""
    global DATA_SOURCE, LAST_UPDATE, LAST_ERROR, SERVICE_ACCOUNT_EMAIL

    try:
        key_path = _abs_path(SERVICE_KEY_FILENAME)
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Secret File nicht gefunden: {key_path}")

        creds = service_account.Credentials.from_service_account_file(key_path)
        SERVICE_ACCOUNT_EMAIL = creds.service_account_email
        client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION, credentials=creds)

        job = client.query(BQ_SQL)
        rows = job.result()
        df_bq = rows.to_dataframe()  # KEIN bqstorage_client -> kein pyarrow nötig

        if df_bq.empty:
            raise ValueError("BigQuery: View/Query lieferte 0 Zeilen.")

        if "date" in df_bq.columns:
            df_bq["date"] = pd.to_datetime(df_bq["date"], errors="coerce")
        for col in ("price_return_7d", "sentiment_score"):
            if col in df_bq.columns:
                df_bq[col] = pd.to_numeric(df_bq[col], errors="coerce")

        DATA_SOURCE = "BigQuery ✅"
        LAST_UPDATE = (
            df_bq["date"].max().strftime("%Y-%m-%d") if "date" in df_bq.columns and df_bq["date"].notna().any()
            else "Unbekannt"
        )
        LAST_ERROR = None
        return df_bq

    except Exception as e:
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
        return pd.DataFrame(columns=["sector", "date", "price_return_7d", "sentiment_score"])


# Initialdaten
df = load_data()
init_sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns and not df.empty else []
init_value = init_sectors[0] if init_sectors else None


# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(fluid=True, children=[
    html.H2("📈 Market Growth Monitor"),
    html.Div(id="data-status", className="mb-3", style={"fontWeight": "bold"}),

    dbc.Row([
        dbc.Col([
            html.Label("Sektor auswählen"),
            dcc.Dropdown(id="sector", options=[{"label": s, "value": s} for s in init_sectors],
                         value=init_value, clearable=False)
        ], width=4),
        dbc.Col(dbc.Button("🔄 Daten neu laden", id="btn-refresh", color="primary"), width=2),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="sentiment-chart"), width=6),
        dbc.Col(dcc.Graph(id="return-chart"), width=6)
    ]),
])


# -----------------------------
# Callback A: Daten neu laden -> Dropdown (Optionen & Value) + Status aktualisieren
# -----------------------------
@app.callback(
    Output("sector", "options"),
    Output("sector", "value"),
    Output("data-status", "children"),
    Input("btn-refresh", "n_clicks"),
    State("sector", "value"),
    prevent_initial_call=False,  # auch beim ersten Rendern ausführen
)
def refresh_and_update_dropdown(_n_clicks, current_value):
    global df
    df = load_data()

    # verfügbare Sektoren aus neuen Daten
    sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns and not df.empty else []
    options = [{"label": s, "value": s} for s in sectors]

    # aktuellen Wert behalten, falls noch vorhanden – sonst ersten nehmen – sonst None
    if current_value in sectors:
        new_value = current_value
    else:
        new_value = sectors[0] if sectors else None

    status = f"📊 Datenquelle: {DATA_SOURCE} | 🕒 Letztes Update: {LAST_UPDATE} | 👤 SA: {SERVICE_ACCOUNT_EMAIL}"
    if LAST_ERROR:
        status += f" | ⚠️ Fehler: {LAST_ERROR}"

    return options, new_value, status


# -----------------------------
# Callback B: Charts zeichnen (reagiert auf Sektor und Reload)
# -----------------------------
@app.callback(
    Output("sentiment-chart", "figure"),
    Output("return-chart", "figure"),
    Input("sector", "value"),
    Input("btn-refresh", "n_clicks"),
)
def draw_charts(selected_sector, _n_clicks):
    if df.empty or "sector" not in df.columns or selected_sector is None:
        empty = px.scatter(title="Keine Daten verfügbar")
        return empty, empty

    sel = df[df["sector"] == selected_sector].copy()
    if sel.empty:
        empty = px.scatter(title=f"Keine Daten für Sektor: {selected_sector}")
        return empty, empty

    if "date" in sel.columns:
        sel = sel.sort_values("date")

    fig_sent = px.line(sel, x="date", y="sentiment_score", title=f"Sentiment Score – {selected_sector}")
    fig_ret  = px.bar(sel, x="date", y="price_return_7d", title=f"7-Tage-Return – {selected_sector}")

    return fig_sent, fig_ret


# -----------------------------
# Start (Render)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
