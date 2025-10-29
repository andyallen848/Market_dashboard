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


# -----------------------------------------
# ✅ Authentifizierung (Render Env Variables)
# -----------------------------------------
server = Flask(__name__)
server.config["BASIC_AUTH_USERNAME"] = os.environ.get("LOGIN_USER", "Allen")
server.config["BASIC_AUTH_PASSWORD"] = os.environ.get("LOGIN_PASS", "Chester01!")
server.config["BASIC_AUTH_FORCE"] = True
basic_auth = BasicAuth(server)


# -----------------------------------------
# ✅ Dash Setup
# -----------------------------------------
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Market Growth Monitor"


# -----------------------------------------
# ✅ Daten-Lader (BigQuery → fallback auf CSV)
# -----------------------------------------
SERVICE_KEY_PATH = "market-growth-monitor-c20a3876d9c9.json"
DATA_SOURCE = "Nicht geladen"

def load_data():
    global DATA_SOURCE, LAST_UPDATE

    # ✅ BigQuery zuerst versuchen
    try:
        if os.path.exists(SERVICE_KEY_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_KEY_PATH

            client = bigquery.Client()
            query = """
            SELECT sector, date, price_return_7d, sentiment_score
            FROM `market-growth-monitor.market_data.market_dashboard_view`
            ORDER BY date DESC
            LIMIT 500
            """
            df = client.query(query).to_dataframe()

            df["date"] = pd.to_datetime(df["date"])
            DATA_SOURCE = "BigQuery ✅"
            LAST_UPDATE = df["date"].max().strftime("%Y-%m-%d")
            return df

        raise FileNotFoundError("Google Key fehlt")

    except Exception as e:
        print("⚠ BigQuery Fehler:", e)
        DATA_SOURCE = "CSV Fallback ⚓"
        if os.path.exists("daily_metrics.csv"):
            df = pd.read_csv("daily_metrics.csv", parse_dates=["date"])
            LAST_UPDATE = df["date"].max().strftime("%Y-%m-%d")
            return df

        # Falls keine CSV existiert → Dummy Daten
        LAST_UPDATE = "Keine Daten"
        return pd.DataFrame({
            "sector": ["Tech", "Health"],
            "date": [datetime.utcnow(), datetime.utcnow()],
            "price_return_7d": [0,0],
            "sentiment_score": [0,0]
        })


df = load_data()
SECTORS = sorted(df["sector"].unique().tolist())


# -----------------------------------------
# ✅ Dashboard Layout
# -----------------------------------------
app.layout = dbc.Container([
    html.H2("📈 Market Growth Monitor"),
    html.Div(id="data-status", className="mb-3", style={"fontWeight": "bold"}),

    dbc.Row([
        dbc.Col([
            html.Label("Sektor auswählen"),
            dcc.Dropdown(
                id="sector-dropdown",
                options=[{"label": s, "value": s} for s in SECTORS],
                value=SECTORS[0],
                clearable=False
            )
        ], width=4),
        dbc.Col([
            dbc.Button("🔄 Daten neu laden", id="refresh")
        ], width=2)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="sentiment-chart"), width=6),
        dbc.Col(dcc.Graph(id="return-chart"), width=6)
    ])
], fluid=True)


# -----------------------------------------
# ✅ Dashboard Callback
# -----------------------------------------
@app.callback(
    Output("sentiment-chart", "figure"),
    Output("return-chart", "figure"),
    Output("data-status", "children"),
    Input("sector-dropdown", "value"),
    Input("refresh", "n_clicks")
)
def update_dashboard(sector, _):
    global df
    df = load_data()  # reload always OK

    sel = df[df["sector"] == sector]

    sentiment_chart = px.line(
        sel, x="date", y="sentiment_score",
        title="Sentiment Score"
    )
    return_chart = px.bar(
        sel, x="date", y="price_return_7d",
        title="Price Return 7d"
    )

    status = f"📊 Datenquelle: {DATA_SOURCE} | 🕒 Letztes Update: {LAST_UPDATE}"

    return sentiment_chart, return_chart, status


# -----------------------------------------
# ✅ Render Server Start
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

