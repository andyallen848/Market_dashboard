# app.py – Render-kompatibel, Dash 3.x, BigQuery Status
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
from flask import Flask
from flask_basicauth import BasicAuth

# --- Config ---
BQ_PROJECT = os.environ.get("BQ_PROJECT", "market-growth-monitor")
BQ_VIEW = os.environ.get("BQ_VIEW", "market_data.market_dashboard_view")
SERVICE_KEY_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
LOGIN_USER = os.environ.get("LOGIN_USER", "Allen")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "Chester01!")
UPDATE_HOUR_UTC = int(os.environ.get("UPDATE_HOUR_UTC", "8"))

# Flask + Basic Auth
server = Flask(__name__)
server.config['BASIC_AUTH_USERNAME'] = LOGIN_USER
server.config['BASIC_AUTH_PASSWORD'] = LOGIN_PASS
server.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(server)

# Dash app
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Growth Monitor"

# --- Load Data ---
DATA_DIR = os.environ.get("DATA_DIR", ".")

def load_from_csvs():
    def safe_read_csv(filename, parse_dates=None):
        try:
            return pd.read_csv(Path(DATA_DIR) / filename, parse_dates=parse_dates)
        except Exception:
            return pd.DataFrame()
    dm = safe_read_csv("daily_metrics.csv", parse_dates=["date"])
    sd = safe_read_csv("sector_details.csv", parse_dates=["date"])
    nf = safe_read_csv("news_feed.csv", parse_dates=["date"])
    ar = safe_read_csv("alert_rules.csv")
    return dm, sd, nf, ar

def load_from_bigquery():
    if not SERVICE_KEY_PATH or not os.path.exists(SERVICE_KEY_PATH):
        raise FileNotFoundError(f"Service key not found: {SERVICE_KEY_PATH}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_KEY_PATH
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
    SELECT date, sector, ticker, price_return_7d, volume_change, sentiment_score,
           metric_name, metric_value
    FROM `{BQ_PROJECT}.{BQ_VIEW}`
    """
    return client.query(query).to_dataframe()

def prepare_data(df_metrics, df_sectors, df_news, df_alerts):
    if not df_metrics.empty and not df_sectors.empty:
        df = df_metrics.merge(df_sectors, on=["date","sector"], how="left")
    else:
        df = df_metrics.copy()

    # Fallbacks
    if 'sector' not in df.columns:
        df['sector'] = 'Unknown'
    for col in ['price_return_7d','volume_change','sentiment_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = pd.Series([0]*len(df))

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date

    df['momentum_score'] = (df['price_return_7d'] + df['sentiment_score'] + df['volume_change'])/3.0

    def cat_sent(x):
        try: x = float(x)
        except: return "Neutral"
        if x >= 0.3: return "Positiv"
        if x <= -0.3: return "Negativ"
        return "Neutral"
    df['sentiment_category'] = df['sentiment_score'].apply(cat_sent)

    df['capex_spike'] = df.apply(lambda r: True if (r.get('metric_name')=='CapEx_Change_Pct' and pd.to_numeric(r.get('metric_value',0), errors='coerce')>10) else False, axis=1)
    df['earnings_beat'] = df.apply(lambda r: True if (r.get('metric_name')=='Earnings_Beat_Pct' and pd.to_numeric(r.get('metric_value',0), errors='coerce')>5) else False, axis=1)
    df['negative_sentiment'] = df['sentiment_score'] < -0.4

    return df, df_news, df_alerts

def load_data():
    try:
        df_metrics = load_from_bigquery()
        df_sectors = pd.DataFrame()
        df_news = pd.DataFrame()
        df_alerts = pd.DataFrame()
    except Exception as e:
        print("BigQuery load failed, falling back to CSVs:", e)
        df_metrics, df_sectors, df_news, df_alerts = load_from_csvs()
    df, df_news, df_alerts = prepare_data(df_metrics, df_sectors, df_news, df_alerts)
    return df, df_news, df_alerts

df, df_news, df_alerts = load_data()
SECTORS = sorted(df['sector'].dropna().unique().tolist()) if not df.empty else ["Unknown"]

# --- Layout ---
def make_kpi_card(title, value, subtitle=None):
    return dbc.Card(dbc.CardBody([html.H6(title), html.H3(value), html.Small(subtitle or '')]), className='mb-2')

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([dbc.Col(html.H2("Market Growth Monitor"), width=8), dbc.Col(html.Div(id='clock'), width=4, style={'textAlign':'right'})]),
    dbc.Row([
        dbc.Col(html.Div(id='bq-status'), width=12)  # NEU: BigQuery Zeilenstatus
    ]),
    dbc.Row([
        dbc.Col([html.Label("Sektor auswählen"),
                 dcc.Dropdown(
                     id='sector-dropdown',
                     options=[{'label':s,'value':s} for s in SECTORS],
                     value=SECTORS[0] if SECTORS else None,
                     clearable=False
                 )], width=4),
        dbc.Col(dbc.Button("Daten neu laden", id='refresh-button', color='primary'), width=4),
        dbc.Col(html.Div(id='status'), width=4)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='heatmap-fig'), width=6), dbc.Col(html.Div(id='kpi-cards'), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id='price-trend'), width=4), dbc.Col(dcc.Graph(id='sentiment-trend'), width=4), dbc.Col(dcc.Graph(id='scatter-fig'), width=4)]),
    dbc.Row([dbc.Col(dcc.Graph(id='alerts-table'), width=6), dbc.Col(dcc.Graph(id='news-table'), width=6)]),
    dcc.Interval(id='auto-refresh', interval=60*60*1000, n_intervals=0)
])

@app.callback(
    Output('heatmap-fig','figure'),
    Output('kpi-cards','children'),
    Output('price-trend','figure'),
    Output('sentiment-trend','figure'),
    Output('scatter-fig','figure'),
    Output('alerts-table','figure'),
    Output('news-table','figure'),
    Output('clock','children'),
    Output('bq-status','children'),  # NEU
    Input('sector-dropdown','value'),
    Input('refresh-button','n_clicks'),
    Input('auto-refresh','n_intervals')
)
def update(selected_sector, n_clicks, n_intervals):
    trigger = callback_context.triggered[0]['prop_id'] if callback_context.triggered else None
    global df, df_news, df_alerts
    if trigger and 'refresh-button' in trigger:
        df, df_news, df_alerts = load_data()

    # BigQuery Zeilen-Status
    if hasattr(df, 'shape'):
        bq_info = f"BigQuery Zeilen: {df.shape[0]}"
    else:
        bq_info = "Keine BigQuery-Daten"

    if df.empty:
        empty = px.scatter(title="Keine Daten")
        return empty, html.Div("Keine Daten"), empty, empty, empty, empty, empty, "", bq_info

    sel_df = df[df['sector']==selected_sector] if selected_sector else df.copy()
    sector_scores = df.groupby('sector')['momentum_score'].mean().reset_index().sort_values('momentum_score', ascending=False)
    heat_fig = px.bar(sector_scores, x='sector', y='momentum_score', title='Sector Momentum', text='momentum_score')
    heat_fig.update_layout(yaxis_title='Momentum Score', xaxis_title='Sektor')

    avg_sent = sel_df['sentiment_score'].mean() if not sel_df.empty else 0
    capex = sel_df[sel_df.get('metric_name')=='CapEx_Change_Pct']['metric_value'].mean() if 'metric_name' in sel_df.columns else None
    earnings = sel_df[sel_df.get('metric_name')=='Earnings_Beat_Pct']['metric_value'].mean() if 'metric_name' in sel_df.columns else None
    kpis = dbc.Row([
        dbc.Col(make_kpi_card("Durchschnitt Sentiment", f"{avg_sent:.2f}", f"Sektor: {selected_sector}")),
        dbc.Col(make_kpi_card("CapEx (avg)", f"{capex:.2f}" if capex is not None and not pd.isna(capex) else "n/a")),
        dbc.Col(make_kpi_card("Earnings Beat (avg)", f"{earnings:.2f}" if earnings is not None and not pd.isna(earnings) else "n/a"))
    ])

    price_trend = px.line(sel_df.sort_values('date'), x='date', y='price_return_7d', markers=True, title='Preis Trend (7d Return)')
    sentiment_trend = px.area(sel_df.sort_values('date'), x='date', y='sentiment_score', title='Sentiment Trend')
    scatter = px.scatter(sel_df, x='sentiment_score', y='price_return_7d', hover_data=['ticker','date'], title='Sentiment vs Return')

    alerts_df = sel_df[(sel_df['capex_spike']) | (sel_df['earnings_beat']) | (sel_df['negative_sentiment'])]
    if alerts_df.empty:
        alerts_fig = px.scatter(title="Keine Alerts")
    else:
        alerts_plot = alerts_df[['date','ticker','capex_spike','earnings_beat','negative_sentiment','metric_name','metric_value']].copy()
        alerts_fig = px.table(alerts_plot.sort_values('date', ascending=False))

    if not df_news.empty and 'sector' in df_news.columns:
        news_plot = df_news[df_news['sector']==selected_sector][['date','headline','sentiment_score','source']].sort_values('date', ascending=False)
        news_fig = px.table(news_plot)
    else:
        news_fig = px.scatter(title="Keine News/Feed")

    clock = f"Letzte Aktualisierung: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    return heat_fig, kpis, price_trend, sentiment_trend, scatter, alerts_fig, news_fig, clock, bq_info

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)

