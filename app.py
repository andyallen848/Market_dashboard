# app.py – Market Growth Monitor (Dash 3.x) with "Top Companies" ranking
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask
from flask_basicauth import BasicAuth

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px

# -----------------------------
# Config (env overrides allowed)
# -----------------------------
BQ_PROJECT        = os.environ.get("BQ_PROJECT", "market-growth-monitor")
BQ_DATASET_VIEW   = os.environ.get("BQ_VIEW", "market_data.market_dashboard_view")
BQ_LOCATION       = os.environ.get("BQ_LOCATION", "EU")  # <— important for EU datasets
SERVICE_KEY_PATH  = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

LOGIN_USER        = os.environ.get("LOGIN_USER", "Allen")
LOGIN_PASS        = os.environ.get("LOGIN_PASS", "Chester01!")

UPDATE_HOUR_UTC   = int(os.environ.get("UPDATE_HOUR_UTC", "8"))
DATA_DIR          = os.environ.get("DATA_DIR", ".")  # CSV fallback directory

# -----------------------------
# Flask + BasicAuth
# -----------------------------
server = Flask(__name__)
server.config['BASIC_AUTH_USERNAME'] = LOGIN_USER
server.config['BASIC_AUTH_PASSWORD'] = LOGIN_PASS
server.config['BASIC_AUTH_FORCE']    = True
basic_auth = BasicAuth(server)

# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Growth Monitor"

# -----------------------------
# Data loaders
# -----------------------------
def load_from_csvs():
    """Fallback: read simple CSVs if BigQuery fails."""
    def safe_read_csv(filename, parse_dates=None):
        try:
            return pd.read_csv(Path(DATA_DIR) / filename, parse_dates=parse_dates)
        except Exception:
            return pd.DataFrame()

    dm = safe_read_csv("daily_metrics.csv", parse_dates=["date"])      # must include sector,ticker,price_return_7d,volume_change,sentiment_score,metric_name,metric_value
    sd = safe_read_csv("sector_details.csv", parse_dates=["date"])     # optional enrich
    nf = safe_read_csv("news_feed.csv", parse_dates=["date"])          # optional news
    ar = safe_read_csv("alert_rules.csv")                              # optional rules
    return dm, sd, nf, ar


def load_from_bigquery():
    """Read the view from BigQuery."""
    if not SERVICE_KEY_PATH or not os.path.exists(SERVICE_KEY_PATH):
        raise FileNotFoundError(f"Service key not found: {SERVICE_KEY_PATH}")

    # Avoid hard import if the package isn't installed yet during build
    from google.cloud import bigquery

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_KEY_PATH
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    query = f"""
    SELECT
      date,
      sector,
      ticker,
      price_return_7d,
      volume_change,
      sentiment_score,
      metric_name,
      metric_value
    FROM `{BQ_PROJECT}.{BQ_DATASET_VIEW}`
    """
    # NOTE: .to_dataframe() may show a 'db-dtypes' warning; harmless for display
    return client.query(query).to_dataframe()


def prepare_data(df_metrics: pd.DataFrame,
                 df_sectors: pd.DataFrame,
                 df_news: pd.DataFrame,
                 df_alerts: pd.DataFrame):
    """Unify, sanitize and derive helper fields/flags."""
    df = df_metrics.copy()

    # Ensure required columns exist
    if 'sector' not in df.columns:
        df['sector'] = 'Unknown'
    if 'ticker' not in df.columns:
        df['ticker'] = 'N/A'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date

    for col in ['price_return_7d', 'volume_change', 'sentiment_score', 'metric_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # Momentum metric used throughout the app
    df['momentum_score'] = df[['price_return_7d', 'sentiment_score', 'volume_change']].mean(axis=1)

    # Sentiment category (for optional coloring)
    def cat_sent(x):
        try:
            x = float(x)
        except Exception:
            return "Neutral"
        if x >= 0.3:
            return "Positiv"
        if x <= -0.3:
            return "Negativ"
        return "Neutral"
    df['sentiment_category'] = df['sentiment_score'].apply(cat_sent)

    # Boolean alert-style flags derived from metric_name/value
    df['capex_spike']        = df.apply(lambda r: bool((r.get('metric_name') == 'CapEx_Change_Pct') and (pd.to_numeric(r.get('metric_value', 0), errors='coerce') > 10)), axis=1)
    df['earnings_beat']      = df.apply(lambda r: bool((r.get('metric_name') == 'Earnings_Beat_Pct') and (pd.to_numeric(r.get('metric_value', 0), errors='coerce') > 5)), axis=1)
    df['negative_sentiment'] = df['sentiment_score'] < -0.4

    # News
    if not df_news.empty and 'date' in df_news.columns:
        df_news['date'] = pd.to_datetime(df_news['date']).dt.date

    return df, df_news, df_alerts


def load_data():
    """Try BigQuery, otherwise CSV fallback."""
    try:
        df_metrics = load_from_bigquery()
        df_sectors = pd.DataFrame()
        df_news    = pd.DataFrame()
        df_alerts  = pd.DataFrame()
    except Exception as e:
        print("BigQuery load failed, falling back to CSVs:", e)
        df_metrics, df_sectors, df_news, df_alerts = load_from_csvs()

    df, df_news, df_alerts = prepare_data(df_metrics, df_sectors, df_news, df_alerts)
    return df, df_news, df_alerts


# Load once at startup
df, df_news, df_alerts = load_data()
SECTORS = sorted(df['sector'].dropna().unique().tolist()) if not df.empty else ["Unknown"]

# -----------------------------
# Helpers
# -----------------------------
def make_kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody([html.H6(title), html.H3(value), html.Small(subtitle or '')]),
        className='mb-2'
    )


def last_n_days_filter(pdf: pd.DataFrame, lookback_days: int):
    """Filter by last N calendar days from max(date) present in df."""
    if pdf.empty or 'date' not in pdf.columns:
        return pdf
    max_day = pd.to_datetime(pdf['date']).max()
    if pd.isna(max_day):
        return pdf
    start_day = (pd.to_datetime(max_day) - timedelta(days=lookback_days - 1)).date()
    return pdf[pd.to_datetime(pdf['date']).dt.date >= start_day].copy()


# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("Market Growth Monitor"), width=8),
        dbc.Col(html.Div(id='clock'), width=4, style={'textAlign': 'right'})
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='bq-status'), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Sektor auswählen"),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': s, 'value': s} for s in SECTORS],
                value=SECTORS[0] if SECTORS else None,
                clearable=False
            )
        ], width=4),

        dbc.Col(dbc.Button("Daten neu laden", id='refresh-button', color='primary'), width=2),

        # --- NEW controls for company ranking ---
        dbc.Col([
            html.Label("Top-N Firmen"),
            dcc.Dropdown(
                id='topn-dropdown',
                options=[{'label': str(n), 'value': n} for n in [5, 10, 15, 20, 25]],
                value=10,
                clearable=False
            )
        ], width=2),
        dbc.Col([
            html.Label("Lookback (Tage)"),
            dcc.Dropdown(
                id='lookback-dropdown',
                options=[{'label': str(n), 'value': n} for n in [5, 7, 10, 14, 21, 30]],
                value=10,
                clearable=False
            )
        ], width=2),

        dbc.Col(html.Div(id='status'), width=2),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='heatmap-fig'), width=6),
        dbc.Col(html.Div(id='kpi-cards'), width=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='price-trend'), width=4),
        dbc.Col(dcc.Graph(id='sentiment-trend'), width=4),
        dbc.Col(dcc.Graph(id='scatter-fig'), width=4)
    ]),

    # --- NEW: ranked companies table
    dbc.Row([
        dbc.Col(dcc.Graph(id='company-table'), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='alerts-table'), width=6),
        dbc.Col(dcc.Graph(id='news-table'), width=6)
    ]),

    dcc.Interval(id='auto-refresh', interval=60 * 60 * 1000, n_intervals=0)
])


# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output('heatmap-fig', 'figure'),
    Output('kpi-cards', 'children'),
    Output('price-trend', 'figure'),
    Output('sentiment-trend', 'figure'),
    Output('scatter-fig', 'figure'),
    Output('alerts-table', 'figure'),
    Output('news-table', 'figure'),
    Output('clock', 'children'),
    Output('bq-status', 'children'),
    Output('company-table', 'figure'),  # NEW
    Input('sector-dropdown', 'value'),
    Input('topn-dropdown', 'value'),
    Input('lookback-dropdown', 'value'),
    Input('refresh-button', 'n_clicks'),
    Input('auto-refresh', 'n_intervals')
)
def update(selected_sector, topn, lookback_days, n_clicks, n_intervals):
    trigger = callback_context.triggered[0]['prop_id'] if callback_context.triggered else None

    global df, df_news, df_alerts
    if trigger and 'refresh-button' in trigger:
        df, df_news, df_alerts = load_data()

    # BigQuery status
    bq_info = f"Datenquelle: BigQuery ✅ | Zeilen: {df.shape[0]}" if not df.empty else "Datenquelle: CSV Fallback ⚠️"

    if df.empty:
        empty = px.scatter(title="Keine Daten")
        return empty, html.Div("Keine Daten"), empty, empty, empty, empty, empty, "", bq_info, empty

    # -------- Sector views --------
    # Heatmap/Bar of sector momentum (mean over all rows)
    sector_scores = (
        df.groupby('sector')[['price_return_7d', 'sentiment_score', 'volume_change']]
          .mean()
          .assign(momentum_score=lambda d: d[['price_return_7d', 'sentiment_score', 'volume_change']].mean(axis=1))
          .reset_index()
          .sort_values('momentum_score', ascending=False)
    )
    heat_fig = px.bar(sector_scores, x='sector', y='momentum_score',
                      title='Sector Momentum', text='momentum_score')
    heat_fig.update_layout(yaxis_title='Momentum Score', xaxis_title='Sektor')

    # Selected sector slice (and last N days for plots/companies)
    if selected_sector:
        sel_df = df[df['sector'] == selected_sector].copy()
    else:
        sel_df = df.copy()

    sel_df_last = last_n_days_filter(sel_df, lookback_days)

    # KPIs
    avg_sent = sel_df_last['sentiment_score'].mean() if not sel_df_last.empty else 0
    capex = sel_df_last.loc[sel_df_last['metric_name']=='CapEx_Change_Pct', 'metric_value'].mean() if 'metric_name' in sel_df_last.columns else np.nan
    earnings = sel_df_last.loc[sel_df_last['metric_name']=='Earnings_Beat_Pct', 'metric_value'].mean() if 'metric_name' in sel_df_last.columns else np.nan

    kpis = dbc.Row([
        dbc.Col(make_kpi_card("Durchschnitt Sentiment", f"{avg_sent:.2f}", f"Sektor: {selected_sector}")),
        dbc.Col(make_kpi_card("CapEx (avg)", "n/a" if pd.isna(capex) else f"{capex:.2f}")),
        dbc.Col(make_kpi_card("Earnings Beat (avg)", "n/a" if pd.isna(earnings) else f"{earnings:.2f}"))
    ])

    # Price & Sentiment trends
    # (Sort by date to produce nice lines even if duplicates exist)
    price_trend = px.line(sel_df_last.sort_values('date'),
                          x='date', y='price_return_7d', markers=True,
                          title=f'Preis Trend (7d Return) – {selected_sector}')
    sentiment_trend = px.area(sel_df_last.sort_values('date'),
                              x='date', y='sentiment_score',
                              title=f'Sentiment Trend – {selected_sector}')
    scatter = px.scatter(sel_df_last, x='sentiment_score', y='price_return_7d',
                         hover_data=['ticker', 'date'], title='Sentiment vs Return')

    # Alerts table (derived booleans)
    alerts_df = sel_df_last[(sel_df_last['capex_spike']) |
                            (sel_df_last['earnings_beat']) |
                            (sel_df_last['negative_sentiment'])].copy()
    if alerts_df.empty:
        alerts_fig = px.scatter(title="Keine Alerts")
    else:
        alerts_plot = alerts_df[['date', 'ticker', 'capex_spike', 'earnings_beat',
                                 'negative_sentiment', 'metric_name', 'metric_value']].copy()
        alerts_plot = alerts_plot.sort_values('date', ascending=False)
        alerts_fig = px.table(alerts_plot)

    # News table
    if not df_news.empty and 'sector' in df_news.columns:
        news_plot = df_news[df_news['sector'] == selected_sector][
            ['date', 'headline', 'sentiment_score', 'source']
        ].sort_values('date', ascending=False)
        news_fig = px.table(news_plot) if not news_plot.empty else px.scatter(title="Keine News/Feed")
    else:
        news_fig = px.scatter(title="Keine News/Feed")

    # Clock
    clock = f"Letzte Aktualisierung: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

    # -------- NEW: Top Companies ranking --------
    # Aggregate ticker stats over lookback window
    comp = sel_df_last.groupby('ticker').agg(
        sector=('sector', 'first'),
        days=('date', 'nunique'),
        avg_return_7d=('price_return_7d', 'mean'),
        avg_sentiment=('sentiment_score', 'mean'),
        avg_volume_change=('volume_change', 'mean'),
        momentum=('momentum_score', 'mean'),
        pos_days=('price_return_7d', lambda s: (s > 0).sum()),
        risk_std=('price_return_7d', 'std'),
        capex_any=('capex_spike', 'any'),
        earn_any=('earnings_beat', 'any'),
        negsent_any=('negative_sentiment', 'any')
    ).reset_index()

    if not comp.empty:
        comp['consistency'] = (comp['pos_days'] / comp['days']).fillna(0.0)
        comp['risk_std']    = comp['risk_std'].fillna(0.0)
        # Simple score that favors high momentum, good sentiment, and lower risk
        comp['score'] = (
            0.6 * comp['momentum'].rank(pct=True) +
            0.3 * comp['avg_sentiment'].rank(pct=True) +
            0.1 * (1 - comp['risk_std'].rank(pct=True))
        )
        # Top-N
        comp = comp.sort_values(['score', 'momentum'], ascending=[False, False]).head(int(topn))

        display_cols = [
            'ticker', 'sector', 'days',
            'momentum', 'avg_return_7d', 'avg_sentiment', 'avg_volume_change',
            'consistency', 'risk_std',
            'capex_any', 'earn_any', 'negsent_any'
        ]
        comp_rounded = comp[display_cols].copy()
        for c in ['momentum', 'avg_return_7d', 'avg_sentiment', 'avg_volume_change',
                  'consistency', 'risk_std']:
            comp_rounded[c] = comp_rounded[c].astype(float).round(4)

        company_fig = px.table(
            comp_rounded.rename(columns={
                'ticker': 'Ticker',
                'sector': 'Sektor',
                'days': 'Tage',
                'momentum': 'Momentum',
                'avg_return_7d': 'Ø Return 7d',
                'avg_sentiment': 'Ø Sentiment',
                'avg_volume_change': 'Ø Volumen-Δ',
                'consistency': 'Positiv-Tage-Quote',
                'risk_std': 'Risiko (Std)',
                'capex_any': 'CapEx-Spike',
                'earn_any': 'Earnings-Beat',
                'negsent_any': 'Neg. Sentiment'
            })
        )
    else:
        company_fig = px.scatter(title="Keine Firmendaten im Fenster")

    return (heat_fig, kpis, price_trend, sentiment_trend, scatter,
            alerts_fig, news_fig, clock, bq_info, company_fig)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)

