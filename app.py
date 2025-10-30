# app.py – Market Growth Monitor (Dash 3.x)
# Adds: Top Companies table with links + trade signals, Trade Ideas panel

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask
from flask_basicauth import BasicAuth

import dash
from dash import dcc, html, Input, Output, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px


# -----------------------------
# Config (env overrides allowed)
# -----------------------------
BQ_PROJECT        = os.environ.get("BQ_PROJECT", "market-growth-monitor")
BQ_DATASET_VIEW   = os.environ.get("BQ_VIEW", "market_data.market_dashboard_view")
BQ_LOCATION       = os.environ.get("BQ_LOCATION", "EU")  # EU dataset region
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
    """Fallback: read CSVs if BigQuery fails."""
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
    """Read the view from BigQuery (EU)."""
    if not SERVICE_KEY_PATH or not os.path.exists(SERVICE_KEY_PATH):
        raise FileNotFoundError(f"Service key not found: {SERVICE_KEY_PATH}")

    from google.cloud import bigquery  # import lazily

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
    return client.query(query).to_dataframe()


def prepare_data(df_metrics: pd.DataFrame,
                 df_sectors: pd.DataFrame,
                 df_news: pd.DataFrame,
                 df_alerts: pd.DataFrame):
    """Unify, sanitize and derive helper fields/flags."""
    df = df_metrics.copy()

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

    # Momentum metric
    df['momentum_score'] = df[['price_return_7d', 'sentiment_score', 'volume_change']].mean(axis=1)

    # Sentiment category
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

    # Flags (from metric_name/value)
    df['capex_spike']        = df.apply(lambda r: bool((r.get('metric_name') == 'CapEx_Change_Pct') and (pd.to_numeric(r.get('metric_value', 0), errors='coerce') > 10)), axis=1)
    df['earnings_beat']      = df.apply(lambda r: bool((r.get('metric_name') == 'Earnings_Beat_Pct') and (pd.to_numeric(r.get('metric_value', 0), errors='coerce') > 5)), axis=1)
    df['negative_sentiment'] = df['sentiment_score'] < -0.4

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
    """Filter by last N calendar days based on max(date) in df."""
    if pdf.empty or 'date' not in pdf.columns:
        return pdf
    max_day = pd.to_datetime(pdf['date']).max()
    if pd.isna(max_day):
        return pdf
    start_day = (pd.to_datetime(max_day) - timedelta(days=lookback_days - 1)).date()
    return pdf[pd.to_datetime(pdf['date']).dt.date >= start_day].copy()


def company_links(ticker: str):
    """Build markdown links to Yahoo Finance and TradingView."""
    yf = f"[YF](https://finance.yahoo.com/quote/{ticker})"
    tv = f"[TV](https://www.tradingview.com/symbols/{ticker})"
    return yf, tv


# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("Market Growth Monitor"), width=8),
        dbc.Col(html.Div(id='clock'), width=4, style={'textAlign': 'right'})
    ]),
    dbc.Row([dbc.Col(html.Div(id='bq-status'), width=12)]),

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

    # NEW: Companies table + Trade ideas region
    dbc.Row([dbc.Col(html.Div(id='company-table'), width=12)], className="mt-2"),
    dbc.Row([dbc.Col(html.Div(id='trade-ideas'), width=12)], className="mt-2"),

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
    Output('company-table', 'children'),   # now returns a component
    Output('trade-ideas', 'children'),     # new panel
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

    bq_info = f"Datenquelle: BigQuery ✅ | Zeilen: {df.shape[0]}" if not df.empty else "Datenquelle: CSV Fallback ⚠️"

    if df.empty:
        empty = px.scatter(title="Keine Daten")
        return empty, html.Div("Keine Daten"), empty, empty, empty, empty, empty, "", bq_info, html.Div(), html.Div()

    # -------- Sector overview --------
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

    # Current sector momentum (for signals)
    sector_mom = None
    if selected_sector and selected_sector in sector_scores['sector'].values:
        sector_mom = float(sector_scores.set_index('sector').loc[selected_sector, 'momentum_score'])

    # Sector slice + lookback
    sel_df = df[df['sector'] == selected_sector].copy() if selected_sector else df.copy()
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

    # Charts
    price_trend = px.line(sel_df_last.sort_values('date'),
                          x='date', y='price_return_7d', markers=True,
                          title=f'Preis Trend (7d Return) – {selected_sector}')
    sentiment_trend = px.area(sel_df_last.sort_values('date'),
                              x='date', y='sentiment_score',
                              title=f'Sentiment Trend – {selected_sector}')
    scatter = px.scatter(sel_df_last, x='sentiment_score', y='price_return_7d',
                         hover_data=['ticker', 'date'], title='Sentiment vs Return')

    # Alerts table
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

    # News
    if not df_news.empty and 'sector' in df_news.columns:
        news_plot = df_news[df_news['sector'] == selected_sector][
            ['date', 'headline', 'sentiment_score', 'source']
        ].sort_values('date', ascending=False)
        news_fig = px.table(news_plot) if not news_plot.empty else px.scatter(title="Keine News/Feed")
    else:
        news_fig = px.scatter(title="Keine News/Feed")

    # Clock
    clock = f"Letzte Aktualisierung: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

    # -------- Companies: ranking + signals + links --------
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

    if comp.empty:
        company_component = html.Div(dbc.Alert("Keine Firmendaten im Fenster", color="secondary"))
        trade_component = html.Div()
    else:
        comp['consistency'] = (comp['pos_days'] / comp['days']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        comp['risk_std']    = comp['risk_std'].fillna(0.0)
        comp['score'] = (
            0.6 * comp['momentum'].rank(pct=True) +
            0.3 * comp['avg_sentiment'].rank(pct=True) +
            0.1 * (1 - comp['risk_std'].rank(pct=True))
        )
        comp['score_pct'] = comp['score'].rank(pct=True)

        # Sector momentum threshold (strong sector if above median)
        sector_is_strong = False
        if sector_mom is not None:
            sector_is_strong = sector_mom > sector_scores['momentum_score'].median()

        # Signals
        cond_buy = (comp['score_pct'] >= 0.70) & (comp['avg_sentiment'] > 0) & (~comp['negsent_any'])
        if sector_is_strong:
            cond_buy = cond_buy | ((comp['score_pct'] >= 0.65) & (comp['momentum'] > comp['momentum'].median()))

        cond_watch = ((comp['score_pct'] >= 0.50) & (comp['score_pct'] < 0.70)) | \
                     ((comp['score_pct'] >= 0.70) & comp['negsent_any'])

        comp['signal'] = np.select(
            [cond_buy, cond_watch],
            ['BUY', 'WATCH'],
            default='AVOID'
        )

        # Links
        links = comp['ticker'].apply(company_links)
        comp['Yahoo'] = links.apply(lambda t: t[0])
        comp['TradingView'] = links.apply(lambda t: t[1])

        # Top-N
        topn = int(topn or 10)
        comp = comp.sort_values(['signal', 'score', 'momentum'], ascending=[True, False, False])
        comp_top = comp.head(topn).copy()

        # Table for display
        view_cols = [
            'ticker', 'sector', 'signal', 'days',
            'momentum', 'avg_return_7d', 'avg_sentiment', 'avg_volume_change',
            'consistency', 'risk_std',
            'capex_any', 'earn_any', 'negsent_any',
            'Yahoo', 'TradingView'
        ]
        view = comp_top[view_cols].rename(columns={
            'ticker': 'Ticker',
            'sector': 'Sektor',
            'signal': 'Signal',
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
        }).copy()

        for c in ['Momentum', 'Ø Return 7d', 'Ø Sentiment', 'Ø Volumen-Δ',
                  'Positiv-Tage-Quote', 'Risiko (Std)']:
            view[c] = view[c].astype(float).round(4)

        # Dash DataTable with markdown links
        company_component = dash_table.DataTable(
            id="company-datatable",
            columns=[
                {"name": "Ticker", "id": "Ticker"},
                {"name": "Sektor", "id": "Sektor"},
                {"name": "Signal", "id": "Signal"},
                {"name": "Tage", "id": "Tage", "type": "numeric"},
                {"name": "Momentum", "id": "Momentum", "type": "numeric"},
                {"name": "Ø Return 7d", "id": "Ø Return 7d", "type": "numeric"},
                {"name": "Ø Sentiment", "id": "Ø Sentiment", "type": "numeric"},
                {"name": "Ø Volumen-Δ", "id": "Ø Volumen-Δ", "type": "numeric"},
                {"name": "Positiv-Tage-Quote", "id": "Positiv-Tage-Quote", "type": "numeric"},
                {"name": "Risiko (Std)", "id": "Risiko (Std)", "type": "numeric"},
                {"name": "CapEx-Spike", "id": "CapEx-Spike"},
                {"name": "Earnings-Beat", "id": "Earnings-Beat"},
                {"name": "Neg. Sentiment", "id": "Neg. Sentiment"},
                {"name": "Yahoo", "id": "Yahoo", "presentation": "markdown"},
                {"name": "TradingView", "id": "TradingView", "presentation": "markdown"},
            ],
            data=view.to_dict("records"),
            style_table={"overflowX": "auto"},
            sort_action="native",
            filter_action="native",
            page_action="native",
            page_size=15,
            style_cell={"padding": "6px", "fontSize": 14},
            style_data_conditional=[
                # Color signals
                {"if": {"filter_query": "{Signal} = 'BUY'"},
                 "backgroundColor": "#e6ffed"},
                {"if": {"filter_query": "{Signal} = 'AVOID'"},
                 "backgroundColor": "#fff0f0"},
            ],
        )

        # ---- Trade Ideas panel (top BUYs) ----
        buys = comp_top[comp_top['signal'] == 'BUY'].copy().sort_values('score', ascending=False).head(5)
        if buys.empty:
            trade_component = dbc.Alert("Keine BUY-Ideen basierend auf den Regeln im aktuellen Fenster.", color="secondary")
        else:
            items = []
            for _, r in buys.iterrows():
                items.append(
                    html.Li([
                        html.Strong(r['ticker']),
                        " – Momentum ", f"{r['momentum']:.3f}",
                        ", Ø Sentiment ", f"{r['avg_sentiment']:.3f}",
                        ", Konsistenz ", f"{(r['consistency']*100):.1f}%",
                        " ",
                        html.Small("(YF / TV: ", style={"color": "#666"}),
                        dcc.Markdown(r['Yahoo'], link_target="_blank", style={"display": "inline"}),
                        html.Span(" | "),
                        dcc.Markdown(r['TradingView'], link_target="_blank", style={"display": "inline"}),
                        html.Small(")", style={"color": "#666"})
                    ])
                )
            headline = f"Trade Ideas – BUY Picks (Sektor {'stark' if sector_is_strong else 'neutral/schwach'})"
            trade_component = dbc.Alert([html.H5(headline), html.Ul(items)], color="success")

    return (heat_fig, kpis, price_trend, sentiment_trend, scatter,
            alerts_fig, news_fig, clock, bq_info, company_component, trade_component)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)


