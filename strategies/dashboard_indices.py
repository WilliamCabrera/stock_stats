"""
Indices Backtest Dashboard  –  Streamlit + Plotly
Run:  streamlit run strategies/dashboard_indices.py
"""
import io
import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indices Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colors ──────────────────────────────────────────────────────────────
GREEN  = "#26a69a"
RED    = "#ef5350"
BLUE   = "#42a5f5"
YELLOW = "#ffca28"
PURPLE = "#ab47bc"
BG     = "#0f1117"
CARD   = "#1a1d27"
TEXT   = "#e0e0e0"
SUB    = "#90a4ae"
ORANGE = "#ffa726"

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #90a4ae; }
    div[data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2e3547;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11),
    xaxis=dict(gridcolor="#2e3547", zerolinecolor="#2e3547"),
    yaxis=dict(gridcolor="#2e3547", zerolinecolor="#2e3547"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor=CARD, bordercolor="#2e3547", borderwidth=1),
)

# ─────────────────────────────────────────────────────────────────────────────
# COMMISSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calc_commission(shares: float, price: float, trade_time) -> float:
    shares = max(0.0, float(shares))
    if shares == 0:
        return 0.0
    if shares < 100:
        return 0.49
    raw = shares * 0.005
    if price < 1.00:
        hour = trade_time.hour if hasattr(trade_time, "hour") else 12
        if 7 <= hour < 20:
            return max(0.49, min(raw, 7.95))
        else:
            return max(0.49, raw)
    return max(0.49, raw)


def commission_per_trade(shares, entry_price, exit_price, entry_time, exit_time):
    return (calc_commission(shares, entry_price, entry_time)
            + calc_commission(shares, exit_price, exit_time))


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE DATA — INDICES
# ─────────────────────────────────────────────────────────────────────────────
INDICES_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "backtest_dataset", "INDICES",
)

def _read_env_key(key: str) -> str:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return os.getenv(key, "")


POLYGON_API_KEY = _read_env_key("MASSIVE_API_KEY")
POLYGON_BASE    = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from_d}/{to_d}"


def _tf_to_polygon(timeframe: str) -> tuple[int, str]:
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1]), "minute"
    if tf.endswith("h"):
        return int(tf[:-1]), "hour"
    if tf.endswith("d"):
        return 1, "day"
    return 5, "minute"


@st.cache_data(show_spinner="Cargando velas locales (INDICES)...")
def load_candles_indices_local(ticker: str, timeframe: str, base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, ticker, timeframe, f"{ticker.lower()}_full_dataset.parquet")
    cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    import pyarrow.parquet as pq
    try:
        schema_names = pq.read_schema(path).names
        for c in ["sma_9", "sma_20", "sma_50"]:
            if c in schema_names:
                cols.append(c)
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        df = pd.read_parquet(path)
        df = df[[c for c in cols if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Descargando datos de Polygon...")
def load_candles_polygon(ticker: str, timeframe: str,
                         from_date: str, to_date: str,
                         api_key: str) -> pd.DataFrame:
    mult, span = _tf_to_polygon(timeframe)
    url = (POLYGON_BASE.format(ticker=ticker, mult=mult, span=span,
                               from_d=from_date, to_d=to_date)
           + f"?adjusted=false&sort=asc&limit=50000&apiKey={api_key}")
    rows, next_url = [], url
    while next_url:
        resp = requests.get(next_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("results", []))
        next_url = data.get("next_url")
        if next_url:
            next_url = f"{next_url}&apiKey={api_key}"
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    })
    df["date"] = (
        pd.to_datetime(df["date"], unit="ms")
          .dt.tz_localize("UTC")
          .dt.tz_convert("America/New_York")
          .dt.tz_localize(None)
    )
    df["ticker"] = ticker
    return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_trade_chart(candles: pd.DataFrame, trade: pd.Series,
                      context_bars: int = 40, timeframe: str = "5m",
                      source_label: str = "local") -> None:
    ticker   = trade["Ticker"]
    entry_dt = pd.to_datetime(trade["Entry"])
    exit_dt  = pd.to_datetime(trade["Exit"])

    tk = candles[candles["ticker"] == ticker].copy() if "ticker" in candles.columns else candles.copy()
    if tk.empty:
        st.warning(f"No hay velas disponibles para **{ticker}**.")
        return

    entry_idx = tk["date"].searchsorted(entry_dt)
    exit_idx  = tk["date"].searchsorted(exit_dt)
    start_idx = max(0, entry_idx - context_bars)
    end_idx   = min(len(tk), exit_idx + context_bars + 1)
    window    = tk.iloc[start_idx:end_idx].reset_index(drop=True)

    if window.empty:
        st.warning(f"Velas fuera del rango para **{ticker}** ({entry_dt.date()}).")
        return

    entry_p  = float(trade["Entry$"])
    exit_p   = float(trade["Exit$"])
    sl       = float(trade["SL"])
    tp_price = float(trade["TP Price"]) if "TP Price" in trade.index and pd.notna(trade["TP Price"]) else None
    is_winner = bool(trade["Win"])

    def ts(dt): return int(pd.Timestamp(dt).timestamp())

    candle_data = [
        {"time": ts(r["date"]), "open": round(r["open"], 4),
         "high": round(r["high"], 4), "low": round(r["low"], 4), "close": round(r["close"], 4)}
        for _, r in window.iterrows()
    ]
    volume_data = [
        {"time": ts(r["date"]), "value": round(r["volume"], 0),
         "color": "rgba(38,166,154,0.35)" if r["close"] >= r["open"] else "rgba(239,83,80,0.35)"}
        for _, r in window.iterrows()
    ]

    def line_series(col, color, title, width=1):
        if col not in window.columns:
            return None
        pts = [{"time": ts(r["date"]), "value": round(r[col], 4)}
               for _, r in window.iterrows() if pd.notna(r.get(col))]
        return {
            "type": "Line", "data": pts,
            "options": {
                "color": color, "lineWidth": width, "lineStyle": 0,
                "priceLineVisible": False, "lastValueVisible": True,
                "title": title, "crosshairMarkerVisible": False,
            },
        }

    def flat_line(price, color, label, style=2, width=1):
        pts = [{"time": ts(r["date"]), "value": round(price, 4)} for _, r in window.iterrows()]
        return {
            "type": "Line", "data": pts,
            "options": {
                "color": color, "lineWidth": width, "lineStyle": style,
                "priceLineVisible": False, "lastValueVisible": True,
                "title": label, "crosshairMarkerVisible": False,
            },
        }

    entry_ts = ts(entry_dt)
    exit_ts  = ts(exit_dt)
    trade_type = str(trade.get("Type", "short")).lower()
    markers = [
        {
            "time": entry_ts,
            "position": "aboveBar" if trade_type == "short" else "belowBar",
            "color": RED if trade_type == "short" else GREEN,
            "shape": "arrowDown" if trade_type == "short" else "arrowUp",
            "text": f"{trade_type.upper()} ${entry_p:.2f}", "size": 2,
        },
        {
            "time": exit_ts,
            "position": "belowBar" if trade_type == "short" else "aboveBar",
            "color": GREEN if is_winner else RED,
            "shape": "arrowUp" if trade_type == "short" else "arrowDown",
            "text": f"EXIT ${exit_p:.2f}", "size": 2,
        },
    ]

    chart_options = {
        "height": 460,
        "layout": {"background": {"type": "solid", "color": "#0f1117"}, "textColor": "#e0e0e0"},
        "grid":   {"vertLines": {"color": "#2e3547"}, "horzLines": {"color": "#2e3547"}},
        "crosshair": {"mode": 1},
        "rightPriceScale": {"borderColor": "#2e3547"},
        "timeScale": {"borderColor": "#2e3547", "timeVisible": True, "secondsVisible": False},
    }

    series = [
        {
            "type": "Candlestick", "data": candle_data,
            "options": {
                "upColor": "#26a69a", "downColor": "#ef5350",
                "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
                "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
            },
            "markers": markers,
        },
        {
            "type": "Histogram", "data": volume_data,
            "options": {
                "priceFormat": {"type": "volume"},
                "priceScaleId": "volume",
                "scaleMargins": {"top": 0.85, "bottom": 0},
            },
        },
        flat_line(sl, "#ef5350", f"SL {sl:.2f}"),
    ]
    if tp_price:
        series.append(flat_line(tp_price, "#26a69a", f"TP {tp_price:.2f}"))
    for col, color, title, width in [
        ("sma_9",  "#ffffff", "SMA9",  1),
        ("sma_20", "#ffca28", "SMA20", 1),
        ("sma_50", "#42a5f5", "SMA50", 1),
    ]:
        s = line_series(col, color, title, width)
        if s:
            series.append(s)

    net_pnl   = float(trade["Net PnL($)"])
    pnl_color = GREEN if net_pnl >= 0 else RED
    src_badge = "🟢 Polygon" if source_label == "polygon" else "💾 Local"
    st.markdown(
        f"**{ticker}** &nbsp;·&nbsp; `{timeframe}` &nbsp;·&nbsp; {src_badge} &nbsp;·&nbsp; "
        f"{entry_dt.strftime('%Y-%m-%d')} &nbsp;·&nbsp; "
        f"Entry **${entry_p:.2f}** → Exit **${exit_p:.2f}** &nbsp;·&nbsp; "
        f"Net PnL <span style='color:{pnl_color}'>**${net_pnl:+.2f}**</span>",
        unsafe_allow_html=True,
    )
    renderLightweightCharts(
        [{"chart": chart_options, "series": series}],
        key=f"chart_{ticker}_{entry_ts}_{source_label}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parámetros")
    st.divider()

    uploaded = st.file_uploader("📂 Cargar archivo .parquet", type=["parquet"])
    st.divider()

    st.subheader("💰 Position Sizing")
    initial_capital = st.number_input(
        "Capital total ($)", min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000, format="%d",
    )
    st.markdown("**% del equity arriesgado por trade**")
    _presets = {"0.25%": 0.25, "0.5%": 0.5, "1%": 1.0, "2%": 2.0, "3%": 3.0, "5%": 5.0}
    _preset_cols = st.columns(len(_presets))
    if "risk_pct_val" not in st.session_state:
        st.session_state.risk_pct_val = 1.0
    for _col, (_label, _val) in zip(_preset_cols, _presets.items()):
        if _col.button(_label, use_container_width=True):
            st.session_state.risk_pct_val = _val
    risk_pct = st.slider(
        "Personalizado", min_value=0.1, max_value=10.0,
        value=st.session_state.risk_pct_val, step=0.05, format="%.2f%%",
        key="risk_slider",
    )
    st.session_state.risk_pct_val = risk_pct
    _risk_dollar = initial_capital * risk_pct / 100
    st.caption(f"≡ **${_risk_dollar:,.2f}** por trade  ·  `shares = riesgo$ / (entry$ × stop%)`")

    sizing_mode = st.radio(
        "Equity de referencia",
        ["Fijo (capital inicial)", "Compuesto (equity actual)"],
        index=0,
    )
    st.divider()

    st.subheader("🏦 Comisiones (IBKR)")
    include_commissions = st.toggle("Incluir comisiones", value=True)
    st.divider()

    st.subheader("🔧 Filtros")

    # Ticker filter — always visible, populated from parquet
    selected_tickers = []
    if uploaded is not None:
        try:
            _tickers = sorted(
                pd.read_parquet(io.BytesIO(uploaded.getvalue()), columns=["ticker"])
                ["ticker"].unique().tolist()
            )
        except Exception:
            _tickers = []
        selected_tickers = st.multiselect("Ticker", _tickers, default=_tickers)
    else:
        st.multiselect("Ticker", [], disabled=True, help="Carga un parquet primero")

    # Date filter
    date_filter = st.checkbox("Filtrar por fechas", value=False)
    if date_filter:
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("Desde")
        with col2:
            to_date = st.date_input("Hasta")
    else:
        from_date = to_date = None

    st.divider()

    st.subheader("🕯️ Gráfico de velas")
    data_source = st.radio("Fuente de datos", ["💾 Local", "🟢 Polygon"], horizontal=True)
    use_polygon = data_source == "🟢 Polygon"

    if use_polygon:
        if POLYGON_API_KEY:
            st.success("API Key cargada desde .env", icon="🔑")
            polygon_key = POLYGON_API_KEY
        else:
            polygon_key = st.text_input("Polygon API Key", type="password")
        pre_days = st.slider("Días previos al trade (Polygon)", 1, 10, 5)
    else:
        polygon_key = ""
        pre_days    = 0
        indices_data_dir = st.text_input(
            "Directorio INDICES",
            value=INDICES_DATA_ROOT,
            help="Carpeta raíz con subcarpetas {TICKER}/{tf}/",
        )

    context_bars = st.slider("Velas de contexto post-exit", 10, 80, 40, 5)
    st.divider()
    st.caption("shares = (capital × risk%) / (entry × stop%)")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_compute(file_bytes: bytes, initial_capital: float,
                     risk_pct: float, compound: bool,
                     with_commissions: bool,
                     from_date, to_date,
                     selected_tickers: tuple) -> pd.DataFrame:

    df = pd.read_parquet(io.BytesIO(file_bytes))
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Coerce columns that may be stored as object dtype
    for _col in ("rvol_daily", "previous_day_close"):
        if _col in df.columns and df[_col].dtype == object:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    if from_date:
        df = df[df["entry_time"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["entry_time"] < pd.Timestamp(to_date) + pd.Timedelta(days=1)]
    if selected_tickers:
        df = df[df["ticker"].isin(selected_tickers)]
    df = df.reset_index(drop=True)

    if df.empty:
        return df

    df["stop_pct_row"] = (
        (df["stop_loss_price"] - df["entry_price"]).abs() / df["entry_price"]
    )

    equity = initial_capital
    gross_pnls, net_pnls, comm_list, shares_list = [], [], [], []

    for _, row in df.iterrows():
        cap    = equity if compound else initial_capital
        r_d    = cap * (risk_pct / 100)
        stop   = row["stop_pct_row"]
        shares = r_d / (row["entry_price"] * stop) if stop > 0 else 1.0
        gross  = row["pnl"] * shares
        comm   = (commission_per_trade(shares, row["entry_price"], row["exit_price"],
                                       row["entry_time"], row["exit_time"])
                  if with_commissions else 0.0)
        net    = gross - comm
        gross_pnls.append(gross)
        net_pnls.append(net)
        comm_list.append(comm)
        shares_list.append(shares)
        equity += net

    df["shares"]     = shares_list
    df["gross_pnl"]  = gross_pnls
    df["commission"] = comm_list
    df["scaled_pnl"] = net_pnls

    df["cum_gross"]    = df["gross_pnl"].cumsum()
    df["equity_gross"] = initial_capital + df["cum_gross"]
    df["cum_pnl"]      = df["scaled_pnl"].cumsum()
    df["equity_curve"] = initial_capital + df["cum_pnl"]

    roll_max            = df["equity_curve"].cummax()
    df["drawdown_abs"]  = df["equity_curve"] - roll_max
    df["drawdown_pct"]  = df["drawdown_abs"] / roll_max * 100

    df["winner"]  = df["pnl"] > 0
    df["ym"]      = df["entry_time"].dt.to_period("M")
    df["year"]    = df["entry_time"].dt.year
    df["hour"]    = df["entry_time"].dt.hour
    df["weekday"] = df["entry_time"].dt.day_name()
    df["ret_pct"] = df["Return"] * 100
    df["hold_h"]  = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
    df["comm_pct_notional"] = df["commission"] / (df["entry_price"] * df["shares"]) * 100

    return df


@st.cache_data(show_spinner="Descargando SPY de Polygon...")
def load_spy_bnh(from_date: str, to_date: str,
                 initial_capital: float, api_key: str) -> pd.DataFrame:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    rows, next_url = [], url
    while next_url:
        resp = requests.get(next_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("results", []))
        next_url = data.get("next_url")
        if next_url:
            next_url = f"{next_url}&apiKey={api_key}"
    if not rows:
        return pd.DataFrame()
    spy = pd.DataFrame(rows)[["t", "c"]].rename(columns={"t": "date", "c": "close"})
    spy["date"] = (pd.to_datetime(spy["date"], unit="ms")
                   .dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                   .dt.tz_localize(None).dt.normalize())
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["bnh_equity"] = initial_capital * (spy["close"] / spy["close"].iloc[0])
    spy["bnh_ret"]    = (spy["bnh_equity"] / initial_capital - 1) * 100
    return spy


def recompute_equity(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["cum_pnl"]      = df["scaled_pnl"].cumsum()
    df["cum_gross"]    = df["gross_pnl"].cumsum()
    df["equity_curve"] = initial_capital + df["cum_pnl"]
    df["equity_gross"] = initial_capital + df["cum_gross"]
    roll_max           = df["equity_curve"].cummax()
    df["drawdown_abs"] = df["equity_curve"] - roll_max
    df["drawdown_pct"] = df["drawdown_abs"] / roll_max * 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pf(x):
    w = x[x > 0].sum(); l = abs(x[x < 0].sum())
    return w / l if l > 0 else np.inf

def fmt_dollar(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"

def compute_streaks(winner: pd.Series) -> pd.Series:
    streak, cur = [], 0
    for w in winner:
        if w:
            cur = cur + 1 if cur > 0 else 1
        else:
            cur = cur - 1 if cur < 0 else -1
        streak.append(cur)
    return pd.Series(streak, index=winner.index, dtype=int)

def color_val(v):
    return GREEN if v >= 0 else RED


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.title("📊 Indices Backtest Dashboard")
    st.info("👈 Carga un archivo **.parquet** de trades en el panel lateral para comenzar.")
    st.stop()

with st.spinner("Calculando..."):
    compound = sizing_mode.startswith("Compuesto")
    df = load_and_compute(
        uploaded.read(), initial_capital, risk_pct, compound,
        include_commissions,
        from_date if date_filter else None,
        to_date   if date_filter else None,
        tuple(selected_tickers),
    )

if df.empty:
    st.warning("No hay trades con los filtros seleccionados.")
    st.stop()

# ── Global KPIs ───────────────────────────────────────────────────────────────
total      = len(df)
wins       = int(df["winner"].sum())
wr         = wins / total * 100
gross_pnl  = df["gross_pnl"].sum()
total_comm = df["commission"].sum()
net_pnl    = df["scaled_pnl"].sum()
final_eq   = initial_capital + net_pnl
total_ret  = net_pnl / initial_capital * 100
pf_val     = pf(df["scaled_pnl"])
mdd_abs    = df["drawdown_abs"].min()
mdd_pct    = df["drawdown_pct"].min()
avg_win    = df.loc[df["winner"],  "scaled_pnl"].mean()
avg_loss   = df.loc[~df["winner"], "scaled_pnl"].mean()
sh         = (df["ret_pct"].mean() / df["ret_pct"].std()) * np.sqrt(252) if df["ret_pct"].std() else 0
avg_hold   = df["hold_h"].mean()
comm_drag  = total_comm / abs(gross_pnl) * 100 if gross_pnl != 0 else 0
expectancy_gross = df["gross_pnl"].mean()
expectancy_net   = df["scaled_pnl"].mean()
expectancy_comm  = df["commission"].mean()

df["streak"]    = compute_streaks(df["winner"])
max_win_streak  = int(df["streak"].max())
max_loss_streak = int(df["streak"].min())
current_streak  = int(df["streak"].iloc[-1])

# ── SPY Buy & Hold ────────────────────────────────────────────────────────────
spy_df = pd.DataFrame()
spy_bnh_ret = spy_bnh_eq = spy_alpha = None
_spy_key = POLYGON_API_KEY or (polygon_key if use_polygon else "")
if _spy_key:
    try:
        _spy_from = df["entry_time"].min().strftime("%Y-%m-%d")
        _spy_to   = df["entry_time"].max().strftime("%Y-%m-%d")
        spy_df    = load_spy_bnh(_spy_from, _spy_to, initial_capital, _spy_key)
        if not spy_df.empty:
            spy_bnh_eq  = spy_df["bnh_equity"].iloc[-1]
            spy_bnh_ret = spy_df["bnh_ret"].iloc[-1]
            spy_alpha   = total_ret - spy_bnh_ret
    except Exception:
        pass

spy_sharpe_bnh = spy_mdd_bnh = None
if not spy_df.empty:
    _spy_daily = spy_df["bnh_equity"].pct_change().dropna() * 100
    if _spy_daily.std() > 0:
        spy_sharpe_bnh = (_spy_daily.mean() / _spy_daily.std()) * np.sqrt(252)
    _spy_rm = spy_df["bnh_equity"].cummax()
    spy_mdd_bnh = ((spy_df["bnh_equity"] - _spy_rm) / _spy_rm * 100).min()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
strategy_name = df["strategy"].iloc[0] if "strategy" in df.columns else uploaded.name
tickers_label = ", ".join(sorted(df["ticker"].unique()))
st.title(f"📊 {strategy_name}")
st.caption(
    f"**{df['entry_time'].min().strftime('%b %Y')} → {df['entry_time'].max().strftime('%b %Y')}**  |  "
    f"Tickers: **{tickers_label}**  |  Capital: **${initial_capital:,}**  |  "
    f"Risk/trade: **{risk_pct}%**  |  Modo: **{'Compuesto' if compound else 'Fijo'}**  |  "
    f"Comisiones: **{'ON ✅' if include_commissions else 'OFF ❌'}**"
)
st.divider()

# ── KPI Tables ────────────────────────────────────────────────────────────────
has_spy = spy_bnh_ret is not None
_cur_label = (f"↑ {current_streak} wins" if current_streak > 0
              else f"↓ {abs(current_streak)} losses" if current_streak < 0
              else "—")
_wl = abs(avg_win / avg_loss) if avg_loss else 0

_strat_rows: list[tuple[str, str]] = [
    ("Capital Inicial",   f"${initial_capital:,}"),
    ("Return Total",      f"{total_ret:+.1f}%"),
    ("Final Equity",      f"${final_eq:,.0f}"),
    ("Max Drawdown",      f"{mdd_pct:.1f}%  (${abs(mdd_abs):,.0f})"),
    ("Sharpe (ann.)",     f"{sh:.2f}"),
    ("Total Trades",      f"{total:,}"),
    ("Win Rate",          f"{wr:.1f}%"),
    ("Profit Factor",     f"{pf_val:.2f}"),
    ("Gross PnL",         fmt_dollar(gross_pnl)),
]
if include_commissions:
    _strat_rows += [
        ("Net PnL",           fmt_dollar(net_pnl)),
        ("Total Comisiones",  f"-${total_comm:,.2f}"),
        ("Avg Comisión",      f"-${expectancy_comm:.2f}"),
        ("Comm. Drag",        f"{comm_drag:.1f}%"),
        ("Expectancy Bruta",  fmt_dollar(expectancy_gross)),
        ("Expectancy Neta",   fmt_dollar(expectancy_net)),
    ]
else:
    _strat_rows += [("Expectancy", fmt_dollar(expectancy_gross))]
_strat_rows += [
    ("Avg Win",          fmt_dollar(avg_win)),
    ("Avg Loss",         fmt_dollar(avg_loss)),
    ("Win/Loss Ratio",   f"{_wl:.2f}"),
    ("Avg Hold",         f"{avg_hold:.1f}h"),
    ("Avg Shares/Trade", f"{df['shares'].mean():,.0f}"),
    ("Max Win Streak",   f"{max_win_streak}"),
    ("Max Loss Streak",  f"{abs(max_loss_streak)}"),
    ("Racha Actual",     _cur_label),
]

_strat_df = pd.DataFrame(_strat_rows, columns=["Métrica", "Estrategia"])
_tbl_strat, _tbl_spy = st.columns(2)
with _tbl_strat:
    st.markdown("**Estrategia**")
    st.dataframe(_strat_df, hide_index=True, use_container_width=True)

if has_spy:
    _spy_rows: list[tuple[str, str]] = [
        ("Return Total",   f"{spy_bnh_ret:+.1f}%"),
        ("Final Equity",   f"${spy_bnh_eq:,.0f}"),
        ("Alpha",          f"{spy_alpha:+.1f}%"),
        ("Max Drawdown",   f"{spy_mdd_bnh:.1f}%" if spy_mdd_bnh is not None else "—"),
        ("Sharpe (ann.)",  f"{spy_sharpe_bnh:.2f}" if spy_sharpe_bnh is not None else "—"),
    ]
    _pad = len(_strat_rows) - len(_spy_rows)
    _spy_rows += [("", "")] * _pad
    _spy_df = pd.DataFrame(_spy_rows, columns=["Métrica", "S&P 500 B&H"])
    with _tbl_spy:
        st.markdown("**S&P 500 B&H**")
        st.dataframe(_spy_df, hide_index=True, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Equity & Drawdown",
    "💸 Comisiones",
    "📅 Por Período",
    "📐 Distribución",
    "🔬 Trade Analysis",
    "🏷️ Por Ticker",
    "📋 Trades",
    "🎲 Montecarlo",
    "⚖️ Comparar",
])


# ── TAB 1: Equity + Drawdown ──────────────────────────────────────────────────
with tab1:
    col_eq, col_dd = st.columns([3, 1])
    with col_eq:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.04,
            subplot_titles=("Equity Curve vs SPY Buy & Hold", "Drawdown (%)"),
        )
        if not spy_df.empty:
            fig.add_trace(go.Scatter(
                x=spy_df["date"], y=spy_df["bnh_equity"],
                mode="lines", name="SPY Buy & Hold",
                line=dict(color="#FF9800", width=1.5, dash="dot"),
            ), row=1, col=1)
        if include_commissions:
            fig.add_trace(go.Scatter(
                x=df["entry_time"], y=df["equity_gross"],
                mode="lines", name="Gross (sin comm.)",
                line=dict(color=BLUE, width=1, dash="dot"),
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["entry_time"], y=df["equity_curve"],
            mode="lines", name="Net equity",
            line=dict(color=GREEN, width=1.8),
            fill="tozeroy", fillcolor="rgba(38,166,154,0.07)",
        ), row=1, col=1)
        fig.add_hline(y=initial_capital, line_dash="dot", line_color=SUB, opacity=0.5, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["entry_time"], y=df["drawdown_pct"],
            mode="lines", name="Drawdown %",
            line=dict(color=RED, width=1),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.25)",
        ), row=2, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=520,
                          yaxis_title="Equity ($)", yaxis2_title="DD (%)")
        fig.update_xaxes(gridcolor="#2e3547")
        fig.update_yaxes(gridcolor="#2e3547")
        st.plotly_chart(fig, use_container_width=True)

    with col_dd:
        st.markdown("#### Resumen")
        st.metric("Capital inicial",  f"${initial_capital:,}")
        st.metric("Capital final",    f"${final_eq:,.0f}",  delta=f"{total_ret:+.1f}%")
        st.metric("Max DD ($)",       f"${mdd_abs:,.2f}",   delta=f"{mdd_pct:.1f}%", delta_color="inverse")
        if spy_bnh_ret is not None:
            st.metric("SPY B&H final",  f"${spy_bnh_eq:,.0f}", delta=f"{spy_bnh_ret:+.1f}%")
            st.metric("Alpha vs SPY",   f"{spy_alpha:+.1f}%",
                      delta="outperform ✅" if spy_alpha > 0 else "underperform ❌",
                      delta_color="normal" if spy_alpha > 0 else "inverse")
        st.divider()
        st.metric("Avg Win",    fmt_dollar(avg_win))
        st.metric("Avg Loss",   fmt_dollar(avg_loss))
        st.metric("Win/Loss",   f"{_wl:.2f}")
        if include_commissions:
            st.metric("Expect. Bruta",  fmt_dollar(expectancy_gross))
            st.metric("Avg Comisión",   f"-${expectancy_comm:.2f}",
                      delta=f"-{comm_drag:.1f}% drag", delta_color="inverse")
            st.metric("Expect. Neta",   fmt_dollar(expectancy_net),
                      delta=fmt_dollar(expectancy_net - expectancy_gross))
        else:
            st.metric("Expectancy", fmt_dollar(expectancy_gross))


# ── TAB 2: Commissions ────────────────────────────────────────────────────────
with tab2:
    if not include_commissions:
        st.info("Activa **Incluir comisiones** en el sidebar para ver este análisis.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_c1 = go.Figure()
            fig_c1.add_trace(go.Scatter(
                x=df["entry_time"], y=df["commission"].cumsum(),
                mode="lines", name="Comm. acumulada",
                line=dict(color=ORANGE, width=1.5),
                fill="tozeroy", fillcolor="rgba(255,167,38,0.1)",
            ))
            fig_c1.update_layout(**PLOTLY_LAYOUT, height=320,
                                 title="Comisión Acumulada ($)", yaxis_title="$")
            st.plotly_chart(fig_c1, use_container_width=True)
        with col_b:
            fig_c2 = go.Figure()
            fig_c2.add_trace(go.Scatter(
                x=df["entry_time"], y=df["equity_gross"],
                mode="lines", name="Gross equity",
                line=dict(color=BLUE, width=1.5, dash="dot"),
            ))
            fig_c2.add_trace(go.Scatter(
                x=df["entry_time"], y=df["equity_curve"],
                mode="lines", name="Net equity",
                line=dict(color=GREEN, width=1.5),
            ))
            fig_c2.add_trace(go.Scatter(
                x=df["entry_time"],
                y=df["equity_gross"] - df["equity_curve"],
                mode="lines", name="Comm. drag ($)",
                line=dict(color=ORANGE, width=1),
                fill="tozeroy", fillcolor="rgba(255,167,38,0.12)",
                yaxis="y2",
            ))
            fig_c2.update_layout(
                **PLOTLY_LAYOUT, height=320, title="Gross vs Net Equity",
                yaxis2=dict(overlaying="y", side="right",
                            title="Comm. drag ($)", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(color=ORANGE)),
            )
            st.plotly_chart(fig_c2, use_container_width=True)

        monthly_comm = df.groupby("ym").agg(
            total_comm=("commission", "sum"),
            avg_comm  =("commission", "mean"),
            gross_pnl =("gross_pnl",  "sum"),
            net_pnl   =("scaled_pnl", "sum"),
        ).reset_index()
        monthly_comm["ym_str"]   = monthly_comm["ym"].astype(str)
        monthly_comm["drag_pct"] = (
            monthly_comm["total_comm"] / monthly_comm["gross_pnl"].abs() * 100
        ).clip(-500, 500)
        tr_by_m = df.groupby("ym")["commission"].count().reset_index().rename(
            columns={"commission": "trades"})
        mc_disp = monthly_comm.merge(tr_by_m, on="ym")
        mc_disp = mc_disp[["ym_str","trades","gross_pnl","total_comm","net_pnl","drag_pct"]].copy()
        mc_disp["gross_pnl"]  = mc_disp["gross_pnl"].map(fmt_dollar)
        mc_disp["total_comm"] = mc_disp["total_comm"].map(lambda x: f"${x:,.2f}")
        mc_disp["net_pnl"]    = mc_disp["net_pnl"].map(fmt_dollar)
        mc_disp["drag_pct"]   = mc_disp["drag_pct"].map(lambda x: f"{x:.1f}%")
        mc_disp.columns = ["Mes","Trades","Gross PnL","Comisiones","Net PnL","Comm Drag%"]
        st.markdown("#### Resumen mensual de comisiones")
        st.dataframe(mc_disp, use_container_width=True, hide_index=True, height=380)


# ── TAB 3: Por Período ────────────────────────────────────────────────────────
with tab3:
    monthly = df.groupby("ym").agg(
        net_pnl  =("scaled_pnl","sum"),
        gross_pnl=("gross_pnl", "sum"),
        trades   =("scaled_pnl","count"),
        win_rate =("winner",    "mean"),
        pf_m     =("scaled_pnl", pf),
        avg_ret  =("ret_pct",   "mean"),
    ).reset_index()
    monthly["ym_str"] = monthly["ym"].astype(str)
    monthly["pf_m"]   = monthly["pf_m"].clip(0, 10)
    monthly["wr_pct"] = monthly["win_rate"] * 100

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig2 = go.Figure()
        if include_commissions:
            fig2.add_trace(go.Bar(
                x=monthly["ym_str"], y=monthly["gross_pnl"],
                marker_color=[GREEN if v >= 0 else RED for v in monthly["gross_pnl"]],
                opacity=0.35, name="Gross PnL",
            ))
        fig2.add_trace(go.Bar(
            x=monthly["ym_str"], y=monthly["net_pnl"],
            marker_color=[GREEN if v >= 0 else RED for v in monthly["net_pnl"]],
            name="Net PnL",
            text=[fmt_dollar(v) for v in monthly["net_pnl"]],
            textposition="outside", textfont=dict(size=7),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=350, barmode="overlay",
                           title="Net PnL por Mes", yaxis_title="$ PnL")
        fig2.update_xaxes(tickangle=45, tickfont=dict(size=7))
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=monthly["ym_str"], y=monthly["wr_pct"],
            marker_color=[GREEN if v >= 50 else RED for v in monthly["wr_pct"]],
        ))
        fig3.add_hline(y=50, line_dash="dash", line_color=YELLOW, opacity=0.7)
        fig3.update_layout(**PLOTLY_LAYOUT, height=350,
                           title="Win Rate % por Mes", yaxis_range=[0, 100])
        fig3.update_xaxes(tickangle=90, tickfont=dict(size=6))
        st.plotly_chart(fig3, use_container_width=True)

    yearly = df.groupby("year").agg(
        net_pnl  =("scaled_pnl","sum"),
        gross_pnl=("gross_pnl", "sum"),
        comm     =("commission","sum"),
        trades   =("scaled_pnl","count"),
        win_rate =("winner",    "mean"),
        pf_y     =("scaled_pnl", pf),
        mdd      =("drawdown_abs","min"),
    ).reset_index()
    yearly["wr_pct"] = yearly["win_rate"] * 100
    yearly["ret_y"]  = yearly["net_pnl"] / initial_capital * 100

    col_c, col_d = st.columns([1, 2])
    with col_c:
        fig4 = go.Figure()
        if include_commissions:
            fig4.add_trace(go.Bar(
                x=yearly["year"].astype(str), y=yearly["gross_pnl"],
                marker_color=[GREEN if v >= 0 else RED for v in yearly["gross_pnl"]],
                opacity=0.35, name="Gross",
            ))
        fig4.add_trace(go.Bar(
            x=yearly["year"].astype(str), y=yearly["net_pnl"],
            marker_color=[GREEN if v >= 0 else RED for v in yearly["net_pnl"]],
            name="Net",
            text=[fmt_dollar(v) for v in yearly["net_pnl"]],
            textposition="outside",
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=320, barmode="overlay",
                           title="Net PnL por Año")
        st.plotly_chart(fig4, use_container_width=True)
    with col_d:
        st.markdown("#### Tabla anual")
        yd = yearly.copy()
        yd["win_rate"] = yd["wr_pct"].map(lambda x: f"{x:.1f}%")
        yd["net_pnl"]  = yd["net_pnl"].map(fmt_dollar)
        yd["gross_pnl"]= yd["gross_pnl"].map(fmt_dollar)
        yd["comm"]     = yd["comm"].map(lambda x: f"${x:,.2f}")
        yd["ret_y"]    = yd["ret_y"].map(lambda x: f"{x:+.1f}%")
        yd["pf_y"]     = yd["pf_y"].map(lambda x: f"{x:.2f}")
        yd["mdd"]      = yd["mdd"].map(lambda x: f"${x:,.0f}")
        cols = ["year","trades","win_rate","gross_pnl","comm","net_pnl","ret_y","pf_y","mdd"]
        st.dataframe(yd[cols].rename(columns=dict(zip(cols,
            ["Año","Trades","WR%","Gross PnL","Comisiones","Net PnL","Ret%","PF","MaxDD"]))),
            use_container_width=True, hide_index=True)

    st.markdown("#### Tabla mensual")
    md = monthly.copy()
    md["net_pnl"]   = md["net_pnl"].map(fmt_dollar)
    md["gross_pnl"] = md["gross_pnl"].map(fmt_dollar)
    md["wr_pct"]    = md["wr_pct"].map(lambda x: f"{x:.1f}%")
    md["pf_m"]      = md["pf_m"].map(lambda x: f"{x:.2f}")
    md["avg_ret"]   = md["avg_ret"].map(lambda x: f"{x:.2f}%")
    show_cols  = ["ym_str","trades","wr_pct","gross_pnl","net_pnl","avg_ret","pf_m"]
    col_labels = ["Mes","Trades","WR%","Gross PnL","Net PnL","Avg Ret","PF"]
    st.dataframe(md[show_cols].rename(columns=dict(zip(show_cols, col_labels))),
                 use_container_width=True, hide_index=True, height=350)


# ── TAB 4: Distribución ───────────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(
            x=df.loc[~df["winner"],"ret_pct"].clip(-120,100),
            nbinsx=50, marker_color=RED, opacity=0.75, name="Losers",
        ))
        fig5.add_trace(go.Histogram(
            x=df.loc[df["winner"],"ret_pct"].clip(-120,100),
            nbinsx=50, marker_color=GREEN, opacity=0.75, name="Winners",
        ))
        fig5.add_vline(x=df["ret_pct"].mean(), line_dash="dash", line_color=YELLOW,
                       annotation_text=f"Mean {df['ret_pct'].mean():.1f}%")
        fig5.add_vline(x=df["ret_pct"].median(), line_dash="dot", line_color=BLUE,
                       annotation_text=f"Med {df['ret_pct'].median():.1f}%")
        fig5.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay",
                           title="Distribución de Retornos (%)", xaxis_title="Return %")
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        clip_lo = df["scaled_pnl"].quantile(0.01)
        clip_hi = df["scaled_pnl"].quantile(0.99)
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(
            x=df.loc[df["scaled_pnl"] >= 0, "scaled_pnl"].clip(clip_lo, clip_hi),
            nbinsx=50, marker_color=GREEN, opacity=0.8, name="Ganadores",
        ))
        fig6.add_trace(go.Histogram(
            x=df.loc[df["scaled_pnl"] < 0, "scaled_pnl"].clip(clip_lo, clip_hi),
            nbinsx=50, marker_color=RED, opacity=0.8, name="Perdedores",
        ))
        fig6.add_vline(x=0, line_color=SUB, line_dash="dot")
        fig6.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay",
                           title="Distribución de Net PnL ($) por Trade",
                           xaxis_title="Net PnL ($)")
        st.plotly_chart(fig6, use_container_width=True)

    bins_ret   = [-999,-20,-10,-5,-1,0,1,5,10,20,50,999]
    labels_ret = ["<-20%","-20/-10%","-10/-5%","-5/-1%","-1/0%",
                  "0/1%","1/5%","5/10%","10/20%","20/50%",">50%"]
    df["ret_bin"] = pd.cut(df["ret_pct"], bins=bins_ret, labels=labels_ret)
    dist = df["ret_bin"].value_counts().reindex(labels_ret).fillna(0)
    fig7 = go.Figure(go.Bar(
        x=labels_ret, y=dist.values,
        marker_color=[RED]*5 + [GREEN]*6,
        text=[f"{v:.0f} ({v/total*100:.1f}%)" for v in dist.values],
        textposition="outside",
    ))
    fig7.update_layout(**PLOTLY_LAYOUT, height=320,
                       title="Frecuencia por Bucket de Retorno", yaxis_title="# Trades")
    st.plotly_chart(fig7, use_container_width=True)


# ── TAB 5: Trade Analysis ─────────────────────────────────────────────────────
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig8 = go.Figure()
        for win, color, name in [(True, GREEN, "Winner"), (False, RED, "Loser")]:
            s = sample[sample["winner"] == win]
            fig8.add_trace(go.Scatter(
                x=s["mae_pct"], y=s["mfe_pct"], mode="markers",
                marker=dict(color=color, size=4, opacity=0.4), name=name,
            ))
        fig8.add_trace(go.Scatter(x=[0,80], y=[0,80], mode="lines",
                                  line=dict(color=SUB, dash="dash", width=1),
                                  showlegend=False))
        fig8.update_layout(**PLOTLY_LAYOUT, height=360, title="MAE % vs MFE %",
                           xaxis_title="MAE % (adverse)", yaxis_title="MFE % (favorable)",
                           xaxis_range=[0,80], yaxis_range=[0,80])
        st.plotly_chart(fig8, use_container_width=True)
    with col2:
        by_hour = df.groupby("hour").agg(
            net_pnl =("scaled_pnl","sum"),
            win_rate=("winner",    "mean"),
            trades  =("scaled_pnl","count"),
        ).reset_index()
        fig9 = make_subplots(specs=[[{"secondary_y": True}]])
        fig9.add_trace(go.Bar(
            x=by_hour["hour"], y=by_hour["net_pnl"],
            marker_color=[GREEN if v >= 0 else RED for v in by_hour["net_pnl"]],
            name="Net PnL",
        ), secondary_y=False)
        fig9.add_trace(go.Scatter(
            x=by_hour["hour"], y=by_hour["win_rate"]*100,
            mode="lines+markers", line=dict(color=YELLOW, width=2),
            marker=dict(size=6), name="Win Rate %",
        ), secondary_y=True)
        fig9.update_layout(**PLOTLY_LAYOUT, height=360,
                           title="Net PnL y Win Rate por Hora (ET)")
        fig9.update_yaxes(title_text="Net PnL ($)",  secondary_y=False, gridcolor="#2e3547")
        fig9.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0,100],
                          gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig9, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # PnL by trade direction (long/short)
        by_type = df.groupby("type").agg(
            net_pnl =("scaled_pnl","sum"),
            win_rate=("winner",    "mean"),
            trades  =("scaled_pnl","count"),
        ).reset_index()
        fig_type = make_subplots(specs=[[{"secondary_y": True}]])
        fig_type.add_trace(go.Bar(
            x=by_type["type"], y=by_type["net_pnl"],
            marker_color=[GREEN if v >= 0 else RED for v in by_type["net_pnl"]],
            name="Net PnL",
            text=[fmt_dollar(v) for v in by_type["net_pnl"]],
            textposition="outside",
        ), secondary_y=False)
        fig_type.add_trace(go.Scatter(
            x=by_type["type"], y=by_type["win_rate"]*100,
            mode="markers+text",
            marker=dict(color=YELLOW, size=14),
            text=[f"{v*100:.1f}%" for v in by_type["win_rate"]],
            textposition="top center",
            name="Win Rate %",
        ), secondary_y=True)
        fig_type.update_layout(**PLOTLY_LAYOUT, height=340,
                               title="Net PnL y Win Rate por Dirección (Long/Short)")
        fig_type.update_yaxes(title_text="Net PnL ($)",  secondary_y=False, gridcolor="#2e3547")
        fig_type.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0,110],
                              gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_type, use_container_width=True)
    with col4:
        fig11 = go.Figure()
        fig11.add_trace(go.Histogram(
            x=df["hold_h"].clip(0, 12), nbinsx=30,
            marker_color=BLUE, opacity=0.8,
        ))
        fig11.add_vline(x=df["hold_h"].mean(), line_dash="dash", line_color=YELLOW,
                        annotation_text=f"Avg {df['hold_h'].mean():.1f}h")
        fig11.update_layout(**PLOTLY_LAYOUT, height=340, title="Hold Time (horas)",
                            xaxis_title="Horas", yaxis_title="# Trades")
        st.plotly_chart(fig11, use_container_width=True)

    heatmap_data = df.groupby(["weekday","hour"])["scaled_pnl"].sum().unstack(fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
    fig12 = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h}h" for h in heatmap_data.columns],
        y=heatmap_data.index.tolist(),
        colorscale=[[0,RED],[0.5,"#1a1d27"],[1,GREEN]],
        zmid=0,
        text=np.round(heatmap_data.values, 0),
        texttemplate="%{text:.0f}",
        textfont=dict(size=8),
    ))
    fig12.update_layout(**PLOTLY_LAYOUT, height=260, title="Heatmap Net PnL: Día × Hora")
    st.plotly_chart(fig12, use_container_width=True)

    # Streak analysis
    st.divider()
    st.markdown("#### 🔗 Rachas (Streaks)")
    str_col1, str_col2 = st.columns(2)
    with str_col1:
        fig_str1 = go.Figure()
        fig_str1.add_trace(go.Bar(
            x=df["entry_time"], y=df["streak"],
            marker_color=[GREEN if v > 0 else RED for v in df["streak"]],
            name="Racha",
        ))
        fig_str1.add_hline(y=0, line_color=SUB, line_width=0.8)
        _idx_max_w = df["streak"].idxmax()
        _idx_max_l = df["streak"].idxmin()
        fig_str1.add_annotation(
            x=df.loc[_idx_max_w, "entry_time"], y=max_win_streak,
            text=f"Max +{max_win_streak}", showarrow=True,
            arrowhead=2, font=dict(color=GREEN, size=10),
            arrowcolor=GREEN, bgcolor=CARD,
        )
        fig_str1.add_annotation(
            x=df.loc[_idx_max_l, "entry_time"], y=max_loss_streak,
            text=f"Max {max_loss_streak}", showarrow=True,
            arrowhead=2, font=dict(color=RED, size=10),
            arrowcolor=RED, bgcolor=CARD,
        )
        fig_str1.update_layout(
            **PLOTLY_LAYOUT, height=340,
            title="Racha consecutiva por trade",
            xaxis_title="Fecha", yaxis_title="Racha (+win / −loss)",
        )
        st.plotly_chart(fig_str1, use_container_width=True)
    with str_col2:
        _winner_list = df["winner"].tolist()
        _win_runs, _loss_runs = [], []
        _cur_len, _cur_type = 1, _winner_list[0]
        for w in _winner_list[1:]:
            if w == _cur_type:
                _cur_len += 1
            else:
                (_win_runs if _cur_type else _loss_runs).append(_cur_len)
                _cur_len, _cur_type = 1, w
        (_win_runs if _cur_type else _loss_runs).append(_cur_len)
        _max_streak_len = max(max(_win_runs, default=0), max(_loss_runs, default=0))
        fig_str2 = go.Figure()
        fig_str2.add_trace(go.Histogram(
            x=_win_runs, xbins=dict(start=0.5, end=_max_streak_len+0.5, size=1),
            marker_color=GREEN, opacity=0.75, name="Rachas ganadoras",
        ))
        fig_str2.add_trace(go.Histogram(
            x=_loss_runs, xbins=dict(start=0.5, end=_max_streak_len+0.5, size=1),
            marker_color=RED, opacity=0.75, name="Rachas perdedoras",
        ))
        fig_str2.update_layout(
            **PLOTLY_LAYOUT, height=340, barmode="overlay",
            title="Distribución de longitud de rachas",
            xaxis_title="Longitud de la racha (# trades)",
            yaxis_title="Frecuencia",
        )
        fig_str2.update_xaxes(dtick=1)
        st.plotly_chart(fig_str2, use_container_width=True)

    _streak_summary = pd.DataFrame({
        "Tipo":          ["Ganadoras 🟢", "Perdedoras 🔴"],
        "Racha Máxima":  [f"{max_win_streak} trades", f"{abs(max_loss_streak)} trades"],
        "Nº de Rachas":  [len(_win_runs),  len(_loss_runs)],
        "Avg Longitud":  [f"{np.mean(_win_runs):.1f}"  if _win_runs  else "—",
                          f"{np.mean(_loss_runs):.1f}" if _loss_runs else "—"],
        "Rachas de 1":   [_win_runs.count(1),  _loss_runs.count(1)],
        "Rachas de 2+":  [sum(1 for r in _win_runs  if r >= 2),
                          sum(1 for r in _loss_runs if r >= 2)],
    })
    st.dataframe(_streak_summary, use_container_width=True, hide_index=True)


# ── TAB 6: Por Ticker ─────────────────────────────────────────────────────────
with tab6:
    by_ticker = df.groupby("ticker").agg(
        trades    =("scaled_pnl","count"),
        net_pnl   =("scaled_pnl","sum"),
        gross_pnl =("gross_pnl", "sum"),
        commission=("commission","sum"),
        win_rate  =("winner",    "mean"),
        avg_ret   =("ret_pct",   "mean"),
        pf_t      =("scaled_pnl", pf),
    ).reset_index().sort_values("net_pnl", ascending=False)
    by_ticker["wr_pct"] = by_ticker["win_rate"] * 100
    by_ticker["pf_t"]   = by_ticker["pf_t"].clip(0, 20)

    fig13 = go.Figure(go.Bar(
        x=by_ticker["ticker"], y=by_ticker["net_pnl"],
        marker_color=[GREEN if v >= 0 else RED for v in by_ticker["net_pnl"]],
        text=[fmt_dollar(v) for v in by_ticker["net_pnl"]],
        textposition="outside",
    ))
    fig13.update_layout(**PLOTLY_LAYOUT, height=360,
                        title="Net PnL por Ticker", yaxis_title="Net PnL ($)")
    st.plotly_chart(fig13, use_container_width=True)

    # Win rate by ticker
    fig_wr = go.Figure(go.Bar(
        x=by_ticker["ticker"], y=by_ticker["wr_pct"],
        marker_color=[GREEN if v >= 50 else RED for v in by_ticker["wr_pct"]],
        text=[f"{v:.1f}%" for v in by_ticker["wr_pct"]],
        textposition="outside",
    ))
    fig_wr.add_hline(y=50, line_dash="dash", line_color=YELLOW, opacity=0.7)
    fig_wr.update_layout(**PLOTLY_LAYOUT, height=300, title="Win Rate por Ticker",
                         yaxis_range=[0, 110])
    st.plotly_chart(fig_wr, use_container_width=True)

    st.markdown("#### Tabla por ticker")
    td = by_ticker.drop(columns=["win_rate"]).copy()
    td = td.rename(columns={
        "ticker":"Ticker","trades":"Trades","net_pnl":"Net PnL","gross_pnl":"Gross PnL",
        "commission":"Comisiones","wr_pct":"WR%","avg_ret":"Avg Ret","pf_t":"PF",
    })
    td["Net PnL"]    = td["Net PnL"].map(fmt_dollar)
    td["Gross PnL"]  = td["Gross PnL"].map(fmt_dollar)
    td["Comisiones"] = td["Comisiones"].map(lambda x: f"${x:,.2f}")
    td["WR%"]        = td["WR%"].map(lambda x: f"{x:.1f}%")
    td["Avg Ret"]    = td["Avg Ret"].map(lambda x: f"{x:.2f}%")
    td["PF"]         = td["PF"].map(lambda x: f"{x:.2f}")
    st.dataframe(
        td[["Ticker","Trades","WR%","Gross PnL","Comisiones","Net PnL","Avg Ret","PF"]],
        use_container_width=True, hide_index=True,
    )


# ── TAB 7: Trades ─────────────────────────────────────────────────────────────
with tab7:
    parquet_cols = [
        "entry_time","exit_time","ticker","timeframe","type",
        "entry_price","exit_price","stop_loss_price","take_profit_price","risk_reward_ratio",
        "pnl","Return","MAE","mae_pct","MFE","mfe_pct","volume","entry_volume",
    ]
    computed_cols = ["shares","gross_pnl","commission","scaled_pnl","winner","hold_h"]
    cols_show = [c for c in parquet_cols if c in df.columns] + computed_cols

    disp = df[cols_show].copy()
    for c in ["entry_price","exit_price","stop_loss_price","take_profit_price",
              "risk_reward_ratio","pnl","Return","MAE","MFE"]:
        if c in disp.columns:
            disp[c] = disp[c].round(4)
    for c in ["mae_pct","mfe_pct"]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)
    for c in ["volume","entry_volume"]:
        if c in disp.columns:
            disp[c] = disp[c].round(0).astype("Int64")
    disp["shares"]     = disp["shares"].round(0).astype("Int64")
    disp["gross_pnl"]  = disp["gross_pnl"].round(2)
    disp["commission"] = disp["commission"].round(2)
    disp["scaled_pnl"] = disp["scaled_pnl"].round(2)
    disp["hold_h"]     = disp["hold_h"].round(2)

    disp = disp.rename(columns={
        "entry_time":"Entry","exit_time":"Exit","ticker":"Ticker","timeframe":"TF","type":"Type",
        "entry_price":"Entry$","exit_price":"Exit$",
        "stop_loss_price":"SL","take_profit_price":"TP Price","risk_reward_ratio":"R/R",
        "pnl":"PnL/sh","Return":"Ret%",
        "MAE":"MAE$","mae_pct":"MAE%","MFE":"MFE$","mfe_pct":"MFE%",
        "volume":"Vol","entry_volume":"Entry Vol",
        "shares":"Shares","gross_pnl":"Gross($)","commission":"Comm($)","scaled_pnl":"Net PnL($)",
        "winner":"Win","hold_h":"Hold(h)",
    })
    disp["Ret%"] = (disp["Ret%"] * 100).round(2)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_only = st.selectbox("Mostrar", ["Todos","Solo Winners","Solo Losers"])
    with col_f2:
        sort_col = st.selectbox("Ordenar por", ["Entry","Net PnL($)","Ret%","Comm($)","Entry$"])
    with col_f3:
        sort_asc = st.radio("Orden", ["↓ Desc","↑ Asc"], horizontal=True) == "↑ Asc"

    if show_only == "Solo Winners":
        disp = disp[disp["Win"]]
    elif show_only == "Solo Losers":
        disp = disp[~disp["Win"]]

    disp = disp.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.caption("👆 Haz clic en una fila para ver el gráfico de velas")
    selection = st.dataframe(
        disp, use_container_width=True, hide_index=True, height=420,
        on_select="rerun", selection_mode="single-row",
    )
    csv = disp.to_csv(index=False).encode()
    st.download_button("⬇️ Descargar CSV", csv, "trades_filtered.csv", "text/csv")

    rows = selection.selection.rows if selection and selection.selection else []
    if rows:
        trade = disp.iloc[rows[0]]
        tf    = str(trade.get("TF", "5m"))
        st.divider()
        try:
            if use_polygon:
                if not polygon_key:
                    st.error("Introduce la Polygon API Key en el sidebar.")
                else:
                    entry_dt  = pd.to_datetime(trade["Entry"])
                    exit_dt   = pd.to_datetime(trade["Exit"])
                    from_d = (entry_dt - pd.Timedelta(days=pre_days)).strftime("%Y-%m-%d")
                    to_d   = (exit_dt  + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
                    candles = load_candles_polygon(trade["Ticker"], tf, from_d, to_d, polygon_key)
                    build_trade_chart(candles, trade, context_bars=context_bars,
                                      timeframe=tf, source_label="polygon")
            else:
                idir = indices_data_dir if not use_polygon else INDICES_DATA_ROOT
                candles = load_candles_indices_local(trade["Ticker"], tf, idir)
                build_trade_chart(candles, trade, context_bars=context_bars,
                                  timeframe=tf, source_label="local")
        except FileNotFoundError as e:
            st.error(f"Dataset local no encontrado: {e}")
        except requests.HTTPError as e:
            st.error(f"Error Polygon API: {e}")
        except Exception as e:
            st.error(f"Error al cargar el gráfico: {e}")


# ── TAB 8: Montecarlo ─────────────────────────────────────────────────────────
with tab8:
    import time as _time

    st.markdown("### 🎲 Simulación de Montecarlo")
    mc1, mc2, mc3, mc4 = st.columns([1, 1, 1, 1])
    n_sims    = mc1.selectbox("Nº simulaciones", [200, 500, 1000, 2000, 5000], index=2)
    ruin_thr  = mc2.slider("Umbral de ruina (% caída)", 10, 90, 50, 5)
    show_paths = mc3.slider("Paths a mostrar", 50, 500, 200, 50)
    use_net   = mc4.radio("PnL a usar", ["Net (c/ comisiones)", "Gross (s/ comisiones)"], index=0)

    run_mc = st.button("▶ Ejecutar nueva simulación", type="primary")

    pnl_series = df["scaled_pnl"].values if "Net" in use_net else df["gross_pnl"].values
    n_trades   = len(pnl_series)
    ruin_level = initial_capital * (1 - ruin_thr / 100)

    if run_mc:
        rng = np.random.default_rng(seed=int(_time.time() * 1000) % (2**32))
        sampled = rng.choice(pnl_series, size=(n_sims, n_trades), replace=True)
        mc_equity = initial_capital + np.cumsum(sampled, axis=1)
        st.session_state["mc_equity"]     = mc_equity
        st.session_state["mc_n_sims"]     = n_sims
        st.session_state["mc_ruin_level"] = ruin_level
        st.session_state["mc_ruin_thr"]   = ruin_thr

    if "mc_equity" not in st.session_state:
        st.info("👆 Pulsa **Ejecutar nueva simulación** para comenzar.")
    else:
        mc_equity   = st.session_state["mc_equity"]
        _ruin_level = st.session_state["mc_ruin_level"]
        _ruin_thr   = st.session_state["mc_ruin_thr"]
        _n_sims     = st.session_state["mc_n_sims"]

        p05 = np.percentile(mc_equity, 5,  axis=0)
        p25 = np.percentile(mc_equity, 25, axis=0)
        p50 = np.percentile(mc_equity, 50, axis=0)
        p75 = np.percentile(mc_equity, 75, axis=0)
        p95 = np.percentile(mc_equity, 95, axis=0)

        actual_equity  = df["equity_curve"].values
        x_axis         = np.arange(1, n_trades + 1)
        final_equities = mc_equity[:, -1]
        actual_final   = actual_equity[-1]
        rank_pct       = (final_equities < actual_final).mean() * 100
        hit_ruin       = (mc_equity.min(axis=1) <= _ruin_level).mean() * 100

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Simulaciones",        f"{_n_sims:,}")
        k2.metric("Equity real final",   fmt_dollar(actual_final))
        k3.metric("Percentil real",      f"{rank_pct:.1f}º")
        k4.metric("Mediana simulaciones",fmt_dollar(np.median(final_equities)))
        k5.metric("Riesgo de Ruina",     f"{hit_ruin:.1f}%",
                  delta=f"umbral -{_ruin_thr}% (${_ruin_level:,.0f})",
                  delta_color="inverse" if hit_ruin > 5 else "off")
        k6.metric("P5 equity final",     fmt_dollar(np.percentile(final_equities, 5)))
        st.divider()

        fig_mc = go.Figure()
        show_n  = min(show_paths, _n_sims)
        indices_sample = np.random.choice(_n_sims, size=show_n, replace=False)
        for i in indices_sample:
            fig_mc.add_trace(go.Scatter(
                x=x_axis, y=mc_equity[i], mode="lines",
                line=dict(color="rgba(144,164,174,0.08)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([x_axis, x_axis[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself", fillcolor="rgba(66,165,245,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="P5–P95", hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([x_axis, x_axis[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself", fillcolor="rgba(66,165,245,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="P25–P75", hoverinfo="skip",
        ))
        for arr, label, color, dash in [
            (p95,"P95","#42a5f5","dot"),(p75,"P75","#42a5f5","dash"),
            (p50,"Mediana","#ffca28","solid"),
            (p25,"P25","#ef5350","dash"),(p05,"P05","#ef5350","dot"),
        ]:
            fig_mc.add_trace(go.Scatter(
                x=x_axis, y=arr, mode="lines", name=label,
                line=dict(color=color, width=1.2, dash=dash),
            ))
        fig_mc.add_trace(go.Scatter(
            x=x_axis, y=actual_equity,
            mode="lines", name=f"Curva real (P{rank_pct:.0f})",
            line=dict(color="#26a69a", width=2.5),
        ))
        fig_mc.add_hline(y=initial_capital, line_dash="dot", line_color=SUB, opacity=0.5,
                         annotation_text="Capital inicial")
        fig_mc.add_hline(y=_ruin_level, line_dash="dash", line_color=RED, opacity=0.7,
                         annotation_text=f"Ruina (-{_ruin_thr}%)")
        fig_mc.update_layout(
            **PLOTLY_LAYOUT, height=500,
            title=f"Montecarlo — {_n_sims:,} simulaciones · Curva real en percentil {rank_pct:.1f}",
            xaxis_title="Trade #", yaxis_title="Equity ($)",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        col_dist, col_ruin = st.columns([2, 1])
        with col_dist:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=final_equities, nbinsx=80,
                marker_color=BLUE, opacity=0.75, name="Equity final",
            ))
            fig_dist.add_vline(x=actual_final, line_color=GREEN, line_width=2,
                               annotation_text=f"Real ${actual_final:,.0f}",
                               annotation_font_color=GREEN)
            fig_dist.add_vline(x=np.median(final_equities), line_color=YELLOW,
                               line_dash="dash", line_width=1.5,
                               annotation_text=f"Mediana ${np.median(final_equities):,.0f}",
                               annotation_font_color=YELLOW)
            fig_dist.add_vline(x=initial_capital, line_color=SUB, line_dash="dot",
                               annotation_text="Capital inicial")
            fig_dist.add_vline(x=_ruin_level, line_color=RED, line_dash="dash",
                               annotation_text=f"Ruina ${_ruin_level:,.0f}",
                               annotation_font_color=RED)
            fig_dist.update_layout(
                **PLOTLY_LAYOUT, height=340,
                title="Distribución de equity final",
                xaxis_title="Equity final ($)", yaxis_title="Frecuencia",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_ruin:
            st.markdown("#### 💀 Riesgo de Ruina")
            ror_color = RED if hit_ruin > 10 else (YELLOW if hit_ruin > 3 else GREEN)
            st.markdown(
                f"<h1 style='color:{ror_color};text-align:center;margin:0'>"
                f"{hit_ruin:.1f}%</h1>"
                f"<p style='text-align:center;color:{SUB};margin:4px 0'>"
                f"de {_n_sims:,} paths tocan la ruina<br>"
                f"(equity &lt; ${_ruin_level:,.0f})</p>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("**Percentiles de equity final:**")
            for p_val, p_label in [(5,"P5"),(10,"P10"),(25,"P25"),(50,"P50"),
                                    (75,"P75"),(90,"P90"),(95,"P95")]:
                val   = np.percentile(final_equities, p_val)
                ret   = (val - initial_capital) / initial_capital * 100
                color = GREEN if val >= initial_capital else RED
                st.markdown(
                    f"`{p_label}` &nbsp; <span style='color:{color}'>"
                    f"**${val:,.0f}** ({ret:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )
            st.divider()
            st.metric("Paths rentables",    f"{(final_equities > initial_capital).mean()*100:.1f}%")
            st.metric("Paths > curva real", f"{(final_equities > actual_final).mean()*100:.1f}%")


# ── TAB 9: Comparar estrategias ───────────────────────────────────────────────
with tab9:
    COMP_COLORS = [
        "#26a69a","#42a5f5","#ffca28","#ab47bc",
        "#ff7043","#66bb6a","#26c6da","#ec407a",
    ]
    st.markdown("### ⚖️ Comparador de Estrategias")

    if "comp_strategies" not in st.session_state:
        st.session_state.comp_strategies = []

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        new_file = st.file_uploader(
            "Añadir estrategia (.parquet)", type=["parquet"],
            key=f"comp_uploader_{len(st.session_state.comp_strategies)}",
        )
    with up_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("➕ Añadir", use_container_width=True, disabled=new_file is None)

    if add_btn and new_file is not None:
        raw = new_file.read()
        fname = new_file.name.replace(".parquet", "")
        existing_names = [s["name"] for s in st.session_state.comp_strategies]
        name = fname; suffix = 2
        while name in existing_names:
            name = f"{fname}_{suffix}"; suffix += 1
        st.session_state.comp_strategies.append({"name": name, "bytes": raw})
        st.rerun()

    if st.session_state.comp_strategies:
        st.markdown(f"**{len(st.session_state.comp_strategies)} estrategia(s) cargada(s):**")
        for i, strat in enumerate(st.session_state.comp_strategies):
            row = st.columns([0.4, 2.5, 1])
            color = COMP_COLORS[i % len(COMP_COLORS)]
            row[0].markdown(
                f"<div style='width:18px;height:18px;background:{color};"
                f"border-radius:3px;margin-top:8px'></div>",
                unsafe_allow_html=True,
            )
            new_name = row[1].text_input(
                "Nombre", value=strat["name"],
                key=f"comp_name_{i}", label_visibility="collapsed",
            )
            st.session_state.comp_strategies[i]["name"] = new_name
            if row[2].button("🗑 Eliminar", key=f"comp_del_{i}", use_container_width=True):
                st.session_state.comp_strategies.pop(i)
                st.rerun()
    else:
        st.info("Carga al menos una estrategia con el selector de arriba.")

    comp_dfs, comp_names = [], []
    if st.session_state.comp_strategies:
        st.divider()
        for strat in st.session_state.comp_strategies:
            try:
                cdf = load_and_compute(
                    strat["bytes"], initial_capital, risk_pct,
                    compound, include_commissions,
                    from_date if date_filter else None,
                    to_date   if date_filter else None,
                    tuple(selected_tickers),
                )
                comp_dfs.append(cdf)
                comp_names.append(strat["name"])
            except Exception as e:
                st.error(f"{strat['name']}: {e}")

    def _strat_metrics(cdf, name):
        _net = cdf["scaled_pnl"].sum()
        _sh  = (cdf["ret_pct"].mean() / cdf["ret_pct"].std() * np.sqrt(252)
                if cdf["ret_pct"].std() else 0)
        return {
            "Estrategia":    name,
            "Período":       (f"{cdf['entry_time'].min().strftime('%Y-%m-%d')} → "
                              f"{cdf['entry_time'].max().strftime('%Y-%m-%d')}"),
            "Trades":        f"{len(cdf):,}",
            "Win Rate":      f"{cdf['winner'].mean()*100:.1f}%",
            "Net PnL":       fmt_dollar(_net),
            "Return":        f"{_net/initial_capital*100:+.1f}%",
            "Comisiones":    f"${cdf['commission'].sum():,.2f}",
            "Profit Factor": f"{pf(cdf['scaled_pnl']):.2f}",
            "Sharpe":        f"{_sh:.2f}",
            "Max DD":        f"{cdf['drawdown_pct'].min():.1f}%",
            "Expectancy":    fmt_dollar(cdf["scaled_pnl"].mean()),
            "Avg Hold":      f"{cdf['hold_h'].mean():.1f}h",
        }

    if comp_dfs:
        metrics_df = pd.DataFrame(
            [_strat_metrics(cdf, name) for cdf, name in zip(comp_dfs, comp_names)]
        ).set_index("Estrategia")
        st.markdown("#### 📋 Tabla comparativa")
        st.dataframe(metrics_df.T, use_container_width=True)
        st.divider()

        show_combined = st.checkbox("⚡ Mostrar equity combinada", value=True)
        fig_comp = go.Figure()
        if not spy_df.empty:
            fig_comp.add_trace(go.Scatter(
                x=spy_df["date"], y=spy_df["bnh_equity"],
                mode="lines", name="SPY Buy & Hold",
                line=dict(color="#FF9800", width=1.8, dash="dot"),
            ))
        for i, (cdf, name) in enumerate(zip(comp_dfs, comp_names)):
            color = COMP_COLORS[i % len(COMP_COLORS)]
            fig_comp.add_trace(go.Scatter(
                x=cdf["entry_time"], y=cdf["equity_curve"],
                mode="lines", name=name,
                line=dict(color=color, width=1.8),
            ))
        if show_combined and len(comp_dfs) > 1:
            combined = pd.concat(
                [cdf[["entry_time","scaled_pnl"]].copy() for cdf in comp_dfs],
                ignore_index=True,
            ).sort_values("entry_time").reset_index(drop=True)
            combined["equity"] = initial_capital + combined["scaled_pnl"].cumsum()
            fig_comp.add_trace(go.Scatter(
                x=combined["entry_time"], y=combined["equity"],
                mode="lines", name="⚡ Combinada",
                line=dict(color="#ffffff", width=2.5),
            ))
        fig_comp.add_hline(y=initial_capital, line_dash="dot",
                           line_color=SUB, opacity=0.4, annotation_text="Capital inicial")
        fig_comp.update_layout(
            **PLOTLY_LAYOUT, height=540,
            title="Comparación de Equity Curves",
            xaxis_title="Fecha", yaxis_title="Equity ($)",
        )
        fig_comp.update_xaxes(gridcolor="#2e3547")
        fig_comp.update_yaxes(gridcolor="#2e3547")
        st.plotly_chart(fig_comp, use_container_width=True)
