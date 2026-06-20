"""
Backtest Results Dashboard  –  Streamlit + Plotly
Run:  streamlit run dashboard.py
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
    page_title="Backtest Dashboard",
    page_icon="📈",
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
    .comm-box {
        background: #1a1d27; border: 1px solid #ffa726;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
    }
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
    """
    Interactive Brokers tiered commission:
      • General:       $0.005/share, min $0.49
      • < 100 shares:  $0.49 flat
      • Sub-$1 stock, 7AM–8PM ET:  $0.005/share, min $0.49, max $7.95 (≤150K shares)
      • Sub-$1 stock, 4AM–7AM ET:  $0.005/share, min $0.49 (no cap)
    """
    shares = max(0.0, float(shares))
    if shares == 0:
        return 0.0

    # < 100 shares → flat $0.49
    if shares < 100:
        return 0.49

    raw = shares * 0.005

    if price < 1.00:
        hour = trade_time.hour if hasattr(trade_time, "hour") else 12
        if 7 <= hour < 20:          # 7 AM – 8 PM ET: cap at $7.95
            return max(0.49, min(raw, 7.95))
        else:                       # 4 AM – 7 AM ET: no cap
            return max(0.49, raw)

    return max(0.49, raw)


def commission_per_trade(shares: float, entry_price: float, exit_price: float,
                         entry_time, exit_time) -> float:
    """Total round-trip commission: entry leg + exit leg."""
    comm_entry = calc_commission(shares, entry_price, entry_time)
    comm_exit  = calc_commission(shares, exit_price,  exit_time)
    return comm_entry + comm_exit


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE DATA
# ─────────────────────────────────────────────────────────────────────────────
CANDLE_BASE_DIR = (
    "/home/local/USHERBROOKE/cabw2601/Documents/Personal/Market"
    "/claude_apps/backtester_api/backtest_dataset/full"
)
def _read_env_key(key: str, env_path: str = None) -> str:
    """Read a key from a .env file, falling back to os.getenv."""
    if env_path is None:
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


def candle_path_for(timeframe: str, base_dir: str) -> str:
    return f"{base_dir}/{timeframe}/full_dataset.parquet"


def _tf_to_polygon(timeframe: str) -> tuple[int, str]:
    """'15m' → (15, 'minute'),  '5m' → (5, 'minute'),  '1h' → (1, 'hour')"""
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1]), "minute"
    if tf.endswith("h"):
        return int(tf[:-1]), "hour"
    return 15, "minute"


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP, reset each calendar day."""
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]
    day = df["date"].dt.date
    return tpv.groupby(day, sort=False).cumsum() / df["volume"].groupby(day, sort=False).cumsum()


def _calc_sma9(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(9, min_periods=1).mean()


@st.cache_data(show_spinner="Cargando velas locales...")
def load_candles_local(timeframe: str, base_dir: str) -> pd.DataFrame:
    path = candle_path_for(timeframe, base_dir)
    cols = ["ticker","date","open","high","low","close","volume"]
    # load vwap/sma_9 if they exist in the file
    import pyarrow.parquet as pq
    schema_names = pq.read_schema(path).names
    for c in ["vwap","sma_9"]:
        if c in schema_names:
            cols.append(c)
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    if "vwap"  not in df.columns: df["vwap"]  = _calc_vwap(df.sort_values(["ticker","date"]))
    if "sma_9" not in df.columns: df["sma_9"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(9, min_periods=1).mean())
    return df


@st.cache_data(show_spinner="Descargando datos de Polygon...")
def load_candles_polygon(ticker: str, timeframe: str,
                         from_date: str, to_date: str,
                         api_key: str) -> pd.DataFrame:
    """Fetch intraday bars from Polygon and compute VWAP + SMA_9."""
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
        "t":"date","o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap_bar",
    })
    # Polygon timestamps are UTC milliseconds → convert to ET naive to match trade times
    df["date"] = (
        pd.to_datetime(df["date"], unit="ms")
          .dt.tz_localize("UTC")
          .dt.tz_convert("America/New_York")
          .dt.tz_localize(None)
    )
    df["ticker"] = ticker
    df = df.sort_values("date").reset_index(drop=True)
    df["vwap"]   = _calc_vwap(df)
    df["sma_9"]  = _calc_sma9(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────

def build_trade_chart(candles: pd.DataFrame, trade: pd.Series,
                      context_bars: int = 40, timeframe: str = "15m",
                      source_label: str = "local") -> None:
    """Render a TradingView Lightweight Chart for a single trade."""
    ticker    = trade["Ticker"]
    entry_dt  = pd.to_datetime(trade["Entry"])
    exit_dt   = pd.to_datetime(trade["Exit"])

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

    pdc       = float(trade["PDC"])
    entry_p   = float(trade["Entry$"])
    exit_p    = float(trade["Exit$"])
    sl        = float(trade["SL"])
    is_winner = bool(trade["Win"])
    tp_price  = float(trade["TP Price"]) if "TP Price" in trade.index and trade["TP Price"] else None

    def ts(dt): return int(pd.Timestamp(dt).timestamp())

    # ── Candlestick ───────────────────────────────────────────────────────────
    candle_data = [
        {"time": ts(r["date"]), "open": round(r["open"],4),
         "high": round(r["high"],4), "low": round(r["low"],4), "close": round(r["close"],4)}
        for _, r in window.iterrows()
    ]

    # ── Volume ────────────────────────────────────────────────────────────────
    volume_data = [
        {"time": ts(r["date"]), "value": round(r["volume"], 0),
         "color": "rgba(38,166,154,0.35)" if r["close"] >= r["open"] else "rgba(239,83,80,0.35)"}
        for _, r in window.iterrows()
    ]

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_data = [
        {"time": ts(r["date"]), "value": round(r["vwap"], 4)}
        for _, r in window.iterrows()
        if pd.notna(r.get("vwap"))
    ] if "vwap" in window.columns else []

    # ── SMA 9 ─────────────────────────────────────────────────────────────────
    sma9_data = [
        {"time": ts(r["date"]), "value": round(r["sma_9"], 4)}
        for _, r in window.iterrows()
        if pd.notna(r.get("sma_9"))
    ] if "sma_9" in window.columns else []

    # ── Markers ───────────────────────────────────────────────────────────────
    entry_ts = ts(entry_dt)
    exit_ts  = ts(exit_dt)
    markers  = [
        {"time": entry_ts, "position": "aboveBar", "color": "#ef5350",
         "shape": "arrowDown", "text": f"SHORT ${entry_p:.2f}", "size": 2},
        {"time": exit_ts,  "position": "belowBar",
         "color": "#26a69a" if is_winner else "#ef5350",
         "shape": "arrowUp", "text": f"EXIT ${exit_p:.2f}", "size": 2},
    ]

    # ── Flat level helper ─────────────────────────────────────────────────────
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

    # ── Chart options ─────────────────────────────────────────────────────────
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
            "type": "Candlestick",
            "data": candle_data,
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
        flat_line(pdc, "#ffca28", f"PDC {pdc:.2f}"),
        flat_line(sl,  "#ef5350", f"SL {sl:.2f}"),
    ]
    if tp_price:
        series.append(flat_line(tp_price, "#26a69a", f"TP {tp_price:.2f}"))
    if vwap_data:
        series.append({
            "type": "Line", "data": vwap_data,
            "options": {
                "color": "#8D6E63", "lineWidth": 2, "lineStyle": 0,
                "priceLineVisible": False, "lastValueVisible": True,
                "title": "VWAP", "crosshairMarkerVisible": False,
            },
        })
    if sma9_data:
        series.append({
            "type": "Line", "data": sma9_data,
            "options": {
                "color": "#ffffff", "lineWidth": 1, "lineStyle": 0,
                "priceLineVisible": False, "lastValueVisible": True,
                "title": "SMA9", "crosshairMarkerVisible": False,
            },
        })

    # ── Header ────────────────────────────────────────────────────────────────
    net_pnl    = float(trade["Net PnL($)"])
    pnl_color  = "#26a69a" if net_pnl >= 0 else "#ef5350"
    src_badge  = "🟢 Polygon" if source_label == "polygon" else "💾 Local"
    st.markdown(
        f"**{ticker}** &nbsp;·&nbsp; `{timeframe}` &nbsp;·&nbsp; {src_badge} &nbsp;·&nbsp; "
        f"{entry_dt.strftime('%Y-%m-%d')} &nbsp;·&nbsp; "
        f"Entry **${entry_p:.2f}** → Exit **${exit_p:.2f}** &nbsp;·&nbsp; "
        f"Net PnL <span style='color:{pnl_color}'>**${net_pnl:+.2f}**</span> &nbsp;·&nbsp; "
        f"PDC **${pdc:.2f}** &nbsp;·&nbsp; PDC→Entry **{trade['PDC→Entry%']:+.1f}%**",
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
        help="Porcentaje del equity total arriesgado en cada trade (= pérdida máxima si se activa el stop loss)",
    )
    st.session_state.risk_pct_val = risk_pct

    _risk_dollar = initial_capital * risk_pct / 100
    st.caption(f"≡ **${_risk_dollar:,.2f}** por trade  ·  `shares = riesgo$ / (entry$ × stop%)`")

    sizing_mode = st.radio(
        "Equity de referencia",
        ["Fijo (capital inicial)", "Compuesto (equity actual)"],
        index=0,
        help="Fijo: siempre calcula el riesgo sobre el capital inicial.\n"
             "Compuesto: recalcula el riesgo sobre la equity actual (crece con las ganancias).",
    )
    st.divider()

    # ── Commissions ──
    st.subheader("🏦 Comisiones (IBKR)")
    include_commissions = st.toggle("Incluir comisiones", value=True)
    if include_commissions:
        with st.expander("📋 Estructura de tarifas", expanded=False):
            st.markdown("""
**Base:** $0.005 / share · min $0.49
**< 100 shares:** $0.49 flat
**Sub-$1 (7AM–8PM ET):** $0.005/sh · min $0.49 · **max $7.95**
**Sub-$1 (4AM–7AM ET):** $0.005/sh · min $0.49
*(Round-trip: entrada + salida)*
            """)
    st.divider()

    st.subheader("🔧 Filtros (opcional)")
    date_filter = st.checkbox("Filtrar por fechas", value=False)
    if date_filter:
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("Desde")
        with col2:
            to_date = st.date_input("Hasta")
    else:
        from_date = to_date = None

    rvol_min = st.number_input("RVOL mínimo", min_value=0.0, max_value=20.0,
                                value=0.0, step=0.5)

    entry_vol_min = st.number_input(
        "Volumen mínimo en entrada (shares)",
        min_value=0, max_value=10_000_000,
        value=0, step=10_000, format="%d",
        help="Filtra trades cuyo volumen en la vela de entrada sea menor a este valor.",
    )
    st.divider()

    st.subheader("🎯 Filtros avanzados")
    st.caption("Se aplican después del cálculo y recalculan todas las métricas.")

    ep_filter = st.checkbox("Entry Price", value=False)
    if ep_filter:
        ep_col1, ep_col2 = st.columns(2)
        with ep_col1:
            ep_min = st.number_input("Min ($)", min_value=0.0, value=0.0,
                                     step=0.5, key="ep_min")
        with ep_col2:
            ep_max = st.number_input("Max ($)", min_value=0.0, value=9999.0,
                                     step=0.5, key="ep_max")
    else:
        ep_min, ep_max = 0.0, 9999.0

    et_filter = st.checkbox("Entry Time (rango de fechas)", value=False)
    if et_filter:
        et_col1, et_col2 = st.columns(2)
        with et_col1:
            et_from = st.date_input("Desde", key="et_from")
        with et_col2:
            et_to = st.date_input("Hasta", key="et_to")
    else:
        et_from = et_to = None

    pdc_filter = st.checkbox("PDC→Entry %", value=False)
    if pdc_filter:
        pdc_col1, pdc_col2 = st.columns(2)
        with pdc_col1:
            pdc_min = st.number_input("Min %", value=-100.0, step=5.0, key="pdc_min")
        with pdc_col2:
            pdc_max = st.number_input("Max %", value=9999.0, step=5.0, key="pdc_max")
    else:
        pdc_min, pdc_max = -9999.0, 9999.0

    ticker_filter = st.checkbox("Ticker", value=False)
    if ticker_filter:
        # Populate options from the already-uploaded file without full compute
        try:
            _tickers = sorted(pd.read_parquet(
                io.BytesIO(uploaded.getvalue()), columns=["ticker"]
            )["ticker"].unique().tolist())
        except Exception:
            _tickers = []
        selected_tickers = st.multiselect("Seleccionar tickers", _tickers)
    else:
        selected_tickers = []

    # ── Market Cap / Float filter ─────────────────────────────────────────────
    st.divider()
    st.subheader("🏦 Market Cap & Float")
    mkcap_filter = False; mkcap_min_m = 0.0;  mkcap_max_m = 99999.0
    float_filter  = False; float_min_m  = 0.0;  float_max_m  = 99999.0

    _fund_sel = st.selectbox(
        "Filtrar por",
        ["— Ninguno —", "Market Cap ($M)", "Float (M shares)"],
        key="fund_filter_sel",
    )
    if _fund_sel == "Market Cap ($M)":
        mkcap_filter = True
        _mc1, _mc2 = st.columns(2)
        mkcap_min_m = _mc1.number_input("Min ($M)", 0.0, value=0.0,   step=10.0, key="mkcap_min")
        mkcap_max_m = _mc2.number_input("Max ($M)", 0.0, value=500.0, step=10.0, key="mkcap_max")
    elif _fund_sel == "Float (M shares)":
        float_filter = True
        _fl1, _fl2 = st.columns(2)
        float_min_m = _fl1.number_input("Min (M)", 0.0, value=0.0,   step=1.0, key="float_min")
        float_max_m = _fl2.number_input("Max (M)", 0.0, value=100.0, step=1.0, key="float_max")

    any_advanced = ep_filter or et_filter or pdc_filter or ticker_filter or mkcap_filter or float_filter
    st.divider()

    st.subheader("🕯️ Gráfico de velas")
    data_source = st.radio("Fuente de datos", ["💾 Local", "🟢 Polygon"], horizontal=True)
    use_polygon = data_source == "🟢 Polygon"

    candle_base_dir = CANDLE_BASE_DIR   # always defined; only used when source=local
    if use_polygon:
        if POLYGON_API_KEY:
            st.success("API Key cargada desde .env", icon="🔑")
            polygon_key = POLYGON_API_KEY
        else:
            polygon_key = st.text_input("Polygon API Key", type="password",
                                        help="No se encontró MASSIVE_API_KEY en el .env")
        pre_days = st.slider("Días previos al trade (Polygon)", 1, 10, 5)
    else:
        polygon_key = ""
        pre_days    = 0
        candle_base_dir = st.text_input(
            "Directorio base datasets",
            value=CANDLE_BASE_DIR,
            help="Carpeta con subcarpetas 5m/ y 15m/ con full_dataset.parquet",
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
                     rvol_min: float,
                     entry_vol_min: int) -> pd.DataFrame:

    df = pd.read_parquet(io.BytesIO(file_bytes))
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Coerce numeric fields that older parquets may have stored as object dtype
    for _col in ("rvol_daily", "previous_day_close"):
        if _col in df.columns and df[_col].dtype == object:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    if from_date:
        df = df[df["entry_time"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["entry_time"] < pd.Timestamp(to_date) + pd.Timedelta(days=1)]
    if rvol_min > 0:
        df = df[df["rvol_daily"] >= rvol_min]
    if entry_vol_min > 0:
        df = df[df["entry_volume"] >= entry_vol_min]
    df = df.reset_index(drop=True)

    # Stop % per trade (from actual SL stored in data)
    df["stop_pct_row"] = (
        (df["stop_loss_price"] - df["entry_price"]).abs() / df["entry_price"]
    )

    equity = initial_capital
    gross_pnls, net_pnls, comm_list, shares_list = [], [], [], []

    for _, row in df.iterrows():
        cap  = equity if compound else initial_capital
        r_d  = cap * (risk_pct / 100)
        stop = row["stop_pct_row"]
        shares = r_d / (row["entry_price"] * stop) if stop > 0 else 1.0

        gross = row["pnl"] * shares

        comm = (
            commission_per_trade(
                shares,
                row["entry_price"], row["exit_price"],
                row["entry_time"],  row["exit_time"],
            )
            if with_commissions else 0.0
        )

        net = gross - comm

        gross_pnls.append(gross)
        net_pnls.append(net)
        comm_list.append(comm)
        shares_list.append(shares)

        equity += net   # compound uses net equity

    df["shares"]        = shares_list
    df["gross_pnl"]     = gross_pnls
    df["commission"]    = comm_list
    df["scaled_pnl"]    = net_pnls          # "scaled_pnl" = net throughout

    # Gross equity (no commissions)
    df["cum_gross"]         = df["gross_pnl"].cumsum()
    df["equity_gross"]      = initial_capital + df["cum_gross"]

    # Net equity (with commissions)
    df["cum_pnl"]           = df["scaled_pnl"].cumsum()
    df["equity_curve"]      = initial_capital + df["cum_pnl"]

    # Drawdown on net equity
    roll_max                = df["equity_curve"].cummax()
    df["drawdown_abs"]      = df["equity_curve"] - roll_max
    df["drawdown_pct"]      = df["drawdown_abs"] / roll_max * 100

    # Derived
    df["winner"]  = df["pnl"] > 0   # based on raw price action, not commission/sizing
    df["ym"]      = df["entry_time"].dt.to_period("M")
    df["year"]    = df["entry_time"].dt.year
    df["hour"]    = df["entry_time"].dt.hour
    df["weekday"] = df["entry_time"].dt.day_name()
    df["ret_pct"] = df["Return"] * 100
    df["hold_h"]  = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600

    # Commission as % of entry notional
    df["comm_pct_notional"] = df["commission"] / (df["entry_price"] * df["shares"]) * 100

    # PDC → Entry %
    df["pct_from_pdc"] = ((df["entry_price"] - df["previous_day_close"])
                          / df["previous_day_close"] * 100).round(2)

    return df


@st.cache_data(show_spinner="Descargando SPY de Polygon...")
def load_spy_bnh(from_date: str, to_date: str,
                 initial_capital: float, api_key: str) -> pd.DataFrame:
    """Fetch SPY daily adjusted closes from Polygon and build a buy-and-hold equity curve."""
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
    spy["date"] = pd.to_datetime(spy["date"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["bnh_equity"] = initial_capital * (spy["close"] / spy["close"].iloc[0])
    spy["bnh_ret"]    = (spy["bnh_equity"] / initial_capital - 1) * 100
    return spy


def recompute_equity(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Recompute cumulative equity and drawdown after filtering."""
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
    """Running streak: +N = N wins consecutivos, -N = N losses consecutivos."""
    streak = []
    cur = 0
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
    st.title("📈 Backtest Dashboard")
    st.info("👈 Carga un archivo **.parquet** de trades en el panel lateral para comenzar.")
    st.stop()

with st.spinner("Calculando..."):
    compound = sizing_mode.startswith("Compuesto")
    df = load_and_compute(
        uploaded.read(), initial_capital, risk_pct, compound,
        include_commissions,
        from_date if date_filter else None,
        to_date   if date_filter else None,
        rvol_min,
        entry_vol_min,
    )

if df.empty:
    st.warning("No hay trades con los filtros seleccionados.")
    st.stop()

# Snapshot sin filtros avanzados — usado en tab10 para ver el universo completo
df_full = df.copy()

# ── Advanced post-filters (recompute equity after) ────────────────────────────
if any_advanced:
    mask = pd.Series(True, index=df.index)
    if ep_filter:
        mask &= df["entry_price"].between(ep_min, ep_max)
    if et_filter and et_from and et_to:
        mask &= df["entry_time"].between(pd.Timestamp(et_from), pd.Timestamp(et_to) + pd.Timedelta(days=1))
    if pdc_filter:
        mask &= df["pct_from_pdc"].between(pdc_min, pdc_max)
    if ticker_filter and selected_tickers:
        mask &= df["ticker"].isin(selected_tickers)
    if mkcap_filter and "market_cap" in df.columns:
        mask &= df["market_cap"].fillna(0).between(mkcap_min_m * 1e6, mkcap_max_m * 1e6)
    if float_filter and "float" in df.columns:
        mask &= df["float"].fillna(0).between(float_min_m * 1e6, float_max_m * 1e6)
    df = recompute_equity(df[mask], initial_capital)
    if df.empty:
        st.warning("No hay trades con los filtros avanzados seleccionados.")
        st.stop()

# ── Global KPIs ──
total       = len(df)
wins        = int(df["winner"].sum())
wr          = wins / total * 100
gross_pnl   = df["gross_pnl"].sum()
total_comm  = df["commission"].sum()
net_pnl     = df["scaled_pnl"].sum()
final_eq    = initial_capital + net_pnl
total_ret   = net_pnl / initial_capital * 100
pf_val      = pf(df["scaled_pnl"])
mdd_abs     = df["drawdown_abs"].min()
mdd_pct     = df["drawdown_pct"].min()
avg_win     = df.loc[df["winner"],  "scaled_pnl"].mean()
avg_loss    = df.loc[~df["winner"], "scaled_pnl"].mean()
sh          = (df["ret_pct"].mean() / df["ret_pct"].std()) * np.sqrt(252) if df["ret_pct"].std() else 0
avg_hold    = df["hold_h"].mean()
comm_drag        = total_comm / abs(gross_pnl) * 100 if gross_pnl != 0 else 0
avg_comm         = df["commission"].mean()
expectancy_gross = df["gross_pnl"].mean()
expectancy_net   = df["scaled_pnl"].mean()
expectancy_comm  = df["commission"].mean()          # avg comm cost per trade

# ── Streaks ───────────────────────────────────────────────────────────────────
df["streak"]      = compute_streaks(df["winner"])
max_win_streak    = int(df["streak"].max())
max_loss_streak   = int(df["streak"].min())          # negative
current_streak    = int(df["streak"].iloc[-1])

# ── SPY Buy & Hold (fetched once, used in KPIs + chart) ───────────────────────
spy_df        = pd.DataFrame()
spy_bnh_ret   = None
spy_bnh_eq    = None
spy_alpha     = None
_spy_key      = POLYGON_API_KEY or (polygon_key if use_polygon else "")
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

# Extra SPY risk metrics derived from the B&H equity curve
spy_sharpe_bnh = None
spy_mdd_bnh    = None
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
st.title(f"📈 {strategy_name}")
st.caption(
    f"**{df['entry_time'].min().strftime('%b %Y')} → {df['entry_time'].max().strftime('%b %Y')}**  |  "
    f"Capital: **${initial_capital:,}**  |  Risk/trade: **{risk_pct}%**  |  "
    f"Modo: **{'Compuesto' if compound else 'Fijo'}**  |  "
    f"Comisiones: **{'ON ✅' if include_commissions else 'OFF ❌'}**  |  "
    f"RVOL mín: **{rvol_min}x**  |  "
    f"Entry Vol mín: **{entry_vol_min:,}**"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── KPI Tables — Estrategia vs S&P 500 ───────────────────────────────────────
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
    _strat_rows += [
        ("Expectancy",        fmt_dollar(expectancy_gross)),
    ]
_strat_rows += [
    ("Avg Win",           fmt_dollar(avg_win)),
    ("Avg Loss",          fmt_dollar(avg_loss)),
    ("Win/Loss Ratio",    f"{_wl:.2f}"),
    ("Avg Hold",          f"{avg_hold:.1f}h"),
    ("Avg Shares/Trade",  f"{df['shares'].mean():,.0f}"),
    ("Max Win Streak",    f"{max_win_streak}"),
    ("Max Loss Streak",   f"{abs(max_loss_streak)}"),
    ("Racha Actual",      _cur_label),
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Equity & Drawdown",
    "💸 Comisiones",
    "📅 Por Período",
    "📐 Distribución",
    "🔬 Trade Analysis",
    "🏷️ Por Ticker",
    "📋 Trades",
    "🎲 Montecarlo",
    "⚖️ Comparar",
    "🏦 Market Cap & Float",
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
        # SPY buy & hold
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
        fig.add_hline(y=initial_capital, line_dash="dot",
                      line_color=SUB, opacity=0.5, row=1, col=1)
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
        st.metric("Avg Win",          fmt_dollar(avg_win))
        st.metric("Avg Loss",         fmt_dollar(avg_loss))
        wl = abs(avg_win / avg_loss) if avg_loss else 0
        st.metric("Win/Loss Ratio",   f"{wl:.2f}")
        if include_commissions:
            st.metric("Expect. Bruta",  fmt_dollar(expectancy_gross))
            st.metric("Avg Comisión",   f"-${expectancy_comm:.2f}",
                      delta=f"-{comm_drag:.1f}% drag", delta_color="inverse")
            st.metric("Expect. Neta",   fmt_dollar(expectancy_net),
                      delta=fmt_dollar(expectancy_net - expectancy_gross))
        else:
            st.metric("Expectancy",     fmt_dollar(expectancy_gross))

# ── TAB 2: Commissions ────────────────────────────────────────────────────────
with tab2:
    if not include_commissions:
        st.info("Activa **Incluir comisiones** en el sidebar para ver este análisis.")
    else:
        # Rolling comm per trade
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
            # Gross vs Net cumulative
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
                **PLOTLY_LAYOUT, height=320,
                title="Gross vs Net Equity",
                yaxis2=dict(overlaying="y", side="right",
                            title="Comm. drag ($)", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(color=ORANGE)),
            )
            st.plotly_chart(fig_c2, use_container_width=True)

        # Commission by price tier
        col_c, col_d = st.columns(2)

        with col_c:
            df["price_tier"] = pd.cut(
                df["entry_price"],
                bins=[0, 1, 5, 10, 25, 50, 100, 999],
                labels=["<$1", "$1-5", "$5-10", "$10-25", "$25-50", "$50-100", ">$100"],
            )
            tier_agg = df.groupby("price_tier", observed=True).agg(
                total_comm=("commission", "sum"),
                avg_comm  =("commission", "mean"),
                trades    =("commission", "count"),
            ).reset_index()
            fig_c3 = go.Figure()
            fig_c3.add_trace(go.Bar(
                x=tier_agg["price_tier"].astype(str),
                y=tier_agg["total_comm"],
                marker_color=ORANGE, name="Total comm.",
                text=[f"${v:,.0f}" for v in tier_agg["total_comm"]],
                textposition="outside",
            ))
            fig_c3.update_layout(**PLOTLY_LAYOUT, height=320,
                                 title="Comisión Total por Tier de Precio",
                                 yaxis_title="$ Total Comisiones")
            st.plotly_chart(fig_c3, use_container_width=True)

        with col_d:
            fig_c4 = go.Figure()
            fig_c4.add_trace(go.Scatter(
                x=df["entry_price"].clip(0, 50),
                y=df["commission"],
                mode="markers",
                marker=dict(color=ORANGE, size=3, opacity=0.3),
                name="Comm/trade",
            ))
            fig_c4.update_layout(**PLOTLY_LAYOUT, height=320,
                                 title="Comisión por Trade vs Precio de Entrada",
                                 xaxis_title="Entry Price ($, capped $50)",
                                 yaxis_title="Commission ($)")
            st.plotly_chart(fig_c4, use_container_width=True)

        # Monthly commission
        monthly_comm = df.groupby("ym").agg(
            total_comm=("commission", "sum"),
            avg_comm  =("commission", "mean"),
            gross_pnl =("gross_pnl",  "sum"),
            net_pnl   =("scaled_pnl", "sum"),
        ).reset_index()
        monthly_comm["ym_str"] = monthly_comm["ym"].astype(str)
        monthly_comm["drag_pct"] = (
            monthly_comm["total_comm"] / monthly_comm["gross_pnl"].abs() * 100
        ).clip(-500, 500)

        fig_c5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c5.add_trace(go.Bar(
            x=monthly_comm["ym_str"], y=monthly_comm["gross_pnl"],
            name="Gross PnL", marker_color=[GREEN if v >= 0 else RED for v in monthly_comm["gross_pnl"]],
            opacity=0.5,
        ), secondary_y=False)
        fig_c5.add_trace(go.Bar(
            x=monthly_comm["ym_str"], y=monthly_comm["net_pnl"],
            name="Net PnL", marker_color=[GREEN if v >= 0 else RED for v in monthly_comm["net_pnl"]],
        ), secondary_y=False)
        fig_c5.add_trace(go.Scatter(
            x=monthly_comm["ym_str"], y=monthly_comm["total_comm"],
            mode="lines+markers", name="Comisión ($)",
            line=dict(color=ORANGE, width=2), marker=dict(size=5),
        ), secondary_y=True)
        fig_c5.update_layout(**PLOTLY_LAYOUT, height=340, barmode="overlay",
                             title="Gross vs Net PnL por Mes + Comisión")
        fig_c5.update_yaxes(title_text="PnL ($)", secondary_y=False, gridcolor="#2e3547")
        fig_c5.update_yaxes(title_text="Comisión ($)", secondary_y=True,
                            gridcolor="rgba(0,0,0,0)", tickfont=dict(color=ORANGE))
        fig_c5.update_xaxes(tickangle=45, tickfont=dict(size=8))
        st.plotly_chart(fig_c5, use_container_width=True)

        # Summary table
        st.markdown("#### Resumen mensual de comisiones")
        mc_disp = monthly_comm[["ym_str","trades" if "trades" in monthly_comm.columns else "total_comm",
                                 "gross_pnl","total_comm","net_pnl","drag_pct"]].copy()
        # recompute trades
        tr_by_m = df.groupby("ym")["commission"].count().reset_index().rename(columns={"commission":"trades"})
        mc_disp = monthly_comm.merge(tr_by_m, on="ym")
        mc_disp = mc_disp[["ym_str","trades","gross_pnl","total_comm","net_pnl","drag_pct"]].copy()
        mc_disp["gross_pnl"]  = mc_disp["gross_pnl"].map(fmt_dollar)
        mc_disp["total_comm"] = mc_disp["total_comm"].map(lambda x: f"${x:,.2f}")
        mc_disp["net_pnl"]    = mc_disp["net_pnl"].map(fmt_dollar)
        mc_disp["drag_pct"]   = mc_disp["drag_pct"].map(lambda x: f"{x:.1f}%")
        mc_disp.columns = ["Mes","Trades","Gross PnL","Comisiones","Net PnL","Comm Drag%"]
        st.dataframe(mc_disp, use_container_width=True, hide_index=True, height=380)

        # ── Break-even analysis ───────────────────────────────────────────────
        st.divider()
        st.subheader("🎯 Análisis Break-Even de Comisiones")
        st.caption(
            "El break-even de comisión es el retorno mínimo que necesita un trade "
            "para cubrir el coste de entrada+salida. Depende del precio de entrada y "
            "del número de shares."
        )

        # Per-trade break-even return %
        # shares >= 100  → comm = 2 × shares × $0.005  → BE = $0.01 / entry_price
        # shares < 100   → comm = $0.98 flat             → BE = $0.98 / gross_notional
        df["notional"]   = df["shares"] * df["entry_price"]
        df["be_ret_pct"] = np.where(
            df["shares"] >= 100,
            0.01 / df["entry_price"] * 100,                  # %: $0.01/share ÷ entry
            df["commission"] / df["notional"] * 100,          # flat fee ÷ notional
        )
        df["covers_comm"] = df["gross_pnl"] >= df["commission"]
        df["ret_pct_abs"] = df["Return"].abs() * 100

        # ── KPIs ─────────────────────────────────────────────────────────────
        be_k1, be_k2, be_k3, be_k4 = st.columns(4)
        covers    = df["covers_comm"].sum()
        not_cover = (~df["covers_comm"]).sum()
        be_k1.metric("Trades que cubren comisión",  f"{covers:,}",
                     delta=f"{covers/total*100:.1f}%")
        be_k2.metric("Trades que NO cubren comisión", f"{not_cover:,}",
                     delta=f"-{not_cover/total*100:.1f}%", delta_color="inverse")
        be_k3.metric("Avg Break-Even Return",
                     f"{df['be_ret_pct'].mean():.2f}%")
        be_k4.metric("Avg Return actual (ganadores)",
                     f"{df.loc[df['winner'],'ret_pct_abs'].mean():.2f}%")

        # ── Break-even curve + scatter ────────────────────────────────────────
        be_col1, be_col2 = st.columns(2)

        with be_col1:
            # Scatter: entry_price vs Return%, colored by covers/not
            sample_be = df.sample(min(3000, len(df)), random_state=42)
            fig_be1 = go.Figure()
            for covers_val, color, name in [
                (True,  GREEN, "Cubre comisión"),
                (False, RED,   "No cubre comisión"),
            ]:
                s = sample_be[sample_be["covers_comm"] == covers_val]
                fig_be1.add_trace(go.Scatter(
                    x=s["entry_price"].clip(0, 30),
                    y=(s["Return"] * 100).clip(-60, 60),
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.35),
                    name=name,
                ))
            # Break-even curve
            prices_curve = np.linspace(0.05, 30, 300)
            be_curve = np.where(
                prices_curve < 2.0,          # shares >= 100 zone (approx)
                0.01 / prices_curve * 100,
                0.98 / (prices_curve * (risk_pct / 100 * initial_capital
                                        / (prices_curve * 0.5))) * 100,
            )
            # Simplified: just use $0.01/price for all (good approximation)
            be_curve_simple = 0.01 / prices_curve * 100
            fig_be1.add_trace(go.Scatter(
                x=prices_curve, y=be_curve_simple,
                mode="lines", name="Break-Even (comisión)",
                line=dict(color=YELLOW, width=2, dash="dash"),
            ))
            fig_be1.add_hline(y=0, line_color=SUB, line_dash="dot", line_width=0.7)
            fig_be1.update_layout(
                **PLOTLY_LAYOUT, height=380,
                title="Return % vs Entry Price · ¿Cubre comisión?",
                xaxis_title="Entry Price ($, capped $30)",
                yaxis_title="Return % (capped ±60)",
            )
            st.plotly_chart(fig_be1, use_container_width=True)

        with be_col2:
            # Bar: by price tier — % trades covering commission
            price_bins  = [0, 0.5, 1, 2, 5, 10, 25, 999]
            price_lbls  = ["<$0.50","$0.50-1","$1-2","$2-5","$5-10","$10-25",">$25"]
            df["ptier"] = pd.cut(df["entry_price"], bins=price_bins, labels=price_lbls)
            tier_be = df.groupby("ptier", observed=True).agg(
                trades        =("covers_comm","count"),
                covers_pct    =("covers_comm","mean"),
                avg_be        =("be_ret_pct", "mean"),
                avg_gross     =("gross_pnl",  "mean"),
                avg_comm      =("commission", "mean"),
                net_pnl_total =("net_pnl",    "sum") if "net_pnl" in df.columns
                                else ("scaled_pnl","sum"),
            ).reset_index()
            tier_be["net_pnl_total"] = df.groupby("ptier", observed=True)["scaled_pnl"].sum().values

            fig_be2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_be2.add_trace(go.Bar(
                x=tier_be["ptier"].astype(str),
                y=tier_be["covers_pct"] * 100,
                marker_color=[GREEN if v >= 50 else RED for v in tier_be["covers_pct"]],
                name="% Trades que cubren comm.",
                text=[f"{v*100:.0f}%" for v in tier_be["covers_pct"]],
                textposition="outside",
            ), secondary_y=False)
            fig_be2.add_trace(go.Scatter(
                x=tier_be["ptier"].astype(str),
                y=tier_be["avg_be"],
                mode="lines+markers", name="BE mínimo (%)",
                line=dict(color=YELLOW, width=2), marker=dict(size=7),
            ), secondary_y=True)
            fig_be2.update_layout(
                **PLOTLY_LAYOUT, height=380,
                title="% Trades que cubren comisión por tier de precio",
            )
            fig_be2.update_yaxes(title_text="% Trades cubriendo comm.",
                                  secondary_y=False, gridcolor="#2e3547", range=[0, 110])
            fig_be2.update_yaxes(title_text="Break-Even mínimo (%)",
                                  secondary_y=True, gridcolor="rgba(0,0,0,0)",
                                  tickfont=dict(color=YELLOW))
            st.plotly_chart(fig_be2, use_container_width=True)

        # ── Net PnL by price tier after commissions ───────────────────────────
        fig_be3 = go.Figure()
        fig_be3.add_trace(go.Bar(
            x=tier_be["ptier"].astype(str),
            y=tier_be["net_pnl_total"],
            marker_color=[GREEN if v >= 0 else RED for v in tier_be["net_pnl_total"]],
            text=[fmt_dollar(v) for v in tier_be["net_pnl_total"]],
            textposition="outside",
            name="Net PnL total",
        ))
        fig_be3.add_hline(y=0, line_color=SUB, line_dash="dot")
        fig_be3.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title="Net PnL total por tier de precio (después de comisiones)",
            yaxis_title="Net PnL ($)",
        )
        st.plotly_chart(fig_be3, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("#### Tabla resumen por tier de precio")
        tier_tbl = tier_be.copy()
        tier_tbl["covers_pct"]    = tier_tbl["covers_pct"].map(lambda x: f"{x*100:.1f}%")
        tier_tbl["avg_be"]        = tier_tbl["avg_be"].map(lambda x: f"{x:.2f}%")
        tier_tbl["avg_gross"]     = tier_tbl["avg_gross"].map(fmt_dollar)
        tier_tbl["avg_comm"]      = tier_tbl["avg_comm"].map(lambda x: f"${x:.2f}")
        tier_tbl["net_pnl_total"] = tier_tbl["net_pnl_total"].map(fmt_dollar)
        tier_tbl.columns = ["Tier Precio","Trades","Cubre Comm%",
                            "BE Mínimo%","Avg Gross","Avg Comm","Net PnL Total"]
        st.dataframe(tier_tbl, use_container_width=True, hide_index=True)

        # ── Recommendation ────────────────────────────────────────────────────
        profitable_tiers = tier_be[tier_be["net_pnl_total"] > 0]["ptier"].astype(str).tolist()
        if profitable_tiers:
            st.success(
                f"✅ **Tiers rentables después de comisiones:** {', '.join(profitable_tiers)}  \n"
                f"Considera filtrar **Entry Price mínimo** en los Filtros Avanzados del sidebar "
                f"para excluir los tiers con Net PnL negativo.",
                icon="💡",
            )
        else:
            st.warning("⚠️ Ningún tier de precio es rentable después de comisiones con la "
                       "configuración actual. Considera aumentar el capital o el % de riesgo.")


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
        labels = ["Año","Trades","WR%","Gross PnL","Comisiones","Net PnL","Ret%","PF","MaxDD"]
        st.dataframe(yd[cols].rename(columns=dict(zip(cols,labels))),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Tabla mensual")
    md = monthly.copy()
    md["net_pnl"]   = md["net_pnl"].map(fmt_dollar)
    md["gross_pnl"] = md["gross_pnl"].map(fmt_dollar)
    md["wr_pct"]    = md["wr_pct"].map(lambda x: f"{x:.1f}%")
    md["pf_m"]      = md["pf_m"].map(lambda x: f"{x:.2f}")
    md["avg_ret"]   = md["avg_ret"].map(lambda x: f"{x:.2f}%")
    show_cols = ["ym_str","trades","wr_pct","gross_pnl","net_pnl","avg_ret","pf_m"]
    col_labels = ["Mes","Trades","WR%","Gross PnL","Net PnL","Avg Ret","PF"]
    st.dataframe(md[show_cols].rename(columns=dict(zip(show_cols,col_labels))),
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
            net_pnl  =("scaled_pnl","sum"),
            win_rate =("winner",    "mean"),
            trades   =("scaled_pnl","count"),
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
        rvol_cap = df["rvol_daily"].clip(0,15)
        rv_bins  = pd.cut(rvol_cap, bins=np.linspace(0,15,16))
        rv_agg   = df.groupby(rv_bins, observed=False).agg(
            net_pnl  =("scaled_pnl","sum"),
            win_rate =("winner",    "mean"),
        )
        bin_centers = [round((b.left+b.right)/2, 1) for b in rv_agg.index]
        fig10 = make_subplots(specs=[[{"secondary_y": True}]])
        fig10.add_trace(go.Bar(
            x=bin_centers, y=rv_agg["net_pnl"],
            marker_color=[GREEN if v >= 0 else RED for v in rv_agg["net_pnl"]],
            name="Net PnL",
        ), secondary_y=False)
        fig10.add_trace(go.Scatter(
            x=bin_centers, y=rv_agg["win_rate"]*100,
            mode="lines+markers", line=dict(color=YELLOW, width=2), name="Win Rate %",
        ), secondary_y=True)
        fig10.update_layout(**PLOTLY_LAYOUT, height=340, title="RVOL Daily vs PnL")
        fig10.update_xaxes(title_text="RVOL Daily (capped 15x)")
        fig10.update_yaxes(title_text="Net PnL ($)", secondary_y=False, gridcolor="#2e3547")
        fig10.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0,100],
                           gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig10, use_container_width=True)

    with col4:
        fig11 = go.Figure()
        fig11.add_trace(go.Histogram(
            x=df["hold_h"].clip(0,12), nbinsx=30,
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
        text=np.round(heatmap_data.values,0),
        texttemplate="%{text:.0f}",
        textfont=dict(size=8),
    ))
    fig12.update_layout(**PLOTLY_LAYOUT, height=260,
                        title="Heatmap Net PnL: Día × Hora")
    st.plotly_chart(fig12, use_container_width=True)

    # ── Streak analysis ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔗 Rachas (Streaks)")

    str_col1, str_col2 = st.columns(2)

    with str_col1:
        # Timeline de la racha corriente a lo largo del tiempo
        fig_str1 = go.Figure()
        fig_str1.add_trace(go.Bar(
            x=df["entry_time"],
            y=df["streak"],
            marker_color=[GREEN if v > 0 else RED for v in df["streak"]],
            name="Racha",
        ))
        fig_str1.add_hline(y=0, line_color=SUB, line_width=0.8)
        # Mark max win and max loss
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
        # Distribución de la longitud de rachas ganadoras y perdedoras
        # Extraer todas las rachas como grupos
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
        _bins = list(range(1, _max_streak_len + 2))

        fig_str2 = go.Figure()
        fig_str2.add_trace(go.Histogram(
            x=_win_runs, xbins=dict(start=0.5, end=_max_streak_len + 0.5, size=1),
            marker_color=GREEN, opacity=0.75, name="Rachas ganadoras",
        ))
        fig_str2.add_trace(go.Histogram(
            x=_loss_runs, xbins=dict(start=0.5, end=_max_streak_len + 0.5, size=1),
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

    # Tabla resumen de rachas
    _streak_summary = pd.DataFrame({
        "Tipo":              ["Ganadoras 🟢", "Perdedoras 🔴"],
        "Racha Máxima":      [f"{max_win_streak} trades", f"{abs(max_loss_streak)} trades"],
        "Nº de Rachas":      [len(_win_runs),  len(_loss_runs)],
        "Avg Longitud":      [f"{np.mean(_win_runs):.1f}"  if _win_runs  else "—",
                              f"{np.mean(_loss_runs):.1f}" if _loss_runs else "—"],
        "Rachas de 1":       [_win_runs.count(1),  _loss_runs.count(1)],
        "Rachas de 2+":      [sum(1 for r in _win_runs  if r >= 2),
                              sum(1 for r in _loss_runs if r >= 2)],
        "Rachas de 5+":      [sum(1 for r in _win_runs  if r >= 5),
                              sum(1 for r in _loss_runs if r >= 5)],
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
        avg_rvol  =("rvol_daily","mean"),
    ).reset_index().sort_values("net_pnl", ascending=False)
    by_ticker["wr_pct"] = by_ticker["win_rate"] * 100
    by_ticker["pf_t"]   = by_ticker["pf_t"].clip(0,20)

    col_top, col_bot = st.columns(2)
    with col_top:
        top15 = by_ticker.head(15)
        fig13 = go.Figure(go.Bar(
            x=top15["ticker"], y=top15["net_pnl"], marker_color=GREEN,
            text=[fmt_dollar(v) for v in top15["net_pnl"]], textposition="outside",
        ))
        fig13.update_layout(**PLOTLY_LAYOUT, height=360, title="Top 15 Tickers — Net PnL")
        st.plotly_chart(fig13, use_container_width=True)

    with col_bot:
        bot15 = by_ticker.tail(15).sort_values("net_pnl")
        fig14 = go.Figure(go.Bar(
            x=bot15["ticker"], y=bot15["net_pnl"], marker_color=RED,
            text=[fmt_dollar(v) for v in bot15["net_pnl"]], textposition="outside",
        ))
        fig14.update_layout(**PLOTLY_LAYOUT, height=360, title="Peores 15 Tickers — Net PnL")
        st.plotly_chart(fig14, use_container_width=True)

    fig15 = px.scatter(
        by_ticker[by_ticker["trades"] >= 2],
        x="trades", y="net_pnl", color="wr_pct",
        size="trades", hover_data=["ticker","avg_ret","pf_t","commission"],
        color_continuous_scale=["red","yellow","green"], range_color=[30,80],
        title="Tickers: # Trades vs Net PnL (color = Win Rate %)",
        labels={"net_pnl":"Net PnL ($)","trades":"# Trades","wr_pct":"WR%"},
    )
    fig15.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig15, use_container_width=True)

    st.markdown("#### Tabla por ticker")
    td = by_ticker.drop(columns=["win_rate"]).copy()
    td = td.rename(columns={
        "ticker":"Ticker","trades":"Trades","net_pnl":"Net PnL","gross_pnl":"Gross PnL",
        "commission":"Comisiones","wr_pct":"WR%","avg_ret":"Avg Ret","pf_t":"PF","avg_rvol":"Avg RVOL",
    })
    td["Net PnL"]    = td["Net PnL"].map(fmt_dollar)
    td["Gross PnL"]  = td["Gross PnL"].map(fmt_dollar)
    td["Comisiones"] = td["Comisiones"].map(lambda x: f"${x:,.2f}")
    td["WR%"]        = td["WR%"].map(lambda x: f"{x:.1f}%")
    td["Avg Ret"]    = td["Avg Ret"].map(lambda x: f"{x:.2f}%")
    td["PF"]         = td["PF"].map(lambda x: f"{x:.2f}")
    td["Avg RVOL"]   = td["Avg RVOL"].map(lambda x: f"{x:.2f}x")
    st.dataframe(
        td[["Ticker","Trades","WR%","Gross PnL","Comisiones","Net PnL","Avg Ret","PF","Avg RVOL"]],
        use_container_width=True, hide_index=True, height=400,
    )


# ── TAB 7: Trades raw ─────────────────────────────────────────────────────────
with tab7:
    # ── Columns from parquet (read as-is) ────────────────────────────────────
    parquet_cols = [
        "entry_time","exit_time","ticker","timeframe","type",
        "previous_day_close","entry_price","exit_price",
        "stop_loss_price","take_profit_price","risk_reward_ratio",
        "pnl","Return","MAE","mae_pct","MFE","mfe_pct",
        "rvol_daily","volume","entry_volume",
    ]
    if "market_cap" in df.columns:
        parquet_cols.append("market_cap")
    if "float" in df.columns:
        parquet_cols.append("float")
    # ── Computed by dashboard ─────────────────────────────────────────────────
    computed_cols = ["pct_from_pdc","shares","gross_pnl","commission","scaled_pnl","winner","hold_h"]

    cols_show = parquet_cols + computed_cols
    disp = df[cols_show].copy()

    # Round parquet price/ratio fields
    for c in ["previous_day_close","entry_price","exit_price","stop_loss_price",
              "take_profit_price","risk_reward_ratio","pnl","Return","MAE","MFE"]:
        disp[c] = disp[c].round(4)
    for c in ["mae_pct","mfe_pct","rvol_daily"]:
        disp[c] = disp[c].round(2)
    for c in ["volume","entry_volume"]:
        disp[c] = disp[c].round(0).astype("Int64")
    # Round computed fields
    for c in ["shares"]:
        disp[c] = disp[c].round(0).astype("Int64")
    for c in ["gross_pnl","commission","scaled_pnl"]:
        disp[c] = disp[c].round(2)
    for c in ["pct_from_pdc","hold_h"]:
        disp[c] = disp[c].round(2)

    disp = disp.rename(columns={
        # parquet
        "entry_time":"Entry","exit_time":"Exit","ticker":"Ticker","timeframe":"TF","type":"Type",
        "previous_day_close":"PDC","entry_price":"Entry$","exit_price":"Exit$",
        "stop_loss_price":"SL","take_profit_price":"TP Price","risk_reward_ratio":"R/R",
        "pnl":"PnL/sh","Return":"Ret%",
        "MAE":"MAE$","mae_pct":"MAE%","MFE":"MFE$","mfe_pct":"MFE%",
        "rvol_daily":"RVOL","volume":"Vol","entry_volume":"Entry Vol",
        # computed
        "pct_from_pdc":"PDC→Entry%","shares":"Shares",
        "gross_pnl":"Gross($)","commission":"Comm($)","scaled_pnl":"Net PnL($)",
        "winner":"Win","hold_h":"Hold(h)",
        "market_cap":"MktCap($M)","float":"Float(M sh)",
    })
    # Ret% is stored as decimal in parquet → display as %
    disp["Ret%"] = (disp["Ret%"] * 100).round(2)
    if "MktCap($M)" in disp.columns:
        disp["MktCap($M)"] = (disp["MktCap($M)"] / 1e6).round(1)
    if "Float(M sh)" in disp.columns:
        disp["Float(M sh)"] = (disp["Float(M sh)"] / 1e6).round(2)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_only = st.selectbox("Mostrar", ["Todos","Solo Winners","Solo Losers"])
    with col_f2:
        sort_col = st.selectbox("Ordenar por",
                                ["Entry","Net PnL($)","Ret%","RVOL","Comm($)","PDC→Entry%","Entry$"])
    with col_f3:
        sort_asc = st.radio("Orden", ["↓ Desc","↑ Asc"], horizontal=True) == "↑ Asc"

    if show_only == "Solo Winners":
        disp = disp[disp["Win"]]
    elif show_only == "Solo Losers":
        disp = disp[~disp["Win"]]

    disp = disp.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.caption("👆 Haz clic en una fila para ver el gráfico de velas")

    selection = st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
    )

    csv = disp.to_csv(index=False).encode()
    st.download_button("⬇️ Descargar CSV", csv, "trades_filtered.csv", "text/csv")

    # ── Candlestick chart for selected trade ──────────────────────────────────
    rows = selection.selection.rows if selection and selection.selection else []
    if rows:
        trade = disp.iloc[rows[0]]
        tf    = str(trade.get("TF", "15m"))
        st.divider()
        try:
            if use_polygon:
                if not polygon_key:
                    st.error("Introduce la Polygon API Key en el sidebar.")
                else:
                    entry_dt  = pd.to_datetime(trade["Entry"])
                    exit_dt   = pd.to_datetime(trade["Exit"])
                    from_date = (entry_dt - pd.Timedelta(days=pre_days)).strftime("%Y-%m-%d")
                    to_date   = (exit_dt  + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
                    candles = load_candles_polygon(
                        trade["Ticker"], tf, from_date, to_date, polygon_key
                    )
                    build_trade_chart(candles, trade, context_bars=context_bars,
                                      timeframe=tf, source_label="polygon")
            else:
                candles = load_candles_local(tf, candle_base_dir)
                build_trade_chart(candles, trade, context_bars=context_bars,
                                  timeframe=tf, source_label="local")
        except FileNotFoundError:
            st.error(f"Dataset local no encontrado: `{candle_path_for(tf, candle_base_dir)}`")
        except requests.HTTPError as e:
            st.error(f"Error Polygon API: {e}")
        except Exception as e:
            st.error(f"Error al cargar el gráfico: {e}")


# ── TAB 8: Montecarlo ─────────────────────────────────────────────────────────
with tab8:
    import time as _time

    st.markdown("### 🎲 Simulación de Montecarlo")
    st.caption(
        "Reordena aleatoriamente (con reemplazo) los PnL reales de cada trade para generar "
        "miles de posibles caminos de equity. Muestra dónde se sitúa tu curva real frente a ellos."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns([1, 1, 1, 1])
    n_sims   = mc1.selectbox("Nº simulaciones", [200, 500, 1000, 2000, 5000], index=2)
    ruin_thr = mc2.slider("Umbral de ruina (% caída)", 10, 90, 50, 5,
                           help="Se considera 'ruina' cuando la equity cae este % desde el capital inicial")
    show_paths = mc3.slider("Paths a mostrar", 50, 500, 200, 50,
                             help="Paths individuales visibles en el gráfico (no afecta el cálculo)")
    use_net  = mc4.radio("PnL a usar", ["Net (c/ comisiones)", "Gross (s/ comisiones)"],
                          index=0, help="Net incluye comisiones; Gross solo precio")

    run_mc = st.button("▶ Ejecutar nueva simulación", type="primary", use_container_width=False)

    pnl_series = df["scaled_pnl"].values if "Net" in use_net else df["gross_pnl"].values
    n_trades   = len(pnl_series)
    ruin_level = initial_capital * (1 - ruin_thr / 100)

    if run_mc:
        rng = np.random.default_rng(seed=int(_time.time() * 1000) % (2**32))
        # Matrix: n_sims × n_trades  — sample with replacement
        sampled = rng.choice(pnl_series, size=(n_sims, n_trades), replace=True)
        mc_equity = initial_capital + np.cumsum(sampled, axis=1)   # (n_sims, n_trades)
        st.session_state["mc_equity"]     = mc_equity
        st.session_state["mc_n_sims"]     = n_sims
        st.session_state["mc_ruin_level"] = ruin_level
        st.session_state["mc_ruin_thr"]   = ruin_thr
        st.session_state["mc_use_net"]    = use_net

    if "mc_equity" not in st.session_state:
        st.info("👆 Pulsa **Ejecutar nueva simulación** para comenzar.")
        _mc_ready = False
    else:
        _mc_ready = True

    if _mc_ready:
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
        k1.metric("Simulaciones",         f"{_n_sims:,}")
        k2.metric("Equity real final",    fmt_dollar(actual_final))
        k3.metric("Percentil real",       f"{rank_pct:.1f}º",
                  help="La curva real supera al X% de las simulaciones en equity final")
        k4.metric("Mediana simulaciones", fmt_dollar(np.median(final_equities)))
        k5.metric("Riesgo de Ruina",      f"{hit_ruin:.1f}%",
                  delta=f"umbral -{_ruin_thr}% (${_ruin_level:,.0f})",
                  delta_color="inverse" if hit_ruin > 5 else "off")
        k6.metric("P5 equity final",      fmt_dollar(np.percentile(final_equities, 5)))

        st.divider()

        fig_mc  = go.Figure()
        show_n  = min(show_paths, _n_sims)
        indices = np.random.choice(_n_sims, size=show_n, replace=False)
        for i in indices:
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
        fig_mc.add_hline(y=initial_capital, line_dash="dot",
                         line_color=SUB, opacity=0.5, annotation_text="Capital inicial")
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
    _comm_badge = "✅ ON" if include_commissions else "❌ OFF"
    st.caption(
        f"Capital **${initial_capital:,}** · Riesgo **{risk_pct}%** · "
        f"Comisiones **{_comm_badge}** · Modo **{'Compuesto' if compound else 'Fijo'}** · "
        f"RVOL mín **{rvol_min}x** · Vol entrada mín **{entry_vol_min:,}**  "
        f"— todos los filtros del sidebar se aplican a cada estrategia."
    )

    if "comp_strategies" not in st.session_state:
        st.session_state.comp_strategies = []

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        new_file = st.file_uploader(
            "Añadir estrategia (.parquet)",
            type=["parquet"],
            key=f"comp_uploader_{len(st.session_state.comp_strategies)}",
        )
    with up_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("➕ Añadir", use_container_width=True,
                            disabled=new_file is None)

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

    if st.session_state.comp_strategies:
        st.divider()

        comp_dfs, comp_names = [], []
        for strat in st.session_state.comp_strategies:
            try:
                cdf = load_and_compute(
                    strat["bytes"], initial_capital, risk_pct,
                    compound, include_commissions,
                    from_date if date_filter else None,
                    to_date   if date_filter else None,
                    rvol_min, entry_vol_min,
                )
                # Apply advanced post-filters + recompute equity
                if any_advanced:
                    mask = pd.Series(True, index=cdf.index)
                    if ep_filter:
                        mask &= cdf["entry_price"].between(ep_min, ep_max)
                    if et_filter and et_from and et_to:
                        mask &= cdf["entry_time"].between(pd.Timestamp(et_from), pd.Timestamp(et_to) + pd.Timedelta(days=1))
                    if pdc_filter:
                        mask &= cdf["pct_from_pdc"].between(pdc_min, pdc_max)
                    if ticker_filter and selected_tickers:
                        mask &= cdf["ticker"].isin(selected_tickers)
                    cdf = recompute_equity(cdf[mask], initial_capital)
                comp_dfs.append(cdf)
                comp_names.append(strat["name"])
            except Exception as e:
                st.error(f"{strat['name']}: {e}")

    def _strat_metrics(cdf, name):
        _net  = cdf["scaled_pnl"].sum()
        _sh   = (cdf["ret_pct"].mean() / cdf["ret_pct"].std() * np.sqrt(252)
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

    if st.session_state.comp_strategies and comp_dfs:
        metrics_df = pd.DataFrame(
            [_strat_metrics(cdf, name) for cdf, name in zip(comp_dfs, comp_names)]
        ).set_index("Estrategia")

        st.markdown("#### 📋 Tabla comparativa")
        st.dataframe(metrics_df.T, use_container_width=True)

        st.divider()

        show_combined = st.checkbox("⚡ Mostrar equity combinada (suma de todas)", value=True)

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
                           line_color=SUB, opacity=0.4,
                           annotation_text="Capital inicial")
        fig_comp.update_layout(
            **PLOTLY_LAYOUT, height=540,
            title="Comparación de Equity Curves",
            xaxis_title="Fecha", yaxis_title="Equity ($)",
        )
        fig_comp.update_xaxes(gridcolor="#2e3547")
        fig_comp.update_yaxes(gridcolor="#2e3547")
        st.plotly_chart(fig_comp, use_container_width=True)

        if show_combined and len(comp_dfs) > 1:
            st.divider()
            st.markdown("#### ⚡ Métricas de la equity combinada")
            all_t = pd.concat(
                [cdf[["entry_time","scaled_pnl","gross_pnl","commission","winner","ret_pct"]].copy()
                 for cdf in comp_dfs],
                ignore_index=True,
            ).sort_values("entry_time").reset_index(drop=True)
            all_t["equity_curve"] = initial_capital + all_t["scaled_pnl"].cumsum()
            roll_max = all_t["equity_curve"].cummax()
            all_t["drawdown_pct"] = (all_t["equity_curve"] - roll_max) / roll_max * 100
            _cn  = all_t["scaled_pnl"].sum()
            _csh = (all_t["ret_pct"].mean() / all_t["ret_pct"].std() * np.sqrt(252)
                    if all_t["ret_pct"].std() else 0)
            ck1,ck2,ck3,ck4,ck5,ck6,ck7,ck8 = st.columns(8)
            ck1.metric("Trades",        f"{len(all_t):,}")
            ck2.metric("Win Rate",      f"{all_t['winner'].mean()*100:.1f}%")
            ck3.metric("Net PnL",       fmt_dollar(_cn))
            ck4.metric("Return",        f"{_cn/initial_capital*100:+.1f}%")
            ck5.metric("Profit Factor", f"{pf(all_t['scaled_pnl']):.2f}")
            ck6.metric("Sharpe",        f"{_csh:.2f}")
            ck7.metric("Max DD",        f"{all_t['drawdown_pct'].min():.1f}%")
            ck8.metric("Comisiones",    f"${all_t['commission'].sum():,.2f}")


# ── TAB 10: Market Cap & Float ────────────────────────────────────────────────
with tab10:
    # Usa df_full (sin filtros avanzados) para ver el universo completo del archivo
    _mkcap_avail = "market_cap" in df_full.columns and df_full["market_cap"].notna().any()
    _float_avail = "float"      in df_full.columns and df_full["float"].notna().any()

    if not _mkcap_avail and not _float_avail:
        st.info(
            "Este archivo no contiene columnas `market_cap` o `float`.  \n"
            "Ejecuta el backtest incremental para que los trades se enriquezcan "
            "automáticamente con datos fundamentales."
        )
        st.stop()

    st.markdown("### 🏦 Análisis por Market Cap & Float")
    st.caption(f"Universo completo: **{len(df_full):,} trades** (sin filtros avanzados aplicados)")

    # ── Coverage KPIs ─────────────────────────────────────────────────────────
    cov1, cov2, cov3, cov4 = st.columns(4)
    if _mkcap_avail:
        cov1.metric("Cobertura Market Cap", f"{df_full['market_cap'].notna().mean()*100:.1f}%")
        cov2.metric("Market Cap Mediana",   f"${df_full['market_cap'].median()/1e6:.1f}M")
    if _float_avail:
        cov3.metric("Cobertura Float",  f"{df_full['float'].notna().mean()*100:.1f}%")
        cov4.metric("Float Mediano",    f"{df_full['float'].median()/1e6:.2f}M shares")

    df_fund = df_full[df_full["market_cap"].notna() | df_full["float"].notna()].copy() if (_mkcap_avail or _float_avail) else df_full.copy()

    # ── Buckets ───────────────────────────────────────────────────────────────
    if _mkcap_avail:
        df_fund["mkcap_M"] = df_fund["market_cap"] / 1e6
        df_fund["mkcap_bucket"] = pd.cut(
            df_fund["mkcap_M"],
            bins=[0, 50, 300, 2000, 1e9],
            labels=["Nano (<$50M)", "Micro ($50-300M)", "Small ($300M-2B)", "Mid+ (>$2B)"],
            right=False,
        )
    if _float_avail:
        df_fund["float_M"] = df_fund["float"] / 1e6
        df_fund["float_bucket"] = pd.cut(
            df_fund["float_M"],
            bins=[0, 5, 20, 50, 1e9],
            labels=["Ultra Low (<5M)", "Low (5-20M)", "Mid (20-50M)", "High (>50M)"],
            right=False,
        )

    st.divider()

    # ── Market Cap section ────────────────────────────────────────────────────
    if _mkcap_avail:
        st.markdown("#### 📊 Rendimiento por Market Cap")

        mkcap_agg = (
            df_fund.groupby("mkcap_bucket", observed=True)
            .agg(
                trades   =("scaled_pnl", "count"),
                net_pnl  =("scaled_pnl", "sum"),
                win_rate =("winner",     "mean"),
                avg_pnl  =("scaled_pnl", "mean"),
                pf_val   =("scaled_pnl", pf),
                med_mkcap=("mkcap_M",    "median"),
            )
            .reset_index()
        )
        mkcap_agg["wr_pct"] = mkcap_agg["win_rate"] * 100
        mkcap_agg["pf_val"] = mkcap_agg["pf_val"].clip(0, 20)

        mc_col1, mc_col2 = st.columns(2)

        with mc_col1:
            fig_mc1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mc1.add_trace(go.Bar(
                x=mkcap_agg["mkcap_bucket"].astype(str),
                y=mkcap_agg["net_pnl"],
                marker_color=[GREEN if v >= 0 else RED for v in mkcap_agg["net_pnl"]],
                name="Net PnL ($)",
                text=[fmt_dollar(v) for v in mkcap_agg["net_pnl"]],
                textposition="outside",
            ), secondary_y=False)
            fig_mc1.add_trace(go.Scatter(
                x=mkcap_agg["mkcap_bucket"].astype(str),
                y=mkcap_agg["wr_pct"],
                mode="lines+markers",
                line=dict(color=YELLOW, width=2),
                marker=dict(size=8),
                name="Win Rate %",
            ), secondary_y=True)
            fig_mc1.add_hline(y=50, line_dash="dash", line_color=YELLOW,
                              opacity=0.4, secondary_y=True)
            fig_mc1.update_layout(**PLOTLY_LAYOUT, height=360,
                                  title="Net PnL y Win Rate por Market Cap")
            fig_mc1.update_yaxes(title_text="Net PnL ($)", secondary_y=False, gridcolor="#2e3547")
            fig_mc1.update_yaxes(title_text="Win Rate (%)", secondary_y=True,
                                  range=[0, 100], gridcolor="rgba(0,0,0,0)",
                                  tickfont=dict(color=YELLOW))
            st.plotly_chart(fig_mc1, use_container_width=True)

        with mc_col2:
            fig_mc2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mc2.add_trace(go.Bar(
                x=mkcap_agg["mkcap_bucket"].astype(str),
                y=mkcap_agg["trades"],
                marker_color=BLUE, opacity=0.7, name="# Trades",
            ), secondary_y=False)
            fig_mc2.add_trace(go.Scatter(
                x=mkcap_agg["mkcap_bucket"].astype(str),
                y=mkcap_agg["pf_val"],
                mode="lines+markers",
                line=dict(color=PURPLE, width=2), marker=dict(size=8),
                name="Profit Factor",
            ), secondary_y=True)
            fig_mc2.add_hline(y=1.0, line_dash="dash", line_color=SUB,
                              opacity=0.6, secondary_y=True)
            fig_mc2.update_layout(**PLOTLY_LAYOUT, height=360,
                                  title="Trades y Profit Factor por Market Cap")
            fig_mc2.update_yaxes(title_text="# Trades", secondary_y=False, gridcolor="#2e3547")
            fig_mc2.update_yaxes(title_text="Profit Factor", secondary_y=True,
                                  gridcolor="rgba(0,0,0,0)", tickfont=dict(color=PURPLE))
            st.plotly_chart(fig_mc2, use_container_width=True)

        # Scatter: market cap vs Return%
        _samp_mc = df_fund[df_fund["mkcap_M"].notna()].sample(
            min(3000, df_fund["mkcap_M"].notna().sum()), random_state=42
        )
        _p98_mc = _samp_mc["mkcap_M"].quantile(0.98)
        fig_mc3 = go.Figure()
        for win, color, name in [(True, GREEN, "Winner"), (False, RED, "Loser")]:
            s = _samp_mc[_samp_mc["winner"] == win]
            fig_mc3.add_trace(go.Scatter(
                x=s["mkcap_M"].clip(upper=_p98_mc),
                y=(s["Return"] * 100).clip(-100, 200),
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.3),
                name=name,
            ))
        fig_mc3.add_hline(y=0, line_color=SUB, line_dash="dot")
        fig_mc3.update_layout(**PLOTLY_LAYOUT, height=320,
                              title="Market Cap ($M) vs Return % por Trade",
                              xaxis_title="Market Cap ($M, capped P98)",
                              yaxis_title="Return % (capped ±200)")
        st.plotly_chart(fig_mc3, use_container_width=True)

        # Box plot winners vs losers
        fig_mc4 = go.Figure()
        for win, color, name in [(True, GREEN, "Winners"), (False, RED, "Losers")]:
            s = df_fund[df_fund["winner"] == win]["mkcap_M"].dropna()
            fig_mc4.add_trace(go.Box(
                y=s.clip(upper=s.quantile(0.95)),
                name=name, marker_color=color,
                boxpoints="outliers", line=dict(color=color),
            ))
        fig_mc4.update_layout(**PLOTLY_LAYOUT, height=300,
                              title="Market Cap ($M): Winners vs Losers",
                              yaxis_title="Market Cap ($M, capped P95)")
        st.plotly_chart(fig_mc4, use_container_width=True)

        st.markdown("#### Tabla por Market Cap")
        _mc_tbl = mkcap_agg[["mkcap_bucket","trades","net_pnl","wr_pct","avg_pnl","pf_val","med_mkcap"]].copy()
        _mc_tbl["net_pnl"]   = _mc_tbl["net_pnl"].map(fmt_dollar)
        _mc_tbl["avg_pnl"]   = _mc_tbl["avg_pnl"].map(fmt_dollar)
        _mc_tbl["wr_pct"]    = _mc_tbl["wr_pct"].map(lambda x: f"{x:.1f}%")
        _mc_tbl["pf_val"]    = _mc_tbl["pf_val"].map(lambda x: f"{x:.2f}")
        _mc_tbl["med_mkcap"] = _mc_tbl["med_mkcap"].map(lambda x: f"${x:.1f}M")
        _mc_tbl.columns = ["Bucket", "Trades", "Net PnL", "WR%", "Avg PnL", "PF", "Median MktCap"]
        st.dataframe(_mc_tbl, use_container_width=True, hide_index=True)

    st.divider()

    # ── Float section ─────────────────────────────────────────────────────────
    if _float_avail:
        st.markdown("#### 📊 Rendimiento por Float")

        float_agg = (
            df_fund.groupby("float_bucket", observed=True)
            .agg(
                trades    =("scaled_pnl", "count"),
                net_pnl   =("scaled_pnl", "sum"),
                win_rate  =("winner",     "mean"),
                avg_pnl   =("scaled_pnl", "mean"),
                pf_val    =("scaled_pnl", pf),
                med_float =("float_M",    "median"),
            )
            .reset_index()
        )
        float_agg["wr_pct"] = float_agg["win_rate"] * 100
        float_agg["pf_val"] = float_agg["pf_val"].clip(0, 20)

        fl_col1, fl_col2 = st.columns(2)

        with fl_col1:
            fig_fl1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fl1.add_trace(go.Bar(
                x=float_agg["float_bucket"].astype(str),
                y=float_agg["net_pnl"],
                marker_color=[GREEN if v >= 0 else RED for v in float_agg["net_pnl"]],
                name="Net PnL ($)",
                text=[fmt_dollar(v) for v in float_agg["net_pnl"]],
                textposition="outside",
            ), secondary_y=False)
            fig_fl1.add_trace(go.Scatter(
                x=float_agg["float_bucket"].astype(str),
                y=float_agg["wr_pct"],
                mode="lines+markers",
                line=dict(color=YELLOW, width=2), marker=dict(size=8),
                name="Win Rate %",
            ), secondary_y=True)
            fig_fl1.add_hline(y=50, line_dash="dash", line_color=YELLOW,
                              opacity=0.4, secondary_y=True)
            fig_fl1.update_layout(**PLOTLY_LAYOUT, height=360,
                                  title="Net PnL y Win Rate por Float")
            fig_fl1.update_yaxes(title_text="Net PnL ($)", secondary_y=False, gridcolor="#2e3547")
            fig_fl1.update_yaxes(title_text="Win Rate (%)", secondary_y=True,
                                  range=[0, 100], gridcolor="rgba(0,0,0,0)",
                                  tickfont=dict(color=YELLOW))
            st.plotly_chart(fig_fl1, use_container_width=True)

        with fl_col2:
            fig_fl2 = go.Figure()
            for win, color, name in [(True, GREEN, "Winners"), (False, RED, "Losers")]:
                s = df_fund[df_fund["winner"] == win]["float_M"].dropna()
                fig_fl2.add_trace(go.Box(
                    y=s.clip(upper=s.quantile(0.95)),
                    name=name, marker_color=color,
                    boxpoints="outliers", line=dict(color=color),
                ))
            fig_fl2.update_layout(**PLOTLY_LAYOUT, height=360,
                                  title="Float (M shares): Winners vs Losers",
                                  yaxis_title="Float (M shares, capped P95)")
            st.plotly_chart(fig_fl2, use_container_width=True)

        # Scatter: float vs Return%
        _samp_fl = df_fund[df_fund["float_M"].notna()].sample(
            min(3000, df_fund["float_M"].notna().sum()), random_state=42
        )
        _p98_fl = _samp_fl["float_M"].quantile(0.98)
        fig_fl3 = go.Figure()
        for win, color, name in [(True, GREEN, "Winner"), (False, RED, "Loser")]:
            s = _samp_fl[_samp_fl["winner"] == win]
            fig_fl3.add_trace(go.Scatter(
                x=s["float_M"].clip(upper=_p98_fl),
                y=(s["Return"] * 100).clip(-100, 200),
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.3),
                name=name,
            ))
        fig_fl3.add_hline(y=0, line_color=SUB, line_dash="dot")
        fig_fl3.update_layout(**PLOTLY_LAYOUT, height=320,
                              title="Float (M shares) vs Return % por Trade",
                              xaxis_title="Float (M shares, capped P98)",
                              yaxis_title="Return % (capped ±200)")
        st.plotly_chart(fig_fl3, use_container_width=True)

        st.markdown("#### Tabla por Float")
        _fl_tbl = float_agg[["float_bucket","trades","net_pnl","wr_pct","avg_pnl","pf_val","med_float"]].copy()
        _fl_tbl["net_pnl"]   = _fl_tbl["net_pnl"].map(fmt_dollar)
        _fl_tbl["avg_pnl"]   = _fl_tbl["avg_pnl"].map(fmt_dollar)
        _fl_tbl["wr_pct"]    = _fl_tbl["wr_pct"].map(lambda x: f"{x:.1f}%")
        _fl_tbl["pf_val"]    = _fl_tbl["pf_val"].map(lambda x: f"{x:.2f}")
        _fl_tbl["med_float"] = _fl_tbl["med_float"].map(lambda x: f"{x:.2f}M")
        _fl_tbl.columns = ["Bucket", "Trades", "Net PnL", "WR%", "Avg PnL", "PF", "Median Float"]
        st.dataframe(_fl_tbl, use_container_width=True, hide_index=True)

    # ── Combined: heatmap + scatter 2D ────────────────────────────────────────
    if _mkcap_avail and _float_avail:
        st.divider()
        st.markdown("#### 🗺️ Heatmap: Market Cap × Float")

        _hm_metric = st.radio(
            "Métrica del heatmap",
            ["Net PnL ($)", "Win Rate (%)", "Profit Factor"],
            horizontal=True,
            key="hm_metric_fund",
        )
        _df_hm = df_fund.dropna(subset=["mkcap_bucket", "float_bucket"])

        if not _df_hm.empty:
            if _hm_metric == "Net PnL ($)":
                hm_vals = _df_hm.groupby(
                    ["mkcap_bucket", "float_bucket"], observed=True
                )["scaled_pnl"].sum().unstack(fill_value=0)
                _zmid = 0
                _cscale = [[0, RED], [0.5, CARD], [1, GREEN]]
            elif _hm_metric == "Win Rate (%)":
                hm_vals = (
                    _df_hm.groupby(["mkcap_bucket", "float_bucket"], observed=True)["winner"]
                    .mean().unstack(fill_value=float("nan")) * 100
                )
                _zmid = 50
                _cscale = [[0, RED], [0.5, YELLOW], [1, GREEN]]
            else:
                def _pf_hm(x):
                    w = x[x > 0].sum(); l = abs(x[x < 0].sum())
                    return w / l if l > 0 else float("nan")
                hm_vals = (
                    _df_hm.groupby(["mkcap_bucket", "float_bucket"], observed=True)["scaled_pnl"]
                    .apply(_pf_hm).unstack(fill_value=float("nan"))
                )
                _zmid = 1
                _cscale = [[0, RED], [0.5, YELLOW], [1, GREEN]]

            fig_hm = go.Figure(go.Heatmap(
                z=hm_vals.values,
                x=[str(c) for c in hm_vals.columns],
                y=[str(r) for r in hm_vals.index],
                colorscale=_cscale,
                zmid=_zmid,
                text=np.round(hm_vals.values, 1),
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(title=_hm_metric),
            ))
            fig_hm.update_layout(
                **PLOTLY_LAYOUT, height=300,
                title=f"Heatmap {_hm_metric}: Market Cap × Float",
                xaxis_title="Float Bucket",
                yaxis_title="Market Cap Bucket",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # 2D Scatter: float vs market_cap, color = Net PnL
        st.markdown("#### Scatter: Float vs Market Cap (color = Net PnL)")
        _df_2d = df_fund.dropna(subset=["mkcap_M", "float_M"])
        if not _df_2d.empty:
            _df_2d = _df_2d.sample(min(2000, len(_df_2d)), random_state=42)
            _p98_mc2 = _df_2d["mkcap_M"].quantile(0.98)
            _p98_fl2 = _df_2d["float_M"].quantile(0.98)
            _df_2d = _df_2d[(_df_2d["mkcap_M"] <= _p98_mc2) & (_df_2d["float_M"] <= _p98_fl2)]

            fig_2d = go.Figure(go.Scatter(
                x=_df_2d["float_M"],
                y=_df_2d["mkcap_M"],
                mode="markers",
                marker=dict(
                    color=_df_2d["scaled_pnl"],
                    colorscale=[[0, RED], [0.5, "#1a1d27"], [1, GREEN]],
                    cmid=0,
                    size=6, opacity=0.5,
                    colorbar=dict(title="Net PnL ($)"),
                    showscale=True,
                ),
                text=_df_2d["ticker"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Float: %{x:.1f}M sh<br>"
                    "MktCap: $%{y:.1f}M<br>"
                    "PnL: $%{marker.color:.2f}"
                    "<extra></extra>"
                ),
            ))
            fig_2d.update_layout(
                **PLOTLY_LAYOUT, height=420,
                title="Float (M shares) vs Market Cap ($M) — color = Net PnL por trade",
                xaxis_title="Float (M shares, capped P98)",
                yaxis_title="Market Cap ($M, capped P98)",
            )
            st.plotly_chart(fig_2d, use_container_width=True)
