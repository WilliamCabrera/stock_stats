"""
Signal Candle Dashboard  –  backside_short_lower_low_fix_stop_iterative
Busca velas VERDES (close > open) con (high-low)/low >= THRESHOLD que aparecen
DESPUÉS de que se cumplen las condiciones de entrada.

Los datos se leen directamente de los ficheros de fecha (YYYY_MM_DD.parquet)
sin ningún parquet pre-computado intermedio.

Run:  streamlit run strategies/signal_candle_dashboard.py
      make signal-candle-dashboard
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Signal Candle Analysis",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
GREEN  = "#26a69a"
RED    = "#ef5350"
BLUE   = "#42a5f5"
YELLOW = "#ffca28"
PURPLE = "#ab47bc"
ORANGE = "#ffa726"
BG     = "#0f1117"
CARD   = "#1a1d27"
TEXT   = "#e0e0e0"
SUB    = "#90a4ae"
GRID   = "#2e3547"

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #90a4ae; }
    div[data-testid="metric-container"] {
        background: #1a1d27; border: 1px solid #2e3547;
        border-radius: 8px; padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    .block-container { padding-top: 1.2rem; }
    .insight-box {
        background: #1a1d27; border-left: 3px solid #ffa726;
        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.87rem; color: #e0e0e0;
    }
    .warn-box {
        background: #1a1d27; border-left: 3px solid #ef5350;
        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.87rem; color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1),
)

DATASET_ROOT = Path(os.path.abspath(".")) / "backtest_dataset"
NY_TZ        = "America/New_York"
PRICE_ORDER  = ["< $1", "$1-5", "$5-10", "$10-20", "$20-50", "> $50"]
SESSION_COLORS = {"Pre-Market": ORANGE, "Market": GREEN, "After-Hours": PURPLE}

WF_FOLDS = [
    ("fold_1", "IN-SAMPLE",     "dates_IS"),
    ("fold_2", "IN-SAMPLE",     "dates_IS"),
    ("fold_3", "IN-SAMPLE",     "dates_IS"),
    ("fold_1", "OUT-OF-SAMPLE", "dates_OOS"),
    ("fold_2", "OUT-OF-SAMPLE", "dates_OOS"),
    ("fold_3", "OUT-OF-SAMPLE", "dates_OOS"),
]
TIER_MAP = {"fold_1": "tier_1", "fold_2": "tier_2", "fold_3": "tier_3"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _session(hora: str) -> str:
    h, m = map(int, hora.split(":"))
    t = h * 60 + m
    if t < 570:  return "Pre-Market"
    if t < 960:  return "Market"
    return "After-Hours"

def _price_bucket(p: float) -> str:
    if p < 1:   return "< $1"
    if p < 5:   return "$1-5"
    if p < 10:  return "$5-10"
    if p < 20:  return "$10-20"
    if p < 50:  return "$20-50"
    return "> $50"

def _parse_period_label(lbl: str) -> tuple[str, str, str]:
    try:
        parts      = lbl.split("  ")
        split_tier = parts[0].strip()
        date_range = parts[1].strip("[]") if len(parts) > 1 else ""
        split, tier = [s.strip() for s in split_tier.split("/")]
        return split, tier, date_range
    except Exception:
        return lbl, "", ""

def _plotly(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

# ── Core detection (runs inline on date files) ────────────────────────────────
def _detect_day(day_df: pd.DataFrame, gap_pct: float, expected_delta: pd.Timedelta) -> tuple[int, list]:
    """Detecta señales y recoge todas las velas VERDES que aparecen después."""
    CLOSE_HOUR = 16
    n_signals  = 0
    records: list[dict] = []

    for ticker, candles in day_df.groupby("ticker"):
        df = candles.reset_index(drop=True)
        if len(df) < 3:
            continue

        before_close = df["date"].dt.hour < CLOSE_HOUR
        if not before_close.any():
            continue
        last_valid_idx = before_close[before_close].index[-1]

        no_gap     = df["date"].diff() == expected_delta
        prev_close = df["close"].shift(1)
        prev_open  = df["open"].shift(1)
        prev_low   = df["low"].shift(1)
        prev_vwap  = df["vwap"].shift(1)

        red        = df["close"] < df["open"]
        green_prev = prev_close > prev_open
        lower_low  = df["close"] < prev_low
        gap_cond   = df["high"] >= df["previous_day_close"] * (1 + gap_pct)
        above_vwap = df["open"] > prev_vwap
        signal     = red & green_prev & lower_low & gap_cond & above_vwap & no_gap

        for i in range(1, last_valid_idx):
            if not signal.iloc[i]:
                continue
            next_i = i + 1
            if next_i > last_valid_idx or not no_gap.iloc[next_i]:
                continue

            n_signals  += 1
            sig_row     = df.iloc[i]
            signal_hora = sig_row["date"].strftime("%H:%M")
            signal_id   = f"{ticker}_{sig_row['date_str']}_{signal_hora}"

            for j in range(next_i, last_valid_idx + 1):
                jrow = df.iloc[j]
                if jrow["close"] <= jrow["open"]:
                    continue
                size_pct = (jrow["high"] - jrow["low"]) / jrow["low"] * 100
                records.append({
                    "signal_id":       signal_id,
                    "ticker":          ticker,
                    "date_str":        sig_row["date_str"],
                    "signal_hora":     signal_hora,
                    "hora":            jrow["date"].strftime("%H:%M"),
                    "bars_after":      j - i,
                    "open":            round(jrow["open"], 4),
                    "high":            round(jrow["high"], 4),
                    "low":             round(jrow["low"], 4),
                    "close":           round(jrow["close"], 4),
                    "candle_size_pct": round(size_pct, 2),
                })
    return n_signals, records


def _process_dir(dates_dir: Path, gap_pct: float, expected_delta: pd.Timedelta, label: str) -> tuple[int, pd.DataFrame]:
    date_files    = sorted(dates_dir.glob("*.parquet"))
    total_signals = 0
    all_records: list[dict] = []

    for pf in date_files:
        try:
            day_df = pd.read_parquet(pf)
        except Exception:
            continue
        n, recs    = _detect_day(day_df, gap_pct, expected_delta)
        total_signals += n
        all_records.extend(recs)

    if not all_records:
        return total_signals, pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.insert(0, "period", label)
    return total_signals, df


@st.cache_data(show_spinner="Analizando ficheros de fecha…")
def _load_analysis(source: str, timeframe: str, gap_pct: float) -> tuple[dict, pd.DataFrame]:
    """
    Lee los ficheros YYYY_MM_DD.parquet directamente y devuelve
    (n_by_period, all_greens_df) donde all_greens_df contiene TODAS las
    velas verdes encontradas después de cada señal (sin filtro de tamaño).
    """
    tf_minutes     = int(timeframe[:-1])
    expected_delta = pd.Timedelta(minutes=tf_minutes)

    if source == "walkforward":
        base        = DATASET_ROOT / "walkforward" / timeframe
        n_by_period: dict[str, int] = {}
        all_dfs: list[pd.DataFrame] = []

        for fold, split, dates_subdir in WF_FOLDS:
            dates_dir  = base / fold / dates_subdir
            if not dates_dir.exists():
                continue
            date_files = sorted(dates_dir.glob("*.parquet"))
            if not date_files:
                continue

            date_from = date_files[0].stem.replace("_", "-")
            date_to   = date_files[-1].stem.replace("_", "-")
            tier      = TIER_MAP[fold]
            label     = f"{split} / {tier}  [{date_from} → {date_to}]"

            n, df_period = _process_dir(dates_dir, gap_pct, expected_delta, label)
            n_by_period[label] = n
            if not df_period.empty:
                all_dfs.append(df_period)

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        return n_by_period, combined

    else:   # full / up-to-date
        dates_dir  = DATASET_ROOT / "full" / timeframe / "dates"
        date_files = sorted(dates_dir.glob("*.parquet"))
        if not date_files:
            return {}, pd.DataFrame()

        date_from = date_files[0].stem.replace("_", "-")
        date_to   = date_files[-1].stem.replace("_", "-")
        label     = f"FULL  [{date_from} → {date_to}]"

        n, df_full = _process_dir(dates_dir, gap_pct, expected_delta, label)
        return {label: n}, df_full


_TF_API = {"5m": (5, "minute"), "15m": (15, "minute"), "1m": (1, "minute"), "1h": (1, "hour")}

@st.cache_data(show_spinner="Descargando velas…", ttl=3600)
def _fetch_day_candles(ticker: str, date_str: str, timeframe: str = "5m") -> pd.DataFrame:
    api_key  = os.getenv("MASSIVE_API_KEY", "")
    base_url = os.getenv("MASSIVE_BASE_URL", "https://api.polygon.io").rstrip("/")
    if not api_key:
        return pd.DataFrame({"_error": ["MASSIVE_API_KEY no encontrada"]})
    mult, span = _TF_API.get(timeframe, (5, "minute"))
    url = (
        f"{base_url}/v2/aggs/ticker/{ticker}/range/{mult}/{span}"
        f"/{date_str}/{date_str}"
        f"?adjusted=false&sort=asc&limit=50000&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})

    results = resp.json().get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.rename(columns={"o": "open", "c": "close", "h": "high",
                        "l": "low",  "v": "volume", "t": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(NY_TZ)
    return df.sort_values("time").reset_index(drop=True)


def _candle_chart(
    candles: pd.DataFrame, ticker: str, date_str: str,
    signal_hora: str, green_hora: str, tf_minutes: int,
    sig_open: float, sig_high: float, sig_low: float, sig_close: float,
    g_open: float,   g_high: float,   g_low: float,   g_close: float,
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])

    fig.add_trace(go.Candlestick(
        x=candles["time"], open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"], name=ticker,
        increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False,
    ), row=1, col=1)

    bar_colors = [GREEN if c >= o else RED for o, c in zip(candles["open"], candles["close"])]
    fig.add_trace(go.Bar(
        x=candles["time"], y=candles["volume"],
        marker_color=bar_colors, opacity=0.6, name="Vol", showlegend=False,
    ), row=2, col=1)

    try:
        # Señal (vela roja) — borde rojo
        sig_start = pd.Timestamp(f"{date_str} {signal_hora}:00").tz_localize(NY_TZ)
        sig_end   = sig_start + pd.Timedelta(minutes=tf_minutes)
        fig.add_shape(type="rect", x0=sig_start, x1=sig_end, y0=0, y1=1,
                      yref="paper", fillcolor=RED, opacity=0.12,
                      line=dict(color=RED, width=2))
        fig.add_annotation(x=sig_start, y=0.94, yref="paper",
                           text=f"  Signal {signal_hora}",
                           font=dict(color=RED, size=11), showarrow=False, xanchor="left")

        # Vela verde grande — resaltada en verde/naranja
        g_size = (g_high - g_low) / g_low * 100
        g_start = pd.Timestamp(f"{date_str} {green_hora}:00").tz_localize(NY_TZ)
        g_end   = g_start + pd.Timedelta(minutes=tf_minutes)
        fig.add_shape(type="rect", x0=g_start, x1=g_end, y0=0, y1=1,
                      yref="paper", fillcolor=ORANGE, opacity=0.20,
                      line=dict(color=ORANGE, width=2))
        fig.add_annotation(x=g_start, y=0.84, yref="paper",
                           text=f"  🟢 {g_size:.1f}%  H={g_high:.3f}  L={g_low:.3f}",
                           font=dict(color=ORANGE, size=11, family="monospace"),
                           showarrow=False, xanchor="left")
    except Exception:
        pass

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{ticker}  ·  {date_str}  ·  signal {signal_hora}  →  vela verde {green_hora}",
        height=520, xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕯️ Signal Candle")
    st.caption("Velas verdes después de la señal")
    st.divider()

    tf = st.selectbox("Timeframe", ["5m", "15m"], index=0)

    source_opts = []
    for src in ["walkforward", "full", "up-to-date"]:
        chk_dir = DATASET_ROOT / ("full" if src != "walkforward" else "walkforward") / tf
        if chk_dir.exists():
            source_opts.append(src)
    if not source_opts:
        st.error("No se encontraron datos.")
        st.stop()

    source    = st.selectbox("Fuente de datos", source_opts)
    gap_pct   = st.slider("Gap pct mínimo", 0.10, 1.00, 0.40, 0.05,
                           help="Condición gap_pct de la estrategia")

    st.divider()
    sessions_sel = st.multiselect(
        "Sesión de la vela verde",
        ["Pre-Market", "Market", "After-Hours"],
        default=["Pre-Market", "Market", "After-Hours"],
    )
    price_sel = st.multiselect("Precio (low) de la vela verde", PRICE_ORDER, default=PRICE_ORDER)
    st.divider()
    st.caption("ℹ️ Primera carga: lee todos los ficheros de fecha (puede tardar ~30s).")

THRESHOLD = 100   # velas verdes con (high-low)/low >= 100% — fijo, no configurable

# ── Load data inline ──────────────────────────────────────────────────────────
n_by_period, df_all = _load_analysis(source, tf, gap_pct)

if df_all.empty and not n_by_period:
    st.error("No se encontraron datos de fecha. Verifica la configuración.")
    st.stop()

# Enriquecer con columnas derivadas
if not df_all.empty:
    df_all["session"]        = df_all["hora"].apply(_session)
    df_all["signal_session"] = df_all["signal_hora"].apply(_session)
    df_all["price_bucket"]   = df_all["low"].apply(_price_bucket)
    df_all["date"]           = pd.to_datetime(df_all["date_str"])
    df_all["weekday"]        = df_all["date"].dt.day_name()
    df_all["month"]          = df_all["date"].dt.to_period("M").astype(str)

# Filtro sidebar
if not df_all.empty:
    df = df_all[
        df_all["session"].isin(sessions_sel) &
        df_all["price_bucket"].isin(price_sel)
    ].copy()
else:
    df = pd.DataFrame()

# Velas grandes (filtradas por THRESHOLD)
big            = df[df["candle_size_pct"] >= THRESHOLD] if not df.empty else pd.DataFrame()
total_signals  = sum(n_by_period.values())
signals_w_big  = big["signal_id"].nunique() if not big.empty else 0
prob           = signals_w_big / total_signals * 100 if total_signals > 0 else 0.0

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Signal Candle — Velas Verdes Después de la Entrada")
st.caption(
    f"Estrategia: `backside_short_lower_low_fix_stop_iterative`  ·  `{tf}`  ·  "
    f"fuente: `{source}`  ·  gap={gap_pct}  ·  umbral verde: `{THRESHOLD}%`"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total señales",                f"{total_signals:,}")
k2.metric(f"Señales con verde ≥{THRESHOLD}%", f"{signals_w_big:,}", f"{prob:.2f}%")
k3.metric("Ocurrencias totales",          f"{len(big):,}")
k4.metric("Tamaño medio (todas verdes)",
          f"{df['candle_size_pct'].mean():.1f}%" if not df.empty else "–")
k5.metric(f"P95 (todas verdes)",
          f"{df['candle_size_pct'].quantile(0.95):.1f}%" if not df.empty else "–")
k6.metric("Máximo",
          f"{df['candle_size_pct'].max():.1f}%" if not df.empty else "–")

st.divider()

tabs = st.tabs(["📊 Overview", "🕐 Horarios", "💲 Precios", "📅 Calendario", "🔁 Walkforward", "📋 Detalle"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if not df.empty:
        premarket_big = big[big["session"] == "Pre-Market"] if not big.empty else pd.DataFrame()
        pct_pm  = len(premarket_big) / len(big) * 100 if len(big) > 0 else 0
        cheap   = big[big["low"] < 5]        if not big.empty else pd.DataFrame()
        pct_ch  = len(cheap) / len(big) * 100 if len(big) > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="warn-box">⚠️ <b>{pct_pm:.0f}%</b> de las velas verdes ≥{THRESHOLD}% '
                f'son en <b>Pre-Market</b> ({len(premarket_big)} de {len(big)})</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="insight-box">💲 <b>{pct_ch:.0f}%</b> de las velas verdes ≥{THRESHOLD}% '
                f'tienen low <b>< $5</b> ({len(cheap)} de {len(big)})</div>',
                unsafe_allow_html=True,
            )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Distribución de tamaño — todas las velas verdes post-señal")
        if not df.empty:
            cap = df["candle_size_pct"].clip(upper=300)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=cap, nbinsx=60, marker_color=GREEN, opacity=0.8, name="Velas verdes",
                hovertemplate="Tamaño: %{x:.0f}%<br>Count: %{y}<extra></extra>",
            ))
            fig.add_vline(x=THRESHOLD, line_color=RED, line_dash="dash",
                          annotation_text=f"{THRESHOLD}%", annotation_font_color=RED)
            med = df["candle_size_pct"].median()
            fig.add_vline(x=med, line_color=YELLOW, line_dash="dot",
                          annotation_text=f"Med {med:.0f}%",
                          annotation_font_color=YELLOW, annotation_position="top left")
            _plotly(fig, title="Histograma (clip 300%)",
                    xaxis_title="(high-low)/low %", yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para el filtro actual.")

    with col_r:
        st.subheader("Velas verdes grandes por sesión")
        if not df.empty:
            sess_all = df.groupby("session").size().rename("total")
            sess_big = big.groupby("session").size().rename("big") if not big.empty else pd.Series(dtype=int)
            sess_df  = pd.concat([sess_all, sess_big], axis=1).fillna(0).astype(int)
            sess_df  = sess_df.reindex(["Pre-Market", "Market", "After-Hours"]).dropna(how="all")
            sess_df["pct"] = (sess_df["big"] / sess_df["total"] * 100).fillna(0)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for sess, color in SESSION_COLORS.items():
                if sess not in sess_df.index:
                    continue
                fig.add_trace(go.Bar(x=[sess], y=[sess_df.loc[sess, "total"]],
                                     name=f"{sess} total", marker_color=color, opacity=0.4),
                              secondary_y=False)
                fig.add_trace(go.Bar(x=[sess], y=[sess_df.loc[sess, "big"]],
                                     name=f"{sess} ≥{THRESHOLD}%", marker_color=color, opacity=1.0),
                              secondary_y=False)
            fig.add_trace(go.Scatter(
                x=sess_df.index.tolist(), y=sess_df["pct"].tolist(),
                name="% grandes", mode="lines+markers",
                line=dict(color=RED, width=2), marker=dict(size=8),
            ), secondary_y=True)
            _plotly(fig, barmode="overlay", title="Velas verdes por sesión")
            fig.update_yaxes(title_text="Ocurrencias", secondary_y=False)
            fig.update_yaxes(title_text=f"% ≥{THRESHOLD}%", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    # Scatter precio vs tamaño
    st.subheader("Precio (low) vs tamaño de la vela verde")
    if not df.empty:
        df_plot          = df.copy()
        df_plot["log_low"] = np.log10(df_plot["low"].clip(lower=0.01))
        df_plot["label"]   = df_plot["ticker"] + "  " + df_plot["date_str"] + "  " + df_plot["hora"]

        fig = go.Figure()
        for sess, color in SESSION_COLORS.items():
            sub = df_plot[df_plot["session"] == sess]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["log_low"], y=sub["candle_size_pct"],
                mode="markers", name=sess,
                marker=dict(color=color, size=4, opacity=0.5),
                text=sub["label"],
                hovertemplate="%{text}<br>Low: $%{customdata:.3f}<br>Tamaño: %{y:.1f}%<extra></extra>",
                customdata=sub["low"],
            ))
        fig.add_hline(y=THRESHOLD, line_color=RED, line_dash="dash",
                      annotation_text=f"{THRESHOLD}%", annotation_font_color=RED)
        tick_vals = [np.log10(v) for v in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]]
        tick_text = ["$0.1", "$0.5", "$1", "$2", "$5", "$10", "$20", "$50", "$100", "$200", "$500"]
        _plotly(fig, title="Log(precio) vs tamaño (todas las velas verdes post-señal)", height=420,
                xaxis=dict(title="Precio low (log)", tickvals=tick_vals, ticktext=tick_text, gridcolor=GRID),
                yaxis=dict(title="(high-low)/low %", gridcolor=GRID))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HORARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if df.empty:
        st.info("Sin datos.")
    else:
        hc1, hc2, hc3 = st.columns(3)
        for sess, col in zip(["Pre-Market", "Market", "After-Hours"], [hc1, hc2, hc3]):
            sub  = df[df["session"] == sess]
            sbig = big[big["session"] == sess] if not big.empty else pd.DataFrame()
            p    = len(sbig) / len(sub) * 100 if len(sub) > 0 else 0
            col.metric(f"{sess}", f"{len(sub):,} verdes", f"{len(sbig)} ≥{THRESHOLD}%  ({p:.2f}%)")

        # Distribución por hora de la vela verde
        hour_all = df.groupby("hora").size().rename("total")
        hour_big = big.groupby("hora").size().rename("big") if not big.empty else pd.Series(dtype=int)
        hour_df  = pd.concat([hour_all, hour_big], axis=1).fillna(0).sort_index()
        hour_df["pct"] = (hour_df["big"] / hour_df["total"] * 100).fillna(0)

        def _h_color(h: str) -> str:
            p = int(h.split(":")[0]) * 60 + int(h.split(":")[1])
            if p < 570: return ORANGE
            if p < 960: return GREEN
            return PURPLE

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=hour_df.index, y=hour_df["total"],
            marker_color=[_h_color(h) for h in hour_df.index], opacity=0.7,
            name="Velas verdes",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hour_df.index, y=hour_df["pct"],
            mode="lines", name=f"% ≥{THRESHOLD}%",
            line=dict(color=RED, width=2),
        ), secondary_y=True)
        for vline, label in [("09:30", "Market open"), ("16:00", "Market close")]:
            fig.add_shape(type="line", x0=vline, x1=vline, y0=0, y1=1, yref="paper",
                          line=dict(dash="dot", color=SUB, width=1))
            fig.add_annotation(x=vline, y=0.98, yref="paper", text=label,
                               font=dict(color=SUB, size=10), showarrow=False,
                               xanchor="left", bgcolor=BG, opacity=0.8)
        _plotly(fig, title="Hora de aparición de la vela verde post-señal", height=420,
                xaxis=dict(title="Hora (ET)", gridcolor=GRID))
        fig.update_yaxes(title_text="Velas verdes", secondary_y=False)
        fig.update_yaxes(title_text=f"% ≥{THRESHOLD}%", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Comparación: hora señal vs hora vela verde
        st.subheader("¿Cuántas barras después del signal aparece la vela verde?")
        if not big.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=big["bars_after"], nbinsx=30,
                marker_color=ORANGE, opacity=0.8, name="Barras después",
                hovertemplate="Barras: %{x}<br>Count: %{y}<extra></extra>",
            ))
            _plotly(fig2, title=f"Distribución barras entre signal y vela verde ≥{THRESHOLD}%",
                    xaxis_title=f"Barras después del signal ({tf})", yaxis_title="Ocurrencias")
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRECIOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if df.empty:
        st.info("Sin datos.")
    else:
        price_all = df.groupby("price_bucket").size().rename("total").reindex(PRICE_ORDER, fill_value=0)
        price_big = big.groupby("price_bucket").size().rename("big").reindex(PRICE_ORDER, fill_value=0) if not big.empty else pd.Series(0, index=PRICE_ORDER)
        price_df  = pd.concat([price_all, price_big], axis=1).fillna(0).astype(int)
        price_df["pct"] = (price_df["big"] / price_df["total"] * 100).fillna(0)

        pc_l, pc_r = st.columns(2)
        with pc_l:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=price_df.index, y=price_df["total"],
                                 marker_color=BLUE, opacity=0.55, name="Total verdes"),
                          secondary_y=False)
            fig.add_trace(go.Bar(x=price_df.index, y=price_df["big"],
                                 marker_color=RED, opacity=0.9, name=f"≥{THRESHOLD}%"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df["pct"],
                                     mode="lines+markers", name="Prob",
                                     line=dict(color=YELLOW, width=2), marker=dict(size=7)),
                          secondary_y=True)
            _plotly(fig, barmode="overlay", title="Velas verdes por rango de precio (low)")
            fig.update_yaxes(title_text="Ocurrencias", secondary_y=False)
            fig.update_yaxes(title_text=f"% ≥{THRESHOLD}%", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        with pc_r:
            fig2 = go.Figure()
            for bucket in PRICE_ORDER:
                sub = df[df["price_bucket"] == bucket]["candle_size_pct"].clip(upper=300)
                if sub.empty:
                    continue
                fig2.add_trace(go.Box(y=sub, name=bucket,
                                      marker_color=GREEN, line_color=GREEN, boxmean=True))
            fig2.add_hline(y=THRESHOLD, line_color=RED, line_dash="dash",
                           annotation_text=f"{THRESHOLD}%", annotation_font_color=RED)
            _plotly(fig2, title="Box plot tamaño por precio (clip 300%)", yaxis_title="Tamaño %")
            st.plotly_chart(fig2, use_container_width=True)

        # Heatmap precio × sesión
        if not big.empty:
            st.subheader(f"Heatmap: velas ≥{THRESHOLD}% por precio × sesión")
            heat = big.groupby(["price_bucket", "session"]).size().unstack(fill_value=0)
            heat = heat.reindex(PRICE_ORDER).fillna(0)
            for sess in ["Pre-Market", "Market", "After-Hours"]:
                if sess not in heat.columns:
                    heat[sess] = 0
            heat = heat[["Pre-Market", "Market", "After-Hours"]]
            fig3 = px.imshow(heat, color_continuous_scale="Reds",
                             labels=dict(x="Sesión", y="Precio", color="Count"), text_auto=True)
            fig3.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CALENDARIO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if df.empty:
        st.info("Sin datos.")
    else:
        DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        col_d, col_m = st.columns([1, 2])

        with col_d:
            dow_all = df.groupby("weekday").size().reindex(DOW_ORDER, fill_value=0)
            dow_big = big.groupby("weekday").size().reindex(DOW_ORDER, fill_value=0) if not big.empty else pd.Series(0, index=DOW_ORDER)
            dow_pct = (dow_big / dow_all * 100).fillna(0)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=DOW_ORDER, y=dow_all.values,
                                 marker_color=BLUE, opacity=0.55, name="Total"),
                          secondary_y=False)
            fig.add_trace(go.Bar(x=DOW_ORDER, y=dow_big.values,
                                 marker_color=RED, opacity=0.9, name=f"≥{THRESHOLD}%"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=DOW_ORDER, y=dow_pct.values,
                                     mode="lines+markers", name="Prob",
                                     line=dict(color=YELLOW, width=2), marker=dict(size=7)),
                          secondary_y=True)
            _plotly(fig, barmode="overlay", title="Día de la semana")
            fig.update_yaxes(title_text="Velas", secondary_y=False)
            fig.update_yaxes(title_text="Prob %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_m:
            mon_all = df.groupby("month").size().rename("total")
            mon_big = big.groupby("month").size().rename("big") if not big.empty else pd.Series(dtype=int)
            mon_df  = pd.concat([mon_all, mon_big], axis=1).fillna(0).sort_index()
            mon_df["pct"] = (mon_df["big"] / mon_df["total"] * 100).fillna(0)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=mon_df.index, y=mon_df["total"],
                                  marker_color=BLUE, opacity=0.5, name="Total verdes"),
                           secondary_y=False)
            fig2.add_trace(go.Bar(x=mon_df.index, y=mon_df["big"],
                                  marker_color=RED, opacity=0.85, name=f"≥{THRESHOLD}%"),
                           secondary_y=False)
            fig2.add_trace(go.Scatter(x=mon_df.index, y=mon_df["pct"],
                                      mode="lines", name="Prob",
                                      line=dict(color=YELLOW, width=2)),
                           secondary_y=True)
            _plotly(fig2, barmode="overlay", title="Evolución mensual",
                    xaxis_tickangle=-45, height=380)
            fig2.update_yaxes(title_text="Velas", secondary_y=False)
            fig2.update_yaxes(title_text="Prob %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Heatmap: señales por hora × día")
        pivot = df.groupby(["weekday", "hora"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(DOW_ORDER).fillna(0)
        fig3  = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                          labels=dict(x="Hora", y="Día", color="Velas verdes"))
        fig3.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WALKFORWARD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    has_wf = any("IN-SAMPLE" in p or "OUT-OF-SAMPLE" in p for p in n_by_period)
    if not has_wf:
        st.info("Selecciona la fuente `walkforward` para ver análisis por período.")
    else:
        st.subheader("Resumen por período walkforward")
        rows = []
        for period, n_sigs in n_by_period.items():
            sub     = df_all[df_all["period"] == period] if not df_all.empty else pd.DataFrame()
            sub_big = sub[sub["candle_size_pct"] >= THRESHOLD] if not sub.empty else pd.DataFrame()
            sw_big  = sub_big["signal_id"].nunique() if not sub_big.empty else 0
            split, tier, date_range = _parse_period_label(period)
            rows.append({
                "Split": split, "Tier": tier, "Período": date_range,
                "Señales": n_sigs,
                f"c/ verde ≥{THRESHOLD}%": sw_big,
                "Prob (%)": round(sw_big / n_sigs * 100, 2) if n_sigs > 0 else 0,
                "Media (todas verdes)": round(sub["candle_size_pct"].mean(), 1) if not sub.empty else 0,
                "P95": round(sub["candle_size_pct"].quantile(0.95), 1) if not sub.empty else 0,
                "Máx": round(sub["candle_size_pct"].max(), 1) if not sub.empty else 0,
            })

        wf_tbl = pd.DataFrame(rows)
        st.dataframe(wf_tbl, use_container_width=True, hide_index=True)

        is_df  = wf_tbl[wf_tbl["Split"] == "IN-SAMPLE"]
        oos_df = wf_tbl[wf_tbl["Split"] == "OUT-OF-SAMPLE"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=is_df["Tier"],  y=is_df["Prob (%)"],
                             name="IN-SAMPLE",     marker_color=BLUE,   opacity=0.8))
        fig.add_trace(go.Bar(x=oos_df["Tier"], y=oos_df["Prob (%)"],
                             name="OUT-OF-SAMPLE", marker_color=ORANGE, opacity=0.8))
        _plotly(fig, barmode="group",
                title=f"Probabilidad vela verde ≥{THRESHOLD}% — IS vs OOS",
                xaxis_title="Tier", yaxis_title="Probabilidad (%)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DETALLE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    tf_minutes = int(tf[:-1])
    st.subheader(f"Velas verdes ≥ {THRESHOLD}% después de la señal")
    st.caption("Haz clic en una fila para ver el gráfico 1m (signal en rojo · vela verde resaltada).")

    if big.empty:
        st.info(f"No hay velas verdes ≥ {THRESHOLD}% en la selección actual.")
    else:
        disp_cols = ["period", "ticker", "date_str", "signal_hora", "hora",
                     "bars_after", "session", "open", "high", "low", "close",
                     "candle_size_pct", "price_bucket"]
        avail      = [c for c in disp_cols if c in big.columns]
        big_sorted = big[avail].sort_values("candle_size_pct", ascending=False).reset_index(drop=True)

        event = st.dataframe(
            big_sorted,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "candle_size_pct": st.column_config.NumberColumn("Tamaño %", format="%.1f%%"),
                "open":  st.column_config.NumberColumn("Open",  format="$%.4f"),
                "high":  st.column_config.NumberColumn("High",  format="$%.4f"),
                "low":   st.column_config.NumberColumn("Low",   format="$%.4f"),
                "close": st.column_config.NumberColumn("Close", format="$%.4f"),
                "bars_after": st.column_config.NumberColumn("Barras después"),
            },
        )

        sel = event.selection.rows
        if sel:
            row = big_sorted.iloc[sel[0]]
            with st.spinner(f"Descargando {row['ticker']} ({row['date_str']})…"):
                candles = _fetch_day_candles(row["ticker"], row["date_str"], tf)

            if "_error" in candles.columns:
                st.error(candles["_error"].iloc[0])
            elif candles.empty:
                st.warning("Sin datos disponibles en la API para esta fecha.")
            else:
                fig_c = _candle_chart(
                    candles, row["ticker"], row["date_str"],
                    signal_hora=row["signal_hora"], green_hora=row["hora"],
                    tf_minutes=tf_minutes,
                    sig_open=float(row.get("open", 0)), sig_high=float(row.get("high", 0)),
                    sig_low=float(row.get("low", 0)),   sig_close=float(row.get("close", 0)),
                    g_open=float(row["open"]),  g_high=float(row["high"]),
                    g_low=float(row["low"]),    g_close=float(row["close"]),
                )
                st.plotly_chart(fig_c, use_container_width=True)

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Ticker",          row["ticker"])
                m2.metric("Fecha",           row["date_str"])
                m3.metric("Signal",          row["signal_hora"])
                m4.metric("Vela verde",      row["hora"])
                m5.metric("Barras después",  int(row["bars_after"]))
                m6.metric("Tamaño",          f"{row['candle_size_pct']:.1f}%")
        else:
            st.info("Selecciona una fila para ver el gráfico.")

        st.divider()

        # Top tickers
        st.subheader("Tickers con más velas verdes extremas")
        tc = big["ticker"].value_counts().head(20)
        fig_t = px.bar(x=tc.index, y=tc.values,
                       labels={"x": "Ticker", "y": "Ocurrencias"},
                       color=tc.values, color_continuous_scale="Greens")
        _plotly(fig_t, title=f"Top 20 tickers con vela verde ≥{THRESHOLD}%", height=320)
        fig_t.update_coloraxes(showscale=False)
        st.plotly_chart(fig_t, use_container_width=True)

    # CDF por sesión (siempre visible si hay datos)
    if not df.empty:
        st.subheader("CDF del tamaño — todas las velas verdes post-señal")
        fig_cdf = go.Figure()
        for sess, color in SESSION_COLORS.items():
            sub = df[df["session"] == sess]["candle_size_pct"].sort_values()
            if sub.empty:
                continue
            cdf = np.arange(1, len(sub) + 1) / len(sub) * 100
            fig_cdf.add_trace(go.Scatter(
                x=sub.values, y=cdf, mode="lines", name=sess,
                line=dict(color=color, width=2),
                hovertemplate=f"{sess}<br>Tamaño: %{{x:.1f}}%<br>Percentil: %{{y:.1f}}%<extra></extra>",
            ))
        fig_cdf.add_vline(x=THRESHOLD, line_color=RED, line_dash="dash",
                          annotation_text=f"{THRESHOLD}%", annotation_font_color=RED)
        _plotly(fig_cdf,
                title="CDF por sesión — ¿qué % de velas verdes están por debajo del umbral?",
                xaxis_title="(high-low)/low %", yaxis_title="Percentil (%)",
                xaxis_range=[0, min(300, df["candle_size_pct"].quantile(0.99) * 1.1)])
        st.plotly_chart(fig_cdf, use_container_width=True)
