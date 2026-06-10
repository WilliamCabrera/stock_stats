"""
Candlestick chart utility.

Fetches OHLCV data from Massive.com and renders an interactive Plotly chart.
Supports entry/exit trade markers and horizontal lines.

Usage:
    from app.utils.charts import plot_candles

    # Basic — one day, regular session only
    plot_candles("NVDA", "2024-03-15")

    # Custom timeframe + date range
    plot_candles("AAPL", "2024-03-11", "2024-03-15", timeframe="1m")

    # With trade markers
    plot_candles(
        "NVDA", "2024-03-15",
        entries=[{"time": 1710499800, "price": 123.45, "label": "entry"}],
        exits  =[{"time": 1710506400, "price": 127.80, "label": "exit"}],
    )

    # With horizontal lines (e.g. prev close, support)
    plot_candles("NVDA", "2024-03-15", hlines=[{"price": 120.0, "label": "prev close"}])

    # Save to file instead of opening browser
    plot_candles("NVDA", "2024-03-15", output="nvda.html")
"""
from __future__ import annotations

import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.massive import fetch_candles, TimeFrame

# Charts are saved here (created automatically if missing).
CHARTS_DIR = Path(__file__).resolve().parents[2] / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Default session (regular market hours ET → UTC) ───────────────────────────
_DEFAULT_SESSION_START = "04:00"
_DEFAULT_SESSION_END   = "20:00"


def plot_candles(
    ticker: str,
    from_date: str,
    to_date: str | None = None,
    timeframe: TimeFrame = "5m",
    session_start: str | None = _DEFAULT_SESSION_START,
    session_end: str | None = _DEFAULT_SESSION_END,
    entries: list[dict[str, Any]] | None = None,
    exits: list[dict[str, Any]] | None = None,
    short_entries: list[dict[str, Any]] | None = None,
    short_exits: list[dict[str, Any]] | None = None,
    hlines: list[dict[str, Any]] | None = None,
    output: str | None = None,
    height: int = 700,
) -> go.Figure:
    """
    Fetch candles and render an interactive candlestick chart.

    Args:
        ticker:        Equity symbol, e.g. "NVDA".
        from_date:     Start date "YYYY-MM-DD".
        to_date:       End date "YYYY-MM-DD". Defaults to from_date (single day).
        timeframe:     "1m", "5m", "15m", "30m", "1h", or "1d".
        session_start: Local time 'HH:MM' for session filter. Default "09:30".
                       Pass None to include pre/after-market.
        session_end:   Local time 'HH:MM' for session filter. Default "16:00".
        entries:       List of dicts with keys:
                         - time  : Unix seconds (int) or "YYYY-MM-DDTHH:MM:SS"
                         - price : float
                         - label : str (optional, shown on hover)
        exits:         Same format as entries.
        hlines:        List of dicts with keys:
                         - price : float
                         - label : str (optional)
                         - color : str (optional, default "gray")
                         - dash  : str (optional, "solid"|"dash"|"dot", default "dash")
        output:        If set, save chart to this HTML file instead of opening browser.
        height:        Chart height in pixels.

    Returns:
        plotly Figure object (already shown or saved depending on `output`).
    """
    if to_date is None:
        to_date = from_date

    # Swap silently if caller passed dates in wrong order
    if from_date > to_date:
        from_date, to_date = to_date, from_date

    # Single day → cap session_end at current ET hour:minute
    if from_date == to_date:
        import zoneinfo
        now_et = datetime.now(tz=zoneinfo.ZoneInfo("America/New_York"))
        session_end = now_et.strftime("%H:%M")

    # ── Fetch ──────────────────────────────────────────────────────────────────
    candles = fetch_candles(
        ticker,
        from_date,
        to_date,
        timeframe=timeframe,
        session_start=session_start,
        session_end=session_end,
    )
    if not candles:
        raise ValueError(f"No candles returned for {ticker} [{from_date} → {to_date}]")

    df = pd.DataFrame(candles)
    title = (
        f"{ticker}  {from_date}"
        + (f" → {to_date}" if to_date != from_date else "")
        + f"  ({timeframe})"
    )
    return plot_candles_df(df, title=title, entries=entries, exits=exits,
                           short_entries=short_entries, short_exits=short_exits,
                           hlines=hlines, output=output, height=height)


def _write_fullscreen_html(fig: go.Figure, path: str) -> None:
    """Write a plotly figure to an HTML file that fills the entire browser window."""
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100vw; height: 100vh; background: #111; overflow: hidden; }}
  #chart-wrapper {{ width: 100vw; height: 100vh; }}
  #chart-wrapper > div {{ width: 100% !important; height: 100% !important; }}
</style>
</head>
<body>
<div id="chart-wrapper">{inner}</div>
<script>
  window.addEventListener("load", function() {{
    var div = document.querySelector(".plotly-graph-div");
    if (div) {{
      Plotly.relayout(div, {{ width: window.innerWidth, height: window.innerHeight }});
    }}
  }});
  window.addEventListener("resize", function() {{
    var div = document.querySelector(".plotly-graph-div");
    if (div) {{
      Plotly.relayout(div, {{ width: window.innerWidth, height: window.innerHeight }});
    }}
  }});
</script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def plot_candles_df(
    df: pd.DataFrame,
    title: str = "",
    entries: list[dict[str, Any]] | None = None,
    exits: list[dict[str, Any]] | None = None,
    short_entries: list[dict[str, Any]] | None = None,
    short_exits: list[dict[str, Any]] | None = None,
    hlines: list[dict[str, Any]] | None = None,
    prev_close: float | None = None,
    indicators: dict[str, pd.Series] | None = None,
    output: str | None = None,
    height: int = 700,
) -> go.Figure:
    """
    Render a candlestick chart from an already-fetched DataFrame.

    The DataFrame must have at minimum these columns:
        time   — Unix seconds (int) in UTC
        open, high, low, close — floats
        volume — numeric

    Args:
        df:            DataFrame with OHLCV data (same format as fetch_candles output).
        title:         Chart title string.
        entries:       Long entries — triangle-up green below candle.
        exits:         Long exits  — triangle-down red above candle.
        short_entries: Short entries — triangle-down red above candle.
        short_exits:   Short exits   — triangle-up green below candle.
                       All marker dicts: {time (unix s or ISO str), price, label?}
        hlines:        List of dicts {price, label?, color?, dash?}.
        indicators:    Dict of {label: pd.Series} plotted as lines on the price panel.
                       e.g. {"VWAP": vwap_series, "ATR band": band_series}
        output:        If set, save chart to this HTML file instead of opening browser.
        height:        Chart height in pixels.

    Returns:
        plotly Figure object.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to plot.")

    df = df.copy()
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # ── Candlesticks ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["dt"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=title or "price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # ── Volume bars ────────────────────────────────────────────────────────────
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["dt"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Previous day close ─────────────────────────────────────────────────────
    if prev_close is not None:
        fig.add_hline(
            y=prev_close,
            line_dash="dash",
            line_color="#ffa726",
            annotation_text=f"prev close  {prev_close:.2f}",
            annotation_position="right",
            row=1, col=1,
        )

    # ── Indicators ─────────────────────────────────────────────────────────────
    _INDICATOR_COLORS = ["#c8a200", "#80cbc4", "#fff176", "#ef9a9a", "#90caf9"]
    for i, (name, series) in enumerate(( indicators or {}).items()):
        fig.add_trace(
            go.Scatter(
                x=df["dt"],
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=_INDICATOR_COLORS[i % len(_INDICATOR_COLORS)], width=1),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Horizontal lines ───────────────────────────────────────────────────────
    for hl in (hlines or []):
        fig.add_hline(
            y=hl["price"],
            line_dash=hl.get("dash", "dash"),
            line_color=hl.get("color", "gray"),
            annotation_text=hl.get("label", ""),
            annotation_position="right",
            row=1, col=1,
        )

    # ── Trade markers ──────────────────────────────────────────────────────────
    import zoneinfo as _zi

    def _to_dt(t: int | str) -> datetime:
        if isinstance(t, (int, float)):
            return datetime.fromtimestamp(t, tz=timezone.utc).astimezone(
                _zi.ZoneInfo("America/New_York")
            )
        return pd.Timestamp(t).tz_localize("America/New_York")

    def _add_markers(markers, name, symbol, color, textposition):
        if not markers:
            return
        fig.add_trace(
            go.Scatter(
                x=[_to_dt(m["time"]) for m in markers],
                y=[m["price"] for m in markers],
                mode="markers+text",
                marker=dict(symbol=symbol, size=14, color=color),
                text=[m.get("label", name) for m in markers],
                textposition=textposition,
                name=name,
                hovertemplate="%{text}<br>%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Long trades: entry = green arrow up below, exit = red arrow down above
    _add_markers(entries,       "Entry",       "triangle-up",   "#00e676", "bottom center")
    _add_markers(exits,         "Exit",        "triangle-down", "#ff1744", "top center")
    # Short trades: entry = red arrow down above, exit = green arrow up below
    _add_markers(short_entries, "Short Entry", "triangle-down", "#ff1744", "top center")
    _add_markers(short_exits,   "Short Exit",  "triangle-up",   "#00e676", "bottom center")

    # ── Styling ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title,
        height=height,
        autosize=True,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=30),
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour"),
        ],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#666666",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        title_text="Price",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#666666",
        spikethickness=1,
        spikedash="dot",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # ── Output ────────────────────────────────────────────────────────────────
    if output:
        _write_fullscreen_html(fig, output)
        print(f"Chart saved → {output}")
    else:
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (title or "chart"))
        safe_title = safe_title.replace(" ", "_")
        path = CHARTS_DIR / f"{safe_title}.html"
        _write_fullscreen_html(fig, str(path))
        print(f"Chart saved → {path}")
        webbrowser.open(path.as_uri())

    return fig


def trades_to_markers(
    trades_df: pd.DataFrame,
    ticker: str | None = None,
    date: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Convert a trades DataFrame into marker lists for plot_candles_df.

    Supports Short and Long trades. Times are converted from UTC to Unix seconds.

    Args:
        trades_df: DataFrame with columns: ticker, type, entry_price, exit_price,
                   entry_time (datetime UTC), exit_time (datetime UTC).
        ticker:    Filter to a single ticker symbol (optional).
        date:      Filter to a single date "YYYY-MM-DD" in ET (optional).

    Returns:
        (entries, exits, short_entries, short_exits) — pass directly to plot_candles_df.

    Example:
        import pandas as pd
        trades = pd.read_parquet("backside_short_in_sample_trades.parquet")
        se, sx, e, x = trades_to_markers(trades, ticker="ABOS", date="2022-09-28")
        plot_candles_df(df, short_entries=se, short_exits=sx)
    """
    
    df = trades_df.copy()

    if ticker:
        df = df[df["ticker"] == ticker]

    if date:
        df = df[df["entry_time"].dt.strftime("%Y-%m-%d") == date]

    entries: list[dict] = []
    exits: list[dict] = []
    short_entries: list[dict] = []
    short_exits: list[dict] = []

    for _, row in df.iterrows():
        # Strip timezone so _to_dt localizes as ET without conversion
        entry_ts = row["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
        exit_ts  = row["exit_time"].strftime("%Y-%m-%d %H:%M:%S")

        pnl_sign = "+" if row["pnl"] >= 0 else ""
        entry_label = f"entry @ {row['entry_price']:.2f}  {row['entry_time'].strftime('%H:%M')}"
        exit_label  = f"exit @ {row['exit_price']:.2f}  {row['exit_time'].strftime('%H:%M')}  ({pnl_sign}{row['pnl']:.2f})"

        if str(row.get("type", "")).lower() == "short":
            short_entries.append({"time": entry_ts, "price": row["entry_price"], "label": entry_label})
            short_exits.append(  {"time": exit_ts,  "price": row["exit_price"],  "label": exit_label})
        else:
            entries.append({"time": entry_ts, "price": row["entry_price"], "label": entry_label})
            exits.append(  {"time": exit_ts,  "price": row["exit_price"],  "label": exit_label})
    return entries, exits, short_entries, short_exits
