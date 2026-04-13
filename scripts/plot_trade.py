"""
Plot a single trade from a trades parquet file.

Usage (from backtester_api/):
    python -m scripts.plot_trade                        # plots trades.iloc[0]
    python -m scripts.plot_trade --index 7              # plots trades.iloc[7]
    python -m scripts.plot_trade --ticker ABOS --date 2022-09-28
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from app.utils.charts import plot_candles_df, trades_to_markers
from app.utils.indicators import compute_close_atr_band, compute_vwap
from app.utils.massive import fetch_candles

TRADES_PATH = "backtest_dataset/in_sample/trades/backside_short/backside_short_in_sample_trades.parquet"
ATR_FACTOR  = 3.5


def plot_trade(index: int = 0, ticker: str | None = None, date: str | None = None):
    trades = pd.read_parquet(TRADES_PATH)

    if ticker and date:
        row = trades[(trades["ticker"] == ticker) &
                     (trades["entry_time"].dt.strftime("%Y-%m-%d") == date)].iloc[0]
    else:
        row = trades.iloc[index]

    ticker = ticker or row["ticker"]
    date   = date   or row["entry_time"].strftime("%Y-%m-%d")

    candles_df = pd.DataFrame(fetch_candles(ticker, date, date))
    entries, exits, short_entries, short_exits = trades_to_markers(trades, ticker=ticker, date=date)

    indicators = {
        "VWAP":           compute_vwap(candles_df),
        "ATR band above": compute_close_atr_band(candles_df, factor=ATR_FACTOR, direction="above"),
    }

    plot_candles_df(
        candles_df,
        title=f"{ticker}  {date}",
        short_entries=short_entries,
        short_exits=short_exits,
        prev_close=row["previous_day_close"],
        indicators=indicators,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a trade from the backtest dataset.")
    parser.add_argument("--index",  type=int,   default=0,    help="Row index in the trades parquet (default: 0)")
    parser.add_argument("--ticker", type=str,   default=None, help="Filter by ticker symbol")
    parser.add_argument("--date",   type=str,   default=None, help="Filter by date YYYY-MM-DD")
    args = parser.parse_args()

    plot_trade(index=args.index, ticker=args.ticker, date=args.date)

#python -m scripts.plot_trade --index 7