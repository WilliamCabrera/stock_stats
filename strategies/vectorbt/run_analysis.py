import os
import sys
sys.path.insert(0, os.path.abspath("."))

from pathlib import Path
import pandas as pd
from app.utils.trade_metrics import analysis_and_plot

DATASET_ROOT = Path("backtest_dataset/full")


def _list_timeframes():
    return sorted(p.name for p in DATASET_ROOT.iterdir() if p.is_dir())


def _list_strategies(timeframe: str):
    trades_dir = DATASET_ROOT / timeframe / "trades"
    return sorted(p.name for p in trades_dir.iterdir() if p.is_dir())


def _list_variants(timeframe: str, strategy: str):
    path = DATASET_ROOT / timeframe / "trades" / strategy / f"{strategy}_full_{timeframe}_trades.parquet"
    df = pd.read_parquet(path, columns=["strategy"])
    return sorted(df["strategy"].dropna().unique().tolist())


def _prompt(question: str, options: list[str], default: str) -> str:
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    raw = input(f"Select [1-{len(options)}] or press Enter for default: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    # Allow typing the value directly
    if raw in options:
        return raw
    print(f"Invalid input, using default: {default}")
    return default


def main():
    print("=" * 50)
    print("  analysis_and_plot — interactive runner")
    print("=" * 50)

    # 1. Timeframe
    timeframes = _list_timeframes()
    timeframe = _prompt("1. Timeframe:", timeframes, timeframes[0])

    # 2. Strategy
    strategies = _list_strategies(timeframe)
    strategy = _prompt("2. Strategy:", strategies, strategies[0])

    # 3. Variant
    variants = _list_variants(timeframe, strategy)
    variant = _prompt("3. Variant (select one or Enter for ALL):", variants, "ALL")

    # Load parquet
    path = DATASET_ROOT / timeframe / "trades" / strategy / f"{strategy}_full_{timeframe}_trades.parquet"
    trades = pd.read_parquet(path)

    if variant != "ALL":
        trades = trades[trades["strategy"] == variant]
        print(f"\nFiltered to variant '{variant}': {len(trades)} trades")
    else:
        print(f"\nUsing all variants: {len(trades)} trades")

    # Capital / risk params
    print()
    raw_capital = input("Initial capital [1000]: ").strip()
    initial_capital = float(raw_capital) if raw_capital else 1000.0

    raw_risk = input("Risk per trade as % of capital, e.g. 1 for 1% [1]: ").strip()
    risk_pct = float(raw_risk) / 100 if raw_risk else 0.01

    print(f"\nRunning analysis_and_plot...")
    analysis_and_plot(trades=trades, initial_capital=initial_capital, risk_pct=risk_pct)


if __name__ == "__main__":
    main()
