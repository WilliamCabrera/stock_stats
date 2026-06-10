import os
import sys
sys.path.insert(0, os.path.abspath("."))

from datetime import datetime
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
    if raw in options:
        return raw
    print(f"Invalid input, using default: {default}")
    return default


def _prompt_multi(question: str, options: list[str]) -> list[str]:
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("  Enter numbers separated by commas (e.g. 1,3), a range (e.g. 2-4), or press Enter for ALL.")
    raw = input("Select: ").strip()
    if not raw:
        return options

    selected = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit():
                for idx in range(int(lo), int(hi) + 1):
                    if 1 <= idx <= len(options):
                        selected.add(options[idx - 1])
        elif part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(options):
                selected.add(options[idx - 1])
        elif part in options:
            selected.add(part)

    if not selected:
        print("No valid selection, using ALL variants.")
        return options
    return [o for o in options if o in selected]


def _prompt_date_range(min_date: str, max_date: str) -> tuple[str | None, str | None]:
    print(f"\n4. Date range (data spans {min_date} → {max_date})")
    print("  Press Enter on both to keep full range.")
    raw_from = input(f"  From [YYYY-MM-DD, default {min_date}]: ").strip()
    raw_to   = input(f"  To   [YYYY-MM-DD, default {max_date}]: ").strip()

    def _parse(raw: str, fallback: str) -> str:
        if not raw:
            return fallback
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            print(f"  Invalid date '{raw}', using {fallback}")
            return fallback

    return _parse(raw_from, min_date), _parse(raw_to, max_date)


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

    # 3. Variant(s)
    variants = _list_variants(timeframe, strategy)
    selected_variants = _prompt_multi("3. Variant(s):", variants)

    # Load parquet
    path = DATASET_ROOT / timeframe / "trades" / strategy / f"{strategy}_full_{timeframe}_trades.parquet"
    trades = pd.read_parquet(path)

    if len(selected_variants) < len(variants):
        trades = trades[trades["strategy"].isin(selected_variants)]
        print(f"\nFiltered to {len(selected_variants)} variant(s): {', '.join(selected_variants)} — {len(trades)} trades")
    else:
        print(f"\nUsing all variants: {len(trades)} trades")

    # 4. Date range
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    min_date = trades["entry_time"].dt.date.min().isoformat()
    max_date = trades["entry_time"].dt.date.max().isoformat()
    date_from, date_to = _prompt_date_range(min_date, max_date)
    trades = trades[
        (trades["entry_time"].dt.date >= datetime.strptime(date_from, "%Y-%m-%d").date()) &
        (trades["entry_time"].dt.date <= datetime.strptime(date_to,   "%Y-%m-%d").date())
    ]
    print(f"\nDate filter {date_from} → {date_to}: {len(trades)} trades")

    # 5. Risk/reward ratio filter
    print(f"\nRisk/reward in data — min: {trades['risk_reward_ratio'].min():.2f}  max: {trades['risk_reward_ratio'].max():.2f}")
    raw_rrr = input("5. Min risk_reward_ratio (press Enter to skip): ").strip()
    if raw_rrr:
        try:
            min_rrr = float(raw_rrr)
            trades = trades[trades["risk_reward_ratio"] > min_rrr]
            print(f"   risk_reward_ratio > {min_rrr}: {len(trades)} trades")
        except ValueError:
            print("   Invalid value, skipping filter.")

    # 6. Volume filter
    print(f"\nVolume in data — min: {trades['volume'].min():,.0f}  max: {trades['volume'].max():,.0f}")
    raw_vol = input("6. Min volume (press Enter to skip): ").strip()
    if raw_vol:
        try:
            min_vol = float(raw_vol)
            trades = trades[trades["volume"] > min_vol]
            print(f"   volume > {min_vol:,.0f}: {len(trades)} trades")
        except ValueError:
            print("   Invalid value, skipping filter.")

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
