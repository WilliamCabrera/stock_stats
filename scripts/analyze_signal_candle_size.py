"""
Análisis de velas verdes después de la señal de backside_short_lower_low_fix_stop_iterative.

Para cada señal detectada, escanea las velas SIGUIENTES del mismo día buscando
velas VERDES (close > open) con (high-low)/low >= threshold.

Calcula la probabilidad de que aparezca al menos una vela verde grande
después de cada señal, y lista todas las ocurrencias.

Uso (desde backtester_api/):
    python -m scripts.analyze_signal_candle_size              # walkforward 5m
    python -m scripts.analyze_signal_candle_size --mode full
    python -m scripts.analyze_signal_candle_size --timeframe 15m
    python -m scripts.analyze_signal_candle_size --threshold 50
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath("."))

from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(os.path.abspath(".")) / "backtest_dataset"


def _detect_signals(
    day_df: pd.DataFrame,
    gap_pct: float,
    timeframe_minutes: int,
) -> tuple[int, list[dict]]:
    """
    Detecta señales y recoge TODAS las velas verdes que aparecen después.

    Returns:
        (n_signals, records)
        - n_signals: total de señales válidas detectadas en day_df
        - records:   lista de velas verdes encontradas tras cada señal,
                     con el contexto de la señal incluido
    """
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    CLOSE_HOUR     = 16
    n_signals      = 0
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

        signal = red & green_prev & lower_low & gap_cond & above_vwap & no_gap

        for i in range(1, last_valid_idx):   # necesita al menos 1 barra después
            if not signal.iloc[i]:
                continue
            next_i = i + 1
            if next_i > last_valid_idx or not no_gap.iloc[next_i]:
                continue

            n_signals += 1
            sig_row     = df.iloc[i]
            signal_hora = sig_row["date"].strftime("%H:%M")
            signal_id   = f"{ticker}_{sig_row['date_str']}_{signal_hora}"

            # Escanear barras siguientes buscando velas VERDES
            for j in range(next_i, last_valid_idx + 1):
                jrow = df.iloc[j]
                if jrow["close"] <= jrow["open"]:   # no es verde
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


def _process_dates_dir(
    dates_dir: Path,
    gap_pct: float,
    timeframe_minutes: int,
    label: str,
) -> tuple[int, pd.DataFrame]:
    date_files = sorted(dates_dir.glob("*.parquet"))
    if not date_files:
        print(f"  [!] No parquet en {dates_dir}")
        return 0, pd.DataFrame()

    total_signals = 0
    all_records:  list[dict] = []

    for pf in date_files:
        try:
            day_df = pd.read_parquet(pf)
        except Exception as e:
            print(f"  [!] Error leyendo {pf.name}: {e}")
            continue
        n, records = _detect_signals(day_df, gap_pct, timeframe_minutes)
        total_signals += n
        all_records.extend(records)

    if not all_records:
        return total_signals, pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.insert(0, "period", label)
    return total_signals, df


def _print_summary(n_signals: int, df: pd.DataFrame, threshold: float) -> None:
    if df.empty:
        big = pd.DataFrame()
    else:
        big = df[df["candle_size_pct"] >= threshold]

    signals_with_big = big["signal_id"].nunique() if not big.empty else 0
    n_occ            = len(big)
    prob             = signals_with_big / n_signals * 100 if n_signals > 0 else 0.0

    print(f"\n  Señales detectadas:              {n_signals}")
    print(f"  Señales con vela verde >= {threshold:.0f}%: {signals_with_big}  ({prob:.2f}%)")
    print(f"  Ocurrencias totales:             {n_occ}")

    if not big.empty:
        print(f"\n  {'Ticker':<8} {'Fecha':<12} {'Signal':>7} {'Hora':>7} {'Bars':>5} {'High':>8} {'Low':>8} {'Size %':>8}")
        print("  " + "-" * 76)
        for _, r in big.sort_values("candle_size_pct", ascending=False).head(20).iterrows():
            print(
                f"  {r['ticker']:<8} {r['date_str']:<12}"
                f" {r['signal_hora']:>7} {r['hora']:>7} {int(r['bars_after']):>5}"
                f" {r['high']:>8.4f} {r['low']:>8.4f} {r['candle_size_pct']:>7.1f}%"
            )


def run_walkforward(timeframe: str, gap_pct: float, threshold: float) -> tuple[dict, pd.DataFrame]:
    tf_minutes = int(timeframe[:-1])
    base       = DATASET_ROOT / "walkforward" / timeframe

    FOLDS = [
        ("fold_1", "IN-SAMPLE",     "dates_IS"),
        ("fold_2", "IN-SAMPLE",     "dates_IS"),
        ("fold_3", "IN-SAMPLE",     "dates_IS"),
        ("fold_1", "OUT-OF-SAMPLE", "dates_OOS"),
        ("fold_2", "OUT-OF-SAMPLE", "dates_OOS"),
        ("fold_3", "OUT-OF-SAMPLE", "dates_OOS"),
    ]
    tier_map = {"fold_1": "tier_1", "fold_2": "tier_2", "fold_3": "tier_3"}

    n_by_period: dict[str, int] = {}
    all_dfs:     list[pd.DataFrame] = []

    for fold, split, dates_subdir in FOLDS:
        dates_dir  = base / fold / dates_subdir
        if not dates_dir.exists():
            continue
        date_files = sorted(dates_dir.glob("*.parquet"))
        if not date_files:
            continue

        date_from = date_files[0].stem.replace("_", "-")
        date_to   = date_files[-1].stem.replace("_", "-")
        tier      = tier_map[fold]
        label     = f"{split} / {tier}  [{date_from} → {date_to}]"

        print(f"\n{'='*72}")
        print(f"  {label}  ({len(date_files)} days)")
        print(f"{'='*72}")

        n, df_period = _process_dates_dir(dates_dir, gap_pct, tf_minutes, label)
        n_by_period[label] = n

        if df_period.empty:
            print("  Sin velas verdes registradas.")
        else:
            all_dfs.append(df_period)

        _print_summary(n, df_period, threshold)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return n_by_period, combined


def run_full(timeframe: str, gap_pct: float, threshold: float) -> tuple[dict, pd.DataFrame]:
    tf_minutes = int(timeframe[:-1])
    dates_dir  = DATASET_ROOT / "full" / timeframe / "dates"
    date_files = sorted(dates_dir.glob("*.parquet"))
    date_from  = date_files[0].stem.replace("_", "-") if date_files else "?"
    date_to    = date_files[-1].stem.replace("_", "-") if date_files else "?"
    label      = f"FULL  [{date_from} → {date_to}]"

    print(f"\n{'='*72}")
    print(f"  {label}  ({len(date_files)} days)")
    print(f"{'='*72}")

    n, df = _process_dates_dir(dates_dir, gap_pct, tf_minutes, label)
    _print_summary(n, df, threshold)

    return {label: n}, df


def _save_results(n_by_period: dict, df: pd.DataFrame, mode: str, timeframe: str, threshold: float) -> None:
    out_dir = Path(os.path.abspath(".")) / "strategies" / "iterative" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # parquet con todas las velas verdes
    out_path = out_dir / f"green_spikes_after_signal_{mode}_{timeframe}.parquet"
    if not df.empty:
        df.to_parquet(out_path, index=False)
        print(f"\nVelas verdes guardadas → {out_path}")

    # metadata de señales totales por período
    meta = pd.DataFrame([{"period": k, "n_signals": v} for k, v in n_by_period.items()])
    meta_path = out_dir / f"green_spikes_after_signal_{mode}_{timeframe}_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"Metadata guardada      → {meta_path}")

    # CSV de ocurrencias grandes
    if not df.empty:
        big      = df[df["candle_size_pct"] >= threshold]
        csv_path = out_dir / f"green_spikes_after_signal_{mode}_{timeframe}_big.csv"
        big.to_csv(csv_path, index=False)
        print(f"Velas >= {threshold:.0f}% guardadas  → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza velas verdes después de la señal")
    parser.add_argument("--mode",      choices=["walkforward", "full", "up-to-date"], default="walkforward")
    parser.add_argument("--timeframe", choices=["5m", "15m"], default="5m")
    parser.add_argument("--gap-pct",   type=float, default=0.40)
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Umbral de tamaño para el resumen (default: 100)")
    args = parser.parse_args()

    print(f"\nbackside_short_lower_low_fix_stop_iterative — Velas verdes después de señal")
    print(f"mode={args.mode}  timeframe={args.timeframe}  gap_pct={args.gap_pct}  threshold={args.threshold}%\n")

    if args.mode == "walkforward":
        n_by_period, df = run_walkforward(args.timeframe, args.gap_pct, args.threshold)
    else:
        n_by_period, df = run_full(args.timeframe, args.gap_pct, args.threshold)

    total_signals = sum(n_by_period.values())
    if total_signals > 0:
        big              = df[df["candle_size_pct"] >= args.threshold] if not df.empty else pd.DataFrame()
        signals_with_big = big["signal_id"].nunique() if not big.empty else 0
        print(f"\n{'='*72}")
        print(f"  RESUMEN GLOBAL")
        print(f"{'='*72}")
        print(f"  Total señales:                   {total_signals}")
        print(f"  Señales con verde >= {args.threshold:.0f}%: {signals_with_big}  ({signals_with_big/total_signals*100:.2f}%)")
        if not df.empty:
            print(f"  Total velas verdes registradas:  {len(df)}")
            print(f"  Tamaño medio (todas verdes):     {df['candle_size_pct'].mean():.1f}%")
            print(f"  Percentil 90:                    {df['candle_size_pct'].quantile(0.90):.1f}%")
            print(f"  Percentil 95:                    {df['candle_size_pct'].quantile(0.95):.1f}%")
            print(f"  Máximo:                          {df['candle_size_pct'].max():.1f}%")

    _save_results(n_by_period, df, args.mode, args.timeframe, args.threshold)


if __name__ == "__main__":
    main()
