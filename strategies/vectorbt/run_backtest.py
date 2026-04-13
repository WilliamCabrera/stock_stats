import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import time as tm
from pathlib import Path

from app.utils.market_utils import append_single_parquet
import strategies.vectorbt.small_caps  as sc 

#backside_short_lower_low, save_trades_to_file, sc.short_push_exhaustion, gap_crap_strategy


def run_backtest(path="backtest_dataset", sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=True):

    file_path = Path(f'{path}/{sample_type}.parquet')

    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return

    if len(df) > 0:
        folder_path = Path(f'{path}/trades/{strategy_fn.__name__}')
        folder_path.mkdir(parents=True, exist_ok=True)

        df['date'] = pd.to_datetime(df['date'])
        groups = df.groupby(['ticker', 'date_str'])

        print(f'Total of groups: {len(groups)}')

        counter = 0
        df_dict = {}
        index = 0
        total_trades = 0

        start_time = tm.perf_counter()

        # -----------------------------
        # MAIN LOOP
        # -----------------------------
        for (ticker, date_str), group in groups:

            group = group.set_index('date')
            len_group = len(group)

            if counter >= 100_000:
                index += 1
                print(
                    f'Processing backtest for {len(df_dict)} tickers '
                    f'at iteration {index}...'
                )

                trades = strategy_fn(df_dict)
                sc.save_trades_to_file(trades, file_path=f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trades.parquet', append=append_trades)
                total_trades += len(trades)
                print(f'Trades generated in iteration {index}: {len(trades)}')

                counter = 0
                df_dict = {}

            if len_group > 50:
                counter += len_group
                if ticker in df_dict:
                    gp = df_dict[ticker]
                    new_group = pd.concat([gp, group], ignore_index=False)
                    new_group.sort_index()
                    df_dict[ticker] = new_group
                else:
                    df_dict[ticker] = group

        # -----------------------------
        # FINAL FLUSH
        # -----------------------------
        if index == 0 and counter > 0 and counter <= 100_000:
            trades = strategy_fn(df_dict)
            save_trades_to_file(trades, file_path=f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trades.parquet', append=append_trades)
            total_trades += len(trades)
            print(f'Trades generated in iteration {index}: {len(trades)}')

        end_time = tm.perf_counter()

        print(
            f"⏰ Tiempo total {sample_type} ({strategy_fn.__name__}): "
            f"{end_time - start_time:.2f}s | "
            f"Total trades: {total_trades}"
        )

        print(f'Finalizing with {index} iterations')


if __name__ == "__main__":

    # 5m folds
    path1_5m = "backtest_dataset/walkforward/5m/fold_1"
    path2_5m = "backtest_dataset/walkforward/5m/fold_2"
    path3_5m = "backtest_dataset/walkforward/5m/fold_3"

    # run_backtest(path=path1_5m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    # run_backtest(path=path2_5m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    # run_backtest(path=path3_5m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)

    # run_backtest(path=path1_5m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    # run_backtest(path=path2_5m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    # run_backtest(path=path3_5m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)

    run_backtest(path=path1_5m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path2_5m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path3_5m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)

    run_backtest(path=path1_5m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path2_5m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path3_5m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)


    run_backtest(path=path1_5m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path2_5m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path3_5m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)

    run_backtest(path=path1_5m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path2_5m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path3_5m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)


    # 15m folds
    path1_15m = "backtest_dataset/walkforward/15m/fold_1"
    path2_15m = "backtest_dataset/walkforward/15m/fold_2"
    path3_15m = "backtest_dataset/walkforward/15m/fold_3"

    run_backtest(path=path1_15m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    run_backtest(path=path2_15m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    run_backtest(path=path3_15m, sample_type="in_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)

    run_backtest(path=path1_15m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    run_backtest(path=path2_15m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)
    run_backtest(path=path3_15m, sample_type="out_of_sample", strategy_fn=sc.backside_short_lower_low, append_trades=False)

    run_backtest(path=path1_15m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path2_15m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path3_15m, sample_type="in_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)

    run_backtest(path=path1_15m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path2_15m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)
    run_backtest(path=path3_15m, sample_type="out_of_sample", strategy_fn=sc.short_push_exhaustion, append_trades=False)

    run_backtest(path=path1_15m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path2_15m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path3_15m, sample_type="in_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)

    run_backtest(path=path1_15m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path2_15m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
    run_backtest(path=path3_15m, sample_type="out_of_sample", strategy_fn=sc.gap_crap_strategy, append_trades=False)
