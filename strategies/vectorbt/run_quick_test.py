import os
import sys
sys.path.insert(0, os.path.abspath("."))

from datetime import datetime
from pathlib import Path
import pandas as pd
from app.utils.trade_metrics import analysis_and_plot, compute_mae_mfe_from_files, compute_mae_mfe_from_files_walkfordward

DATASET_ROOT = Path("backtest_dataset/full")
DATASET_ROOT_WF = Path("backtest_dataset/walkforward")


def main(timeframe='5m', strategy='backside_short_lower_low', variant="backside_short_lower_low_0.15_3.5_0.5", initial_capital=1000, risk_pct=0.01, return_filter=0.50):
   
   
    print(f'Running quick test with timeframe={timeframe}, strategy={strategy}, variant={variant}, initial_capital={initial_capital}, risk_pct={risk_pct}, return_filter={return_filter} ... ')

    # Load parquet
    path = DATASET_ROOT / timeframe / "trades" / strategy / f"{strategy}_full_{timeframe}_trades.parquet"
    trades = pd.read_parquet(path)
    
    grouped = trades.groupby("strategy")
    print(f"Available strategy variants: {grouped.groups.keys()}")
    
    trades = trades[trades["strategy"] == variant]
    #trades = trades[trades["Return"] > return_filter]
    
    winners = trades[trades['Return'] > 0]
    losers = trades[trades['Return'] < 0]
    
    print('--- Winners  ---')
    print(winners[['ticker', 'entry_price', 'exit_price', 'stop_loss_price', 'previous_day_close','entry_time', 'exit_time','Return','MAE','MFE']])
    print('------ lossers  ---')
    print(losers[['ticker', 'entry_price', 'exit_price', 'stop_loss_price', 'previous_day_close','entry_time', 'exit_time','Return','MAE','MFE']])

    print(f"\nRunning analysis_and_plot...")
    analysis_and_plot(trades=trades, initial_capital=initial_capital, risk_pct=risk_pct)


def testing_ticker(ticker):
    from strategies.vectorbt.small_caps import  backside_short_lower_low_fix_stop, backside_short_lower_low, gap_crap_strategy
    
    date = "2025-09-03"
    timeframe = "5m"
    
    file_apth =  DATASET_ROOT / timeframe / "tickers" / f"{ticker}.parquet"
    
    df_day = pd.read_parquet(file_apth)
    filtered = df_day[df_day['date_str'] == date]
    filtered['gap_pct'] = (filtered['open'] - filtered['previous_day_close']) / filtered['previous_day_close']
    
    #print(" ********** 50% gap or more ********** ")
    #print(filtered[filtered['gap_pct'] > 0.5][['ticker','date','date_str','open','close','previous_day_close','gap_pct']])
    
    filtered = filtered.set_index('date')
    f_dict ={ticker:filtered}
    
    #print(f"Running strategy on {ticker} for date {date} and timeframe {timeframe}...")
    #print(filtered.columns)
    #print(filtered[['ticker','date','date_str','open','close','previous_day_close','gap_pct']])
    trades = backside_short_lower_low_fix_stop(f_dict)
    trades['gap_pct'] = (trades['entry_price'] - trades['previous_day_close']) / trades['previous_day_close']
    print(trades[['ticker', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'stop_loss_price', 'previous_day_close','gap_pct', 'Return', 'strategy']])

if __name__ == "__main__":
    #backside_short_lower_low_fix_stop_0.15_0.4_0.5
    main(timeframe='5m', strategy='backside_short_lower_low_fix_stop', variant="backside_short_lower_low_fix_stop_0.2_0.5_0.5", initial_capital=1000, risk_pct=0.01, return_filter=0)
    #main(timeframe='5m', strategy='backside_short_lower_low', variant="backside_short_lower_low_0.15_3.5_0.5", initial_capital=1000, risk_pct=0.01, return_filter=0.50)
    #main(timeframe='5m', strategy='backside_short_lower_low', variant="backside_short_lower_low_0.15_3.5_0.5", initial_capital=1000, risk_pct=0.01, return_filter=0.40)
    #main(timeframe='5m', strategy='backside_short_lower_low', variant="backside_short_lower_low_0.15_3.5_0.5", initial_capital=1000, risk_pct=0.01, return_filter=0.0)
   
    # compute_mae_mfe_from_files(
    #     trades_path=DATASET_ROOT / "15m" / "trades" / "backside_short_lower_low_fix_stop" / "backside_short_lower_low_fix_stop_full_15m_trades.parquet",
    #     tickers_folder=DATASET_ROOT / "15m" / "tickers"

    # )
    # compute_mae_mfe_from_files(
    #     trades_path=DATASET_ROOT / "5m" / "trades" / "backside_short_lower_low_fix_stop" / "backside_short_lower_low_fix_stop_full_5m_trades.parquet",
    #     tickers_folder=DATASET_ROOT / "5m" / "tickers"

    # )
    
    # compute_mae_mfe_from_files_walkfordward(
    #     base_path=DATASET_ROOT_WF,
    #     strategy= "backside_short_lower_low"
    # )
    
    # compute_mae_mfe_from_files_walkfordward(
    #     base_path=DATASET_ROOT_WF,
    #     strategy= "backside_short_lower_low_fix_stop"
    # )
    # compute_mae_mfe_from_files_walkfordward(
    #     base_path=DATASET_ROOT_WF,
    #     strategy= "orb_short"
    # )
    # compute_mae_mfe_from_files_walkfordward(
    #     base_path=DATASET_ROOT_WF,
    #     strategy= "short_push_exhaustion"
    # )
    # compute_mae_mfe_from_files_walkfordward(
    #     base_path=DATASET_ROOT_WF,
    #     strategy= "gap_crap_strategy"
    # )
    
    #testing_ticker("AIHS")