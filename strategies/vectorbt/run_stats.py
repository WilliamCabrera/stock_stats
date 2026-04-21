
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
from app.utils.trade_metrics import (analysis_and_plot, summary_report)



# trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/short_push_exhaustion/short_push_exhaustion_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# #print(trades[trades['pnl'] < 0])
# #print(trades)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.02)

# trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/short_push_exhaustion/short_push_exhaustion_out_of_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# #print(trades[trades['pnl'] < 0])
# #print(trades)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.02)

trades_path =  'backtest_dataset/full/15m/trades/backside_short_lower_low/backside_short_lower_low_full_15m_trades.parquet'
#trades_path =  'backtest_dataset/full/15m/trades/short_push_exhaustion/short_push_exhaustion_full_15m_trades.parquet'
#trades_path =  'backtest_dataset/full/5m/trades/gap_crap_strategy/gap_crap_strategy_full_5m_trades.parquet'
trades =  pd.read_parquet(trades_path)
#print(trades[trades['pnl'] < 0])
#print(trades)
analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.005)

# data_path =  'backtest_dataset/full/5m/full_dataset.parquet'
# data =  pd.read_parquet(data_path)
# print(data)