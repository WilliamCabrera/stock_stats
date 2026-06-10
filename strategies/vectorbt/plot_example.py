
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
from app.utils.trade_metrics import (analysis_and_plot, summary_report)

import strategies.vectorbt.small_caps  as sc 

sc.example_with_api_data(ticker="TDS", date="2023-08-04",timeframe="5m", ATR_FACTOR=[3.5, 2.0], strategy_func=sc.short_push_exhaustion)
