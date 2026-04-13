import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

import numpy as np
#from utils import helpers
#from utils import helpers as utils_helpers, trade_metrics as tme
import itertools
import time as tm
from functools import wraps
#from small_caps_strategies import commons
from pprint import pprint
from pathlib import Path
from app.utils.market_utils import  append_single_parquet
from app.utils.charts import plot_candles_df, trades_to_markers
from app.utils.indicators import compute_close_atr_band, compute_vwap
from app.utils.massive import fetch_candles
from app.utils.trade_metrics import (analysis_and_plot, summary_report)


# ====== helper functions ======

def prepare_params_and_vectors(f_dict, tp_list, sl_list ):
    
    all_params = {}
    assert len(tp_list) == len(sl_list)
    tp_sl_pairs = list(zip(tp_list, sl_list))
    n_params = len(tp_sl_pairs)
    

    # --------------------------------------------------
    # 1. Construir índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])

    for df in f_dict.values():
        index_master = index_master.union(df.index)

    index_master = pd.to_datetime(index_master.sort_values())

    tickers = list(f_dict.keys())
    n_tickers = len(tickers)
    n_cols = n_tickers * n_params
    n_bars = len(index_master)
    

    # --------------------------------------------------
    # 2. Crear arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)
    atr_arr   = np.full_like(open_arr, np.nan)
    volume_arr = np.full_like(open_arr, np.nan)
    rvol_arr   = np.full_like(open_arr, np.nan)
    prev_day_close_arr = np.full_like(open_arr, np.nan)
    exhaustion_score_arr = np.full_like(open_arr, np.nan)
    sma_volume_20_5m_arr = np.full_like(open_arr, np.nan)
    vwap_arr = np.full_like(open_arr, np.nan)
    
    col = 0
    col_meta = []   # para mapear trades → parámetros
    
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)

        for tp, sl in tp_sl_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'sl': sl,
                'column': col    
            })

            col += 1
    all_params.update({
        "n_params":n_params,
        "tp_sl_pairs":tp_sl_pairs,
        "index_master":index_master,
        "n_tickers":n_tickers,
        "n_cols":n_cols,
        "n_bars": n_bars,
        "open_arr":open_arr,
        "high_arr":high_arr,
        "low_arr":low_arr,
        "close_arr":close_arr,
        "atr_arr": atr_arr,
        "volume_arr": volume_arr,
        "rvol_arr":rvol_arr,
        "prev_day_close_arr": prev_day_close_arr,
        "sma_volume_20_5m_arr":sma_volume_20_5m_arr,
        "vwap_arr": vwap_arr,
        "col":col,
        "col_meta":col_meta
    })
    
    return all_params

# gap % is passed as parameter
def prepare_params_and_vectors_for_gappers(f_dict, gap_list =[], tp_list = [], sl_list= []):
    
    # Verificar que todas tengan el mismo tamaño
    assert len(gap_list) == len(tp_list) == len(sl_list)
    
    # Creamos la lista de pares “paralelos”
    tp_sl_gap_pairs = list(zip(tp_list, sl_list, gap_list))
    n_params = len(tp_sl_gap_pairs)
    

    # --------------------------------------------------
    # 1. Construir índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])

    for df in f_dict.values():
        index_master = index_master.union(df.index)

    index_master = pd.to_datetime(index_master.sort_values())

    tickers = list(f_dict.keys())
    n_tickers = len(tickers)
    n_cols = n_tickers * n_params
    n_bars = len(index_master)
    

    # --------------------------------------------------
    # 2. Crear arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)
    atr_arr   = np.full_like(open_arr, np.nan)
    volume_arr = np.full_like(open_arr, np.nan)
    rvol_arr   = np.full_like(open_arr, np.nan)
    prev_day_close_arr = np.full_like(open_arr, np.nan)
    exhaustion_score_arr = np.full_like(open_arr, np.nan)
    sma_volume_20_5m_arr = np.full_like(open_arr, np.nan)
    vwap_arr = np.full_like(open_arr, np.nan)
    
    col = 0
    col_meta = []   # para mapear trades → parámetros
    
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)

        for tp, sl, gap in tp_sl_gap_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'sl': sl,
                'gap': gap,
                'column': col    
            })

            col += 1
    
    max_entry_time =  '15:45'
    # # -----------------------
    # # Filtro horario (<15:45)
    # # -----------------------
    # time_mask = np.array([
    #     t.strftime("%H:%M") < max_entry_time
    #     for t in index_master
    # ])
    
    hours   = np.array([t.hour for t in index_master])
    minutes = np.array([t.minute for t in index_master])

    time_mask = (
        ( (hours > 8) | ((hours == 8) & (minutes >= 15))) &
        ((hours < 15) | ((hours == 15) & (minutes <= 45)))
    )
    
    time_mask = (
        # 04:00 - 07:59
        (
            (hours >= 4) & (hours < 8)
        )
        |
        # 08:30 - 14:00
        (
            ((hours == 8) & (minutes >= 30)) |
            ((hours > 8) & (hours < 14)) |
            ((hours == 14) & (minutes == 0))
        )
    )
    
    
    all_params ={}
    all_params.update({
        "n_params":n_params,
        "tp_sl_gap_pairs":tp_sl_gap_pairs,
        "index_master":index_master,
        "n_tickers":n_tickers,
        "n_cols":n_cols,
        "n_bars": n_bars,
        "open_arr":open_arr,
        "high_arr":high_arr,
        "low_arr":low_arr,
        "close_arr":close_arr,
        "atr_arr": atr_arr,
        "volume_arr": volume_arr,
        "rvol_arr":rvol_arr,
        "prev_day_close_arr": prev_day_close_arr,
        "sma_volume_20_5m_arr":sma_volume_20_5m_arr,
        "vwap_arr": vwap_arr,
        "col":col,
        "col_meta":col_meta,
        "max_entry_time": max_entry_time,
        "time_mask":time_mask
    })
    
    return all_params

# this add dochain channel 5 bars, offset 1 for using as trailing stop
def prepare_params_and_vectors_for_gappers_with_trailing(f_dict, gap_list =[], tp_list = []):
    
    # Verificar que todas tengan el mismo tamaño
    assert len(gap_list) == len(tp_list)
    
    # Creamos la lista de pares “paralelos”
    tp_sl_gap_pairs = list(zip(tp_list, gap_list))
    n_params = len(tp_sl_gap_pairs)
    

    # --------------------------------------------------
    # 1. Construir índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])

    for df in f_dict.values():
        index_master = index_master.union(df.index)

    index_master = pd.to_datetime(index_master.sort_values())

    tickers = list(f_dict.keys())
    n_tickers = len(tickers)
    n_cols = n_tickers * n_params
    n_bars = len(index_master)
    

    # --------------------------------------------------
    # 2. Crear arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)
    atr_arr   = np.full_like(open_arr, np.nan)
    volume_arr = np.full_like(open_arr, np.nan)
    rvol_arr   = np.full_like(open_arr, np.nan)
    prev_day_close_arr = np.full_like(open_arr, np.nan)
    exhaustion_score_arr = np.full_like(open_arr, np.nan)
    sma_volume_20_5m_arr = np.full_like(open_arr, np.nan)
    vwap_arr = np.full_like(open_arr, np.nan)
    donchian_upper_arr = np.full_like(open_arr, np.nan)
    donchian_basis_arr = np.full_like(open_arr, np.nan)
    donchian_lower_arr = np.full_like(open_arr, np.nan)
    
    col = 0
    col_meta = []   # para mapear trades → parámetros
    
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)

        for tp, gap in tp_sl_gap_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            donchian_upper_arr[idx, col] = df.loc[idx, 'donchian_upper'].values
            donchian_basis_arr[idx, col] = df.loc[idx, 'donchian_basis'].values
            donchian_lower_arr[idx, col] = df.loc[idx, 'donchian_lower'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'gap': gap,
                'column': col    
            })

            col += 1
    
    max_entry_time =  '15:45'
    # # -----------------------
    # # Filtro horario (<15:45)
    # # -----------------------
    # time_mask = np.array([
    #     t.strftime("%H:%M") < max_entry_time
    #     for t in index_master
    # ])
    
    hours   = np.array([t.hour for t in index_master])
    minutes = np.array([t.minute for t in index_master])

    time_mask = (
        ( (hours > 8) | ((hours == 8) & (minutes >= 15))) &
        ((hours < 15) | ((hours == 15) & (minutes <= 45)))
    )
    
    time_mask = (
        # 04:00 - 07:59
        (
            (hours >= 4) & (hours < 8)
        )
        |
        # 08:30 - 14:00
        (
            ((hours == 8) & (minutes >= 30)) |
            ((hours > 8) & (hours < 14)) |
            ((hours == 14) & (minutes == 0))
        )
    )
    
    
    all_params ={}
    all_params.update({
        "n_params":n_params,
        "tp_sl_gap_pairs":tp_sl_gap_pairs,
        "index_master":index_master,
        "n_tickers":n_tickers,
        "n_cols":n_cols,
        "n_bars": n_bars,
        "open_arr":open_arr,
        "high_arr":high_arr,
        "low_arr":low_arr,
        "close_arr":close_arr,
        "atr_arr": atr_arr,
        "volume_arr": volume_arr,
        "rvol_arr":rvol_arr,
        "prev_day_close_arr": prev_day_close_arr,
        "sma_volume_20_5m_arr":sma_volume_20_5m_arr,
        "vwap_arr": vwap_arr,
        "col":col,
        "col_meta":col_meta,
        "max_entry_time": max_entry_time,
        "time_mask":time_mask,
        "donchian_upper_arr": donchian_upper_arr,   
        "donchian_basis_arr": donchian_basis_arr,
        "donchian_lower_arr": donchian_lower_arr
    })
    
    return all_params


# this prepare the trades dataframe given by vectorbt after when running the backtest
def modify_trades_columns(params=None, strategy_name_prefix = "strategy"):
    if params is None:
        return pd.DataFrame([])
    
    trades = params['trades']
    col_meta = params['col_meta']
    index_master = params['index_master']
    atr_arr = params['atr_arr']
    rvol_arr = params['rvol_arr']
    prev_day_close_arr = params['prev_day_close_arr']
    volume_arr = params['volume_arr']
    
    trades = trades = (
    trades
    .replace([np.inf, -np.inf], np.nan)
    .dropna()).copy()

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades['strategy'] = f'{strategy_name_prefix}_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]

    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
  
# this prepare the trades dataframe given by vectorbt after when running the backtest  but includes donchain channels data
def modify_trades_columns_trailing(params=None, strategy_name_prefix = "strategy"):
    if params is None:
        return pd.DataFrame([])
    
    trades = params['trades']
    col_meta = params['col_meta']
    index_master = params['index_master']
    atr_arr = params['atr_arr']
    rvol_arr = params['rvol_arr']
    prev_day_close_arr = params['prev_day_close_arr']
    volume_arr = params['volume_arr']
    donchian_upper_arr = params['donchian_upper_arr']
    
    trades = trades = (
    trades
    .replace([np.inf, -np.inf], np.nan)
    .dropna()).copy()

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades['strategy'] = f'{strategy_name_prefix}_'+ trades['tp'].astype(str) + "_"+ trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = donchian_upper_arr[entry_idx, col_idx] 
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]

    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
    

def removing_imposible_trades(trades):
    """
    Quitando todos los trades que no debieron ocurrir.
    Sucede: cuando hay hueco en los datos y el backtest no logra cerrar el trade:
    Ejemplo; ticker SANA, 2022-06-22, no tiene datos entre 15:45 y 16:05. esto provoco que el trade se cierre al dia siguiente
    
    normalmente deberia cerrarse en la primera vela con datos en  el caso de:  ticker SANA, 2022-06-22, habian datos a las 16:05 
    pero tenemos filtros que solo ejecutan operaciones en intervalos especificos (pre-market y regular hours)
    """
    
    # trades dentro del mismo dia.(Intradia), nunca se hace hold overnight
    return trades[trades["entry_time"].dt.normalize() == trades["exit_time"].dt.normalize()]


def reduce_trades_columns(trades):
    """ filter to obtain relevants columns

    Args:
        trades (dataframe): trades 

    Returns:
        _type_: _description_
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame([])
    
    trades = removing_imposible_trades(trades)
    
    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
  
  
def generate_signal_to_force_close_EOD(forced_exit, index_master):
    
    """generate signals to close trades at 15:50 at latest

    Returns:
        numpy array[bool, ..]: true at the position where time is 15:50
    """
    dates = pd.to_datetime(index_master.date)  # fecha sin hora
    unique_dates = np.unique(dates)

    for d in unique_dates:
        # tomamos todas las barras del día
        mask_day = dates == d
        
        # todas las barras del día que sean <= 15:50
        mask_before_1550 = mask_day & (index_master.time <= pd.to_datetime("15:50").time())
        
        if np.any(mask_before_1550):
            # cerramos en la última barra antes de las 15:50
            last_bar_idx = np.where(mask_before_1550)[0][-1]
        else:
            # si no hay barra antes de 15:50, cerrar en la última del día
            last_bar_idx = np.where(mask_day)[0][-1]
            
        forced_exit[last_bar_idx, :] = True  # cerramos todas las posiciones abiertas

        
    return forced_exit

def save_trades_to_file(trades, file_path="vectorbt_trades", append=True):
    """
    save trades grouped by strategies, meaning that if there are different names in strategy column
    it will create a file for each strategy and save the trades as separated strategies.
    Args:
        trades (dataframe): trades
        file_path (str, optional): path to target file. Defaults to vectorbt_trades"
        append (bool, optional): if append to an existing file. Defaults to True.
    """
    if trades is None or isinstance(trades, pd.DataFrame) == False or len(trades) == 0 :
        return
    
    if 'strategy' in trades.columns:
        
        # Agrupar por la columna 'strategy'
        grouped = trades.groupby('strategy')

        # Iterar por cada grupo
        for strategy_name, group_df in grouped:
            #print("Strategy:", strategy_name)
            #print(group_df)  # Aquí tienes el DataFrame solo de esa estrategia
            if append:
                append_single_parquet(df=group_df, path=file_path)
            else:  
                group_df.to_parquet(path=file_path)
        
    else:
        print("====== el dataframe trades no tiene la column: strategy , la cual contiene el nombre de la estrategia que se esta probando")
    
    
    
    return


# ====== short strategies =======

def gap_crap_strategy(f_dict):
    """
    short at the open 
    1 take profit at 15%
    1 stop loss at 3.5 ATR
    params: f_dict: {ticker: df_5m, ...}
    """
    
    #print("Starting gap_crap_strategy backtest...")

    tp_list = [0.15, 0.20, 1.00, 0.15, 0.20, 1.00]       # TP relativos
    sl_list = [3.50, 3.50, 3.50, 2.00, 2.00, 2.00]         # SL en múltiplos de ATR
    gap_list =[0.50, 0.50, 0.50, 0.50, 0.50, 0.50]         # GAPs list to 
    
    all_params = prepare_params_and_vectors_for_gappers(f_dict,gap_list, tp_list, sl_list)
   
    tp_sl_gap_pairs = all_params['tp_sl_gap_pairs']
    n_params = all_params['n_params']
    index_master = all_params['index_master']
    n_tickers = all_params['n_tickers']
    n_cols = all_params['n_cols']
    n_bars = all_params['n_bars']
    open_arr  = all_params['open_arr']
    high_arr  = all_params['high_arr']
    low_arr   = all_params['low_arr']
    close_arr = all_params['close_arr']
    atr_arr   = all_params['atr_arr']
    volume_arr = all_params['volume_arr']
    rvol_arr   = all_params['rvol_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    col = all_params['col']
    col_meta = all_params['col_meta']
    
    # --------------------------------------------------
    # Inicializar entradas
    # --------------------------------------------------
    entries = np.zeros((n_bars, n_cols), dtype=bool)

    # Usamos 9:25 para evitar lookahead bias (close 9:25 = open 9:30)
    mask_930 = np.array([t.strftime("%H:%M") == "09:25" for t in index_master])

    # --------------------------------------------------
    # Gap % vs previous_day_close
    # --------------------------------------------------
    gap_vals = np.array([m["gap"] for m in col_meta])

    gap_pct = np.divide(
        close_arr - prev_day_close_arr,
        prev_day_close_arr,
        out=np.zeros_like(close_arr, dtype=float),
        where=prev_day_close_arr > 0
    )

    # Condición de gap: cada columna usa su GAP específico
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

    # Opcionales: volumen y rvol
    vol_cond  = volume_arr > 40_000
    rvol_cond = rvol_arr >= 3

    # --------------------------------------------------
    # Entradas
    # --------------------------------------------------
    entries[mask_930, :] = (
        ~np.isnan(open_arr[mask_930, :]) &
        gap_cond[mask_930, :] &
        vol_cond[mask_930, :] &
        rvol_cond[mask_930, :]
    )

    # --------------------------------------------------
    # FIX look-ahead: fill at the OPEN of the NEXT bar (= the actual 9:30 open).
    # The signal is detected at the 9:25 bar close; the earliest realistic fill
    # is the open of the following bar.
    # --------------------------------------------------
    next_open = np.roll(open_arr, -1, axis=0)
    next_open[-1, :] = np.nan   # last bar has no next bar
    entries[-1, :]   = False

    # --------------------------------------------------
    # TAKE PROFIT y STOP LOSS  (based on next_open = actual fill price)
    # --------------------------------------------------
    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_price = next_open * (1 - tp_vals)
    sl_price = next_open + sl_vals * atr_arr

    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[mask_930, :] = (next_open[mask_930, :] - tp_price[mask_930, :]) / next_open[mask_930, :]
    sl_stop[mask_930, :] = (sl_price[mask_930, :] - next_open[mask_930, :]) / next_open[mask_930, :]

    forced_exit = np.zeros_like(entries, dtype=bool)
    forced_exit = generate_signal_to_force_close_EOD(forced_exit, index_master)

    pf = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        entries=entries,
        price=next_open,        # fill at next bar's open (9:30 open)
        exits=forced_exit,
        size=1,
        direction='shortonly',
        tp_stop=tp_stop,
        sl_stop=sl_stop,
        init_cash=0,
        freq='5min'
    )
        
    trades = pf.trades.records_readable

    trades = trades = (
    trades
    .replace([np.inf, -np.inf], np.nan)
    .dropna()).copy()

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades['strategy'] = f'gap_and_crap_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]
    
   
    return reduce_trades_columns(trades)   


def backside_short(f_dict):
    """
    Short cuando el momentum comienza a fallar (backside)
    condiciones:
    - vela previa verde
    - vela actual roja (cuerpo mayor cuerpo de la vela previa)
    - lower low (cierre actual < mínimo previo) o upper tail grande (vela roja)
    - gap mínimo vs previous day close
    1 take profit at 15%
    1 stop loss at 3.5 ATR from close of signal
    params: f_dict: {ticker: df_5m, ...}
    
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 1.00, 1.00]        # TP relativos (short)
        sl_list  = [2.00, 3.50, 2.00, 3.50, 2.00, 3.50, 2.00, 3.50]          # SL en múltiplos de ATR
        gap_list = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]          # Gap mínimo vs prev day close

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers(
            f_dict,
            gap_list,
            tp_list,
            sl_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']

        # ==================================================
        # INICIALIZAR ENTRADAS
        # ==================================================
        entries = np.zeros((n_bars, n_cols), dtype=bool)

        # ==================================================
        # VELAS (VECTORIAL)
        # ==================================================
        red_curr = close_arr < open_arr

        prev_open  = np.roll(open_arr, 1, axis=0)
        prev_close = np.roll(close_arr, 1, axis=0)
        prev_high  = np.roll(high_arr, 1, axis=0)
        prev_low   = np.roll(low_arr, 1, axis=0)

        green_prev = prev_close > prev_open
        
        # ================================================
        # VELA ACTUAL MAYOR QUE VELA PREVIA         
        # =================================================
        prev_open_2  = np.roll(open_arr, 2, axis=0)
        prev_close_2 = np.roll(close_arr, 2, axis=0)
        prev_high_2  = np.roll(high_arr, 2, axis=0)
        prev_low_2   = np.roll(low_arr, 2, axis=0)

        green_prev_2 = prev_close_2 > prev_open_2

        # ==================================================
        # LOWER LOW (definición exacta)
        # ==================================================
        lower_low_1 = (
            green_prev &
            red_curr &
            (close_arr < prev_low)
        )
        
        lower_low_2 = (
            green_prev_2 &
            red_curr &
            (close_arr < prev_low_2)
        )

        lower_low = lower_low_1 | lower_low_2
        
        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE (POR COLUMNA)
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)
        
        gap_pct_with_high = np.divide(
            high_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(high_arr, dtype=float),
            where=prev_day_close_arr > 0
        )
        
        gap_cond_for_tail = (gap_pct_with_high >= gap_vals) & (gap_pct_with_high <= 5.0)
        
        #print("Gap condition calculated")   
        #print(gap_cond)
        
      
        
        # ==================================================
        # UPPER TAIL GRANDE (vela roja)
        # ==================================================
        body = np.abs(close_arr - open_arr)
        upper_tail = high_arr - np.maximum(open_arr, close_arr)

        big_upper_tail = (
            red_curr &
            (upper_tail >= 5.0 * body) & 
            (prev_high < high_arr)
        ) 

        gap_filter = np.where(
            big_upper_tail,
            gap_cond_for_tail,   # cuando hay upper tail
            gap_cond             # caso contrario
        )
        
        # ==================================================
        # CONDICION: PRECIO > VWAP
        # ==================================================
        
    
        is_above_vwap = close_arr > vwap_arr
       
        #print("is_above_vwap condition calculated")   
        #print(is_above_vwap)

        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_filter &
            is_above_vwap &
            time_mask[:, None] &
            (
                lower_low |
                big_upper_tail
            )
        )

        # ==================================================
        # TAKE PROFIT / STOP LOSS (POR COLUMNA)
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])
    
        tp_price = close_arr * (1 - tp_vals)        # short
        sl_price = close_arr + sl_vals * atr_arr    # SL ATR
       

        tp_stop = np.full_like(open_arr, np.nan)
        tp_stop[entries] = (close_arr[entries] - tp_price[entries]) / close_arr[entries]
        
        sl_stop = np.full_like(close_arr, np.nan)
        sl_stop[entries] = (
            (sl_price[entries] - close_arr[entries]) 
            / close_arr[entries]
        )

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            #price=open_arr,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )
        
        trades = pf.trades.records_readable
        trades = trades = (
        trades
        .replace([np.inf, -np.inf], np.nan)
        .dropna()).copy()

        trades = trades.rename(columns={
            'Avg Entry Price': 'entry_price',
            'Avg Exit Price': 'exit_price',
            'PnL': 'pnl',
            'Direction':'type'
        })
        
        col_meta_df = pd.DataFrame(col_meta).set_index('column')
        trades = trades.join(col_meta_df, on='Column')
        trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
        trades['strategy'] = f'backside_short_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 
        
    except Exception as e:
        print(" error found in   --- backside_short --- ")
        print(e)
        return pd.DataFrame([])
    

def backside_short_lower_low(f_dict):
    """
    Short cuando el momentum comienza a fallar (backside)
    condiciones:
    - vela previa verde
    - vela actual roja (cuerpo mayor cuerpo de la vela previa)
    - lower low (cierre actual < mínimo previo) 
    - gap mínimo vs previous day close
    1 take profit at 15%
    1 stop loss at 3.5 ATR from close of signal
    params: f_dict: {ticker: df_5m, ...}
    
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 1.00, 1.00]        # TP relativos (short)
        sl_list  = [2.00, 3.50, 2.00, 3.50, 2.00, 3.50, 2.00, 3.50]          # SL en múltiplos de ATR
        gap_list = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]          # Gap mínimo vs prev day close

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers(
            f_dict,
            gap_list,
            tp_list,
            sl_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']

        # ==================================================
        # INICIALIZAR ENTRADAS
        # ==================================================
        entries = np.zeros((n_bars, n_cols), dtype=bool)

        # ==================================================
        # VELAS (VECTORIAL)
        # ==================================================
        red_curr = close_arr < open_arr

        # FIX: mask cross-day comparisons so np.roll never leaks the last bar
        # of day N into the first bar of day N+1.
        dates_arr  = np.array([t.date() for t in index_master])
        prev_dates = np.roll(dates_arr, 1)
        same_day   = (dates_arr == prev_dates)          # False at the first bar of every day

        prev_open  = np.where(same_day[:, None], np.roll(open_arr,  1, axis=0), np.nan)
        prev_close = np.where(same_day[:, None], np.roll(close_arr, 1, axis=0), np.nan)
        prev_high  = np.where(same_day[:, None], np.roll(high_arr,  1, axis=0), np.nan)
        prev_low   = np.where(same_day[:, None], np.roll(low_arr,   1, axis=0), np.nan)

        green_prev = prev_close > prev_open

        # ==================================================
        # LOWER LOW (definición exacta)
        # ==================================================
        lower_low = (
            green_prev &
            red_curr &
            (close_arr < prev_low)
        )


        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE (POR COLUMNA)
        # FIX: use open_arr instead of close_arr — the open is known at bar start,
        # using close would require the bar to finish first (look-ahead).
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            open_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(open_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

        # ==================================================
        # CONDICION: PRECIO > VWAP
        # ==================================================
        is_above_vwap = open_arr > vwap_arr

        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_cond &
            is_above_vwap &
            time_mask[:, None] &
            lower_low
        )

        # ==================================================
        # FIX look-ahead bias: enter at the OPEN of the NEXT bar, not at the
        # close of the signal bar.  The signal is detected when bar i closes,
        # so the earliest realistic fill is bar i+1's open.
        # ==================================================
        next_open = np.roll(open_arr, -1, axis=0)
        next_open[-1, :] = np.nan   # last bar has no next bar
        entries[-1, :]   = False    # can't trade without a next bar

        # ==================================================
        # TAKE PROFIT / STOP LOSS (POR COLUMNA)
        # Levels computed from next_open (actual fill price).
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])

        tp_price = next_open * (1 - tp_vals)
        sl_price = next_open + sl_vals * atr_arr

        tp_stop = np.full_like(open_arr, np.nan)
        tp_stop[entries] = (next_open[entries] - tp_price[entries]) / next_open[entries]

        sl_stop = np.full_like(close_arr, np.nan)
        sl_stop[entries] = (sl_price[entries] - next_open[entries]) / next_open[entries]

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            price=next_open,        # fill at next bar's open
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )
        
        trades = pf.trades.records_readable
        trades = trades = (
        trades
        .replace([np.inf, -np.inf], np.nan)
        .dropna()).copy()

        trades = trades.rename(columns={
            'Avg Entry Price': 'entry_price',
            'Avg Exit Price': 'exit_price',
            'PnL': 'pnl',
            'Direction':'type'
        })
        
        col_meta_df = pd.DataFrame(col_meta).set_index('column')
        trades = trades.join(col_meta_df, on='Column')
        trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
        trades['strategy'] = f'backside_short_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 
        
    except Exception as e:
        print(" error found in   --- backside_short --- ")
        print(e)
        return pd.DataFrame([])


def short_push_exhaustion(f_dict):
    """
    Short on exhaustion candle (topping tail + volume spike).

    Entry conditions:
      - Gap >= 50% vs previous day close (gapped-up stock)
      - Red candle  (close < open)
      - Upper tail  >= 1.5 × candle body  (topping/rejection wick)
      - Volume      >= 1.3 × average of previous 4 bars' volume
      - Price (open) > VWAP

    Entries fill at the OPEN of the next bar (no look-ahead bias).
    TP / SL are relative to that fill price.
    """
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 1.00, 1.00]
        sl_list  = [2.00, 3.50, 2.00, 3.50, 2.00, 3.50, 2.00, 3.50]
        gap_list = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers(f_dict, gap_list, tp_list, sl_list)

        index_master       = all_params['index_master']
        n_cols             = all_params['n_cols']
        n_bars             = all_params['n_bars']
        open_arr           = all_params['open_arr']
        high_arr           = all_params['high_arr']
        low_arr            = all_params['low_arr']
        close_arr          = all_params['close_arr']
        atr_arr            = all_params['atr_arr']
        volume_arr         = all_params['volume_arr']
        rvol_arr           = all_params['rvol_arr']
        prev_day_close_arr = all_params['prev_day_close_arr']
        col_meta           = all_params['col_meta']
        vwap_arr           = all_params['vwap_arr']
        time_mask          = all_params['time_mask']

        # ==================================================
        # MASCARA DE LIMITE DE DÍA (evita que np.roll cruce dias)
        # ==================================================
        dates_arr = np.array([t.date() for t in index_master])

        def _same_day_mask(shift):
            return (dates_arr == np.roll(dates_arr, shift))

        same_day_1 = _same_day_mask(1)
        same_day_2 = _same_day_mask(2)
        same_day_3 = _same_day_mask(3)
        same_day_4 = _same_day_mask(4)

        # ==================================================
        # SEÑAL 1: VELA ROJA CON TOPPING TAIL
        # ==================================================
        red_curr   = close_arr < open_arr
        body       = open_arr - close_arr                          # > 0 for red candle
        upper_tail = high_arr - np.maximum(open_arr, close_arr)   # wick above body top
        lower_tail = np.minimum(open_arr, close_arr) - low_arr    # wick below body bottom
        # upper tail >= 1.5x body AND lower tail (if any) is at least 1.5x smaller than upper tail
        topping_tail_cond = (
            red_curr &
            (upper_tail >= 1.5 * body) &
            (lower_tail * 1.5 <= upper_tail)
        )

        # ==================================================
        # SEÑAL 2: VOLUMEN 1.3x MAYOR QUE PROMEDIO 4 BARRAS PREVIAS
        # ==================================================
        prev_vol_1 = np.where(same_day_1[:, None], np.roll(volume_arr, 1, axis=0), np.nan)
        prev_vol_2 = np.where(same_day_2[:, None], np.roll(volume_arr, 2, axis=0), np.nan)
        prev_vol_3 = np.where(same_day_3[:, None], np.roll(volume_arr, 3, axis=0), np.nan)
        prev_vol_4 = np.where(same_day_4[:, None], np.roll(volume_arr, 4, axis=0), np.nan)

        vol_stack   = np.stack([prev_vol_1, prev_vol_2, prev_vol_3, prev_vol_4], axis=0)
        with np.errstate(all="ignore"):
            vol_avg_4 = np.nanmean(vol_stack, axis=0)             # NaN where no prior same-day bars
        volume_cond = volume_arr >= 1.3 * vol_avg_4

        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE  (usa open — sin look-ahead)
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])
        gap_pct  = np.divide(
            open_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(open_arr, dtype=float),
            where=prev_day_close_arr > 0,
        )
        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

        # ==================================================
        # CONDICION: PRECIO > VWAP
        # ==================================================
        is_above_vwap = open_arr > vwap_arr

        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_cond         &
            is_above_vwap    &
            time_mask[:, None] &
            topping_tail_cond &
            volume_cond
        )

        # ==================================================
        # FIX look-ahead: fill al OPEN de la barra siguiente
        # ==================================================
        next_open          = np.roll(open_arr, -1, axis=0)
        next_open[-1, :]   = np.nan
        entries[-1, :]     = False

        # ==================================================
        # TAKE PROFIT / STOP LOSS
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])

        tp_price = next_open * (1 - tp_vals)
        sl_price = next_open + sl_vals * atr_arr

        tp_stop = np.full_like(open_arr, np.nan)
        tp_stop[entries] = (next_open[entries] - tp_price[entries]) / next_open[entries]

        sl_stop = np.full_like(close_arr, np.nan)
        sl_stop[entries] = (sl_price[entries] - next_open[entries]) / next_open[entries]

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(forced_exit, index_master)

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            price=next_open,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            size=1,
            init_cash=0,
            freq='5min',
        )

        trades = pf.trades.records_readable
        trades = (trades.replace([np.inf, -np.inf], np.nan).dropna()).copy()

        trades = trades.rename(columns={
            'Avg Entry Price': 'entry_price',
            'Avg Exit Price':  'exit_price',
            'PnL':             'pnl',
            'Direction':       'type',
        })

        col_meta_df = pd.DataFrame(col_meta).set_index('column')
        trades = trades.join(col_meta_df, on='Column')
        trades = trades.replace([np.inf, -np.inf], np.nan).dropna()

        trades['strategy']     = 'short_push_exhaustion_' + trades['tp'].astype(str) + '_' + trades['sl'].astype(str) + '_' + trades['gap'].astype(str)
        trades['entry_time']   = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']    = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values

        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price']   = trades['entry_price'] + trades['sl'] * atr_entry
        trades['rvol_daily']        = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close']= prev_day_close_arr[entry_idx, col_idx]
        trades['volume']            = volume_arr[entry_idx, col_idx]

        return reduce_trades_columns(trades)

    except Exception as e:
        print(" error found in   --- short_push_exhaustion --- ")
        print(e)
        return pd.DataFrame([])


# ========= test =======
# usa datos del parquet local
def exemple_with_local_data():
    
    
    date_str0 =  "2022-07-21"
    ticker =  'ADXN'
    
    date_str0 =  "2022-06-28"
    ticker =  'AGRX'
    
    # date_str0 =  "2025-12-31"
    # ticker =  'INBS'
    
    # date_str0 =  "2026-01-05"
    # ticker =  'INBS'
    
    df0  = pd.read_parquet('backtest_dataset/in_sample/gappers_backtest_dataset_5min_in_sample.parquet')
    df0 = df0.dropna(
        subset=["donchian_upper", "donchian_lower", "donchian_basis"]
    )
    
    # df0["date"].dt.date == pd.to_datetime(date_str0).date()
    df_day_0 = df0[(df0['ticker'] == ticker) & (df0["date"].dt.date == pd.to_datetime(date_str0).date()) ]
    df_day_0['date'] = pd.to_datetime(df_day_0['date'])
    df1 =  df_day_0.copy()
   
    df1 = df1.drop(columns=["time"])
    
    df_day_0 = df_day_0.set_index('date')
   
    df =  df_day_0[['ticker','date_str','open','high','low','close', 'volume', 'donchian_upper', 'donchian_lower', 'donchian_basis', 'previous_day_close']]
    print(df)
    #print(df_day_0.between_time("9:30","16:00"))
    f_dict ={ticker:df_day_0}
    
    
    #trades = gap_crap_strategy(f_dict)
    trades= backside_short_lower_low(f_dict)
    # trades = short_exhaustion_strategy(f_dict)
    # trades = short_vwap_pop_strategy(f_dict)
    # trades = short_explosives_pops(f_dict)
    # trades = backside_short(f_dict)
    # trades = backside_short_tp_dchain_stop(f_dict)
    # trades = small_range_breakout_long_strategy(f_dict)
    #trades = small_range_breakout_long_strategy_with_tp_factor(f_dict)
    
    # save_trades(trades)
    print(trades[['ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return','entry_time','exit_time','strategy']])
  

def example_with_api_data(ticker: str, date: str,timeframe: str| None = None, ATR_FACTOR: list[float] | None = None, strategy_func = backside_short_lower_low):
    """
    Run backside_short_lower_low on live Massive.com data for a given ticker/date.
    Fetches 5-min candles, computes all needed indicators, runs the strategy,
    prints the trades, and plots the chart with VWAP and trade markers.

    Args:
        ticker: Equity symbol, e.g. "AGRX"
        date:   Trading day in "YYYY-MM-DD" format
        timeframe: Timeframe for the candles, e.g. "5m"
        ATR_FACTOR: List of ATR factors to test, e.g. [3.5, 2.0]

    """
    from datetime import datetime, timedelta
    from app.utils.massive import fetch_candles
    from app.utils.indicators import compute_vwap, compute_atr, compute_rvol, compute_sma, compute_close_atr_band
    from app.utils.charts import plot_candles_df, trades_to_markers

    if timeframe is None:
        timeframe = "5m"

   
    if ATR_FACTOR is None:
        ATR_FACTOR = [3.5]

    # --------------------------------------------------
    # 1. Fetch 5min candles — lookback for rolling indicators + prev_day_close
    # --------------------------------------------------
    lookback_from = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")

    candles = fetch_candles(
        ticker,
        lookback_from,
        date,
        timeframe="5m",
        session_start="04:00",
        session_end="16:00",
    )

    if not candles:
        print(f"No data found for {ticker} on {date}")
        return

    df_all = pd.DataFrame(candles)

    # --------------------------------------------------
    # 2. Build tz-naive ET datetime index (consistent with parquet data format)
    # --------------------------------------------------
    dt_et = pd.to_datetime(df_all["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df_all.index = pd.to_datetime(dt_et.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df_all.index.name = "date"

    # --------------------------------------------------
    # 3. Compute indicators on full lookback window (better ATR/RVOL warmup)
    # --------------------------------------------------
    df_all["vwap"]             = compute_vwap(df_all).values
    df_all["atr"]              = compute_atr(df_all).values
    df_all["RVOL_daily"]       = compute_rvol(df_all).values
    df_all["SMA_VOLUME_20_5m"] = compute_sma(df_all, window=20, column="volume").values

    # --------------------------------------------------
    # 4. Compute previous_day_close for each bar
    # --------------------------------------------------
    df_all["_day"] = df_all.index.date
    daily_close    = df_all.groupby("_day")["close"].last()
    sorted_days    = sorted(daily_close.index)
    day_to_prev_close = {
        day: (daily_close[sorted_days[i - 1]] if i > 0 else np.nan)
        for i, day in enumerate(sorted_days)
    }
    df_all["previous_day_close"] = df_all["_day"].map(day_to_prev_close)

    # --------------------------------------------------
    # 5. Filter to target date
    # --------------------------------------------------
    target_date = pd.to_datetime(date).date()
    df_day = df_all[df_all["_day"] == target_date].drop(columns=["_day", "day"], errors="ignore").copy()

    if df_day.empty:
        print(f"No bars found for {ticker} on {date}")
        return

    # --------------------------------------------------
    # 6. Run strategy
    # --------------------------------------------------
    #trades = backside_short_lower_low({ticker: df_day})
    trades = strategy_func({ticker: df_day})
    
    print(f"Trades for {ticker} on {date}:")
   
    if not trades.empty:
        print(trades[["ticker", "type", "entry_price", "exit_price",
                  "stop_loss_price", "pnl", "Return", 'volume', "entry_time", "exit_time", "strategy"]])

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    prev_close_val = df_day["previous_day_close"].iloc[0]
    prev_close = float(prev_close_val) if pd.notna(prev_close_val) else None

    if not trades.empty:
        entries_m, exits_m, short_entries_m, short_exits_m = trades_to_markers(
            trades, ticker=ticker, date=date
        )
    else:
        entries_m = exits_m = short_entries_m = short_exits_m = []

    # Use the same ATR already computed with the full lookback window (same as the strategy).
    # Recomputing ATR from df_day alone would give different (cold) values for early bars.
    atr_series = df_day["atr"]
    indicators = {"VWAP": pd.Series(df_day["vwap"].values)}
    for factor in ATR_FACTOR:
        indicators[f"ATR {factor}x"] = pd.Series(
            (df_day["close"] + factor * atr_series).values
        )

    plot_candles_df(
        df_day.reset_index(drop=True),
        title=f"{ticker}  {date}",
        short_entries=short_entries_m,
        short_exits=short_exits_m,
        prev_close=prev_close,
        indicators=indicators,
    )




# ============================
#exemple_with_local_data()

# trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/short_push_exhaustion/short_push_exhaustion_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# print(trades)

#example_with_api_data(ticker="CREG", date="2026-04-10",timeframe="5m", ATR_FACTOR=[3.5, 2.0])
#example_with_api_data(ticker="ABOS", date="2022-09-28",timeframe="5m", ATR_FACTOR=[3.5, 2.0], strategy_func=short_push_exhaustion)

# trades_path =  'backtest_dataset/walkforward/5m/fold_1/trades/backside_short_lower_low/backside_short_lower_low_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/5m/fold_1/trades/backside_short_lower_low/backside_short_lower_low_out_of_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/5m/fold_2/trades/backside_short_lower_low/backside_short_lower_low_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/5m/fold_2/trades/backside_short_lower_low/backside_short_lower_low_out_of_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/5m/fold_3/trades/backside_short_lower_low/backside_short_lower_low_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/5m/fold_3/trades/backside_short_lower_low/backside_short_lower_low_out_of_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)


#trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/backside_short_lower_low/backside_short_lower_low_in_sample_trades.parquet'
#trades =  pd.read_parquet(trades_path)
#analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

#trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/backside_short_lower_low/backside_short_lower_low_out_of_sample_trades.parquet'
#trades =  pd.read_parquet(trades_path)
#analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/short_push_exhaustion/short_push_exhaustion_in_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# print(trades)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

# trades_path =  'backtest_dataset/walkforward/15m/fold_3/trades/short_push_exhaustion/short_push_exhaustion_out_of_sample_trades.parquet'
# trades =  pd.read_parquet(trades_path)
# analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)

