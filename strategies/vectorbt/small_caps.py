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
from app.utils.trade_metrics import (analysis_and_plot, summary_report, get_mae_mfe)


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
    trades['take_profit_price'] = trades['entry_price'] * (1 - trades['tp'])
    trades['risk_reward_ratio'] = (
        (trades['entry_price'] - trades['take_profit_price']) /
        (trades['stop_loss_price'] - trades['entry_price'])
    )
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]

    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price', 'take_profit_price', 'risk_reward_ratio', 'pnl',
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
    'exit_price','stop_loss_price', 'take_profit_price', 'risk_reward_ratio', 'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]

def _format_trades_full(pf, all_params, strategy_name=""):

    trades = pf.trades.records_readable.copy()
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    
    #print(trades.columns)
    
    # print(trades[['Size', 'Entry Timestamp', 'Avg Entry Price',
    #    'Entry Fees', 'Exit Timestamp', 'Avg Exit Price', 'Exit Fees', 'PnL',
    #    'Return']])

    index_master = all_params['index_master']
    col_meta     = all_params['col_meta']
    atr_arr      = all_params['atr_arr']
    rvol_arr     = all_params['rvol_arr']
    volume_arr   = all_params['volume_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']

    # ============================================
    # RENOMBRE BASE
    # ============================================
    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction': 'type'
    })

    # ============================================
    # TIME
    # ============================================
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]

    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values

    # ============================================
    # JOIN PARAMETROS (tp, sl, gap)
    # ============================================
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')

    # ============================================
    # ATR (SIN LOOK-AHEAD)
    # ============================================
    atr_prev = np.roll(atr_arr, 1, axis=0)
    atr_prev[0, :] = np.nan

    atr_entry = atr_prev[entry_idx, col_idx]

    trades['stop_loss_price'] = trades['entry_price'] + trades['sl'] * atr_entry
    trades['take_profit_price'] = trades['entry_price'] * (1 - trades['tp'])
    trades['risk_reward_ratio'] = (
        (trades['entry_price'] - trades['take_profit_price']) /
        (trades['stop_loss_price'] - trades['entry_price'])
    )

    # ============================================
    # FEATURES
    # ============================================
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]
    

    # ============================================
    # STRATEGY NAME DINAMICO
    # ============================================
    trades['strategy'] = (
        strategy_name + "_" +
        trades['tp'].astype(str) + "_" +
        trades['sl'].astype(str) + "_" +
        trades['gap'].astype(str)
    )

    # ============================================
    # RETURN %
    # ============================================
    trades['return_pct'] = np.where(
        trades['type'] == 'short',
        (trades['entry_price'] - trades['exit_price']) / trades['entry_price'],
        (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    )

    # ============================================
    # LIMPIEZA FINAL
    # ============================================
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()

    return trades

def _format_trades_full_fix_stop(pf, all_params, strategy_name=""):

    trades = pf.trades.records_readable.copy()
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    
    #print(trades.columns)
    
    # print(trades[['Size', 'Entry Timestamp', 'Avg Entry Price',
    #    'Entry Fees', 'Exit Timestamp', 'Avg Exit Price', 'Exit Fees', 'PnL',
    #    'Return']])

    index_master = all_params['index_master']
    col_meta     = all_params['col_meta']
    atr_arr      = all_params['atr_arr']
    rvol_arr     = all_params['rvol_arr']
    volume_arr   = all_params['volume_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']

    # ============================================
    # RENOMBRE BASE
    # ============================================
    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction': 'type'
    })

    # ============================================
    # TIME
    # ============================================
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]

    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values

    # ============================================
    # JOIN PARAMETROS (tp, sl, gap)
    # ============================================
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')

    # ============================================
    # ATR (SIN LOOK-AHEAD)
    # ============================================
    atr_prev = np.roll(atr_arr, 1, axis=0)
    atr_prev[0, :] = np.nan


    trades['stop_loss_price'] = trades['entry_price'] * (1 + trades['sl'])
    trades['take_profit_price'] = trades['entry_price'] * (1 - trades['tp'])
    trades['risk_reward_ratio'] = (
        (trades['entry_price'] - trades['take_profit_price']) /
        (trades['stop_loss_price'] - trades['entry_price'])
    )

    # ============================================
    # FEATURES
    # ============================================
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]
    

    # ============================================
    # STRATEGY NAME DINAMICO
    # ============================================
    trades['strategy'] = (
        strategy_name + "_" +
        trades['tp'].astype(str) + "_" +
        trades['sl'].astype(str) + "_" +
        trades['gap'].astype(str)
    )

    # ============================================
    # RETURN %
    # ============================================
    trades['return_pct'] = np.where(
        trades['type'] == 'short',
        (trades['entry_price'] - trades['exit_price']) / trades['entry_price'],
        (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    )

    # ============================================
    # LIMPIEZA FINAL
    # ============================================
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()

    return trades
  
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

# ======= version v1 correciones ========


def gap_crap_strategy(f_dict):

    tp_list = [0.15, 0.20, 1.00, 0.15, 0.20, 1.00]
    sl_list = [3.50, 3.50, 3.50, 2.00, 2.00, 2.00]
    gap_list =[0.50]*6

    all_params = prepare_params_and_vectors_for_gappers(f_dict,gap_list, tp_list, sl_list)

    open_arr  = all_params['open_arr']
    high_arr  = all_params['high_arr']
    low_arr   = all_params['low_arr']
    close_arr = all_params['close_arr']
    atr_arr   = all_params['atr_arr']
    volume_arr = all_params['volume_arr']
    rvol_arr   = all_params['rvol_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']

    n_bars, n_cols = open_arr.shape

    # === FIXES ===
    atr_prev = np.roll(atr_arr, 1, axis=0)
    atr_prev[0,:] = np.nan

    mask_925 = np.array([t.strftime("%H:%M") == "09:25" for t in index_master])

    gap_vals = np.array([m["gap"] for m in col_meta])

    gap_pct = (open_arr - prev_day_close_arr) / prev_day_close_arr
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

    vol_cond  = volume_arr > 40_000
    rvol_cond = rvol_arr >= 3

    entries = np.zeros((n_bars, n_cols), dtype=bool)

    entries[mask_925,:] = (
        gap_cond[mask_925,:] &
        vol_cond[mask_925,:] &
        rvol_cond[mask_925,:]
    )

    next_open = np.roll(open_arr, -1, axis=0)
    next_open[-1,:] = np.nan
    entries[-1,:] = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (sl_vals * (atr_prev / next_open))[entries]

    # 🚫 evitar TP/SL misma vela
    entries_shifted = np.roll(entries,1,axis=0)
    entries_shifted[0,:] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
         freq=None
    )

    _trades = _format_trades_full(pf, all_params, strategy_name="gap_crap_strategy")
    return reduce_trades_columns(_trades)

def backside_short_lower_low(f_dict):

    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.50]*8,
        [0.10,0.10,0.15,0.15,0.20,0.20,1,1],
        [2,3.5,2,3.5,2,3.5,2,3.5]
    )

    open_arr = all_params['open_arr']
    high_arr = all_params['high_arr']
    low_arr  = all_params['low_arr']
    close_arr= all_params['close_arr']
    atr_arr  = all_params['atr_arr']
    vwap_arr = all_params['vwap_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']
    time_mask = all_params['time_mask']

    n_bars, n_cols = open_arr.shape

    # === FIXES ===
    atr_prev  = np.roll(atr_arr,1,axis=0)
    vwap_prev = np.roll(vwap_arr,1,axis=0)
    atr_prev[0,:] = np.nan
    vwap_prev[0,:] = np.nan

    red = close_arr < open_arr

    prev_close = np.roll(close_arr,1,axis=0)
    prev_open  = np.roll(open_arr,1,axis=0)
    prev_low   = np.roll(low_arr,1,axis=0)

    green_prev = prev_close > prev_open

    lower_low = green_prev & red & (close_arr < prev_low)

    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct = (open_arr - prev_day_close_arr) / prev_day_close_arr
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5)

    is_above_vwap = open_arr > vwap_prev

    entries = gap_cond & is_above_vwap & time_mask[:,None] & lower_low

    next_open = np.roll(open_arr,-1,axis=0)
    next_open[-1,:] = np.nan
    entries[-1,:] = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr,np.nan)
    sl_stop = np.full_like(open_arr,np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (sl_vals * (atr_prev / next_open))[entries]

    entries_shifted = np.roll(entries,1,axis=0)
    entries_shifted[0,:] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
          freq=None
    )

    _trades = _format_trades_full(pf, all_params, strategy_name="backside_short_lower_low")
    return reduce_trades_columns(_trades)


def backside_short_lower_low_fix_stop(f_dict):

    
    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.50]*8,
        [0.10,0.10,0.15,0.15,0.20,0.20,1.00,1.00],
        [0.40,0.50,0.40,0.50,0.40,0.50,0.40,0.50]
    )

    open_arr = all_params['open_arr']
    high_arr = all_params['high_arr']
    low_arr  = all_params['low_arr']
    close_arr= all_params['close_arr']
    atr_arr  = all_params['atr_arr']
    vwap_arr = all_params['vwap_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']
    time_mask = all_params['time_mask']

    n_bars, n_cols = open_arr.shape

    # === FIXES ===
    atr_prev  = np.roll(atr_arr,1,axis=0)
    vwap_prev = np.roll(vwap_arr,1,axis=0)
    atr_prev[0,:] = np.nan
    vwap_prev[0,:] = np.nan

    red = close_arr < open_arr

    prev_close = np.roll(close_arr,1,axis=0)
    prev_open  = np.roll(open_arr,1,axis=0)
    prev_low   = np.roll(low_arr,1,axis=0)

    green_prev = prev_close > prev_open

    lower_low = green_prev & red & (close_arr < prev_low)

    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct = (open_arr - prev_day_close_arr) / prev_day_close_arr
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5)
    
    is_above_vwap = open_arr > vwap_prev

    entries = gap_cond & is_above_vwap & time_mask[:,None] & lower_low

    next_open = np.roll(open_arr,-1,axis=0)
    next_open[-1,:] = np.nan
    entries[-1,:] = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr,np.nan)
    sl_stop = np.full_like(open_arr,np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (next_open * (1 + sl_vals))[entries]
   
    entries_shifted = np.roll(entries,1,axis=0)
    entries_shifted[0,:] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan
    

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
          freq=None
    )

    _trades = _format_trades_full_fix_stop(pf, all_params, strategy_name="backside_short_lower_low_fix_stop")
    return reduce_trades_columns(_trades)

def backside_short_big_green(f_dict):
    """
    Short after a large green candle (range >= 40% of price) is rejected.

    Entry conditions:
    - Previous bar: green AND (high - low) / close >= 0.40
    - Signal bar:   red (close < open)
    - Signal bar volume > previous bar volume
    - Gap >= 50% vs previous day close
    - Open above VWAP
    """

    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.30]*4,
        [0.10,0.15,0.20,1],
        [3.5]*4
    )

    open_arr   = all_params['open_arr']
    high_arr   = all_params['high_arr']
    low_arr    = all_params['low_arr']
    close_arr  = all_params['close_arr']
    atr_arr    = all_params['atr_arr']
    vwap_arr   = all_params['vwap_arr']
    volume_arr = all_params['volume_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta   = all_params['col_meta']
    time_mask  = all_params['time_mask']

    atr_prev  = np.roll(atr_arr, 1, axis=0)
    vwap_prev = np.roll(vwap_arr, 1, axis=0)
    atr_prev[0, :]  = np.nan
    vwap_prev[0, :] = np.nan

    prev_open   = np.roll(open_arr,   1, axis=0)
    prev_close  = np.roll(close_arr,  1, axis=0)
    prev_high   = np.roll(high_arr,   1, axis=0)
    prev_low    = np.roll(low_arr,    1, axis=0)
    prev_volume = np.roll(volume_arr, 1, axis=0)

    green_prev = prev_close > prev_open
    big_move   = (prev_high - prev_open) / (prev_open + 1e-9) >= 0.20
    big_green  = green_prev & big_move

    red      = close_arr < open_arr
    vol_cond = volume_arr > prev_volume

    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct  = (open_arr - prev_day_close_arr) / (prev_day_close_arr + 1e-9)
   
    is_above_vwap = open_arr > vwap_prev
    
    # debug_mask = big_green & red  # bars that match the core candle pattern
    # if debug_mask.any():
    #     idxs = np.argwhere(debug_mask[:, 0])
    #     for i in idxs[:, 0]:
    #         t = index_master[i]
    #         print(f"[debug] {t}  big_green={big_green[i,0]}  red={red[i,0]}  "
    #               f"vol_cond={vol_cond[i,0]} "
    #               f"vwap={is_above_vwap[i,0]}  time={time_mask[i]}  "
    #               f"gap_pct={gap_pct[i,0]:.2%}")

    entries = big_green & red & vol_cond  & is_above_vwap & time_mask[:, None]

    next_open = np.roll(open_arr, -1, axis=0)
    next_open[-1, :] = np.nan
    entries[-1, :]   = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (sl_vals * (atr_prev / next_open))[entries]

    entries_shifted = np.roll(entries, 1, axis=0)
    entries_shifted[0, :] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
        freq=None
    )

    _trades = _format_trades_full(pf, all_params, strategy_name="backside_short_big_green")
    return reduce_trades_columns(_trades)


def short_push_exhaustion(f_dict):

    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.50]*8,
        [0.10,0.10,0.15,0.15,0.20,0.20,1,1],
        [2,3.5,2,3.5,2,3.5,2,3.5]
    )

    open_arr = all_params['open_arr']
    high_arr = all_params['high_arr']
    low_arr  = all_params['low_arr']
    close_arr= all_params['close_arr']
    atr_arr  = all_params['atr_arr']
    volume_arr = all_params['volume_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    vwap_arr = all_params['vwap_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']
    time_mask = all_params['time_mask']

    n_bars, n_cols = open_arr.shape

    atr_prev  = np.roll(atr_arr,1,axis=0)
    vwap_prev = np.roll(vwap_arr,1,axis=0)

    red = close_arr < open_arr
    body = open_arr - close_arr
    upper_tail = high_arr - np.maximum(open_arr, close_arr)

    topping = red & (upper_tail >= 1.5*body)

    vol_cond = volume_arr > np.roll(volume_arr,1,axis=0)

    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct = (open_arr - prev_day_close_arr) / prev_day_close_arr
    gap_cond = (gap_pct >= gap_vals)

    is_above_vwap = open_arr > vwap_prev

    entries = gap_cond & is_above_vwap & time_mask[:,None] & topping & vol_cond

    next_open = np.roll(open_arr,-1,axis=0)
    next_open[-1,:] = np.nan
    entries[-1,:] = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr,np.nan)
    sl_stop = np.full_like(open_arr,np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (sl_vals * (atr_prev / next_open))[entries]

    entries_shifted = np.roll(entries,1,axis=0)
    entries_shifted[0,:] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
         freq=None
    )

    _trades = _format_trades_full(pf, all_params,strategy_name="short_push_exhaustion")
    return reduce_trades_columns(_trades)


def backside_short_dynamic_size(f_dict):

    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.50]*8,
        [0.10,0.10,0.15,0.15,0.20,0.20,1,1],
        [2,3.5,2,3.5,2,3.5,2,3.5]
    )

    open_arr = all_params['open_arr']
    high_arr = all_params['high_arr']
    low_arr  = all_params['low_arr']
    close_arr= all_params['close_arr']
    atr_arr  = all_params['atr_arr']
    vwap_arr = all_params['vwap_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']
    time_mask = all_params['time_mask']

    n_bars, n_cols = open_arr.shape

    # =========================
    # FIXES (shifted values)
    # =========================
    atr_prev  = np.roll(atr_arr, 1, axis=0)
    vwap_prev = np.roll(vwap_arr, 1, axis=0)
    atr_prev[0, :] = np.nan
    vwap_prev[0, :] = np.nan

    prev_close = np.roll(close_arr, 1, axis=0)
    prev_open  = np.roll(open_arr, 1, axis=0)
    prev_low   = np.roll(low_arr, 1, axis=0)

    green_prev = prev_close > prev_open
    red = close_arr < open_arr

    lower_low = green_prev & red & (close_arr < prev_low)

    # =========================
    # GAP CONDITIONS
    # =========================
    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct = (open_arr - prev_day_close_arr) / prev_day_close_arr

    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5)

    is_above_vwap = open_arr > vwap_prev

    # =========================
    # VOLATILITY REGIME
    # =========================
    atr_ratio = atr_arr / (close_arr + 1e-9)

    # high volatility regime (dangerous squeeze zone)
    high_vol = (
        (gap_pct > 0.15) &
        (atr_ratio > np.nanpercentile(atr_ratio, 80)) &
        (((high_arr - low_arr) / (close_arr + 1e-9)) > 0.10)
    )

    # reduced regime
    reduced_vol = (
        (atr_ratio > np.nanpercentile(atr_ratio, 60)) &
        (~high_vol)
    )

    # =========================
    # BASE ENTRY LOGIC
    # =========================
    base_entries = (
        gap_cond &
        is_above_vwap &
        time_mask[:, None] &
        lower_low
    )

    # filter out dangerous regimes
    entries = base_entries & (~high_vol)

    # =========================
    # POSITION SIZING
    # =========================
    size_matrix = np.ones_like(open_arr, dtype=float)

    size_matrix[reduced_vol] = 0.5
    size_matrix[high_vol] = 0.0

    # =========================
    # TP / SL
    # =========================
    next_open = np.roll(open_arr, -1, axis=0)
    next_open[-1, :] = np.nan

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]

    # optional improvement: cap ATR to reduce extreme stops
    atr_prev_clipped = np.minimum(
        atr_prev,
        np.nanpercentile(atr_prev, 80)
    )

    sl_stop[entries] = (
        sl_vals * (atr_prev_clipped / (next_open + 1e-9))
    )[entries]

    # prevent leakage to next bar
    entries_shifted = np.roll(entries, 1, axis=0)
    entries_shifted[0, :] = False

    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    # =========================
    # FORCED EXIT EOD
    # =========================
    forced_exit = generate_signal_to_force_close_EOD(
        np.zeros_like(entries),
        index_master
    )

    # =========================
    # BACKTEST
    # =========================
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
        size=size_matrix,
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
        freq=None
    )

    _trades = _format_trades_full(
        pf,
        all_params,
        strategy_name="backside_short_dynamic_size"
    )

    return reduce_trades_columns(_trades)


def orb_short(f_dict):
    """ORB short: if the 9:30am candle closes red, short on the next open."""

    all_params = prepare_params_and_vectors_for_gappers(
        f_dict,
        [0.50]*4,
        [0.10,0.15,0.20,1],
        [3.5]*4
    )

    open_arr  = all_params['open_arr']
    high_arr  = all_params['high_arr']
    low_arr   = all_params['low_arr']
    close_arr = all_params['close_arr']
    atr_arr   = all_params['atr_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    index_master = all_params['index_master']
    col_meta = all_params['col_meta']

    n_bars, n_cols = open_arr.shape

    atr_prev = np.roll(atr_arr, 1, axis=0)
    atr_prev[0, :] = np.nan

    # Identify the 9:30am candle (first candle of the regular session)
    hours   = np.array([t.hour   for t in index_master])
    minutes = np.array([t.minute for t in index_master])
    is_930 = (hours == 9) & (minutes == 30)

    red = close_arr < open_arr

    # Gap-up condition (same as other strategies)
    gap_vals = np.array([m["gap"] for m in col_meta])
    gap_pct  = (open_arr - prev_day_close_arr) / prev_day_close_arr
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5)

    # Enter short only on the 9:30 candle if it closes red and stock gapped up
    entries = is_930[:, None] & red & gap_cond

    next_open = np.roll(open_arr, -1, axis=0)
    next_open[-1, :] = np.nan
    entries[-1, :] = False

    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([m["sl"] for m in col_meta])

    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[entries] = np.broadcast_to(tp_vals, open_arr.shape)[entries]
    sl_stop[entries] = (sl_vals * (atr_prev / next_open))[entries]

    entries_shifted = np.roll(entries, 1, axis=0)
    entries_shifted[0, :] = False
    tp_stop[entries_shifted] = np.nan
    sl_stop[entries_shifted] = np.nan

    forced_exit = generate_signal_to_force_close_EOD(np.zeros_like(entries), index_master)

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
        init_cash=100_000,
        fees=0.0015,
        slippage=0.001,
        freq=None
    )

    _trades = _format_trades_full(pf, all_params, strategy_name="orb_short")
    return reduce_trades_columns(_trades)
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
  
def example_with_local_data_fix_stop(ticker: str, date: str,timeframe: str| None = None, ATR_FACTOR: list[float] | None = None, strategy_func = backside_short_lower_low):
    
    from datetime import datetime, timedelta
    from app.utils.massive import fetch_candles
    from app.utils.indicators import compute_vwap, compute_atr, compute_rvol, compute_sma, compute_close_atr_band
    from app.utils.charts import plot_candles_df, trades_to_markers

    if timeframe is None:
        timeframe = "5m"
        
    lookback_from = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")

    candles = fetch_candles(
        ticker,
        lookback_from,
        date,
        timeframe=timeframe,
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
    df_day['gap_pct'] = (df_day['close'] - df_day['previous_day_close']) / df_day['previous_day_close']
    print(df_day[['open','close','volume','previous_day_close','gap_pct']])
    trades = strategy_func({ticker: df_day})
    df_day['ticker'] = ticker
    
    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    prev_close_val = df_day["previous_day_close"].iloc[0]
    prev_close = float(prev_close_val) if pd.notna(prev_close_val) else None
   
    
    print(f"Trades for {ticker} on {date}:")
   
    if not trades.empty:
        print(trades[["ticker",  "entry_price", "exit_price", "take_profit_price",
                  "stop_loss_price",  "Return",  "entry_time", "exit_time", "strategy"]])
        
        
    plot_candles_df(
        df_day.reset_index(drop=True),
        title=f"{ticker}  {date}",
        short_entries=[],
        short_exits=[],
        prev_close=prev_close,
        indicators={},
    )
    
    return


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
        timeframe=timeframe,
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
    df_day['gap_pct'] = (df_day['close'] - df_day['previous_day_close']) / df_day['previous_day_close']
    print(df_day[['open','close','volume','previous_day_close','gap_pct']])
    trades = strategy_func({ticker: df_day})
    df_day['ticker'] = ticker
   
    
    print(f"Trades for {ticker} on {date}:")
   
    if not trades.empty:
        print(trades[["ticker",  "entry_price", "exit_price", "take_profit_price",
                  "stop_loss_price",  "Return",  "entry_time", "exit_time", "strategy"]])

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
    indicators = {}

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
#example_with_api_data(ticker="AIXI", date="2026-04-10",timeframe="15m", ATR_FACTOR=[3.5, 5.0], strategy_func=short_push_exhaustion)
#example_with_api_data(ticker="SAGT", date="2026-04-29",timeframe="15m", ATR_FACTOR=[3.5, 5.0], strategy_func=orb_short)
#example_with_api_data(ticker="LABT", date="2026-05-01",timeframe="5m", ATR_FACTOR=[3.5, 5.0], strategy_func=backside_short_big_green)
example_with_api_data(ticker="TDIC", date="2026-05-13",timeframe="15m", ATR_FACTOR=[3.5, 5.0], strategy_func=backside_short_lower_low_fix_stop)
#example_with_api_data(ticker="MASK", date="2026-05-05",timeframe="5m", ATR_FACTOR=[3.5, 5.0], strategy_func=short_push_exhaustion)
#example_with_local_data_fix_stop(ticker="QUCY", date="2026-05-15",timeframe="5m", ATR_FACTOR=[3.5, 5.0], strategy_func=backside_short_lower_low_fix_stop)

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

