
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import multiprocessing
from typing import List, Tuple, Dict, Any
import os
import sys
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

import httpx

from app.config import get_settings
from app.utils.time_utils import local_to_ms, local_time_to_utc_str, tz_label, tz_offset_hours
from app.utils.market_utils import (
    sync_tickers,
    ticker_chunks,
    process_data_minutes,
    sync_data_with_prev_day_close,
    _save_state,
)
from app.utils.trade_metrics import (analysis_and_plot, summary_report)
import json
import time

MAX_CONCURRENT_REQUESTS = 300
logger = logging.getLogger(__name__)

# Función A: Fetch a una API (Asíncrona)
async def fetch_data_1_min(session: aiohttp.ClientSession, params) :
    # URL de ejemplo
    
    (ticker, start_date, end_date) = params

    # Polygon.io API details
    API_KEY = os.getenv("MASSIVE_API_KEY", "none")  # Replace with your Polygon.io API key
    
    BASE_URL_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_min = BASE_URL_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
    logger.debug("Fetching %s [%s → %s]", ticker, start_date, end_date)
    try:
        async with session.get(url_min) as response:
            response.raise_for_status()
            res = await response.json()
            data = res.get("results", [])
            df = pd.DataFrame(data)
            df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
            return df

    except aiohttp.ClientResponseError as e:
        # Error HTTP 4xx / 5xx
        try:
            error_details = await response.text()
        except Exception:
            error_details = "No se pudo leer el cuerpo de la respuesta."
        logger.error(
            "HTTP %s en %s [%s → %s]: %s",
            e.status, ticker, start_date, end_date, error_details,
            exc_info=True,
        )
        return False

    except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
        logger.error("Timeout en %s [%s → %s]: %s", ticker, start_date, end_date, e, exc_info=True)
        return False

    except aiohttp.ClientConnectionError as e:
        logger.error("Error de conexión en %s [%s → %s]: %s", ticker, start_date, end_date, e, exc_info=True)
        return False

    except aiohttp.ClientError as e:
        logger.error("Error de aiohttp en %s [%s → %s]: %s", ticker, start_date, end_date, e, exc_info=True)
        return False

    except Exception as e:
        logger.error("Error inesperado en %s [%s → %s]: %s", ticker, start_date, end_date, e, exc_info=True)
        return False


def group_parameters_by_ticker(parameters: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Agrupa la lista plana de (ticker, start_date, end_date) por ticker."""
    
    # Asumimos que los parámetros son (ticker, date1, date2)
    temp_df = pd.DataFrame(parameters, columns=["ticker", "date1", "date2"])
    
    # Agrupa y convierte de nuevo a un diccionario
    grouped_params = {
        ticker: list(df_group.itertuples(index=False, name=None))
        for ticker, df_group in temp_df.groupby("ticker")
    }
    return grouped_params


def partition_tickers_into_batches(grouped_params_dict: Dict[str, List], num_batches: int) -> List[Dict]:
    """Divide los tickers agrupados en N lotes, asegurando que cada ticker 
       permanezca en un único lote."""
    
    tickers = list(grouped_params_dict.keys())
    
    # Inicializa los N lotes (cada lote es un diccionario)
    batches = [{} for _ in range(num_batches)]
    
    # Asigna cada ticker (y sus parámetros) a un lote en rotación (round-robin)
    for i, ticker in enumerate(tickers):
        batch_index = i % num_batches
        batches[batch_index][ticker] = grouped_params_dict[ticker]
        
    return batches

# ==============================================================================
#                      2. FUNCIONES ASÍNCRONAS 
# ==============================================================================

async def fetch_and_process(session: aiohttp.ClientSession, params, api_semaphore: asyncio.Semaphore):
    """
    Función de tarea individual que realiza el Fetch (A) y el Process (B).
    Devuelve un DataFrame si tiene éxito, o None si hay fallo.
    """
    (ticker, start_date, end_date) = params
    
    
    
    # Usamos el semáforo para limitar la concurrencia a la API
    async with api_semaphore:
        try:
            # Función A
            raw_data = await fetch_data_1_min(session, params)
            if not isinstance(raw_data, pd.DataFrame):
                logger.warning("fetch falló para %s [%s → %s]", ticker, start_date, end_date)
                return None

            # Función B
            processed_data = process_data_minutes(raw_data)
            if processed_data is None or processed_data.empty:
                logger.warning("process_data devolvió vacío para %s [%s → %s]", ticker, start_date, end_date)
                return None

            # Añadir información clave para la consolidación
            processed_data['ticker'] = ticker
            return processed_data

        except Exception as e:
            logger.error("Error en FETCH/PROCESS para %s [%s → %s]: %s", ticker, start_date, end_date, e, exc_info=True)
            return None
        
        
async def _fetch_split(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    ticker: str,
    api_key: str,
    base_url: str,
) -> list[dict]:
    """
    Fetch all stock-split events for a ticker via the Massive/Polygon API.

    Returns a list of dicts with keys:
        execution_date               (ISO date string, e.g. "2020-08-31")
        historical_adjustment_factor (float = split_from / split_to)

    Returns [] on any error so enrichment can still proceed without adjustments.
    """
    url: str | None = (
        f"{base_url}/v3/reference/splits"
        f"?ticker={ticker}&limit=1000&apiKey={api_key}"
    )
    splits: list[dict] = []
    _MAX_RETRIES    = 4
    _RETRY_STATUSES = {429, 500, 502, 503, 504}

    async with semaphore:
        while url:
            for attempt in range(_MAX_RETRIES):
                try:
                    async with session.get(url) as resp:
                        if resp.status in _RETRY_STATUSES:
                            wait = 2 ** attempt
                            if attempt < _MAX_RETRIES - 1:
                                logger.warning(
                                    "SPLIT_HTTP_%d  ticker=%-6s  attempt=%d  retry_in=%ds",
                                    resp.status, ticker, attempt + 1, wait,
                                )
                                await asyncio.sleep(wait)
                                continue
                            else:
                                logger.error("SPLIT_GIVE_UP  ticker=%-6s  HTTP %d", ticker, resp.status)
                                url = None
                                break

                        if resp.status >= 400:
                            logger.error("SPLIT_HTTP_%d  ticker=%-6s", resp.status, ticker)
                            url = None
                            break

                        data = await resp.json()
                        for item in data.get("results") or []:
                            split_from = item.get("split_from", 1) or 1
                            split_to   = item.get("split_to",   1) or 1
                            splits.append({
                                "execution_date":               item["execution_date"],
                                "historical_adjustment_factor": split_from / split_to,
                            })

                        next_url = data.get("next_url")
                        url = f"{next_url}&apiKey={api_key}" if next_url else None
                        break  # success

                except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "SPLIT_TIMEOUT  ticker=%-6s  attempt=%d  retry_in=%ds  %s",
                        ticker, attempt + 1, wait, exc,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                    else:
                        logger.error("SPLIT_GIVE_UP  ticker=%-6s  all %d attempts failed", ticker, _MAX_RETRIES)
                        url = None

                except aiohttp.ClientConnectionError as exc:
                    logger.warning("SPLIT_CONN_ERROR  ticker=%-6s  %s", ticker, exc)
                    url = None
                    break

                except Exception as exc:
                    logger.error("SPLIT_ERROR  ticker=%-6s  %s: %s", ticker, type(exc).__name__, exc)
                    url = None
                    break

    return splits


async def _fetch_daily_ohlc(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    ticker: str,
    api_key: str,
    base_url: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Fetch daily (unadjusted) OHLC bars for a ticker.

    The range is extended 7 calendar days before from_date so the first
    intraday trading day always has a previous_close available.

    Returns a DataFrame with columns [date_str, previous_close] sorted
    ascending, where previous_close is the close of the prior trading day
    (shift(1) on daily bars — correct even when intraday data has gaps).
    Returns an empty DataFrame on error.
    """
    # Extend start to capture the close before the first intraday bar
    start = (date.fromisoformat(from_date) - timedelta(days=7)).isoformat()

    url: str | None = (
        f"{base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{to_date}"
        f"?adjusted=false&sort=asc&limit=50000&apiKey={api_key}"
    )
    bars: list[dict] = []
    _MAX_RETRIES    = 4
    _RETRY_STATUSES = {429, 500, 502, 503, 504}

    async with semaphore:
        while url:
            for attempt in range(_MAX_RETRIES):
                try:
                    async with session.get(url) as resp:
                        if resp.status in _RETRY_STATUSES:
                            wait = 2 ** attempt
                            if attempt < _MAX_RETRIES - 1:
                                logger.warning(
                                    "DAILY_HTTP_%d  ticker=%-6s  attempt=%d  retry_in=%ds",
                                    resp.status, ticker, attempt + 1, wait,
                                )
                                await asyncio.sleep(wait)
                                continue
                            else:
                                logger.error("DAILY_GIVE_UP  ticker=%-6s  HTTP %d", ticker, resp.status)
                                url = None
                                break

                        if resp.status >= 400:
                            logger.error("DAILY_HTTP_%d  ticker=%-6s", resp.status, ticker)
                            url = None
                            break

                        data = await resp.json()
                        bars.extend(
                            {"t": b["t"], "c": b["c"]}
                            for b in (data.get("results") or [])
                        )
                        next_url = data.get("next_url")
                        url = f"{next_url}&apiKey={api_key}" if next_url else None
                        break  # success

                except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "DAILY_TIMEOUT  ticker=%-6s  attempt=%d  retry_in=%ds  %s",
                        ticker, attempt + 1, wait, exc,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                    else:
                        logger.error("DAILY_GIVE_UP  ticker=%-6s  all %d attempts failed", ticker, _MAX_RETRIES)
                        url = None

                except aiohttp.ClientConnectionError as exc:
                    logger.warning("DAILY_CONN_ERROR  ticker=%-6s  %s", ticker, exc)
                    url = None
                    break

                except Exception as exc:
                    logger.error("DAILY_ERROR  ticker=%-6s  %s: %s", ticker, type(exc).__name__, exc)
                    url = None
                    break

    if not bars:
        return pd.DataFrame()

    daily = pd.DataFrame(bars)
    daily["date_str"] = (
        pd.to_datetime(daily["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    daily = daily.sort_values("date_str").reset_index(drop=True)
    # shift(1) on daily bars is correct: only actual trading days have rows,
    # so the previous row is always the previous trading session.
    daily["previous_close"] = daily["c"].shift(1)
    return daily[["date_str", "previous_close"]]


def _apply_gap_logic(
    df: pd.DataFrame,
    daily_df: pd.DataFrame,
    splits: list[dict],
) -> pd.DataFrame:
    """
    Set previous_close from daily OHLC bars, apply split overrides, then
    compute gap / range derived columns.

    previous_close priority:
        1. daily_df match on date_str  (robust against intraday data gaps)
        2. pm_open when > 0            (4am candle open)
        3. df["open"]                  (9:30 open as last resort)

    After previous_close is established, sync_data_with_prev_day_close
    overrides it on split execution days.
    """
    # 1. Set previous_close from daily bars (merge on date_str)
    if not daily_df.empty:
        df = df.merge(daily_df, on="date_str", how="left")
    else:
        df["previous_close"] = float("nan")

    # Fallback for rows with no match in daily_df (e.g. very recent dates)
    pm_open = df.get("pm_open", pd.Series(dtype=float))
    fallback = pm_open.where(pm_open > 0, df["open"])
    df["previous_close"] = df["previous_close"].fillna(fallback)

    # 2. Override previous_close on split execution days
    df = sync_data_with_prev_day_close(df, fetch_split=lambda _: splits)

    # 3. Gap and range derived columns
    prev = df["previous_close"]
    df["gap"]           = (df["open"] - prev).round(3)
    df["gap_perc"]      = np.where(
        prev > 0,
        ((df["open"] - prev) * 100 / prev).round(3),
        0.0,
    )
    df["daily_range"]   = np.where(
        prev > 0,
        (df["high_mh"] - prev).round(3),
        0.0,
    )
    df["day_range_perc"] = np.where(
        prev > 0,
        ((df["high_mh"] - prev) * 100 / prev).round(3),
        0.0,
    )

    return df


# ==============================================================================
#                      3. WORKER DE MULTIPROCESAMIENTO (Nivel Síncrono)
# ==============================================================================

def process_batch_worker(batch_id: int, ticker_batch: Dict, connectionParams: Dict):
    """
    Función síncrona que corre en un proceso separado (un núcleo de CPU).
    Contiene el ciclo de asyncio.
    """
    logger.info("[Worker %s] Iniciando. Contiene %s Tickers.", batch_id, len(ticker_batch))

    # El worker necesita un ciclo de eventos de asyncio
    try:
        asyncio.run(
            async_batch_runner(batch_id, ticker_batch, connectionParams)
        )
    except Exception as e:
        logger.error("[Worker %s] Error fatal en el runner: %s", batch_id, e, exc_info=True)
    
async def async_batch_runner(batch_id: int, ticker_batch: Dict, connectionParams: Dict):

    # Semáforos para el control de recursos
    api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) # Límite de peticiones a la API
    ingest_semaphore = asyncio.Semaphore(3) # Límite de conexiones a la DB

    failed_tickers: list[dict] = []

    settings = get_settings()
    api_key  = settings.massive_api_key
    base_url = settings.massive_base_url

    # Usamos una sesión para todo el worker
    async with aiohttp.ClientSession() as session:

        # Iterar secuencialmente sobre CADA Ticker dentro del batch
        for ticker, ticker_param_list in ticker_batch.items():

            # 1. Ejecutar A y B concurrentemente para UN SOLO Ticker
            fetch_tasks = [
                fetch_and_process(session, params, api_semaphore)
                for params in ticker_param_list
            ]

            processed_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Detectar chunks que fallaron en fetch o en process_data
            for params, r in zip(ticker_param_list, processed_results):
                if r is None or isinstance(r, Exception):
                    failed_tickers.append({
                        "ticker":    ticker,
                        "from_date": str(params[1]),
                        "to_date":   str(params[2]),
                        "error":     str(r) if isinstance(r, Exception) else "fetch/process returned None",
                    })

            # 2. Consolidar el DataFrame (Paso B final)
            valid_dataframes = [df for df in processed_results if isinstance(df, pd.DataFrame) and not df.empty]

            if not valid_dataframes:
                logger.warning("[Worker %s] No hay datos válidos para %s.", batch_id, ticker)
                continue

            df = pd.concat(valid_dataframes, ignore_index=True)
            df = df.sort_values(by="time").reset_index(drop=True)

            # 3. Fetch daily bars + splits en paralelo
            from_date = min(p[1] for p in ticker_param_list)
            to_date   = max(p[2] for p in ticker_param_list)
            daily_df, splits = await asyncio.gather(
                _fetch_daily_ohlc(session, api_semaphore, ticker, api_key, base_url, from_date, to_date),
                _fetch_split(session, api_semaphore, ticker, api_key, base_url),
            )

            # 4. previous_close + ajuste por splits + gap/range
            df = await asyncio.to_thread(_apply_gap_logic, df, daily_df, splits)
            df = df.drop(columns=['day'], errors='ignore')

            logger.debug("[Worker %s] %s — %s filas procesadas.\n%s", batch_id, ticker, len(df), df.to_string())

            # 5. Guardar en DB
            try:
                await save_ticker_to_db(df)
            except Exception as exc:
                logger.error("[Worker %s] DB insert falló para %s: %s", batch_id, ticker, exc)
                failed_tickers.append({
                    "ticker":    ticker,
                    "from_date": str(min(p[1] for p in ticker_param_list)),
                    "to_date":   str(max(p[2] for p in ticker_param_list)),
                    "error":     str(exc),
                })

    if failed_tickers:
        _save_state(os.path.join(_PROJECT_ROOT, "logs", "ticker_fails", f"failed_tickers_worker_{batch_id}.json"), failed_tickers)
        
           
# ==============================================================================
#                      STORAGE
# ==============================================================================

# Columns expected by upsert_stock_data (matches create.sql exactly)
_DB_COLS: list[str] = [
    "ticker", "date_str",
    "gap", "gap_perc", "daily_range", "day_range_perc",
    "previous_close",
    "open", "high", "low", "close", "volume",
    "premarket_volume", "market_hours_volume",
    "high_pm", "low_pm", "pm_open", "highest_in_pm", "high_pm_time",
    "high_mh",
    "ah_open", "ah_close", "ah_high", "ah_low", "ah_range", "ah_range_perc", "ah_volume",
    "market_cap", "stock_float", "daily_200_sma",
    "split_date_str", "split_adjust_factor",
    "time",
]

# Columns that must be integer in PostgreSQL (jsonb_to_recordset rejects floats for bigint)
_BIGINT_COLS: frozenset[str] = frozenset({
    "volume", "premarket_volume", "market_hours_volume",
    "ah_volume", "high_pm_time", "time",
})


async def save_ticker_to_db(
    df: pd.DataFrame,
    batch_size: int = 5_000,
) -> None:
    """
    Upsert a ticker's daily DataFrame into PostgreSQL via the PostgREST
    ``upsert_stock_data`` RPC.

    Calls:
        POST {postgrest_url}/rpc/upsert_stock_data
        Authorization: Bearer <web_admin JWT>
        {"p_data": [{...row...}, ...]}

    Columns missing from df (e.g. market_cap, stock_float, daily_200_sma)
    are filled with -1 before sending.  Extra columns not in the DB schema
    are silently dropped.  bigint columns are cast to int64 so PostgreSQL's
    jsonb_to_recordset does not reject float-formatted integers.

    Args:
        df:         Daily summary DataFrame produced by the pipeline.
        batch_size: Rows per HTTP request (default 5 000).
    """
    if df.empty:
        return

    settings      = get_settings()
    postgrest_url = settings.postgrest_url.rstrip("/") + "/rpc/upsert_stock_data"
    token         = settings.postgrest_token
    ticker        = df["ticker"].iloc[0]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }

    # Add placeholder columns if absent
    for col in ("market_cap", "stock_float", "daily_200_sma"):
        if col not in df.columns:
            df[col] = -1

    # Select only the columns the DB function expects (in schema order)
    present = [c for c in _DB_COLS if c in df.columns]
    df = df[present].copy()

    # Cast bigint columns: pandas stores them as float64 after fillna(-1)
    for col in _BIGINT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype("int64")

    df = df.fillna(-1)
    records = df.to_dict(orient="records")
    total   = len(records)
    n_batches = (total + batch_size - 1) // batch_size

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client:
        for i in range(0, total, batch_size):
            batch   = records[i : i + batch_size]
            batch_n = i // batch_size + 1
            try:
                resp = await client.post(postgrest_url, headers=headers, json={"p_data": batch})
                resp.raise_for_status()
                logger.info(
                    "db  ticker=%-6s  batch=%d/%d  rows=%d  status=%d",
                    ticker, batch_n, n_batches, len(batch), resp.status_code,
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "db  ticker=%-6s  batch=%d/%d  HTTP %d — %s",
                    ticker, batch_n, n_batches, exc.response.status_code, exc.response.text,
                )
                raise
            except Exception as exc:
                logger.error(
                    "db  ticker=%-6s  batch=%d/%d  %s: %s",
                    ticker, batch_n, n_batches, type(exc).__name__, exc,
                )
                raise


def save_ticker_parquet(df: pd.DataFrame, output_dir: str) -> None:
    """
    Upsert a ticker's daily DataFrame into a per-ticker Parquet file.

    Layout:
        {output_dir}/{ticker}.parquet

    If the file already exists, rows whose date_str appears in df are replaced
    (upsert semantics), so re-running the pipeline for a date range is safe.
    New rows are appended and the result is sorted by date_str ascending.

    Args:
        df:         Daily summary DataFrame with at least columns
                    ``ticker`` and ``date_str``.
        output_dir: Directory where per-ticker .parquet files are stored.
                    Created automatically if it does not exist.
    """
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq

    if df.empty:
        return

    ticker = df["ticker"].iloc[0]
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path   = out / f"{ticker}.parquet"

    if path.exists():
        existing = pq.read_table(path).to_pandas()
        # Drop rows that will be replaced by the incoming data
        existing = existing[~existing["date_str"].isin(df["date_str"])]
        df = pd.concat([existing, df], ignore_index=True)

    df = df.sort_values("date_str").reset_index(drop=True)

    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="zstd",
    )
    logger.info("saved  ticker=%-6s  rows=%d  path=%s", ticker, len(df), path)


# ==============================================================================
#                      4. FUNCIÓN PRINCIPAL DE LANZAMIENTO
# ==============================================================================

def main_multiprocess_pipeline(parameters: List[Tuple], num_processes: int, connectionParams: Dict):
    
    logger.info("Iniciando pipeline con %s tareas usando %s procesos.", len(parameters), num_processes)
    
    #utils.update_stock_list_DB(connectionParams)
    
    # ⏱️ 1. CAPTURAR EL TIEMPO DE INICIO
    start_time = time.perf_counter()
    
    # 1. Agrupar por ticker
    grouped_params = group_parameters_by_ticker(parameters)
    
    # 2. Particionar en N batches
    batches = partition_tickers_into_batches(grouped_params, num_processes)
    
    processes = []
    
    # 3. Lanzar N procesos (uno por batch)
    for i, batch in enumerate(batches):
        if not batch:
            continue
            
        p = multiprocessing.Process(
            target=process_batch_worker, 
            args=(i + 1, batch, connectionParams)
        )
        processes.append(p)
        p.start()
        
    # 4. Esperar a que todos los procesos terminen
    for p in processes:
        p.join()
        
    # ⏱️ 5. CAPTURAR EL TIEMPO FINAL
    end_time = time.perf_counter()
    
    # 6. CALCULAR Y MOSTRAR LA DURACIÓN
    total_duration = end_time - start_time
    
    logger.info("Pipeline completado en %.2f segundos.", total_duration)


def run_sample_v1():
    
    logger.warning("syncing ticker list before spawning workers...")
    tickers_df = sync_tickers()
  
    tickers = tickers_df["ticker"].dropna().unique().tolist()
    
    all_chunks = ticker_chunks(tickers=tickers, months_back=59)
    #all_chunks = [("RVPH","2026-03-01","2026-03-31")]
    #all_chunks = [("DRCT","2026-01-01","2026-02-01")]
    
    main_multiprocess_pipeline(parameters=all_chunks, num_processes=8, connectionParams={})
    
    
    
    #print(all_chunks)
    #print("Ejecutando función de muestra...")
    
    return

async def _fetch_latest_stock_date() -> date | None:
    """
    Fetch the latest `time` value from stock_data via PostgREST.
    Equivalent to: SELECT * FROM stock_data ORDER BY time DESC LIMIT 1;

    Returns the corresponding date in NY timezone, or None on error.
    """
    settings = get_settings()
    url = f"{settings.postgrest_url.rstrip('/')}/stock_data?order=time.desc&limit=1"
    headers = {
        "Authorization": f"Bearer {settings.postgrest_token}",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                logger.warning("stock_data is empty — no latest date found.")
                return None
            time_ms = rows[0]["time"]
            latest_date = (
                pd.Timestamp(time_ms, unit="ms", tz="UTC")
                .tz_convert("America/New_York")
                .date()
            )
            logger.info("Latest date in stock_data: %s", latest_date)
            return latest_date
    except Exception as exc:
        logger.error("Failed to fetch latest stock_data date: %s", exc, exc_info=True)
        return None


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_stock_data_filtered(page_size: int = 10_000) -> pd.DataFrame:
    """
    Fetch all rows from the stock_data_filtered materialized view via PostgREST.

    The view contains daily OHLCV + derived fields for stocks that satisfy:
        previous_close > 0.10
        AND (gap_perc > 40% OR day_range_perc > 40%)

    Pagination is handled automatically — all pages are fetched and merged.

    Args:
        page_size: Rows per request (default 10 000). Raise if the view is large
                   and you want fewer round-trips; lower if PostgREST times out.

    Returns:
        pd.DataFrame with all rows, or an empty DataFrame on error.
    """
    settings   = get_settings()
    base_url   = settings.postgrest_url.rstrip("/") + "/stock_data_filtered"
    headers    = {
        "Authorization": f"Bearer {settings.postgrest_token}",
        "Accept":        "application/json",
    }

    all_rows: list[dict] = []
    offset = 0

    with httpx.Client(timeout=60) as client:
        while True:
            url  = f"{base_url}?limit={page_size}&offset={offset}"
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:   # last (possibly partial) page
                break
            offset += page_size

    if not all_rows:
        logger.warning("fetch_stock_data_filtered: no rows returned.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info("fetch_stock_data_filtered: %d rows loaded.", len(df))
    return df


def run_retry_failures(failures_dir: str | None = None, num_processes: int = 8):
    """
    Lee todos los archivos de failures en failures_dir y reintenta el fetch
    para cada (ticker, from_date, to_date) que haya fallado.
    """
    import glob

    if failures_dir is None:
        failures_dir = os.path.join(_PROJECT_ROOT, "logs", "ticker_fails")

    json_files = glob.glob(os.path.join(failures_dir, "failed_tickers_worker_*.json"))
    if not json_files:
        logger.info("run_retry_failures: no hay archivos de failures en %s", failures_dir)
        return

    seen: set[tuple] = set()
    parameters: list[tuple[str, str, str]] = []

    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("run_retry_failures: no se pudo leer %s — %s", path, exc)
            continue

        for item in data.get("failures", []):
            key = (item["ticker"], item["from_date"], item["to_date"])
            if key not in seen:
                seen.add(key)
                parameters.append(key)

    if not parameters:
        logger.info("run_retry_failures: todos los archivos estaban vacíos.")
        return

    logger.info("run_retry_failures: reintentando %d chunks de %d archivos.", len(parameters), len(json_files))
    main_multiprocess_pipeline(parameters=parameters, num_processes=num_processes, connectionParams={})


def run_incremental_v1():
    """
    Like run_sample_v1 but fetches only data newer than the latest date
    already stored in stock_data (determined via PostgREST).
    """
    run_retry_failures()

    from_date: date | None = asyncio.run(_fetch_latest_stock_date())

    if from_date is None:
        logger.warning("Could not determine latest date — falling back to run_sample_v1 behaviour.")
        run_sample_v1()
        return

    to_date = date.today()
    if from_date >= to_date:
        logger.info("stock_data is already up to date (latest=%s). Nothing to fetch.", from_date)
        return

    logger.warning("Incremental sync from %s to %s — syncing ticker list...", from_date, to_date)
    tickers_df = sync_tickers()
    tickers = tickers_df["ticker"].dropna().unique().tolist()

    from app.utils.market_utils import chunk_date_range
    all_chunks: list[tuple[str, str, str]] = []
    for ticker in tickers:
        all_chunks.extend(chunk_date_range(ticker, from_date, to_date))

    logger.info("Incremental pipeline: %d chunks across %d tickers.", len(all_chunks), len(tickers))
    main_multiprocess_pipeline(parameters=all_chunks, num_processes=8, connectionParams={})


if __name__ == "__main__":
    from app.utils.logging_config import setup_logging
    setup_logging()
    run_incremental_v1()
    
    #trades_path = "backtest_dataset/in_sample/trades/backside_short/backside_short_in_sample_trades.parquet"
    # data_path = 'backtest_dataset/in_sample/gappers_backtest_dataset_5min_in_sample.parquet'

    #trades =  pd.read_parquet(trades_path)
    #analysis_and_plot(trades=trades, initial_capital=10000, risk_pct=0.01)
    
    