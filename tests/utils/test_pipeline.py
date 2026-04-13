"""
Tests for app.utils.pipeline internals.

Covers _enrich which runs after process_data_minutes and computes:
  - previous_close  (with pm_open / open fallback logic)
  - gap, gap_perc, daily_range, day_range_perc
  - split_adjust_factor, split_date_str  (via sync_data_with_prev_day_close)
  - placeholder columns: market_cap, stock_float, daily_200_sma

Also covers save_to_db which pushes pipeline output to PostgREST.
"""
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import pytest

from app.utils.pipeline import (
    _enrich,
    _OUTPUT_COLS,
    _PIPELINE_STATE_FILE,
    _load_failed_tickers,
    _save_state,
    save_to_db,
)

ET = ZoneInfo("America/New_York")


# ── helpers ───────────────────────────────────────────────────────────────────

def _midnight_ms(date_str: str) -> int:
    """ET midnight for the given date → UTC milliseconds."""
    dt = datetime.fromisoformat(date_str).replace(tzinfo=ET)
    return int(dt.timestamp() * 1000)


def _make_df(*rows) -> pd.DataFrame:
    """
    Build a minimal daily-summary DataFrame suitable for _enrich.

    Each element of *rows* is a dict with any subset of:
        date_str, ticker, open, close, high, low, pm_open

    Sensible defaults are applied for omitted keys so every test only
    specifies the fields it cares about.
    """
    records = []
    for row in rows:
        date_str = row["date_str"]
        records.append({
            "ticker":   row.get("ticker",  "TEST"),
            "date_str": date_str,
            "time":     _midnight_ms(date_str),
            "open":     row.get("open",    100.0),
            "close":    row.get("close",   110.0),
            "high":     row.get("high",    115.0),
            "low":      row.get("low",      95.0),
            "pm_open":  row.get("pm_open",  np.nan),
        })
    return pd.DataFrame(records)


# ── TestEnrich ────────────────────────────────────────────────────────────────

class TestEnrich:

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=[
            "ticker", "date_str", "time", "open", "close", "high", "low", "pm_open",
        ])
        result = _enrich(df, splits=[])
        assert result.empty

    def test_all_new_columns_present(self):
        df = _make_df({"date_str": "2024-01-02"})
        result = _enrich(df, splits=[])
        for col in (
            "previous_close", "gap", "gap_perc", "daily_range",
            "day_range_perc", "market_cap", "stock_float", "daily_200_sma",
            "split_adjust_factor", "split_date_str",
        ):
            assert col in result.columns, f"missing column: {col}"

    # ── previous_close — first row ────────────────────────────────────────────

    def test_first_row_uses_pm_open_as_previous_close(self):
        """First row has no prior day → previous_close = pm_open (session open)."""
        df = _make_df({"date_str": "2024-01-02", "open": 100.0, "pm_open": 98.0})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["previous_close"] == pytest.approx(98.0)

    def test_first_row_uses_open_when_pm_open_is_nan(self):
        """No pre-market data → fallback to regular 09:30 open."""
        df = _make_df({"date_str": "2024-01-02", "open": 100.0, "pm_open": np.nan})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["previous_close"] == pytest.approx(100.0)

    def test_first_row_uses_open_when_pm_open_is_zero(self):
        """pm_open = 0 is treated as absent → fallback to open."""
        df = _make_df({"date_str": "2024-01-02", "open": 100.0, "pm_open": 0.0})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["previous_close"] == pytest.approx(100.0)

    # ── previous_close — consecutive days ─────────────────────────────────────

    def test_consecutive_days_use_prior_close(self):
        """Mon→Tue (1-day gap): second row's previous_close = Monday's close."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 110.0},
            {"date_str": "2024-01-03", "open": 112.0, "pm_open": 111.0},
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["previous_close"] == pytest.approx(110.0)

    def test_friday_to_monday_uses_prior_close(self):
        """Fri→Mon (3-day gap): within the ≤3 threshold → uses prior close."""
        df = _make_df(
            {"date_str": "2024-01-05", "close": 105.0},           # Friday
            {"date_str": "2024-01-08", "open": 106.0, "pm_open": 104.0},  # Monday
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["previous_close"] == pytest.approx(105.0)

    # ── previous_close — gap > 3 days ─────────────────────────────────────────

    def test_gap_over_3_days_falls_back_to_pm_open(self):
        """A gap >3 days (halt/holiday) → falls back to that day's pm_open."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 110.0},
            {"date_str": "2024-01-09", "open": 100.0, "pm_open": 97.0},  # 7-day gap
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["previous_close"] == pytest.approx(97.0)

    def test_gap_over_3_days_no_pm_open_falls_back_to_open(self):
        """A gap >3 days with no pm_open → falls back to that day's open."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 110.0},
            {"date_str": "2024-01-09", "open": 100.0, "pm_open": np.nan},
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["previous_close"] == pytest.approx(100.0)

    # ── derived columns ───────────────────────────────────────────────────────

    def test_gap_column(self):
        """gap = open - previous_close."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 100.0},
            {"date_str": "2024-01-03", "open":  105.0},
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["gap"] == pytest.approx(5.0, abs=1e-3)

    def test_gap_perc_column(self):
        """gap_perc = (open - previous_close) / previous_close * 100."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 100.0},
            {"date_str": "2024-01-03", "open":  105.0},
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["gap_perc"] == pytest.approx(5.0, abs=1e-3)

    def test_daily_range_column(self):
        """daily_range = high - low."""
        df = _make_df({"date_str": "2024-01-02", "high": 120.0, "low": 95.0})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["daily_range"] == pytest.approx(25.0)

    def test_day_range_perc_column(self):
        """day_range_perc = (high - low) / previous_close * 100."""
        # First row: pm_open = 100 → previous_close = 100
        df = _make_df({"date_str": "2024-01-02", "open": 100.0, "pm_open": 100.0, "high": 110.0, "low": 90.0})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["day_range_perc"] == pytest.approx(20.0, abs=1e-3)

    def test_gap_perc_negative_gap(self):
        """Negative gap (open < prev close) produces a negative gap_perc."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 100.0},
            {"date_str": "2024-01-03", "open":   90.0},
        )
        result = _enrich(df, splits=[])
        assert result.iloc[1]["gap_perc"] == pytest.approx(-10.0, abs=1e-3)

    # ── division-by-zero guard ────────────────────────────────────────────────

    def test_gap_perc_zero_when_previous_close_is_zero(self):
        """Guard: gap_perc and day_range_perc must be 0 when previous_close = 0."""
        df = _make_df(
            {"date_str": "2024-01-02", "close": 0.0},
            {"date_str": "2024-01-03", "open":  10.0, "pm_open": np.nan},
        )
        result = _enrich(df, splits=[])
        # Row 1: consecutive → previous_close = prior close = 0.0
        assert result.iloc[1]["gap_perc"]       == pytest.approx(0.0)
        assert result.iloc[1]["day_range_perc"] == pytest.approx(0.0)

    # ── placeholder columns ───────────────────────────────────────────────────

    def test_placeholder_columns(self):
        df = _make_df({"date_str": "2024-01-02"})
        result = _enrich(df, splits=[])
        assert result.iloc[0]["market_cap"]    == -1
        assert result.iloc[0]["stock_float"]   == -1
        assert result.iloc[0]["daily_200_sma"] == pytest.approx(-1.0)

    def test_placeholder_columns_all_rows(self):
        """Placeholder values must be set on every row, not just the first."""
        df = _make_df(
            {"date_str": "2024-01-02"},
            {"date_str": "2024-01-03"},
            {"date_str": "2024-01-04"},
        )
        result = _enrich(df, splits=[])
        assert (result["market_cap"]    == -1).all()
        assert (result["stock_float"]   == -1).all()
        assert (result["daily_200_sma"] == -1.0).all()

    # ── split adjustment integration ──────────────────────────────────────────

    def test_no_splits_sets_split_adjust_factor_to_one(self):
        """With an empty splits list every row gets split_adjust_factor = 1."""
        df = _make_df(
            {"date_str": "2024-01-02"},
            {"date_str": "2024-01-03"},
        )
        result = _enrich(df, splits=[])
        assert (result["split_adjust_factor"] == 1).all()

    # ── column selection and cleanup ──────────────────────────────────────────

    def test_day_column_is_dropped(self):
        """The internal 'day' column from process_data_minutes must not appear."""
        df = _make_df({"date_str": "2024-01-02"})
        df["day"] = "2024-01-02"   # simulate process_data_minutes output
        result = _enrich(df, splits=[])
        assert "day" not in result.columns

    def test_output_column_order_matches_schema(self):
        """Columns present in the result must follow _OUTPUT_COLS order."""
        df = _make_df({"date_str": "2024-01-02"})
        result = _enrich(df, splits=[])
        # Only check columns that are actually present (minimal test df)
        expected_order = [c for c in _OUTPUT_COLS if c in result.columns]
        assert list(result.columns) == expected_order

    def test_nan_values_filled_with_minus_one(self):
        """Any remaining NaN after enrichment must be replaced with -1."""
        df = _make_df({"date_str": "2024-01-02", "pm_open": float("nan")})
        # pm_open is used internally but not in _OUTPUT_COLS;
        # verify no NaN leaks through to the output columns
        result = _enrich(df, splits=[])
        assert not result.isnull().any().any()

    def test_zero_open_close_rows_are_removed(self):
        """Rows where both open=0 and close=0 must be filtered out."""
        df = _make_df(
            {"date_str": "2024-01-02", "open": 100.0, "close": 110.0},
            {"date_str": "2024-01-03", "open":   0.0, "close":   0.0},  # bad row
            {"date_str": "2024-01-04", "open": 102.0, "close": 105.0},
        )
        result = _enrich(df, splits=[])
        assert len(result) == 2
        assert "2024-01-03" not in result["date_str"].values

    def test_nonzero_open_zero_close_row_kept(self):
        """A row with open>0 and close=0 (unusual but possible) is kept."""
        df = _make_df({"date_str": "2024-01-02", "open": 5.0, "close": 0.0})
        result = _enrich(df, splits=[])
        assert len(result) == 1

    def test_index_reset_after_row_filter(self):
        """After filtering, the integer index must start at 0 with no gaps."""
        df = _make_df(
            {"date_str": "2024-01-02", "open": 100.0, "close": 110.0},
            {"date_str": "2024-01-03", "open":   0.0, "close":   0.0},
            {"date_str": "2024-01-04", "open": 102.0, "close": 105.0},
        )
        result = _enrich(df, splits=[])
        assert list(result.index) == [0, 1]

    def test_split_overrides_previous_close_on_execution_date(self):
        """On the split execution date previous_close must equal that day's open."""
        split_date = "2024-01-03"
        df = _make_df(
            {"date_str": "2024-01-02", "close": 200.0},
            # On the split date previous_close should be overridden to open
            {"date_str": split_date,   "open":  105.0, "close": 108.0},
            {"date_str": "2024-01-04", "open":  109.0},
        )
        splits = [{
            "execution_date":               split_date,
            "historical_adjustment_factor": 2.0,   # 2-for-1 split
        }]
        result = _enrich(df, splits=splits)
        split_row = result[result["date_str"] == split_date].iloc[0]
        assert split_row["previous_close"] == pytest.approx(split_row["open"])


# ── TestSaveToDb ──────────────────────────────────────────────────────────────

def _make_enriched_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal enriched DataFrame with *n* rows (enough for save_to_db)."""
    dates = [f"2024-01-{i+2:02d}" for i in range(n)]
    return pd.DataFrame({
        "ticker":   ["TEST"] * n,
        "date_str": dates,
        "open":     [100.0] * n,
        "close":    [110.0] * n,
        "high":     [115.0] * n,
        "low":      [ 95.0] * n,
        "time":     [i * 86_400_000 for i in range(n)],
    })


class TestSaveToDb:
    """Tests for save_to_db — all HTTP calls are mocked."""

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mock_response(self, status_code: int = 204) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.raise_for_status = MagicMock()
        return resp

    # ── happy path ────────────────────────────────────────────────────────────

    def test_single_batch_posts_once(self):
        """All records fit in one batch → exactly one POST is made."""
        results = {"TEST": _make_enriched_df(3)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
                batch_size=10,
            ))

        mock_client.post.assert_called_once()

    def test_multiple_batches(self):
        """10 rows with batch_size=4 → 3 POST calls."""
        results = {"TEST": _make_enriched_df(10)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
                batch_size=4,
            ))

        assert mock_client.post.call_count == 3

    def test_correct_endpoint_called(self):
        """The POST must hit /rpc/upsert_stock_data."""
        results = {"TEST": _make_enriched_df(1)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
            ))

        url_called = mock_client.post.call_args[0][0]
        assert url_called == "http://test:3031/rpc/upsert_stock_data"

    def test_auth_header_sent(self):
        """Authorization: Bearer <token> header must be present."""
        results = {"TEST": _make_enriched_df(1)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="my-secret-token",
            ))

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-token"

    def test_body_contains_p_data_key(self):
        """Request body must be {"p_data": [...]}."""
        results = {"TEST": _make_enriched_df(2)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
            ))

        body = mock_client.post.call_args[1]["json"]
        assert "p_data" in body
        assert isinstance(body["p_data"], list)
        assert len(body["p_data"]) == 2

    def test_bigint_columns_are_integers_not_floats(self):
        """BIGINT columns (volume, time, …) must serialize as int, not float.

        Pandas uses float64 for columns that had NaN values, so 175326 becomes
        175326.0 in to_dict().  PostgreSQL's jsonb_to_recordset rejects floats
        for bigint fields with «invalid input syntax for type bigint».
        """
        df = _make_enriched_df(1)
        # Add the bigint columns as floats to simulate the real pipeline output
        df["volume"]              = 175326.0
        df["premarket_volume"]    = 500.0
        df["market_hours_volume"] = 174826.0
        df["ah_volume"]           = 1000.0
        df["high_pm_time"]        = -1.0
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                {"TEST": df},
                postgrest_url="http://test:3031",
                token="tok",
            ))

        record = mock_client.post.call_args[1]["json"]["p_data"][0]
        for col in ("volume", "premarket_volume", "market_hours_volume", "ah_volume", "high_pm_time", "time"):
            if col in record:
                assert isinstance(record[col], int), f"{col} should be int, got {type(record[col])}"

    def test_multiple_tickers_combined(self):
        """Records from all tickers in the dict are sent in a single stream."""
        results = {
            "AAPL": _make_enriched_df(3),
            "MSFT": _make_enriched_df(3),
        }
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
                batch_size=10,
            ))

        body = mock_client.post.call_args[1]["json"]
        assert len(body["p_data"]) == 6

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_results_dict_makes_no_requests(self):
        """An empty results dict must not make any HTTP calls."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock()
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                {},
                postgrest_url="http://test:3031",
                token="tok",
            ))

        mock_client.post.assert_not_called()

    def test_none_df_in_results_is_skipped(self):
        """A ticker mapped to None (no data) must be skipped gracefully."""
        results = {"AAPL": None, "MSFT": _make_enriched_df(2)}
        resp = self._mock_response(204)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=resp)
            mock_cls.return_value  = mock_client

            asyncio.run(save_to_db(
                results,
                postgrest_url="http://test:3031",
                token="tok",
            ))

        body = mock_client.post.call_args[1]["json"]
        assert len(body["p_data"]) == 2

    def test_http_error_raises(self):
        """A non-2xx response must propagate as an exception."""
        results = {"TEST": _make_enriched_df(1)}

        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 500
        bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=bad_resp
        )
        bad_resp.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client.post       = AsyncMock(return_value=bad_resp)
            mock_cls.return_value  = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(save_to_db(
                    results,
                    postgrest_url="http://test:3031",
                    token="tok",
                ))


# ── TestStateFile ──────────────────────────────────────────────────────────────

class TestStateFile:
    """Tests for _load_failed_tickers and _save_state."""

    def test_load_returns_empty_set_when_file_missing(self, tmp_path):
        """No state file → empty set, no exception."""
        result = _load_failed_tickers(str(tmp_path / "missing.json"))
        assert result == set()

    def test_save_then_load_roundtrip(self, tmp_path):
        """Saved tickers must be recovered exactly on the next load."""
        path = str(tmp_path / "state.json")
        tickers = {"AAPL", "MSFT", "TSLA"}
        _save_state(path, tickers)
        loaded = _load_failed_tickers(path)
        assert loaded == tickers

    def test_save_writes_valid_json(self, tmp_path):
        """State file must be valid JSON with expected keys."""
        path = str(tmp_path / "state.json")
        _save_state(path, {"AAPL", "GOOG"})
        with open(path) as f:
            data = json.load(f)
        assert "failed_tickers" in data
        assert "count" in data
        assert "timestamp" in data
        assert data["count"] == 2
        assert set(data["failed_tickers"]) == {"AAPL", "GOOG"}

    def test_failed_tickers_sorted_in_file(self, tmp_path):
        """Tickers must be stored in sorted order for deterministic diffs."""
        path = str(tmp_path / "state.json")
        _save_state(path, {"TSLA", "AAPL", "MSFT"})
        with open(path) as f:
            data = json.load(f)
        assert data["failed_tickers"] == sorted(["TSLA", "AAPL", "MSFT"])

    def test_save_empty_set(self, tmp_path):
        """Saving an empty set must produce count=0 and an empty list."""
        path = str(tmp_path / "state.json")
        _save_state(path, set())
        loaded = _load_failed_tickers(path)
        assert loaded == set()
        with open(path) as f:
            data = json.load(f)
        assert data["count"] == 0
        assert data["failed_tickers"] == []

    def test_load_returns_empty_set_on_corrupt_file(self, tmp_path):
        """A corrupt state file must not raise — returns empty set with a warning."""
        path = tmp_path / "state.json"
        path.write_text("not valid json{{{")
        result = _load_failed_tickers(str(path))
        assert result == set()

    def test_load_returns_empty_set_when_key_missing(self, tmp_path):
        """A state file without 'failed_tickers' key returns empty set."""
        path = tmp_path / "state.json"
        path.write_text('{"count": 0}')
        result = _load_failed_tickers(str(path))
        assert result == set()

    def test_save_overwrites_previous_state(self, tmp_path):
        """Each save must overwrite the previous file completely."""
        path = str(tmp_path / "state.json")
        _save_state(path, {"OLD1", "OLD2"})
        _save_state(path, {"NEW1"})
        loaded = _load_failed_tickers(path)
        assert loaded == {"NEW1"}
        assert "OLD1" not in loaded

    def test_pipeline_state_file_constant_is_string(self):
        """_PIPELINE_STATE_FILE must be a non-empty string."""
        assert isinstance(_PIPELINE_STATE_FILE, str)
        assert len(_PIPELINE_STATE_FILE) > 0
