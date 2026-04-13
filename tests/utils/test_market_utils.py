"""
Tests for app.utils.market_utils
"""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.utils.market_utils import chunk_date_range, daily_sessions, process_data_minutes

FIXTURES = Path(__file__).parent / "fixtures_1m.parquet"

ET = ZoneInfo("America/New_York")


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(dt_str: str) -> int:
    """ET datetime string → UTC milliseconds."""
    return int(datetime.fromisoformat(dt_str).replace(tzinfo=ET).timestamp() * 1000)


def _candle(ticker, dt_et, o, h, l, c, v):
    return dict(ticker=ticker, time=_ms(dt_et), open=o, high=h, low=l, close=c, volume=v)


# ─────────────────────────────────────────────────────────────────────────────
#  chunk_date_range
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkDateRange:

    def test_single_month_exact(self):
        chunks = chunk_date_range("AAPL", "2024-01-01", "2024-01-31")
        assert chunks == [("AAPL", "2024-01-01", "2024-01-31")]

    def test_single_month_partial_start(self):
        chunks = chunk_date_range("AAPL", "2024-01-15", "2024-01-31")
        assert chunks == [("AAPL", "2024-01-15", "2024-01-31")]

    def test_single_month_partial_end(self):
        chunks = chunk_date_range("AAPL", "2024-01-01", "2024-01-20")
        assert chunks == [("AAPL", "2024-01-01", "2024-01-20")]

    def test_three_months_partial_both_ends(self):
        chunks = chunk_date_range("NVDA", "2020-03-12", "2020-05-13")
        assert chunks == [
            ("NVDA", "2020-03-12", "2020-03-31"),
            ("NVDA", "2020-04-01", "2020-04-30"),
            ("NVDA", "2020-05-01", "2020-05-13"),
        ]

    def test_leap_year_february(self):
        chunks = chunk_date_range("X", "2024-02-01", "2024-02-29")
        assert chunks == [("X", "2024-02-01", "2024-02-29")]

    def test_non_leap_year_february(self):
        chunks = chunk_date_range("X", "2023-02-01", "2023-02-28")
        assert chunks == [("X", "2023-02-01", "2023-02-28")]

    def test_year_boundary_december_to_january(self):
        chunks = chunk_date_range("SPY", "2023-12-15", "2024-01-15")
        assert chunks == [
            ("SPY", "2023-12-15", "2023-12-31"),
            ("SPY", "2024-01-01", "2024-01-15"),
        ]

    def test_accepts_date_objects(self):
        chunks = chunk_date_range("X", date(2024, 3, 1), date(2024, 4, 30))
        assert chunks == [
            ("X", "2024-03-01", "2024-03-31"),
            ("X", "2024-04-01", "2024-04-30"),
        ]

    def test_single_day(self):
        chunks = chunk_date_range("TSLA", "2024-06-15", "2024-06-15")
        assert chunks == [("TSLA", "2024-06-15", "2024-06-15")]

    def test_large_range_chunk_count(self):
        chunks = chunk_date_range("NVDA", "2020-01-01", "2025-04-04")
        assert chunks[0]  == ("NVDA", "2020-01-01", "2020-01-31")
        assert chunks[-1] == ("NVDA", "2025-04-01", "2025-04-04")
        assert len(chunks) == 64

    def test_last_day_of_month_is_included(self):
        """to_date on last calendar day → single chunk, no extra empty chunk."""
        chunks = chunk_date_range("X", "2024-03-01", "2024-03-31")
        assert len(chunks) == 1
        assert chunks[0][2] == "2024-03-31"


# ─────────────────────────────────────────────────────────────────────────────
#  daily_sessions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def one_day_df():
    """One trading day (2024-01-02) with candles in all three sessions."""
    rows = [
        # Pre-market
        _candle("AAPL", "2024-01-02 05:00", 183.0, 184.0, 182.5, 183.5, 10_000),
        _candle("AAPL", "2024-01-02 07:30", 183.5, 185.0, 183.0, 184.8, 15_000),
        # Market hours
        _candle("AAPL", "2024-01-02 09:30", 185.0, 187.0, 184.0, 186.5, 500_000),
        _candle("AAPL", "2024-01-02 12:00", 186.5, 188.0, 186.0, 187.0, 300_000),
        _candle("AAPL", "2024-01-02 15:55", 187.0, 187.5, 186.8, 187.2, 200_000),
        # After hours
        _candle("AAPL", "2024-01-02 16:05", 187.2, 188.5, 187.0, 188.0,  50_000),
        _candle("AAPL", "2024-01-02 18:00", 188.0, 189.0, 187.8, 188.5,  30_000),
    ]
    return pd.DataFrame(rows)


class TestDailySessions:

    def test_returns_one_row_per_ticker_date(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
        assert str(result.iloc[0]["date"]) == "2024-01-02"

    # ── Pre-market ────────────────────────────────────────────────────────────

    def test_premarket_high(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["high_pm"] == pytest.approx(185.0)

    def test_premarket_low(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["low_pm"] == pytest.approx(182.5)

    def test_premarket_volume(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["premarket_volume"] == 25_000

    # ── Market hours ──────────────────────────────────────────────────────────

    def test_day_open_is_first_candle_open(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["day_open"] == pytest.approx(185.0)

    def test_day_close_is_last_candle_close(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["day_close"] == pytest.approx(187.2)

    def test_day_high(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["high_mh"] == pytest.approx(188.0)

    def test_day_low(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["day_low"] == pytest.approx(184.0)

    # ── After hours ───────────────────────────────────────────────────────────

    def test_ah_open_is_first_candle_open(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_open"] == pytest.approx(187.2)

    def test_ah_close_is_last_candle_close(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_close"] == pytest.approx(188.5)

    def test_ah_high(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_high"] == pytest.approx(189.0)

    def test_ah_low(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_low"] == pytest.approx(187.0)

    def test_ah_volume(self, one_day_df):
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_volume"] == 80_000

    def test_ah_range(self, one_day_df):
        # ah_range = ah_high - ah_open = 189.0 - 187.2 = 1.8
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_range"] == pytest.approx(1.8)

    def test_ah_range_pct(self, one_day_df):
        # ah_range_pct = 1.8 / 187.2 * 100 ≈ 0.9615...
        result = daily_sessions(one_day_df)
        assert result.iloc[0]["ah_range_pct"] == pytest.approx(1.8 / 187.2 * 100)

    # ── Multi-day / multi-ticker ───────────────────────────────────────────────

    def test_two_days_produce_two_rows(self):
        rows = [
            _candle("AAPL", "2024-01-02 09:30", 185.0, 187.0, 184.0, 186.5, 500_000),
            _candle("AAPL", "2024-01-02 15:55", 186.5, 187.5, 186.0, 187.2, 200_000),
            _candle("AAPL", "2024-01-03 09:30", 188.0, 190.0, 187.5, 189.0, 400_000),
            _candle("AAPL", "2024-01-03 15:55", 189.0, 189.5, 188.0, 189.2, 150_000),
        ]
        result = daily_sessions(pd.DataFrame(rows))
        assert len(result) == 2
        dates = sorted(str(d) for d in result["date"])
        assert dates == ["2024-01-02", "2024-01-03"]

    def test_two_tickers_same_day(self):
        rows = [
            _candle("AAPL", "2024-01-02 09:30", 185.0, 187.0, 184.0, 186.5, 500_000),
            _candle("NVDA", "2024-01-02 09:30", 500.0, 510.0, 499.0, 505.0, 300_000),
        ]
        result = daily_sessions(pd.DataFrame(rows))
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "NVDA"}

    def test_missing_session_produces_nan(self):
        """A day with no after-hours candles → ah_* columns are NaN."""
        rows = [
            _candle("AAPL", "2024-01-02 09:30", 185.0, 187.0, 184.0, 186.5, 500_000),
            _candle("AAPL", "2024-01-02 15:55", 186.5, 187.5, 186.0, 187.2, 200_000),
        ]
        result = daily_sessions(pd.DataFrame(rows))
        row = result.iloc[0]
        assert pd.isna(row["ah_open"])
        assert pd.isna(row["ah_high"])
        assert pd.isna(row["ah_volume"])
        assert pd.isna(row["ah_range"])

    def test_session_boundary_930_is_market_hours(self):
        """A candle exactly at 09:30 ET belongs to market hours, not pre-market."""
        rows = [
            _candle("X", "2024-01-02 09:29", 100.0, 101.0, 99.0, 100.5, 1_000),  # pm
            _candle("X", "2024-01-02 09:30", 100.5, 102.0, 100.0, 101.5, 5_000),  # mh
        ]
        result = daily_sessions(pd.DataFrame(rows))
        row = result.iloc[0]
        assert row["premarket_volume"] == 1_000
        assert row["day_open"] == pytest.approx(100.5)

    def test_session_boundary_1600_is_after_hours(self):
        """A candle exactly at 16:00 ET belongs to after hours, not market hours."""
        rows = [
            _candle("X", "2024-01-02 15:59", 100.0, 101.0, 99.5, 100.8,  5_000),  # mh
            _candle("X", "2024-01-02 16:00", 100.8, 102.0, 100.5, 101.5, 10_000),  # ah
        ]
        result = daily_sessions(pd.DataFrame(rows))
        row = result.iloc[0]
        assert row["day_close"] == pytest.approx(100.8)
        assert row["ah_open"] == pytest.approx(100.8)

    def test_time_unit_seconds(self):
        """time_unit='s' produces the same result as milliseconds / 1000."""
        rows_ms = [
            _candle("AAPL", "2024-01-02 09:30", 185.0, 187.0, 184.0, 186.5, 500_000),
        ]
        df_ms = pd.DataFrame(rows_ms)
        df_s  = df_ms.copy()
        df_s["time"] = df_s["time"] // 1000

        result_ms = daily_sessions(df_ms, time_unit="ms")
        result_s  = daily_sessions(df_s,  time_unit="s")

        assert result_ms.iloc[0]["day_open"] == result_s.iloc[0]["day_open"]
        assert str(result_ms.iloc[0]["date"]) == str(result_s.iloc[0]["date"])


# ─────────────────────────────────────────────────────────────────────────────
#  process_data_minutes  (tests use real 1-min data from fixtures_1m.parquet)
# ─────────────────────────────────────────────────────────────────────────────

def _load_fixture(ticker: str) -> list[dict]:
    """
    Load fixture parquet and convert to the raw Polygon/Massive format that
    process_data_minutes expects: keys o/c/h/l/v/t  (t in UTC milliseconds).
    """
    df = pd.read_parquet(FIXTURES)
    sub = df[df["ticker"] == ticker].copy()
    sub["t"] = sub["time"] * 1000          # seconds → milliseconds
    return sub.rename(columns={
        "open": "o", "close": "c", "high": "h", "low": "l", "volume": "v",
    })[["o", "c", "h", "l", "v", "t"]].to_dict("records")


@pytest.fixture(scope="module")
def batl_result():
    """process_data_minutes output for BATL (single day: 2026-02-27)."""
    return process_data_minutes(_load_fixture("BATL"))


@pytest.fixture(scope="module")
def cdio_result():
    """process_data_minutes output for CDIO (8 days: 2026-02-18 → 2026-02-27)."""
    return process_data_minutes(_load_fixture("CDIO"))


class TestProcessDataMinutes:

    # ── Output shape and columns ──────────────────────────────────────────────

    def test_returns_dataframe(self, batl_result):
        assert isinstance(batl_result, pd.DataFrame)

    def test_batl_one_row_per_day(self, batl_result):
        assert len(batl_result) == 1

    def test_cdio_correct_day_count(self, cdio_result):
        """CDIO has 8 trading days in the fixture range (weekends excluded)."""
        assert len(cdio_result) == 8

    def test_expected_columns_present(self, batl_result):
        expected = {
            "date_str", "open", "close", "high", "low", "volume",
            "high_pm", "low_pm", "pm_open", "premarket_volume",
            "high_mh", "low_mh", "market_hours_volume",
            "ah_open", "ah_close", "ah_high", "ah_low", "ah_volume",
            "ah_range", "ah_range_perc",
            "high_pm_time", "highest_in_pm", "time",
        }
        assert expected.issubset(set(batl_result.columns))

    def test_empty_input_returns_empty_dataframe(self):
        result = process_data_minutes([])
        assert result is not None
        assert len(result) == 0

    # ── BATL 2026-02-27 — market hours ───────────────────────────────────────

    def test_batl_date_str(self, batl_result):
        assert batl_result.iloc[0]["date_str"] == "2026-02-27"

    def test_batl_day_open(self, batl_result):
        assert batl_result.iloc[0]["open"] == pytest.approx(5.11)

    def test_batl_day_close(self, batl_result):
        assert batl_result.iloc[0]["close"] == pytest.approx(5.51)

    def test_batl_high_mh(self, batl_result):
        assert batl_result.iloc[0]["high_mh"] == pytest.approx(6.0)

    def test_batl_low_mh(self, batl_result):
        assert batl_result.iloc[0]["low_mh"] == pytest.approx(4.41)

    # ── BATL 2026-02-27 — pre-market ─────────────────────────────────────────

    def test_batl_high_pm(self, batl_result):
        assert batl_result.iloc[0]["high_pm"] == pytest.approx(6.13)

    def test_batl_low_pm(self, batl_result):
        assert batl_result.iloc[0]["low_pm"] == pytest.approx(4.16)

    def test_batl_premarket_volume(self, batl_result):
        assert batl_result.iloc[0]["premarket_volume"] == pytest.approx(13_540_750, rel=1e-3)

    # ── BATL 2026-02-27 — after hours ────────────────────────────────────────

    def test_batl_ah_open(self, batl_result):
        assert batl_result.iloc[0]["ah_open"] == pytest.approx(5.51)

    def test_batl_ah_close(self, batl_result):
        assert batl_result.iloc[0]["ah_close"] == pytest.approx(5.52)

    def test_batl_ah_high(self, batl_result):
        assert batl_result.iloc[0]["ah_high"] == pytest.approx(5.89)

    def test_batl_ah_low(self, batl_result):
        assert batl_result.iloc[0]["ah_low"] == pytest.approx(5.42)

    def test_batl_ah_range(self, batl_result):
        # ah_range = ah_high - ah_open = 5.89 - 5.51 = 0.38
        assert batl_result.iloc[0]["ah_range"] == pytest.approx(0.38, abs=1e-4)

    def test_batl_ah_range_perc(self, batl_result):
        # ah_range_perc = 0.38 / 5.51 * 100
        expected = 0.38 / 5.51 * 100
        assert batl_result.iloc[0]["ah_range_perc"] == pytest.approx(expected, rel=1e-3)

    # ── BATL 2026-02-27 — derived fields ─────────────────────────────────────

    def test_batl_high_is_max_of_pm_and_mh(self, batl_result):
        # high_pm=6.13 > high_mh=6.0 → high should be 6.13
        assert batl_result.iloc[0]["high"] == pytest.approx(6.13)

    def test_batl_low_is_min_of_pm_and_mh(self, batl_result):
        # low_pm=4.16 < low_mh=4.41 → low should be 4.16
        assert batl_result.iloc[0]["low"] == pytest.approx(4.16)

    def test_batl_highest_in_pm_true(self, batl_result):
        # high_pm (6.13) >= high_mh (6.0) → True
        assert batl_result.iloc[0]["highest_in_pm"]

    def test_batl_time_is_et_midnight_epoch_ms(self, batl_result):
        """time column must be midnight America/New_York in UTC milliseconds."""
        et = ZoneInfo("America/New_York")
        expected_ms = int(
            datetime(2026, 2, 27, 0, 0, 0, tzinfo=et).timestamp() * 1000
        )
        assert batl_result.iloc[0]["time"] == expected_ms

    def test_batl_high_pm_time_is_epoch_ms(self, batl_result):
        """high_pm_time must be a valid epoch ms (positive integer)."""
        t = batl_result.iloc[0]["high_pm_time"]
        assert t > 0
        # Must correspond to a time on 2026-02-27
        dt = datetime.fromtimestamp(t / 1000, tz=ZoneInfo("UTC"))
        assert dt.year == 2026 and dt.month == 2 and dt.day == 27

    # ── CDIO — multi-day correctness ─────────────────────────────────────────

    def test_cdio_date_range(self, cdio_result):
        dates = sorted(cdio_result["date_str"].tolist())
        assert dates[0] == "2026-02-18"
        assert dates[-1] == "2026-02-27"

    def test_cdio_no_weekend_rows(self, cdio_result):
        """Sat 2026-02-21 and Sun 2026-02-22 must not appear."""
        assert "2026-02-21" not in cdio_result["date_str"].values
        assert "2026-02-22" not in cdio_result["date_str"].values

    def test_cdio_feb26_highest_in_pm_true(self, cdio_result):
        """On 2026-02-26 the pm high exceeded the market-hours high."""
        row = cdio_result[cdio_result["date_str"] == "2026-02-26"].iloc[0]
        assert row["highest_in_pm"]

    def test_cdio_feb18_highest_in_pm_false(self, cdio_result):
        row = cdio_result[cdio_result["date_str"] == "2026-02-18"].iloc[0]
        assert not row["highest_in_pm"]

    def test_cdio_feb26_ah_range_perc(self, cdio_result):
        """2026-02-26 had a large AH move (ah_range_perc ≈ 96.65%)."""
        row = cdio_result[cdio_result["date_str"] == "2026-02-26"].iloc[0]
        assert row["ah_range_perc"] == pytest.approx(96.653543, rel=1e-3)

    def test_cdio_all_rows_have_market_hours(self, cdio_result):
        """Every trading day must have a valid market-hours open and close."""
        assert cdio_result["open"].isna().sum() == 0
        assert cdio_result["close"].isna().sum() == 0

    def test_cdio_premarket_volume_non_negative(self, cdio_result):
        assert (cdio_result["premarket_volume"] >= 0).all()

    # ── Midnight-to-04:00 candles are excluded from pre-market ───────────────

    def test_pre_market_excludes_candles_before_0400(self):
        """A candle at 03:00 ET must NOT be counted in pre-market stats."""
        et = ZoneInfo("America/New_York")

        def raw_ms(dt_str):
            return int(datetime.fromisoformat(dt_str).replace(tzinfo=et).timestamp() * 1000)

        data = [
            {"t": raw_ms("2026-02-27 03:00"), "o": 50.0, "h": 60.0, "l": 49.0, "c": 55.0, "v": 9_999},
            {"t": raw_ms("2026-02-27 05:00"), "o": 10.0, "h": 11.0, "l": 9.0,  "c": 10.5, "v": 1_000},
            {"t": raw_ms("2026-02-27 09:30"), "o": 10.5, "h": 12.0, "l": 10.0, "c": 11.5, "v": 50_000},
        ]
        result = process_data_minutes(data)
        row = result.iloc[0]
        # high_pm must be 11.0 (05:00 candle), not 60.0 (03:00 candle)
        assert row["high_pm"] == pytest.approx(11.0)
        # premarket_volume must be 1_000, not 10_999
        assert row["premarket_volume"] == pytest.approx(1_000)

    # ── pm_open ───────────────────────────────────────────────────────────────

    def test_pm_open_is_first_premarket_candle_open(self):
        """pm_open must equal the open of the earliest valid pre-market candle."""
        et = ZoneInfo("America/New_York")

        def raw_ms(dt_str):
            return int(datetime.fromisoformat(dt_str).replace(tzinfo=et).timestamp() * 1000)

        data = [
            # excluded (before 04:00)
            {"t": raw_ms("2026-02-27 03:00"), "o": 99.0, "h": 100.0, "l": 98.0, "c": 99.5, "v": 1_000},
            # first valid pre-market candle → pm_open should be 55.0
            {"t": raw_ms("2026-02-27 04:00"), "o": 55.0, "h": 57.0,  "l": 54.0, "c": 56.0, "v": 2_000},
            {"t": raw_ms("2026-02-27 05:00"), "o": 56.0, "h": 58.0,  "l": 55.0, "c": 57.0, "v": 1_000},
            {"t": raw_ms("2026-02-27 09:30"), "o": 57.0, "h": 60.0,  "l": 56.0, "c": 59.0, "v": 50_000},
        ]
        result = process_data_minutes(data)
        assert result.iloc[0]["pm_open"] == pytest.approx(55.0)

    def test_pm_open_nan_when_no_premarket(self):
        """When a day has no pre-market candles pm_open must be NaN."""
        et = ZoneInfo("America/New_York")

        def raw_ms(dt_str):
            return int(datetime.fromisoformat(dt_str).replace(tzinfo=et).timestamp() * 1000)

        data = [
            {"t": raw_ms("2026-02-27 09:30"), "o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0, "v": 100_000},
            {"t": raw_ms("2026-02-27 10:00"), "o": 103.0, "h": 106.0, "l": 102.0, "c": 104.0, "v": 80_000},
        ]
        result = process_data_minutes(data)
        assert pd.isna(result.iloc[0]["pm_open"])

    def test_batl_pm_open_is_positive(self, batl_result):
        """BATL has pre-market activity → pm_open must be a valid positive price."""
        pm_open = batl_result.iloc[0]["pm_open"]
        assert not pd.isna(pm_open)
        assert pm_open > 0
