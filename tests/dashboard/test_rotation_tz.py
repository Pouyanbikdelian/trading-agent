"""The Rotation tab died on every refresh for a tz mismatch.

``_load_history`` draws from two sources — the parquet cache (tz-aware,
per the house rule) and a yfinance fallback for anything not cached
(tz-naive). ``pd.DataFrame(closes)`` cannot align the two:

    rotation compute failed: Cannot join tz-naive with tz-aware DatetimeIndex

Observed live 2026-08-11, once every refresh. Not intermittent: several
sector ETFs sit outside the traded universe and are therefore ALWAYS
fetched, so the mix was permanent. ``build_rotation`` swallows the
exception ("degraded tab beats a dead dashboard"), which is why this
presented as an empty tab rather than an error — the operator was told
nothing at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading.dashboard.rotation import _daily_utc


def _naive(n: int = 5, price: float = 100.0) -> pd.Series:
    """A yfinance-shaped series: tz-naive, stamped at midnight."""
    return pd.Series(
        [price + i for i in range(n)],
        index=pd.date_range("2026-08-01", periods=n, freq="D"),
    )


def _aware(n: int = 5, price: float = 200.0, hour: int = 20) -> pd.Series:
    """A parquet-shaped series: tz-aware, stamped at the close."""
    return pd.Series(
        [price + i for i in range(n)],
        index=pd.date_range(f"2026-08-01 {hour}:00", periods=n, freq="D", tz="UTC"),
    )


class TestTheJoinThatFailed:
    def test_mixing_the_two_sources_no_longer_raises(self) -> None:
        """The regression, in the exact shape it occurred."""
        closes = {"XLK": _aware(), "URA": _naive()}

        frame = pd.DataFrame({k: _daily_utc(v) for k, v in closes.items()})

        assert list(frame.columns) == ["XLK", "URA"]

    def test_the_raw_mix_really_does_raise(self) -> None:
        """Pins the premise — if pandas ever stops raising, this test
        stops being evidence and should be revisited."""
        with pytest.raises(Exception, match=r"tz-naive|tz-aware"):
            pd.DataFrame({"XLK": _aware(), "URA": _naive()}).to_numpy()
            pd.concat([_aware(), _naive()], axis=1)

    def test_both_sources_land_on_the_same_rows(self) -> None:
        """Unifying the timezone alone would leave two rows per day and
        every cross-symbol comparison reading NaN."""
        frame = pd.DataFrame({"XLK": _daily_utc(_aware()), "URA": _daily_utc(_naive())})

        assert len(frame) == 5
        assert frame.notna().all().all()


class TestNormalization:
    def test_a_naive_index_becomes_utc(self) -> None:
        assert str(_daily_utc(_naive()).index.tz) == "UTC"

    def test_an_aware_index_stays_utc(self) -> None:
        assert str(_daily_utc(_aware()).index.tz) == "UTC"

    def test_a_non_utc_index_is_converted_not_relabelled(self) -> None:
        """A New York close at 16:00 is 20:00 UTC — the same day, not the
        day before."""
        s = pd.Series([1.0], index=pd.DatetimeIndex(["2026-08-03 16:00"], tz="America/New_York"))

        assert _daily_utc(s).index[0] == pd.Timestamp("2026-08-03", tz="UTC")

    def test_intraday_stamps_collapse_to_the_date(self) -> None:
        assert _daily_utc(_aware(hour=20)).index[0] == pd.Timestamp("2026-08-01", tz="UTC")

    def test_values_are_preserved(self) -> None:
        assert list(_daily_utc(_naive()).to_numpy()) == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_the_input_series_is_not_mutated(self) -> None:
        s = _naive()
        before = s.index.tz

        _daily_utc(s)

        assert s.index.tz is before

    def test_two_stamps_on_one_day_keep_the_last(self) -> None:
        """Normalizing can collide — a duplicate index breaks reindexing
        downstream, so the later bar wins."""
        s = pd.Series(
            [1.0, 2.0],
            index=pd.DatetimeIndex(["2026-08-03 13:30", "2026-08-03 20:00"], tz="UTC"),
        )

        out = _daily_utc(s)

        assert len(out) == 1
        assert out.iloc[0] == 2.0

    def test_an_empty_series_does_not_raise(self) -> None:
        assert len(_daily_utc(pd.Series([], dtype=float, index=pd.DatetimeIndex([])))) == 0


def test_load_history_normalizes_before_building_the_frame() -> None:
    """The fix has to land before pd.DataFrame(), not after."""
    from pathlib import Path

    src = Path("src/trading/dashboard/rotation.py").read_text()
    norm = src.index("closes = {k: _daily_utc(v) for k, v in closes.items()}")
    frame = src.index("return pd.DataFrame(closes), dollar_vol")

    assert norm < frame
