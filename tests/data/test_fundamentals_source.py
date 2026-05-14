"""Fundamentals source tests — Parquet round-trip + fetch with a fake yf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trading.data.fundamentals_source import (
    fetch_fundamentals_yf,
    read_fundamentals_cache,
    write_fundamentals_cache,
)
from trading.selection import Fundamentals


class _FakeTicker:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


class _FakeYf:
    def __init__(self, infos: dict[str, dict[str, Any]]) -> None:
        self._infos = infos

    def Ticker(self, symbol: str) -> _FakeTicker:
        return _FakeTicker(self._infos.get(symbol, {}))


def test_fetch_maps_yfinance_keys() -> None:
    fake = _FakeYf(
        {
            "AAPL": {
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "marketCap": 3_500_000_000_000,
                "returnOnEquity": 0.40,
                "debtToEquity": 137.0,  # yfinance returns % (137 = 137%)
                "profitMargins": 0.25,
                "freeCashflow": 90_000_000_000,
            },
        }
    )
    out = fetch_fundamentals_yf(["AAPL"], downloader=fake, pause_seconds=0.0)
    assert "AAPL" in out
    aapl = out["AAPL"]
    assert aapl.sector == "Technology"
    assert aapl.market_cap == 3_500_000_000_000
    assert aapl.roe == 0.40
    # 137% → 1.37 after the normalization.
    assert aapl.debt_to_equity == pytest.approx(1.37, rel=1e-9)
    assert aapl.free_cash_flow_positive is True


def test_fetch_handles_missing_info_gracefully() -> None:
    fake = _FakeYf({"GHOST": {}})
    out = fetch_fundamentals_yf(["GHOST"], downloader=fake, pause_seconds=0.0)
    assert "GHOST" in out
    g = out["GHOST"]
    assert g.sector is None
    assert g.roe is None


def test_fetch_swallows_per_symbol_errors() -> None:
    """A yfinance exception on one symbol shouldn't kill the whole batch."""

    class _FailingYf:
        def Ticker(self, symbol: str) -> Any:
            if symbol == "BROKEN":
                raise RuntimeError("yfinance is sad today")
            return _FakeTicker({"sector": "Technology"})

    out = fetch_fundamentals_yf(
        ["AAPL", "BROKEN"],
        downloader=_FailingYf(),
        pause_seconds=0.0,
    )
    assert "AAPL" in out
    assert "BROKEN" not in out


def test_write_read_round_trip(tmp_path: Path) -> None:
    funds = {
        "AAPL": Fundamentals(
            symbol="AAPL",
            sector="Technology",
            roe=0.40,
            debt_to_equity=1.5,
            free_cash_flow_positive=True,
        ),
        "MSFT": Fundamentals(
            symbol="MSFT", sector="Technology", roe=0.30, free_cash_flow_positive=True
        ),
    }
    path = tmp_path / "fundamentals.parquet"
    write_fundamentals_cache(path, funds)
    assert path.exists()
    out = read_fundamentals_cache(path)
    assert set(out) == {"AAPL", "MSFT"}
    assert out["AAPL"].roe == pytest.approx(0.40)


def test_read_missing_file_returns_empty(tmp_path: Path) -> None:
    out = read_fundamentals_cache(tmp_path / "does-not-exist.parquet")
    assert out == {}


def test_write_empty_dict_skips(tmp_path: Path) -> None:
    path = tmp_path / "fundamentals.parquet"
    write_fundamentals_cache(path, {})
    assert not path.exists()
