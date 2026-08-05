"""Candidate ladder — hermetic: fixture parquet cache, no network.

This is the agents' only channel for a name the desk does not already
hold, so the tests care most about the two failure modes that would put
the system back where it was on 2026-08-05: a ladder that silently
degrades to nothing, and one that raises and takes the cycle with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.agents.candidates import MIN_BARS, build_candidate_ladder
from trading.core.types import AssetClass, Instrument
from trading.data.cache import ParquetCache

UNIVERSE_YAML = """
universes:
  tiny:
    asset_class: equity
    description: "test"
    symbols: [AAA, BBB, CCC]
  mixed:
    asset_class: equity
    description: "equities plus an ETF, as the real universes are"
    symbols: [AAA, BBB, CCC, DDD]
"""


def _write_prices(
    data_dir: Path,
    symbol: str,
    *,
    bars: int,
    drift: float,
    asset_class: AssetClass = AssetClass.EQUITY,
) -> None:
    """A clean, deterministic ramp — the ranking must be a function of the
    drift, so a flaky ordering means a real bug, not RNG."""
    idx = pd.date_range("2023-01-02", periods=bars, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.arange(bars) * drift)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(bars, 1_000_000.0),
            "adj_close": close,
        },
        index=idx,
    )
    df.index.name = "ts"
    ParquetCache(data_dir).write(Instrument(symbol=symbol, asset_class=asset_class), "1D", df)


@pytest.fixture
def desk(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "universes.yaml").write_text(UNIVERSE_YAML)
    monkeypatch.setattr("trading.core.config.PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(
        "trading.core.universes.DEFAULT_UNIVERSES_PATH", tmp_path / "config" / "universes.yaml"
    )
    monkeypatch.setattr(
        "trading.core.universes.GENERATED_UNIVERSES_PATH", tmp_path / "config" / "none.yaml"
    )
    from trading.core.universes import clear_cache

    clear_cache()
    monkeypatch.setenv("UNIVERSE", "tiny")
    monkeypatch.setenv("STRATEGY", "top_k_momentum")
    data_dir = tmp_path / "data"
    _write_prices(data_dir, "AAA", bars=400, drift=0.0010)
    _write_prices(data_dir, "BBB", bars=400, drift=0.0005)
    _write_prices(data_dir, "CCC", bars=400, drift=0.0001)
    yield data_dir
    clear_cache()


def test_ladder_ranks_by_strategy_score(desk: Path) -> None:
    ladder = build_candidate_ladder(desk, top_n=3)
    assert ladder is not None
    assert [r["symbol"] for r in ladder["ranked"]] == ["AAA", "BBB", "CCC"]
    assert [r["rank"] for r in ladder["ranked"]] == [1, 2, 3]
    assert ladder["strategy"] == "top_k_momentum" and ladder["universe"] == "tiny"


def test_ladder_carries_52w_percentile(desk: Path) -> None:
    """Rank without position-in-range invites exactly the top-ticking the
    quant charter's first hard rule warns about."""
    ladder = build_candidate_ladder(desk, top_n=3)
    assert ladder is not None
    # Monotone ramps sit at their 52-week high by construction.
    assert all(r["pctile_52w"] == pytest.approx(1.0) for r in ladder["ranked"])


def test_etf_cached_under_the_etf_dir_is_still_found(desk: Path) -> None:
    """The trading universes mix equities and ETFs, and the cache files
    them under different asset directories. A first cut hardcoded
    AssetClass.EQUITY and found nothing at all — the ladder came back
    empty against a fully backfilled cache, silently.
    """
    _write_prices(desk, "DDD", bars=400, drift=0.0020, asset_class=AssetClass.ETF)
    (desk / "equity" / "DDD").exists()  # sanity: it is NOT under equity/
    ladder = build_candidate_ladder(desk, top_n=4, universe="mixed")
    assert ladder is not None
    assert "DDD" in [r["symbol"] for r in ladder["ranked"]]


def test_lowercase_freq_filename_is_still_found(desk: Path) -> None:
    """Older CLI fetches wrote ``1d.parquet``. macOS hides the case
    difference from ``1D.parquet``; the Linux VPS does not."""
    src = desk / "equity" / "BBB" / "1D.parquet"
    src.rename(src.with_name("1d.parquet"))
    ladder = build_candidate_ladder(desk, top_n=3)
    assert ladder is not None
    assert "BBB" in [r["symbol"] for r in ladder["ranked"]]


def test_short_history_symbols_are_excluded(tmp_path: Path, desk: Path) -> None:
    _write_prices(desk, "AAA", bars=MIN_BARS - 10, drift=0.002)
    ladder = build_candidate_ladder(desk, top_n=3)
    assert ladder is not None
    assert "AAA" not in [r["symbol"] for r in ladder["ranked"]]


def test_empty_cache_degrades_to_none(tmp_path: Path, monkeypatch) -> None:
    """No ladder is a survivable cycle; an exception is not."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "universes.yaml").write_text(UNIVERSE_YAML)
    monkeypatch.setattr(
        "trading.core.universes.DEFAULT_UNIVERSES_PATH", tmp_path / "config" / "universes.yaml"
    )
    monkeypatch.setattr(
        "trading.core.universes.GENERATED_UNIVERSES_PATH", tmp_path / "config" / "none.yaml"
    )
    from trading.core.universes import clear_cache

    clear_cache()
    monkeypatch.setenv("UNIVERSE", "tiny")
    assert build_candidate_ladder(tmp_path / "empty") is None
    clear_cache()


def test_unknown_universe_degrades_to_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UNIVERSE", "no_such_universe")
    assert build_candidate_ladder(tmp_path / "data") is None


def test_unknown_strategy_degrades_to_none(desk: Path, monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY", "no_such_strategy")
    assert build_candidate_ladder(desk) is None
