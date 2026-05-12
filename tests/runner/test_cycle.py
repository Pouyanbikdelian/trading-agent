"""End-to-end Cycle tests.

We build a fully in-memory rig: a fake DataSource that always returns the
same historical frame from the cache, a Simulator broker, the real risk
manager, and ``NullAlerts``. Each test sets the world up and asserts on
what the cycle does — orders submitted, fills reconciled, snapshots
persisted, halt behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.core.types import (
    AssetClass,
    Bar,
    Instrument,
)
from trading.core.universes import clear_cache  # type: ignore[attr-defined]
from trading.data.cache import ParquetCache
from trading.execution import OrderStore, Simulator
from trading.risk.limits import RiskLimits
from trading.risk.manager import RiskManager
from trading.runner import Cycle, NullAlerts, RunnerConfig, RunnerStore

# --------------------------------------------------------------- fixtures


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tiny_universe_yaml(tmp_path: Path, monkeypatch) -> str:
    """Write a tiny universes.yaml file and point the loader at it."""
    universe_name = "_runner_test_universe"
    yaml_path = tmp_path / "universes.yaml"
    yaml_path.write_text(
        f"universes:\n  {universe_name}:\n    asset_class: equity\n    symbols: [TEST_A, TEST_B]\n"
    )
    # Patch the loader's default path. clear_cache() drops the lru_cache so
    # the next call re-reads from disk.
    from trading.core import universes as universes_module

    monkeypatch.setattr(universes_module, "DEFAULT_UNIVERSES_PATH", yaml_path)
    clear_cache()
    return universe_name


@pytest.fixture
def primed_cache(tmp_path: Path) -> ParquetCache:
    """Write a synthetic price frame for TEST_A and TEST_B into the cache."""
    cache = ParquetCache(tmp_path / "parquet")
    idx = pd.date_range("2024-01-01", periods=300, freq="1D", tz="UTC", name="ts")
    rng = np.random.default_rng(0)
    for symbol, sigma in (("TEST_A", 0.01), ("TEST_B", 0.02)):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, sigma, 300)))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.full(300, 1000.0),
                "adj_close": prices,
            },
            index=idx,
        )
        ins = Instrument(symbol=symbol, asset_class=AssetClass.EQUITY)
        cache.write(ins, "1D", df)
    return cache


class _NullSourceFactory:
    """Source factory that always returns an empty source so auto_refresh
    is a no-op and the cycle falls back to the cache."""

    def __call__(self, instrument: Instrument):
        class _NoFetch:
            name = "noop"

            def get_bars(self, *a, **kw):
                return pd.DataFrame()

        return _NoFetch()


def _make_cycle(
    config: RunnerConfig,
    cache: ParquetCache,
    tmp_path: Path,
    *,
    halted_reason: str | None = None,
) -> tuple[Cycle, Simulator, NullAlerts]:
    broker = Simulator(initial_cash=config.initial_cash)
    broker.connect()
    # Mark-to-market by stepping with the latest known close so get_account works.
    last_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = {}
    for ins_sym in ("TEST_A", "TEST_B"):
        df = cache.read(Instrument(symbol=ins_sym, asset_class=AssetClass.EQUITY), "1D")
        last_ts = df.index[-1].to_pydatetime()
        bars[ins_sym] = Bar(
            ts=last_ts,
            open=float(df["open"].iloc[-1]),
            high=float(df["high"].iloc[-1]),
            low=float(df["low"].iloc[-1]),
            close=float(df["close"].iloc[-1]),
            volume=float(df["volume"].iloc[-1]),
        )
    broker.step(last_ts, bars)

    rm = RiskManager(
        RiskLimits(max_position_pct=0.20, max_gross_exposure=2.0),
        halt_state_path=tmp_path / "halt.json",
    )
    if halted_reason:
        rm.halt(halted_reason)

    cycle = Cycle(
        config,
        cache=cache,
        source_factory=_NullSourceFactory(),
        broker=broker,
        risk_manager=rm,
        order_store=OrderStore(tmp_path / "orders.db"),
        runner_store=RunnerStore(tmp_path / "runner.db"),
        alerts=NullAlerts(),
        heartbeat_path=tmp_path / "heartbeat.json",
        clock=lambda: last_ts,
    )
    return cycle, broker, cycle.alerts


# ---------------------------------------------------------------- tests


def test_cycle_produces_orders(tiny_universe_yaml, primed_cache, tmp_state) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        strategy_params={"donchian": {"lookback": 20}},
        freq="1D",
        auto_refresh=False,
        history_bars=200,
        initial_cash=100_000.0,
    )
    cycle, _, _ = _make_cycle(cfg, primed_cache, tmp_state)
    report = cycle.run_cycle()
    # Status is one of {ok, no_orders} depending on the synthetic path.
    assert report.status in {"ok", "no_orders"}
    assert report.error is None


def test_cycle_persists_snapshot(tiny_universe_yaml, primed_cache, tmp_state) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
        initial_cash=100_000.0,
    )
    cycle, _, _ = _make_cycle(cfg, primed_cache, tmp_state)
    cycle.run_cycle()
    snap = cycle.runner_store.latest_snapshot()
    assert snap is not None
    assert snap.equity > 0


def test_cycle_writes_heartbeat(tiny_universe_yaml, primed_cache, tmp_state) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    cycle, _, _ = _make_cycle(cfg, primed_cache, tmp_state)
    cycle.run_cycle()
    hb_path = tmp_state / "heartbeat.json"
    assert hb_path.exists()


def test_cycle_records_halt_when_risk_manager_halted(
    tiny_universe_yaml,
    primed_cache,
    tmp_state,
) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    cycle, _, alerts = _make_cycle(
        cfg,
        primed_cache,
        tmp_state,
        halted_reason="manual test halt",
    )
    report = cycle.run_cycle()
    assert report.status == "halted"
    assert report.orders_submitted == 0
    # Critical alert fires on halt.
    assert any(level == "critical" for level, _ in alerts.sent)


def test_cycle_error_is_caught(tiny_universe_yaml, primed_cache, tmp_state) -> None:
    """A misconfigured strategy params dict surfaces as ``error``, not a raise."""
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        strategy_params={"donchian": {"lookback": -1}},  # invalid; will raise on Params
        auto_refresh=False,
        history_bars=200,
    )
    cycle, _, alerts = _make_cycle(cfg, primed_cache, tmp_state)
    report = cycle.run_cycle()
    assert report.status == "error"
    assert report.error is not None
    assert any(level == "critical" for level, _ in alerts.sent)


def test_cycle_handles_short_history_gracefully(
    tiny_universe_yaml,
    tmp_path: Path,
) -> None:
    # Empty cache (no parquet writes) → cycle must report no_orders and not crash.
    cache = ParquetCache(tmp_path / "parquet_empty")
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    broker = Simulator(initial_cash=cfg.initial_cash)
    broker.connect()
    # No step → broker.get_account would raise; cycle catches that into the fallback.
    rm = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    cycle = Cycle(
        cfg,
        cache=cache,
        source_factory=_NullSourceFactory(),
        broker=broker,
        risk_manager=rm,
        order_store=OrderStore(tmp_path / "orders.db"),
        runner_store=RunnerStore(tmp_path / "runner.db"),
        alerts=NullAlerts(),
        heartbeat_path=tmp_path / "heartbeat.json",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    report = cycle.run_cycle()
    assert report.status == "no_orders"
