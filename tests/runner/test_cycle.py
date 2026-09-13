"""End-to-end Cycle tests.

We build a fully in-memory rig: a fake DataSource that always returns the
same historical frame from the cache, a Simulator broker, the real risk
manager, and ``NullAlerts``. Each test sets the world up and asserts on
what the cycle does — orders submitted, fills reconciled, snapshots
persisted, halt behavior.
"""

from __future__ import annotations

import json
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


def test_cycle_ticks_simulator_and_fills_match_submissions(
    tiny_universe_yaml, primed_cache, tmp_state
) -> None:
    """The cycle must drive the Simulator's clock so paper-trade fills
    materialize in the same cycle they were submitted in. Regression for
    the original bug where Simulator orders stayed in PENDING forever."""
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["risk_parity"],
        strategy_params={"risk_parity": {"vol_lookback": 30, "rebalance": 1}},
        auto_refresh=False,
        history_bars=200,
        initial_cash=100_000.0,
    )
    cycle, _, _ = _make_cycle(cfg, primed_cache, tmp_state)
    report = cycle.run_cycle()
    assert report.status == "ok"
    assert report.orders_submitted > 0
    # Every submitted order must have a matching fill — no orphan PENDINGs.
    assert report.fills_received == report.orders_submitted


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


def test_cycle_turns_active_halt_into_non_executable_live_account_review(
    tiny_universe_yaml,
    primed_cache,
    tmp_state,
    monkeypatch,
) -> None:
    """A halted `/cycle` still shows a fresh plan but cannot touch execution state."""
    from trading.core import config as config_module

    monkeypatch.setattr(
        config_module,
        "settings",
        config_module.settings.model_copy(
            update={"state_dir": tmp_state, "require_cycle_approval": True}
        ),
    )
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["risk_parity"],
        strategy_params={"risk_parity": {"vol_lookback": 30, "rebalance": 1}},
        auto_refresh=False,
        history_bars=200,
    )
    cycle, broker, alerts = _make_cycle(
        cfg,
        primed_cache,
        tmp_state,
        halted_reason="manual test halt",
    )

    submissions: list[object] = []
    order_writes: list[object] = []

    monkeypatch.setattr(broker, "submit_order", lambda order: submissions.append(order))
    monkeypatch.setattr(
        cycle.order_store, "save_order", lambda order, **_kw: order_writes.append(order)
    )
    monkeypatch.setattr(
        cycle,
        "_request_cycle_approval",
        lambda *_args, **_kwargs: pytest.fail("a halted review must not open approval"),
    )

    report = cycle.run_cycle()

    assert report.status == "halted_review"
    assert report.orders_submitted == 0
    assert submissions == []
    assert order_writes == []
    assert cycle.order_store.load_orders() == []
    assert not (tmp_state / cycle.APPROVAL_PENDING_FILE).exists()
    assert not (tmp_state / cycle.APPROVAL_DECISION_FILE).exists()
    artifact = json.loads((tmp_state / cycle.HALTED_REVIEW_FILE).read_text())
    assert artifact["kind"] == "halted_review"
    assert artifact["will_never_auto_execute"] is True
    assert artifact["account"]["base_currency"] == "USD"
    # The normal risk breach remains visible; it is not treated as recovery.
    assert any(level == "critical" for level, _ in alerts.sent)


def test_live_cycle_with_no_trusted_open_baseline_is_review_only(
    tiny_universe_yaml,
    primed_cache,
    tmp_state,
    monkeypatch,
) -> None:
    """A noon/restart live cycle must not invent a daily open and trade."""
    from trading.core import config as config_module

    monkeypatch.setattr(
        config_module,
        "settings",
        config_module.settings.model_copy(
            update={
                "state_dir": tmp_state,
                "trading_env": "live",
                "allow_live_trading": True,
                "require_cycle_approval": True,
            }
        ),
    )
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["risk_parity"],
        strategy_params={"risk_parity": {"vol_lookback": 30, "rebalance": 1}},
        auto_refresh=False,
        history_bars=200,
    )
    cycle, broker, _alerts = _make_cycle(cfg, primed_cache, tmp_state)
    submissions: list[object] = []
    monkeypatch.setattr(broker, "submit_order", lambda order: submissions.append(order))

    report = cycle.run_cycle()

    assert report.status == "halted_review"
    assert report.orders_submitted == 0
    assert submissions == []
    assert not cycle.risk_manager.is_halted()
    review = json.loads((tmp_state / cycle.HALTED_REVIEW_FILE).read_text())
    assert "baseline" in review["halt_reason"].lower() or "session" in review["halt_reason"].lower()


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


def test_cycle_with_smart_combiner_runs_end_to_end(
    tiny_universe_yaml, primed_cache, tmp_state
) -> None:
    """Multi-strategy cycle through a risk-aware combiner must complete
    without raising. We don't pin numerical output here — the strategies
    themselves are already covered in their own tests."""
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian", "ema_cross"],
        combiner="dsr_weighted",
        strategy_params={
            "donchian": {"lookback": 20},
            "ema_cross": {"fast_span": 5, "slow_span": 20},
        },
        auto_refresh=False,
        history_bars=200,
        initial_cash=100_000.0,
    )
    cycle, _, _ = _make_cycle(cfg, primed_cache, tmp_state)
    report = cycle.run_cycle()
    assert report.status in {"ok", "no_orders"}
    assert report.error is None


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


def test_short_history_symbol_does_not_truncate_price_matrix(
    tiny_universe_yaml, primed_cache, tmp_state, tmp_path
) -> None:
    """Regression for the June-2026 dead-month: one freshly listed symbol
    with ~26 bars inner-joined the whole 300-bar matrix down to 26 rows,
    starving the momentum lookback — the strategy silently held for
    weeks. Short-history symbols must be dropped, not the matrix rows."""
    idx = pd.date_range("2024-10-01", periods=26, freq="1D", tz="UTC", name="ts")
    prices = np.linspace(50.0, 55.0, 26)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.full(26, 1000.0),
            "adj_close": prices,
        },
        index=idx,
    )
    baby = Instrument(symbol="TEST_IPO", asset_class=AssetClass.EQUITY)
    primed_cache.write(baby, "1D", df)

    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        freq="1D",
        auto_refresh=False,
        history_bars=250,
    )
    cycle, _broker, _alerts = _make_cycle(cfg, primed_cache, tmp_path)
    instruments = [
        Instrument(symbol=s, asset_class=AssetClass.EQUITY)
        for s in ("TEST_A", "TEST_B", "TEST_IPO")
    ]
    ts = datetime(2024, 10, 26, tzinfo=timezone.utc)
    wide = cycle._load_prices(instruments, ts)

    assert "TEST_IPO" not in wide.columns  # baby symbol excluded…
    assert set(wide.columns) == {"TEST_A", "TEST_B"}
    assert len(wide) >= 200  # …and the matrix keeps its full history


def test_missing_fundamentals_warns_sector_cap_disabled(
    tiny_universe_yaml, primed_cache, tmp_state, tmp_path
) -> None:
    """The 30% sector cap silently never bound in production because no
    fundamentals cache existed (found 2026-07-14 with ~90% in one
    sector). A cycle without sector tags must alert the operator that
    the cap is NOT enforced — failing open is allowed, silence is not."""
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        strategy_params={"donchian": {"lookback": 20}},
        freq="1D",
        auto_refresh=False,
        history_bars=250,
        fundamentals_path=str(tmp_path / "nope" / "fundamentals.parquet"),
    )
    cycle, _broker, alerts = _make_cycle(cfg, primed_cache, tmp_path)
    cycle.run_cycle()
    assert any(
        lvl == "warning" and "sector cap" in msg.lower() and "not enforced" in msg.lower()
        for lvl, msg in alerts.sent
    )


@pytest.fixture
def pm_prep_cycle(tiny_universe_yaml, primed_cache, tmp_state, monkeypatch):
    """The PM's input stage must remain independent of all execution state."""
    from trading.core import config as config_module
    from trading.runner import cycle as cycle_module

    monkeypatch.setattr(
        config_module,
        "settings",
        config_module.settings.model_copy(update={"state_dir": tmp_state}),
    )
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["top_k_momentum"],
        strategy_params={"top_k_momentum": {"k": 1}},
        freq="1D",
        auto_refresh=True,
        history_bars=250,
    )
    cycle, _, alerts = _make_cycle(cfg, primed_cache, tmp_state)

    def forbidden(*_args, **_kwargs):
        pytest.fail("PM candidate preparation entered the account/order path")

    class _NoExecutionAccess:
        def __getattr__(self, name):
            pytest.fail(f"PM candidate preparation accessed execution member {name}")

    for attribute in ("broker", "risk_manager", "order_store", "runner_store"):
        monkeypatch.setattr(cycle, attribute, _NoExecutionAccess())
    for method in ("_run_inner", "_fetch_account", "_request_cycle_approval", "_add_pm_targets"):
        monkeypatch.setattr(cycle, method, forbidden)
    monkeypatch.setattr(cycle_module, "execution_lock", forbidden)
    return cycle, alerts


def test_pm_candidate_prep_refreshes_week_old_prices_before_publishing(
    pm_prep_cycle, primed_cache, tmp_state, monkeypatch
) -> None:
    """A weekly PM must rank newly fetched bars before its next trade cycle."""
    from trading.agents.candidates import (
        candidate_snapshot_path,
        write_runner_candidate_snapshot,
    )
    from trading.runner.playbook import Playbook, PlaybookRule

    cycle, alerts = pm_prep_cycle
    today = pd.Timestamp.now(tz="UTC").normalize()
    last_week = today - pd.Timedelta(days=7)
    monkeypatch.setattr(cycle, "_clock", lambda: today.to_pydatetime())
    old_snapshot = {"as_of": str(last_week.date()), "ranked": [{"symbol": "OLD"}]}
    write_runner_candidate_snapshot(tmp_state, old_snapshot)
    refreshed_frames = {}
    for symbol in ("TEST_A", "TEST_B"):
        ins = Instrument(symbol=symbol, asset_class=AssetClass.EQUITY)
        frame = primed_cache.read(ins, "1D")
        frame.index = pd.date_range(end=last_week, periods=len(frame), freq="1D", name="ts")
        primed_cache.write(ins, "1D", frame)
        fresh = frame.tail(7).copy()
        fresh.index = pd.date_range(end=today, periods=7, freq="1D", name="ts")
        refreshed_frames[symbol] = fresh

    fetched_symbols = set()

    class _FreshSource:
        name = "synthetic_refresh"

        def get_bars(self, instrument, start, end, freq):
            frame = refreshed_frames[instrument.symbol]
            result = frame.loc[(frame.index >= start) & (frame.index <= end)]
            if not result.empty:
                fetched_symbols.add(instrument.symbol)
            return result

    monkeypatch.setattr(cycle, "source_factory", lambda _ins: _FreshSource())
    # Verify the snapshot carries the active playbook's parameters, while
    # preserving the transition notification for the actual trading cycle.
    active_params = {"top_k_momentum": {"k": 2, "lookback": 126, "skip": 21, "rebalance": 63}}
    cycle._playbook = Playbook(rules={"risk_on": PlaybookRule(strategy_params=active_params)})
    cycle._regime_label_fn = lambda _ts: "risk_on"

    result = cycle.refresh_candidate_snapshot_for_pm()

    assert result.ok, result.reason
    assert result.as_of == str(today.date())
    assert fetched_symbols == {"TEST_A", "TEST_B"}
    snapshot = json.loads(candidate_snapshot_path(tmp_state).read_text())
    assert result.snapshot == snapshot
    assert snapshot["as_of"] == result.as_of
    assert snapshot["age_days"] == 0
    assert "staleness_warning" not in snapshot
    assert snapshot["strategy_params"] == active_params
    assert snapshot["universe_size"] == 2
    assert {row["symbol"] for row in snapshot["ranked"]} <= {"TEST_A", "TEST_B"}
    assert cycle._last_regime is None
    assert cycle._cycle_count == 0
    assert alerts.sent == []
    assert not (tmp_state / "heartbeat.json").exists()
    assert not (tmp_state / cycle.APPROVAL_PENDING_FILE).exists()


@pytest.mark.parametrize("age_days", [7, None])
def test_pm_candidate_prep_preserves_snapshot_when_freshness_unverified(
    pm_prep_cycle, tmp_state, monkeypatch, age_days
) -> None:
    """Rewriting cached bars must not turn a stale ladder into a fresh one."""
    from trading.agents import candidates

    cycle, alerts = pm_prep_cycle
    old_snapshot = {"as_of": "2024-10-01", "ranked": [{"symbol": "OLD"}]}
    path = candidates.write_runner_candidate_snapshot(tmp_state, old_snapshot)
    original_bytes = path.read_bytes()
    monkeypatch.setattr(candidates, "_age_days", lambda _last_bar: age_days)

    result = cycle.refresh_candidate_snapshot_for_pm()

    assert not result.ok
    assert result.reason
    assert result.snapshot is None
    assert path.read_bytes() == original_bytes
    assert alerts.sent == []


def test_pm_candidate_prep_refuses_force_flatten_playbook(
    pm_prep_cycle, tmp_state, monkeypatch
) -> None:
    from trading.agents.candidates import write_runner_candidate_snapshot
    from trading.runner.playbook import Playbook, PlaybookRule

    cycle, alerts = pm_prep_cycle
    path = write_runner_candidate_snapshot(tmp_state, {"as_of": "prior snapshot"})
    original_bytes = path.read_bytes()
    cycle._playbook = Playbook(rules={"crisis": PlaybookRule(force_flatten=True)})
    cycle._regime_label_fn = lambda _ts: "crisis"
    monkeypatch.setattr(
        cycle,
        "_load_prices",
        lambda *_args: pytest.fail("a flat-book playbook should stop before refreshing"),
    )

    result = cycle.refresh_candidate_snapshot_for_pm()

    assert not result.ok
    assert "flat book" in result.reason
    assert result.snapshot is None
    assert path.read_bytes() == original_bytes
    assert cycle._last_regime is None
    assert alerts.sent == []
