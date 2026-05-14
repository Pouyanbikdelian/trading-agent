"""Playbook tests — YAML round-trip + rule resolution + cycle integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.data.cache import ParquetCache
from trading.execution import OrderStore, Simulator
from trading.risk.limits import RiskLimits
from trading.risk.manager import RiskManager
from trading.runner import (
    Cycle,
    NullAlerts,
    Playbook,
    PlaybookRule,
    RunnerConfig,
    RunnerStore,
    load_playbook,
    rule_for,
)

# ---------------------------------------------------------------- Playbook ----


def test_load_playbook_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "playbook.yaml"
    yaml_path.write_text(
        "classifier: vix\n"
        "default_rule: mid_vol\n"
        "rules:\n"
        "  low_vol:\n"
        "    strategies: [donchian]\n"
        "    universe: nasdaq100\n"
        "    vol_target: 0.15\n"
        "  mid_vol:\n"
        "    strategies: [risk_parity]\n"
        "    universe: sp500\n"
    )
    pb = load_playbook(yaml_path)
    assert pb.classifier == "vix"
    assert pb.default_rule == "mid_vol"
    assert pb.rules["low_vol"].vol_target == 0.15
    assert pb.rules["mid_vol"].universe == "sp500"


def test_load_playbook_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_playbook(tmp_path / "absent.yaml")


def test_rule_for_resolves_label() -> None:
    pb = Playbook(
        classifier="vix",
        default_rule="mid_vol",
        rules={
            "low_vol": PlaybookRule(strategies=["donchian"], universe="nasdaq100"),
            "mid_vol": PlaybookRule(strategies=["risk_parity"], universe="sp500"),
        },
    )
    assert rule_for(pb, "low_vol").universe == "nasdaq100"
    # Unknown label → default_rule.
    assert rule_for(pb, "unknown_label").universe == "sp500"


def test_rule_for_no_default_returns_none() -> None:
    pb = Playbook(
        classifier="vix",
        rules={"low_vol": PlaybookRule(strategies=["donchian"])},
    )
    assert rule_for(pb, "low_vol") is not None
    assert rule_for(pb, "missing") is None


def test_force_flatten_rule_round_trip() -> None:
    pb = Playbook(
        classifier="vix",
        rules={
            "crisis": PlaybookRule(strategies=[], force_flatten=True),
        },
    )
    rule = rule_for(pb, "crisis")
    assert rule is not None
    assert rule.force_flatten is True
    assert rule.strategies == []


# ---------------------------------------------- Cycle integration with playbook


@pytest.fixture
def primed_cache(tmp_path: Path) -> ParquetCache:
    cache = ParquetCache(tmp_path / "parquet")
    idx = pd.date_range("2024-01-01", periods=300, freq="1D", tz="UTC", name="ts")
    rng = np.random.default_rng(0)
    for symbol in ("TEST_A", "TEST_B"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300)))
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


@pytest.fixture
def tiny_universe_yaml(tmp_path: Path, monkeypatch) -> str:
    yaml_path = tmp_path / "universes.yaml"
    yaml_path.write_text(
        "universes:\n"
        "  _runner_test_universe:\n"
        "    asset_class: equity\n"
        "    symbols: [TEST_A, TEST_B]\n"
    )
    from trading.core import universes as universes_module
    from trading.core.universes import clear_cache

    monkeypatch.setattr(universes_module, "DEFAULT_UNIVERSES_PATH", yaml_path)
    clear_cache()
    return "_runner_test_universe"


def _build_cycle(
    cfg: RunnerConfig,
    cache: ParquetCache,
    tmp_path: Path,
    *,
    playbook: Playbook | None,
    regime_label_fn=None,
) -> tuple[Cycle, Simulator, NullAlerts]:
    from trading.core.types import Bar

    broker = Simulator(initial_cash=cfg.initial_cash)
    broker.connect()
    last_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars: dict[str, Bar] = {}
    for sym in ("TEST_A", "TEST_B"):
        df = cache.read(Instrument(symbol=sym, asset_class=AssetClass.EQUITY), "1D")
        last_ts = df.index[-1].to_pydatetime()
        bars[sym] = Bar(
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

    class _NullSource:
        name = "noop"

        def get_bars(self, *a, **kw):
            return pd.DataFrame()

    cycle = Cycle(
        cfg,
        cache=cache,
        source_factory=lambda ins: _NullSource(),
        broker=broker,
        risk_manager=rm,
        order_store=OrderStore(tmp_path / "orders.db"),
        runner_store=RunnerStore(tmp_path / "runner.db"),
        alerts=NullAlerts(),
        heartbeat_path=tmp_path / "heartbeat.json",
        clock=lambda: last_ts,
        playbook=playbook,
        regime_label_fn=regime_label_fn,
    )
    return cycle, broker, cycle.alerts


def test_playbook_overrides_universe_and_strategies(
    tiny_universe_yaml, primed_cache, tmp_path: Path
) -> None:
    """A playbook rule's `strategies` must replace cfg.strategies for the
    cycle. We tell the cycle the rule says 'ema_cross' even though config
    says 'donchian'; the cycle must dispatch on ema_cross."""
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        strategy_params={"donchian": {"lookback": 20}},
        auto_refresh=False,
        history_bars=200,
    )
    pb = Playbook(
        classifier="vix",
        rules={
            "high_vol": PlaybookRule(
                strategies=["ema_cross"],
                strategy_params={"ema_cross": {"fast_span": 5, "slow_span": 20}},
            ),
        },
    )
    cycle, _, _ = _build_cycle(
        cfg,
        primed_cache,
        tmp_path,
        playbook=pb,
        regime_label_fn=lambda ts: "high_vol",
    )
    eff_cfg, force_flat = cycle._effective_config(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert eff_cfg.strategies == ["ema_cross"]
    assert "ema_cross" in eff_cfg.strategy_params
    assert force_flat is False


def test_playbook_force_flatten_triggers_close_only_cycle(
    tiny_universe_yaml, primed_cache, tmp_path: Path
) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    pb = Playbook(
        classifier="vix",
        rules={"crisis": PlaybookRule(strategies=[], force_flatten=True)},
    )
    cycle, _, _ = _build_cycle(
        cfg,
        primed_cache,
        tmp_path,
        playbook=pb,
        regime_label_fn=lambda ts: "crisis",
    )
    _, force_flat = cycle._effective_config(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert force_flat is True
    # No positions yet → no_orders.
    report = cycle.run_cycle()
    assert report.status == "no_orders"


def test_playbook_unknown_label_falls_back_to_default(
    tiny_universe_yaml, primed_cache, tmp_path: Path
) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    pb = Playbook(
        classifier="vix",
        default_rule="mid_vol",
        rules={"mid_vol": PlaybookRule(strategies=["risk_parity"])},
    )
    cycle, _, _ = _build_cycle(
        cfg,
        primed_cache,
        tmp_path,
        playbook=pb,
        regime_label_fn=lambda ts: "completely_unknown",
    )
    eff_cfg, _ = cycle._effective_config(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert eff_cfg.strategies == ["risk_parity"]


def test_playbook_classifier_exception_falls_back_to_static_config(
    tiny_universe_yaml, primed_cache, tmp_path: Path
) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    pb = Playbook(
        classifier="vix",
        rules={"low_vol": PlaybookRule(strategies=["ema_cross"])},
    )

    def boom(ts):
        raise RuntimeError("yfinance is down")

    cycle, _, _ = _build_cycle(cfg, primed_cache, tmp_path, playbook=pb, regime_label_fn=boom)
    eff_cfg, _ = cycle._effective_config(datetime(2024, 1, 1, tzinfo=timezone.utc))
    # Classifier exploded → static cfg unchanged.
    assert eff_cfg.strategies == ["donchian"]


def test_no_playbook_returns_static_config(
    tiny_universe_yaml, primed_cache, tmp_path: Path
) -> None:
    cfg = RunnerConfig(
        universe=tiny_universe_yaml,
        strategies=["donchian"],
        auto_refresh=False,
        history_bars=200,
    )
    cycle, _, _ = _build_cycle(cfg, primed_cache, tmp_path, playbook=None)
    eff_cfg, force_flat = cycle._effective_config(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert eff_cfg is cfg
    assert force_flat is False
