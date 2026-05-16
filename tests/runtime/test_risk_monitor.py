r"""Tests for the multi-trigger risk monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.runtime.mode import Mode
from trading.runtime.risk_monitor import MonitorConfig, Severity, evaluate, is_clean


def _make_spy(prices: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="1D", tz="UTC")
    return pd.Series(prices, index=idx, name="SPY", dtype=float)


def test_no_triggers_in_calm_uptrend() -> None:
    spy = _make_spy(np.linspace(100, 200, 400).tolist())
    triggers = evaluate(spy)
    assert triggers == []


def test_slow_grind_fires_on_death_cross() -> None:
    # Uptrend, then a sustained decline that crosses 50 below 200.
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    spy = _make_spy(up + down)
    triggers = evaluate(spy)
    names = {t.name for t in triggers}
    assert "slow_grind" in names
    grind = next(t for t in triggers if t.name == "slow_grind")
    assert grind.suggested_mode == Mode.DEFENSE
    assert grind.severity == Severity.LIGHT


def test_fast_crash_needs_both_drop_and_vix_floor() -> None:
    base = np.linspace(100, 200, 300).tolist()
    crash = [200, 198, 195, 190, 185, 180]  # ~-10% in 5 bars
    spy = _make_spy(base + crash)
    # Without VIX → no fast_crash trigger
    triggers = evaluate(spy)
    assert "fast_crash" not in {t.name for t in triggers}

    # With elevated VIX → fast_crash fires
    vix = pd.Series([15.0] * (len(spy) - 1) + [35.0], index=spy.index, dtype=float)
    triggers = evaluate(spy, vix)
    names = {t.name for t in triggers}
    assert "fast_crash" in names
    crash_t = next(t for t in triggers if t.name == "fast_crash")
    assert crash_t.suggested_mode == Mode.BEAR
    assert crash_t.severity == Severity.HEAVY


def test_vol_spike_extreme_fires_alone() -> None:
    spy = _make_spy(np.linspace(100, 200, 400).tolist())
    vix = pd.Series([20.0] * (len(spy) - 1) + [45.0], index=spy.index, dtype=float)
    triggers = evaluate(spy, vix)
    names = {t.name for t in triggers}
    assert "vol_spike" in names
    spike = next(t for t in triggers if t.name == "vol_spike")
    assert spike.severity == Severity.HEAVY


def test_vol_spike_jump_fires_on_50pct_move() -> None:
    spy = _make_spy(np.linspace(100, 200, 400).tolist())
    vix_vals = [18.0] * (len(spy) - 2) + [18.0, 30.0]  # +66% jump from 18 → 30
    vix = pd.Series(vix_vals, index=spy.index, dtype=float)
    triggers = evaluate(spy, vix)
    assert any(t.name == "vol_spike" for t in triggers)


def test_combined_extreme_only_when_both_fire() -> None:
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 195).tolist()
    crash = [200, 198, 195, 190, 185, 180]
    spy = _make_spy(up + down + crash)
    vix = pd.Series([15.0] * (len(spy) - 1) + [35.0], index=spy.index, dtype=float)
    triggers = evaluate(spy, vix)
    names = {t.name for t in triggers}
    assert "combined_extreme" in names
    extreme = next(t for t in triggers if t.name == "combined_extreme")
    assert extreme.severity == Severity.EXTREME
    assert extreme.suggested_mode == Mode.BEAR


def test_is_clean_after_recovery() -> None:
    # 400 bars of pure uptrend → no triggers anywhere → clean.
    spy = _make_spy(np.linspace(100, 200, 400).tolist())
    cfg = MonitorConfig(recovery_clean_bars=3)
    assert is_clean(spy, cfg=cfg) is True


def test_is_clean_false_during_active_trigger() -> None:
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 180, 200).tolist()
    spy = _make_spy(up + down)
    assert is_clean(spy) is False
