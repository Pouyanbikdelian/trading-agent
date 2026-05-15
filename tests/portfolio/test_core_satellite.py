r"""Tests for the core-satellite framework."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading.portfolio import (
    CoreSpec,
    CoreTheme,
    apply_core_satellite,
    build_core_weights,
    load_core_spec,
)


def _spec_50pct_core() -> CoreSpec:
    return CoreSpec(
        core_allocation=0.50,
        core_rebalance_days=63,
        core_holdings={
            "nuclear": CoreTheme(weight=0.30, symbols=["URA", "CCJ", "NLR"]),
            "utilities": CoreTheme(weight=0.20, symbols=["XLU", "NEE"]),
        },
    )


def test_build_core_weights_returns_total_equity_fractions() -> None:
    spec = _spec_50pct_core()
    cols = ["URA", "CCJ", "NLR", "XLU", "NEE", "AAPL"]  # AAPL is satellite-only
    w = build_core_weights(spec, cols)
    # Nuclear: 3 names sharing 0.30 of core (0.15 of total) → 0.05 each
    assert w["URA"] == pytest.approx(0.50 * 0.30 / 3)
    assert w["CCJ"] == pytest.approx(0.50 * 0.30 / 3)
    assert w["NLR"] == pytest.approx(0.50 * 0.30 / 3)
    # Utilities: 2 names sharing 0.20 of core → 0.10 / 2 = 0.05 each
    assert w["XLU"] == pytest.approx(0.50 * 0.20 / 2)
    assert w["NEE"] == pytest.approx(0.50 * 0.20 / 2)
    # AAPL is not in any theme
    assert w["AAPL"] == 0.0


def test_build_core_weights_sum_matches_allocation_minus_cash() -> None:
    spec = _spec_50pct_core()
    cols = ["URA", "CCJ", "NLR", "XLU", "NEE"]
    w = build_core_weights(spec, cols)
    # Themes sum to 0.30 + 0.20 = 0.50 of the core sleeve.
    # Core sleeve is 50% of equity. So total deployed = 0.50 * 0.50 = 0.25.
    assert w.sum() == pytest.approx(0.25)


def test_theme_weights_over_one_rejected() -> None:
    spec = CoreSpec(
        core_allocation=0.50,
        core_holdings={
            "a": CoreTheme(weight=0.70, symbols=["X"]),
            "b": CoreTheme(weight=0.50, symbols=["Y"]),
        },
    )
    with pytest.raises(ValueError, match="leverage"):
        spec.validate_theme_weights()


def test_apply_core_satellite_scales_satellite_down() -> None:
    spec = CoreSpec(
        core_allocation=0.50,
        core_holdings={"util": CoreTheme(weight=0.40, symbols=["XLU"])},
    )
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    # Satellite is fully invested in AAPL at 100% gross
    sat = pd.DataFrame({"AAPL": [1.0] * 5}, index=idx)
    core_w = build_core_weights(spec, ["AAPL", "XLU"])
    final = apply_core_satellite(sat, core_w, spec)
    # AAPL scaled to 0.50 (the satellite share); XLU sits at the static
    # core weight = 0.50 * 0.40 = 0.20
    assert final["AAPL"].iloc[-1] == pytest.approx(0.50)
    assert final["XLU"].iloc[-1] == pytest.approx(0.20)
    # Total gross stays under 1.0
    assert final.abs().sum(axis=1).max() == pytest.approx(0.70)


def test_apply_core_satellite_overlap_adds() -> None:
    """If the algo wants to BUY MORE of a core name, the weights add."""
    spec = CoreSpec(
        core_allocation=0.50,
        core_holdings={"a": CoreTheme(weight=0.20, symbols=["AAPL"])},
    )
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    sat = pd.DataFrame({"AAPL": [0.5] * 3}, index=idx)  # 50% gross AAPL
    core_w = build_core_weights(spec, ["AAPL"])
    final = apply_core_satellite(sat, core_w, spec)
    # Satellite 0.5 * satellite_share 0.5 = 0.25, plus core 0.5 * 0.2 = 0.10
    assert final["AAPL"].iloc[-1] == pytest.approx(0.35)


def test_core_allocation_zero_disables_core() -> None:
    spec = CoreSpec(
        core_allocation=0.0,
        core_holdings={"a": CoreTheme(weight=1.0, symbols=["IGNORED"])},
    )
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    sat = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=idx)
    core_w = build_core_weights(spec, ["AAPL", "IGNORED"])
    final = apply_core_satellite(sat, core_w, spec)
    # No core: satellite at 100%, no IGNORED position
    assert final["AAPL"].iloc[-1] == pytest.approx(1.0)
    assert final["IGNORED"].iloc[-1] == pytest.approx(0.0)


def test_load_core_spec_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text(
        "core_allocation: 0.4\n"
        "core_rebalance_days: 21\n"
        "core_holdings:\n"
        "  energy:\n"
        "    weight: 0.5\n"
        "    symbols: [XOM, CVX]\n"
    )
    spec = load_core_spec(yaml_path)
    assert spec.core_allocation == 0.4
    assert spec.core_rebalance_days == 21
    assert spec.core_holdings["energy"].symbols == ["XOM", "CVX"]


def test_load_core_spec_rejects_unknown_keys(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text("core_allocation: 0.5\nnonsense_field: 42\n")
    with pytest.raises(Exception):
        load_core_spec(yaml_path)
