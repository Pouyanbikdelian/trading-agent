"""Tests for Strategy base + registry."""

from __future__ import annotations

import pandas as pd
import pytest

from trading.strategies import (
    Donchian,
    DonchianParams,
    available_strategies,
    get_strategy,
)
from trading.strategies.base import Strategy, StrategyParams, register


def test_registry_lists_all_builtins() -> None:
    for name in (
        "donchian",
        "ema_cross",
        "xsec_momentum",
        "rsi2",
        "zscore_meanrev",
        "risk_parity",
    ):
        assert name in available_strategies()


def test_get_strategy_returns_class() -> None:
    cls = get_strategy("donchian")
    assert cls is Donchian


def test_get_strategy_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown strategy"):
        get_strategy("not_a_real_strategy")


def test_kwargs_or_params_but_not_both() -> None:
    Donchian(lookback=10)  # kwargs path
    Donchian(params=DonchianParams(lookback=10))  # params path
    with pytest.raises(TypeError, match="either"):
        Donchian(params=DonchianParams(lookback=10), lookback=10)


def test_params_are_frozen() -> None:
    p = DonchianParams(lookback=10)
    with pytest.raises(Exception):
        p.lookback = 20  # type: ignore[misc]


def test_params_reject_unknown_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        DonchianParams(lookback=10, nonsense_field=42)  # type: ignore[call-arg]


def test_register_rejects_duplicate_name() -> None:
    class FakeParams(StrategyParams):
        pass

    class FakeStrat(Strategy):
        name = "donchian"  # already registered
        Params = FakeParams

        def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
            return prices * 0

    with pytest.raises(ValueError, match="duplicate"):
        register(FakeStrat)


def test_register_requires_name() -> None:
    class P(StrategyParams):
        pass

    class S(Strategy):
        name = ""
        Params = P

        def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
            return prices * 0

    with pytest.raises(ValueError, match="must set"):
        register(S)
