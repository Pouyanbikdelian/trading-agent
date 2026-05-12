"""Strategy interface + registry.

A Strategy maps a price history to a frame of target portfolio weights:

    weights = strategy.generate(prices)

The frame has the same index/columns as ``prices``; each cell is the target
weight at the end of that bar to be held during the *next* bar (matches the
backtester convention — see ``trading.backtest.engine``).

Why target-weight (not signed-quantity, not buy/sell deltas)
------------------------------------------------------------
Target weights are linear in capital and trivially aggregable across many
strategies — the portfolio combiner just averages them (Phase 5). They are
also the natural input to the risk manager, which sizes them into orders
(Phase 7). The risk manager never sees the strategy directly.

Strategies are *parametric*. Each subclass declares a frozen pydantic
``Params`` inner class so the CLI / config layer can construct a strategy
from a flat dict without ad-hoc parsing per strategy.

Registry
--------
``register`` decorates a strategy class with a stable name and stores it in
``STRATEGY_REGISTRY``. The CLI looks up strategies by name; tests can read
the registry directly. Names must be unique.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict


class StrategyParams(BaseModel):
    """Base class for strategy parameter blocks. Frozen so a Strategy
    can't accidentally mutate its config mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Strategy(ABC):
    """Abstract base. Subclasses implement ``generate``.

    A subclass MUST define:
      * ``name``        — class variable, unique, used by the CLI.
      * ``Params``      — a ``StrategyParams`` subclass describing config.
      * ``generate``    — the vectorized weight producer.

    Subclasses MAY define ``regime_scale_map`` (``{state_id: multiplier}``)
    to opt into the default ``modulate`` behavior, e.g.::

        class MyTrend(Strategy):
            # de-risk in the high-vol regime, full size in low/mid.
            regime_scale_map = {0: 1.0, 1: 1.0, 2: 0.0}

    Construction takes a ``Params`` instance; we keep it as ``self.params``
    so subclasses don't need to copy each field into ``self``.
    """

    name: ClassVar[str] = ""
    Params: ClassVar[type[StrategyParams]]
    regime_scale_map: ClassVar[dict[int, float]] = {}

    def __init__(self, params: StrategyParams | None = None, **kwargs: Any) -> None:
        if params is None:
            params = self.Params(**kwargs)
        elif kwargs:
            raise TypeError("pass either a Params instance or keyword args, not both")
        if not isinstance(params, self.Params):
            raise TypeError(
                f"{type(self).__name__} expects {self.Params.__name__}, got {type(params).__name__}"
            )
        self.params = params

    @abstractmethod
    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target weights aligned to ``prices.index`` / ``prices.columns``.

        Implementations must:
          * Use only data at or before each row (no lookahead).
          * Fill leading bars with 0 (no position) while indicators warm up.
          * Be deterministic given the same ``prices`` input.
        """

    def modulate(self, weights: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
        """Apply a regime overlay to a weights frame.

        Default behavior: if the subclass declares ``regime_scale_map``,
        scale each row by the corresponding multiplier (states absent from
        the map and the warm-up sentinel ``-1`` are left at full size).
        Otherwise return the weights unchanged.

        Override this in a subclass that needs richer regime logic — e.g.
        flipping signs in mean-reverting vs. trending regimes.
        """
        if not self.regime_scale_map:
            return weights
        # Local import keeps the strategies module decoupled from the
        # regime module's import order.
        from trading.regime.base import regime_scale

        return regime_scale(weights, regime, self.regime_scale_map)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params!r})"


# ---------------------------------------------------------------------- registry


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: register a Strategy subclass under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class-level 'name'")
    if cls.name in STRATEGY_REGISTRY:
        raise ValueError(
            f"duplicate strategy name {cls.name!r}: "
            f"{STRATEGY_REGISTRY[cls.name].__name__} vs {cls.__name__}"
        )
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    """Look up a Strategy class by name. Raises KeyError if missing."""
    if name not in STRATEGY_REGISTRY:
        known = ", ".join(sorted(STRATEGY_REGISTRY)) or "(none)"
        raise KeyError(f"unknown strategy {name!r}. Known: {known}")
    return STRATEGY_REGISTRY[name]


def available_strategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY)
