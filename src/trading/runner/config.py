"""RunnerConfig — frozen pydantic model describing one runner instance.

A runner is a single (universe, strategies, broker, schedule) bundle. To
run two strategies on different schedules, instantiate two runners.

Many fields are optional with sensible defaults so a minimal config like::

    RunnerConfig(universe="us_large_cap", strategies=["donchian"])

starts working immediately (against the cache and simulator).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trading.data.base import CANONICAL_FREQUENCIES, Frequency

CombinerName = Literal["equal_weight", "inverse_vol", "min_variance"]


class RunnerConfig(BaseModel):
    """Static configuration for a runner instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- What to trade ---------------------------------------------------
    universe: str
    """Name from config/universes.yaml."""

    strategies: list[str] = Field(default_factory=lambda: ["donchian"])
    """Names from STRATEGY_REGISTRY."""

    strategy_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-strategy overrides keyed by strategy name."""

    combiner: CombinerName = "equal_weight"
    """How to blend multi-strategy weights. Only ``equal_weight`` is robust
    in the live runner; ``inverse_vol`` / ``min_variance`` need historical
    per-strategy returns which the runner doesn't track yet (Phase 5 callers
    pass them in directly)."""

    # --- Data ------------------------------------------------------------
    freq: Frequency = "1D"
    history_bars: int = Field(default=252, ge=10)
    """Bars to keep in the working price frame each cycle."""

    price_column: str = "adj_close"
    auto_refresh: bool = True
    """If True, the cycle fetches fresh bars before reading the cache;
    if False, the cycle reads whatever is already cached."""

    # --- Regime + risk ---------------------------------------------------
    use_regime: bool = False
    regime_lookback: int = Field(default=20, ge=2)

    vol_target: float | None = None
    """Annualized volatility target. None disables the overlay."""

    vol_lookback: int = Field(default=60, ge=2)
    max_leverage: float = Field(default=2.0, gt=0.0)

    periods_per_year: int = Field(default=252, ge=1)

    # --- Execution -------------------------------------------------------
    initial_cash: float = Field(default=100_000.0, gt=0.0)
    use_simulator: bool = True
    """Phase 8's CLI flips this; the runner itself is broker-agnostic."""

    # --- Scheduling ------------------------------------------------------
    schedule_cron: str = "0 16 * * MON-FRI"
    """Crontab string for APScheduler. Default: weekdays at 16:00 ``schedule_tz``."""
    schedule_tz: str = "UTC"

    # --- Persistence + observability -------------------------------------
    state_db_path: str | None = None
    """Defaults to ``<settings.state_dir>/runner.db`` at construction time."""

    order_db_path: str | None = None
    """Defaults to ``<settings.state_dir>/orders.db``."""

    halt_state_path: str | None = None
    """Defaults to ``<settings.state_dir>/halt.json``."""

    heartbeat_path: str | None = None
    """Defaults to ``<settings.state_dir>/heartbeat.json``."""

    # --- Sectoring (optional) -------------------------------------------
    sector_map: dict[str, str] = Field(default_factory=dict)
    """``instrument.key -> sector_name``. Empty disables sector caps."""

    # ---- validation -----------------------------------------------------

    def freq_is_canonical(self) -> bool:
        return self.freq in CANONICAL_FREQUENCIES
