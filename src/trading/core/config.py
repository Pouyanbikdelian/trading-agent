"""Centralized, env-driven config.

We read defaults from ``.env`` (via pydantic-settings) and let YAML overrides
in ``config/*.yaml`` extend them for things that don't belong in env (universes,
strategy params).

Usage::

    from trading.core.config import settings
    print(settings.trading_env)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TradingEnv = Literal["research", "paper", "live"]


class Settings(BaseSettings):
    """Process-wide settings. Frozen after instantiation."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ---- Environment ----
    trading_env: TradingEnv = Field(default="research", alias="TRADING_ENV")

    # ---- IBKR ----
    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(default=7497, alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=17, alias="IBKR_CLIENT_ID")

    # Live trading must be explicitly enabled in BOTH env flag and trading_env.
    allow_live_trading: bool = Field(default=False, alias="ALLOW_LIVE_TRADING")

    # ---- Data ----
    tiingo_api_key: str | None = Field(default=None, alias="TIINGO_API_KEY")
    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")

    # ---- Notifications ----
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # ---- Storage ----
    data_dir: Path = Field(default=PROJECT_ROOT / "data" / "parquet", alias="DATA_DIR")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")
    state_dir: Path = Field(default=PROJECT_ROOT / "state", alias="STATE_DIR")

    # ---- Risk ----
    max_gross_exposure: float = Field(default=1.0, alias="MAX_GROSS_EXPOSURE")
    max_position_pct: float = Field(default=0.10, alias="MAX_POSITION_PCT")
    max_daily_loss_pct: float = Field(default=0.02, alias="MAX_DAILY_LOSS_PCT")
    max_drawdown_pct: float = Field(default=0.15, alias="MAX_DRAWDOWN_PCT")

    @field_validator("data_dir", "log_dir", "state_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: str | Path) -> Path:
        p = Path(v).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def is_live_armed(self) -> bool:
        """True only when BOTH the env flag and the trading_env say live."""
        return self.trading_env == "live" and self.allow_live_trading

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.log_dir, self.state_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lazily load settings once per process. Tests can ``cache_clear()`` it."""
    return Settings()  # type: ignore[call-arg]


# Convenience accessor — most callers want this.
settings = get_settings()
