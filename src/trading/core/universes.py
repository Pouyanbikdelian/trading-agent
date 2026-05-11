"""Universe loader: ``config/universes.yaml`` -> ``list[Instrument]``.

A universe is a named symbol set. Strategies declare which universes they
consume; data backfills iterate one universe at a time. Keeping this in YAML
(rather than env or code) means non-coders can curate symbol lists without
touching Python.

Example::

    from trading.core.universes import load_universe
    instruments = load_universe("us_large_cap")

The loader is intentionally strict about asset_class so adapters can dispatch
on it without runtime type checks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from trading.core.config import PROJECT_ROOT
from trading.core.types import AssetClass, Instrument

DEFAULT_UNIVERSES_PATH = PROJECT_ROOT / "config" / "universes.yaml"


@lru_cache(maxsize=4)
def _load_yaml(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"universes file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    universes = raw.get("universes")
    if not isinstance(universes, dict):
        raise ValueError(f"{path} must contain a top-level 'universes:' mapping")
    return universes


def available_universes(path: Path | None = None) -> list[str]:
    """Names of universes defined in the YAML, in declaration order."""
    return list(_load_yaml(path or DEFAULT_UNIVERSES_PATH).keys())


def load_universe(name: str, path: Path | None = None) -> list[Instrument]:
    """Resolve a universe name to a list of frozen ``Instrument`` records.

    Raises ``KeyError`` if the universe isn't defined and ``ValueError`` if a
    required field is missing or malformed.
    """
    universes = _load_yaml(path or DEFAULT_UNIVERSES_PATH)
    if name not in universes:
        known = ", ".join(universes.keys())
        raise KeyError(f"universe '{name}' not in YAML. Available: {known}")

    spec = universes[name]
    asset_class_str = spec.get("asset_class")
    if not asset_class_str:
        raise ValueError(f"universe '{name}' is missing 'asset_class'")
    try:
        asset_class = AssetClass(asset_class_str)
    except ValueError as e:
        raise ValueError(f"universe '{name}' has unknown asset_class={asset_class_str!r}") from e

    symbols = spec.get("symbols")
    if not symbols or not isinstance(symbols, list):
        raise ValueError(f"universe '{name}' must declare a non-empty 'symbols:' list")

    # exchange / currency / multiplier / min_tick can be set at the universe level
    # and inherited by every instrument; per-symbol override isn't worth the
    # complexity in v1 (we'd just enumerate dicts instead of strings).
    exchange = spec.get("exchange")
    currency = spec.get("currency", "USD")

    return [
        Instrument(
            symbol=str(sym),
            asset_class=asset_class,
            exchange=exchange,
            currency=currency,
        )
        for sym in symbols
    ]


def clear_cache() -> None:
    """Drop the loader's lru_cache. Tests use this when they rewrite the YAML."""
    _load_yaml.cache_clear()
