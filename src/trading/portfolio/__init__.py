"""Portfolio composition layer (core-satellite).

Sits between the strategy/combiner stack and the risk manager. The
strategy decides what to *trade* (the satellite); this module overlays
the operator's static buy-and-hold-forever sleeve (the core).
"""

from __future__ import annotations

from trading.portfolio.core_satellite import (
    CoreSpec,
    CoreTheme,
    apply_core_satellite,
    build_core_weights,
    load_core_spec,
)

__all__ = [
    "CoreSpec",
    "CoreTheme",
    "apply_core_satellite",
    "build_core_weights",
    "load_core_spec",
]
