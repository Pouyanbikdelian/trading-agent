"""Risk manager — the hard-blocking gate between strategies and the broker.

Public surface::

    from trading.risk import RiskManager, RiskLimits, HaltState
"""

from __future__ import annotations

from trading.risk.limits import HaltState, RiskLimits
from trading.risk.manager import RiskManager

__all__ = ["HaltState", "RiskLimits", "RiskManager"]
