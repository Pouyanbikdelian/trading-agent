r"""Multi-trigger risk monitor — the auto-detector that flags regime
stress and suggests a defensive mode, without ever auto-executing.

Lives outside the trading runner cycle so it can run on a faster
schedule (hourly during market hours). Its only output is a list of
``Trigger`` instances, each with a ``severity`` and ``suggested_mode``.
The bot decides what to do with that — typically: push a Telegram alert
with a suggestion the operator must confirm.

Why two timescales
------------------
Modern systematic crashes happen on either:

* **Slow grind** — the market tops, rolls over, and bleeds out for
  6-8 weeks (2018-Q4, 2022 H1). Hallmark: SPY 50-day SMA crosses below
  the 200-day, or 60-day trailing return turns sharply negative.

* **Fast crash** — a 5-10 day waterfall (Feb 2020, Aug 2024). Hallmark:
  SPY -7%+ in 5 days *and* a VIX spike past 30.

A single-trigger detector catches one of these, never both. The
composite below catches either. We deliberately allow some redundancy —
``vol_spike`` will often fire alongside ``fast_crash`` and that's fine,
the consumer picks the most severe.

Severity is ordinal, not a probability — ``LIGHT < HEAVY < EXTREME`` —
so the bot can rank multiple firing triggers and only act on the
strongest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Literal

import numpy as np
import pandas as pd

from trading.runtime.mode import Mode

TriggerName = Literal["slow_grind", "fast_crash", "vol_spike", "combined_extreme"]


class Severity(IntEnum):
    NONE = 0
    LIGHT = 1
    HEAVY = 2
    EXTREME = 3


@dataclass(frozen=True)
class Trigger:
    name: TriggerName
    severity: Severity
    suggested_mode: Mode
    detail: str  # one-line human-readable explanation
    fired_at: str  # ISO 8601

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "severity": int(self.severity),
            "suggested_mode": self.suggested_mode.value,
            "detail": self.detail,
            "fired_at": self.fired_at,
        }


@dataclass(frozen=True)
class MonitorConfig:
    """Tunable thresholds. Defaults pass the smell test on 2015-2026 history.

    All windows are in trading bars (daily by default).
    """

    # Slow grind
    slow_grind_sma_long: int = 200
    slow_grind_sma_fast: int = 50
    slow_grind_ret_window: int = 60
    slow_grind_ret_threshold: float = -0.08  # 60-day return < -8%

    # Fast crash
    fast_crash_window: int = 5  # bars
    fast_crash_threshold: float = -0.07  # 5-day return < -7%
    fast_crash_vix_floor: float = 30.0

    # Vol spike
    vol_spike_vix_extreme: float = 40.0
    vol_spike_vix_jump_pct: float = 0.50  # 1-day VIX move > +50%
    vol_spike_vix_min: float = 25.0  # only counts above this base

    # Recovery — how many consecutive clean bars before suggesting return to NEUTRAL
    recovery_clean_bars: int = 5


def _confirmed(series: pd.Series, window: int) -> pd.Series:
    """Boolean series, True iff all of the last ``window`` bars are True.

    Implemented as a rolling-min on the 0/1 cast — cheapest hysteresis
    operator in pandas-land. NaN windows yield NaN, which we treat as
    False at the call site.
    """
    return series.astype(int).rolling(window, min_periods=window).min() == 1


def evaluate(
    spy: pd.Series,
    vix: pd.Series | None = None,
    *,
    cfg: MonitorConfig | None = None,
    as_of: datetime | None = None,
) -> list[Trigger]:
    r"""Run all triggers against the latest bar of ``spy`` (+ optional VIX).

    The series should be tz-aware, sorted, and ideally daily. The
    function only inspects values up to the last index — it never peeks
    at the as-of timestamp.

    Returns a list of currently-active triggers (zero or more). Caller
    typically reports the most-severe one to the operator.
    """
    cfg = cfg or MonitorConfig()
    as_of = as_of or datetime.now(tz=timezone.utc)
    if len(spy) < max(cfg.slow_grind_sma_long, cfg.slow_grind_ret_window) + 5:
        return []

    out: list[Trigger] = []
    iso = as_of.isoformat()

    # --- slow grind ----------------------------------------------------
    sma_long = spy.rolling(cfg.slow_grind_sma_long, min_periods=cfg.slow_grind_sma_long).mean()
    sma_fast = spy.rolling(cfg.slow_grind_sma_fast, min_periods=cfg.slow_grind_sma_fast).mean()
    cross_below = (sma_fast.iloc[-1] < sma_long.iloc[-1]) and np.isfinite(sma_long.iloc[-1])

    ret_60 = float(spy.iloc[-1] / spy.iloc[-cfg.slow_grind_ret_window] - 1.0)
    grind_ret_hit = ret_60 < cfg.slow_grind_ret_threshold

    if cross_below or grind_ret_hit:
        causes = []
        if cross_below:
            causes.append("SMA(50) crossed below SMA(200)")
        if grind_ret_hit:
            causes.append(f"60-day return {ret_60:+.2%}")
        out.append(
            Trigger(
                name="slow_grind",
                severity=Severity.LIGHT,
                suggested_mode=Mode.DEFENSE,
                detail="; ".join(causes),
                fired_at=iso,
            )
        )

    # --- fast crash ----------------------------------------------------
    if len(spy) > cfg.fast_crash_window:
        ret_short = float(spy.iloc[-1] / spy.iloc[-cfg.fast_crash_window - 1] - 1.0)
        vix_now = float(vix.iloc[-1]) if vix is not None and len(vix) > 0 else 0.0
        if ret_short < cfg.fast_crash_threshold and vix_now >= cfg.fast_crash_vix_floor:
            out.append(
                Trigger(
                    name="fast_crash",
                    severity=Severity.HEAVY,
                    suggested_mode=Mode.BEAR,
                    detail=f"SPY {cfg.fast_crash_window}-day return {ret_short:+.2%}, VIX {vix_now:.1f}",
                    fired_at=iso,
                )
            )

    # --- vol spike -----------------------------------------------------
    if vix is not None and len(vix) >= 2:
        vix_now = float(vix.iloc[-1])
        vix_prev = float(vix.iloc[-2])
        jump = (vix_now / vix_prev - 1.0) if vix_prev > 0 else 0.0
        if vix_now > cfg.vol_spike_vix_extreme:
            out.append(
                Trigger(
                    name="vol_spike",
                    severity=Severity.HEAVY,
                    suggested_mode=Mode.DEFENSE,
                    detail=f"VIX {vix_now:.1f} > {cfg.vol_spike_vix_extreme}",
                    fired_at=iso,
                )
            )
        elif jump > cfg.vol_spike_vix_jump_pct and vix_now >= cfg.vol_spike_vix_min:
            out.append(
                Trigger(
                    name="vol_spike",
                    severity=Severity.LIGHT,
                    suggested_mode=Mode.DEFENSE,
                    detail=f"VIX jumped {jump:+.1%} to {vix_now:.1f}",
                    fired_at=iso,
                )
            )

    # --- combined extreme ---------------------------------------------
    names = {t.name for t in out}
    if "slow_grind" in names and ("fast_crash" in names or "vol_spike" in names):
        out.append(
            Trigger(
                name="combined_extreme",
                severity=Severity.EXTREME,
                suggested_mode=Mode.BEAR,
                detail="multiple risk regimes confirmed",
                fired_at=iso,
            )
        )

    return out


def is_clean(
    spy: pd.Series, vix: pd.Series | None = None, *, cfg: MonitorConfig | None = None
) -> bool:
    """True iff *no* trigger has fired for ``cfg.recovery_clean_bars`` bars.

    Used by the recovery-suggestion logic — the bot only nudges the
    operator toward NEUTRAL after a stretch of clean conditions, to
    avoid whipsaw on a 2-day bounce that doesn't hold.
    """
    cfg = cfg or MonitorConfig()
    if len(spy) < cfg.slow_grind_sma_long + cfg.recovery_clean_bars + 5:
        return False
    for i in range(cfg.recovery_clean_bars):
        idx = len(spy) - 1 - i
        if idx < 0:
            return False
        sub_spy = spy.iloc[: idx + 1]
        sub_vix = vix.iloc[: idx + 1] if vix is not None else None
        if evaluate(sub_spy, sub_vix, cfg=cfg):
            return False
    return True
