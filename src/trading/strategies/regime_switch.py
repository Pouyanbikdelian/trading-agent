"""Regime-switch ETF allocator — the winner of the Nov-2026 strategy zoo.

``scripts/zoo/`` tested 83 strategies (2019-2026, costs, train 2018-22 /
judge 2023-26, deflated Sharpe). No single rule beat the market in every
environment; the robust result was SWITCHING by a regime that is knowable
the day before:

=================  =========================================================
regime             legs (share of this sleeve; equal-risk on 2018-22 vols)
=================  =========================================================
bull / calm        QQQ trend (EMA50 > EMA200) 28% + risk parity, core-6 72%
bull / stressed    GTAA-5 39% + Donchian 100/50 breakout 35% + min-var 25%
bear               GTAA-5 100%
=================  =========================================================

* regime: SPY vs its 200d MA (bull/bear); in a bull, SPY 21d realized vol vs
  its trailing 252d median (calm/stressed). Bear is not split — below the
  200d the tape is calm on 16 of 1,830 days.
* bear leg: the zoo's bear winner was next-day index mean reversion (IBS),
  which needs daily market-on-close execution the desk does not have yet.
  GTAA-5 is the stand-in: its own trend filter moves it to bonds/gold/cash.
* core-6 = SPY EFA IEF TLT GLD DBC. GTAA-5 = SPY EFA IEF DBC GLD, each 20%
  only while above its 10-month SMA (else cash).
* Long-only, gross <= 1, no leverage (the zoo's Donchian leg was vol-targeted
  with up to 2x; live it is not).

``generate`` returns a daily weight frame like every strategy;
``latest_allocation`` is what the live sleeve reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register

CORE6 = ("SPY", "EFA", "IEF", "TLT", "GLD", "DBC")
GTAA5 = ("SPY", "EFA", "IEF", "DBC", "GLD")
DONCHIAN = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC")
ALL_SYMBOLS = tuple(sorted(set(CORE6) | set(GTAA5) | set(DONCHIAN) | {"QQQ"}))

LEG_WEIGHTS: dict[str, dict[str, float]] = {
    "bull/calm": {"qqq_trend": 0.28, "risk_parity": 0.72},
    "bull/stressed": {"gtaa5": 0.39, "donchian": 0.35, "min_variance": 0.26},
    "bear": {"gtaa5": 1.0},
}


class RegimeSwitchParams(StrategyParams):
    trend_ma: int = Field(default=200, ge=20)
    vol_window: int = Field(default=21, ge=5)
    vol_median_window: int = Field(default=252, ge=21)
    cov_window: int = Field(default=252, ge=60)
    min_var_cap: float = Field(default=0.5, gt=0.0, le=1.0)


def _shrink_cov(x: np.ndarray, a: float = 0.5) -> np.ndarray:
    s = np.cov(x, rowvar=False)
    sd = np.sqrt(np.diag(s))
    c = s / np.outer(sd, sd)
    n = len(c)
    rbar = (c.sum() - n) / (n * n - n)
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, sd**2)
    return a * target + (1 - a) * s


def _erc(cov: np.ndarray, iters: int = 200) -> np.ndarray:
    w = 1 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        rc = w * (cov @ w)
        w *= (rc.mean() / rc) ** 0.5
        w /= w.sum()
    return w


def _min_var(cov: np.ndarray, cap: float) -> np.ndarray:
    from scipy.optimize import minimize

    n = len(cov)
    res = minimize(
        lambda w: w @ cov @ w,
        np.full(n, 1 / n),
        jac=lambda w: 2 * cov @ w,
        bounds=[(0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
        options={"maxiter": 200},
    )
    return res.x if res.success else np.full(n, 1 / n)


def regime_series(prices: pd.DataFrame, p: RegimeSwitchParams) -> pd.Series:
    """Regime label per row, from data up to and including that row."""
    spy = prices["SPY"]
    ma = spy.rolling(p.trend_ma).mean()
    rv = spy.pct_change().rolling(p.vol_window).std()
    med = rv.rolling(p.vol_median_window, min_periods=p.vol_median_window // 2).median()
    lab = pd.Series(np.where(rv > med, "bull/stressed", "bull/calm"), index=spy.index)
    lab[spy <= ma] = "bear"
    lab[ma.isna() | med.isna()] = np.nan
    return lab


def _month_end_rows(idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(idx, index=idx)
    last = s.groupby([idx.year, idx.month]).transform("max")
    return pd.Series(idx == last.values, index=idx)


class _Legs:
    """Daily leg weights (each leg sums to <= 1)."""

    def __init__(self, prices: pd.DataFrame, p: RegimeSwitchParams) -> None:
        self.px = prices
        self.p = p
        self.ret = prices.pct_change()

    def qqq_trend(self) -> pd.DataFrame:
        q = self.px["QQQ"]
        on = (q.ewm(span=50).mean() > q.ewm(span=200).mean()).astype(float)
        return pd.DataFrame({"QQQ": on})

    def gtaa5(self) -> pd.DataFrame:
        px = self.px[list(GTAA5)]
        me = _month_end_rows(px.index)
        monthly = px[me.values]
        sig = (monthly > monthly.rolling(10).mean()).astype(float) * 0.2
        return sig.reindex(px.index).ffill().fillna(0.0)

    def donchian(self) -> pd.DataFrame:
        px = self.px[list(DONCHIAN)]
        hi = px.rolling(100).max().shift(1)
        lo = px.rolling(50).min().shift(1)
        st = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
        st[px >= hi] = 1.0
        st[px <= lo] = 0.0
        st = st.ffill().fillna(0.0)
        iv = 1 / self.ret[list(DONCHIAN)].rolling(60).std()
        w = (st * iv).div(iv.sum(axis=1), axis=0)
        return w.fillna(0.0)

    def _cov_leg(self, kind: str, step: int = 5) -> pd.DataFrame:
        cols = list(CORE6)
        r = self.ret[cols]
        out = pd.DataFrame(np.nan, index=r.index, columns=cols)
        n = self.p.cov_window
        for i in range(n, len(r), step):
            x = r.iloc[i - n + 1 : i + 1].dropna()
            if len(x) < int(n * 0.8):
                continue
            cov = _shrink_cov(x.values)
            w = _erc(cov) if kind == "erc" else _min_var(cov, self.p.min_var_cap)
            out.iloc[i] = w
        # the latest row is always computed, so the live sleeve is current
        if len(r) > n:
            x = r.iloc[-n:].dropna()
            if len(x) >= int(n * 0.8):
                cov = _shrink_cov(x.values)
                out.iloc[-1] = _erc(cov) if kind == "erc" else _min_var(cov, self.p.min_var_cap)
        return out.ffill().fillna(0.0)

    def risk_parity(self) -> pd.DataFrame:
        return self._cov_leg("erc")

    def min_variance(self) -> pd.DataFrame:
        return self._cov_leg("minvar")


@register
class RegimeSwitch(Strategy):
    name = "regime_switch"
    Params = RegimeSwitchParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        missing = [s for s in ALL_SYMBOLS if s not in prices.columns]
        if missing:
            raise ValueError(f"regime_switch needs prices for {missing}")
        px = prices[list(ALL_SYMBOLS)].ffill()
        legs_obj = _Legs(px, self.params)
        legs = {
            name: getattr(legs_obj, name)()
            for name in ("qqq_trend", "risk_parity", "gtaa5", "donchian", "min_variance")
        }
        lab = regime_series(px, self.params)
        out = pd.DataFrame(0.0, index=px.index, columns=list(ALL_SYMBOLS))
        for regime, mix in LEG_WEIGHTS.items():
            mask = (lab == regime).values
            if not mask.any():
                continue
            blend = pd.DataFrame(0.0, index=px.index, columns=list(ALL_SYMBOLS))
            for leg, share in mix.items():
                blend = blend.add(
                    legs[leg].reindex(columns=ALL_SYMBOLS, fill_value=0.0) * share, fill_value=0.0
                )
            out.loc[mask] = blend.loc[mask].values
        gross = out.sum(axis=1)
        out = out.div(gross.clip(lower=1.0), axis=0)  # never above 100% of the sleeve
        return out.reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)


def latest_allocation(
    prices: pd.DataFrame, params: RegimeSwitchParams | None = None
) -> dict[str, Any]:
    """{regime, weights{sym: w}, asof} for the newest row. Raises on missing data."""
    p = params or RegimeSwitchParams()
    strat = RegimeSwitch(p)
    w = strat.generate(prices)
    lab = regime_series(prices[list(ALL_SYMBOLS)].ffill(), p)
    last = w.iloc[-1]
    regime = lab.iloc[-1]
    if not isinstance(regime, str):
        raise ValueError("regime_switch: not enough history to classify the regime")
    return {
        "asof": str(w.index[-1].date()),
        "regime": regime,
        "legs": LEG_WEIGHTS[regime],
        "weights": {s: round(float(v), 4) for s, v in last.items() if abs(v) > 1e-4},
    }
