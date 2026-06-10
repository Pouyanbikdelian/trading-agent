r"""Options-structure risk monitor — the "vol desk" advisor.

Watches the *shape* of SPY's implied-vol surface for the stress
signatures that precede or accompany regime breaks, using only free
yfinance option-chain data (per the ≤$20/mo data budget; true net-gamma
/ order-flow feeds are paid products — see docs note in the report):

* ``atm_iv``        — near-dated (~30d) at-the-money IV level.
* ``put_skew``      — 25-delta-ish put IV minus ATM IV (crash insurance
                      bid). We proxy 25-delta with the strike at 0.95x spot,
                      which avoids needing a greeks engine.
* ``term_slope``    — far (~90d) ATM IV minus near ATM IV. Negative
                      (inverted backwardation) = front-loaded fear.
* ``pc_oi_ratio``   — put/call open-interest ratio across the near
                      expiry (positioning, crude order-flow proxy).

Each metric maps to a named trigger with a threshold; alerts are
debounced through ``state/options_monitor.json`` exactly like the
SPY/VIX advisor — alert on new/escalated, recovery message when all
clear. Advisory only; never touches the order path.

The math lives in pure functions over plain DataFrames so tests are
hermetic — only ``fetch_chain_metrics`` talks to the network.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger

STATE_FILENAME = "options_monitor.json"


@dataclass(frozen=True)
class OptionsMetrics:
    """Snapshot of the surface-shape metrics for one underlier."""

    underlier: str
    asof: datetime
    atm_iv: float  # near-dated ATM implied vol (decimal, e.g. 0.18)
    put_skew: float  # IV(0.95x-spot put) minus IV(ATM), decimal points
    term_slope: float  # IV(far ATM) minus IV(near ATM), decimal points
    pc_oi_ratio: float  # put OI / call OI, near expiry

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlier": self.underlier,
            "asof": self.asof.isoformat(),
            "atm_iv": self.atm_iv,
            "put_skew": self.put_skew,
            "term_slope": self.term_slope,
            "pc_oi_ratio": self.pc_oi_ratio,
        }


@dataclass(frozen=True)
class OptionsThresholds:
    """Trigger levels. Defaults are deliberately *loose* — this is a
    smoke detector, not a strategy. Tune in the field with data."""

    atm_iv_high: float = 0.30  # ~VIX 30: elevated vol regime
    put_skew_high: float = 0.08  # 8 vol pts of crash premium
    term_slope_inverted: float = -0.02  # ≥2 pts backwardation
    pc_oi_high: float = 1.60  # heavy put positioning


def _nearest_strike_iv(chain: pd.DataFrame, target_strike: float) -> float:
    """IV at the strike closest to ``target_strike``. Expects yfinance's
    chain frame (columns: strike, impliedVolatility, openInterest, ...)."""
    if chain.empty:
        return float("nan")
    idx = (chain["strike"] - target_strike).abs().idxmin()
    return float(chain.loc[idx, "impliedVolatility"])


def compute_metrics(
    *,
    underlier: str,
    spot: float,
    near_calls: pd.DataFrame,
    near_puts: pd.DataFrame,
    far_calls: pd.DataFrame,
    asof: datetime | None = None,
) -> OptionsMetrics:
    """Pure computation over already-fetched chains (hermetic, testable)."""
    atm_call_iv = _nearest_strike_iv(near_calls, spot)
    atm_put_iv = _nearest_strike_iv(near_puts, spot)
    # ATM IV: average the call/put ATM reads — cheap smile-noise filter.
    pair = [v for v in (atm_call_iv, atm_put_iv) if pd.notna(v)]
    atm_iv = float(sum(pair) / len(pair)) if pair else float("nan")

    otm_put_iv = _nearest_strike_iv(near_puts, 0.95 * spot)
    put_skew = float(otm_put_iv - atm_iv) if pd.notna(otm_put_iv) and pd.notna(atm_iv) else 0.0

    far_atm_iv = _nearest_strike_iv(far_calls, spot)
    term_slope = float(far_atm_iv - atm_iv) if pd.notna(far_atm_iv) and pd.notna(atm_iv) else 0.0

    put_oi = float(near_puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
    call_oi = float(near_calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
    pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 0.0

    return OptionsMetrics(
        underlier=underlier,
        asof=asof or datetime.now(tz=timezone.utc),
        atm_iv=atm_iv if pd.notna(atm_iv) else 0.0,
        put_skew=put_skew,
        term_slope=term_slope,
        pc_oi_ratio=pc_oi_ratio,
    )


def evaluate(
    metrics: OptionsMetrics, thresholds: OptionsThresholds | None = None
) -> list[tuple[str, str]]:
    """Map metrics to (trigger_name, human detail) pairs."""
    th = thresholds or OptionsThresholds()
    out: list[tuple[str, str]] = []
    if metrics.atm_iv >= th.atm_iv_high:
        out.append(
            (
                "ATM_IV_ELEVATED",
                f"{metrics.underlier} ~30d ATM IV {metrics.atm_iv:.0%} ≥ {th.atm_iv_high:.0%}",
            )
        )
    if metrics.put_skew >= th.put_skew_high:
        out.append(
            (
                "PUT_SKEW_STEEP",
                f"25Δ-proxy put skew {metrics.put_skew * 100:.1f} vol pts — crash insurance bid",
            )
        )
    if metrics.term_slope <= th.term_slope_inverted:
        out.append(
            (
                "TERM_STRUCTURE_INVERTED",
                f"IV term slope {metrics.term_slope * 100:+.1f} pts (near > far) — front-loaded fear",
            )
        )
    if metrics.pc_oi_ratio >= th.pc_oi_high:
        out.append(
            (
                "PUT_CALL_OI_HEAVY",
                f"put/call OI {metrics.pc_oi_ratio:.2f} ≥ {th.pc_oi_high:.2f} — defensive positioning",
            )
        )
    return out


def fetch_chain_metrics(underlier: str = "SPY") -> OptionsMetrics | None:
    """Network path: pull near (~30d) and far (~90d) chains via yfinance.
    Returns None on any failure — a flaky chain fetch must never break
    the runner loop."""
    try:
        import yfinance as yf

        tkr = yf.Ticker(underlier)
        expiries = list(tkr.options or [])
        if len(expiries) < 2:
            return None
        today = datetime.now(tz=timezone.utc).date()

        def _pick(target_days: int) -> str:
            return min(
                expiries,
                key=lambda e: abs(
                    (datetime.strptime(e, "%Y-%m-%d").date() - today).days - target_days
                ),
            )

        near_exp, far_exp = _pick(30), _pick(90)
        if near_exp == far_exp and len(expiries) > 1:
            far_exp = expiries[min(len(expiries) - 1, expiries.index(near_exp) + 1)]

        hist = tkr.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

        near = tkr.option_chain(near_exp)
        far = tkr.option_chain(far_exp)
        return compute_metrics(
            underlier=underlier,
            spot=spot,
            near_calls=near.calls,
            near_puts=near.puts,
            far_calls=far.calls,
        )
    except Exception as e:
        logger.bind(component="options_monitor").info(
            f"chain fetch failed for {underlier}: {type(e).__name__}: {e}"
        )
        return None


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _format_alert(active: list[tuple[str, str]], metrics: OptionsMetrics) -> str:
    if not active:
        return (
            "✅ *Options structure normalized* — all vol-surface triggers cleared "
            f"({metrics.underlier}: ATM IV {metrics.atm_iv:.0%}, "
            f"skew {metrics.put_skew * 100:+.1f}, slope {metrics.term_slope * 100:+.1f})."
        )
    lines = [f"🌡️ *Vol-surface signal* — {metrics.underlier} options structure:"]
    for name, detail in active:
        lines.append(f"  • `{name}`: {detail}")
    lines.append("")
    lines.append(
        f"_Snapshot: ATM IV {metrics.atm_iv:.0%} | put skew "
        f"{metrics.put_skew * 100:+.1f} pts | term slope {metrics.term_slope * 100:+.1f} pts | "
        f"P/C OI {metrics.pc_oi_ratio:.2f}._\n"
        "_Advisory only — pairs with the SPY/VIX advisor. Consider `/mode "
        "defense` if multiple triggers stack._"
    )
    return "\n".join(lines)


async def poll_and_alert(
    *,
    metrics: OptionsMetrics | None = None,
    underlier: str = "SPY",
    thresholds: OptionsThresholds | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """One poll. ``metrics`` injectable for tests; fetched when None."""
    state_path = state_path or (settings.state_dir / STATE_FILENAME)
    if metrics is None:
        metrics = fetch_chain_metrics(underlier)
    if metrics is None:
        return {"polled": False, "alert_sent": False}

    active = evaluate(metrics, thresholds)
    now_active = {name for name, _ in active}
    prior = _read_state(state_path)
    prior_active: set[str] = set(prior.get("active", []))

    new = now_active - prior_active
    cleared = bool(prior_active) and not now_active

    sent = False
    if new:
        sent = await _send_telegram(_format_alert(active, metrics))
    elif cleared:
        sent = await _send_telegram(_format_alert([], metrics))

    _write_state(
        state_path,
        {
            "active": sorted(now_active),
            "metrics": metrics.to_dict(),
            "last_polled_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return {
        "polled": True,
        "metrics": metrics.to_dict(),
        "active": sorted(now_active),
        "new": sorted(new),
        "cleared": cleared,
        "alert_sent": bool(sent),
    }


async def _send_telegram(text: str) -> bool:
    try:
        from trading.bot.notifier import send_message
    except Exception:
        logger.warning("options_monitor: cannot import telegram notifier; alert dropped")
        return False
    return await send_message(text)
