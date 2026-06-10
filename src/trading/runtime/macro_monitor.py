r"""Macro financial-conditions monitor — the "economist" advisor.

Daily advisory derived from docs/research_macro_leadlag.md (2018-2026
study). Watches the four channels that empirically led or inversely
tracked NDX, each as a z-score of a trailing move vs its own 1-year
history:

* ``rates_shock``  — 5d change in the 5y yield (fastest channel;
                     t≈-2 on NDX next 5d in-sample).
* ``dollar``       — 5d DXY move (most consistent inverse since 2020).
* ``energy``       — 21d natgas+WTI momentum (slow-burn; led 3 of the
                     4 major NDX peaks by 2-6 weeks).
* ``btc_confirm``  — 21d BTC momentum (risk-ON confirm; positive
                     correlation, NOT a hedge).

A composite (mean of the three risk-off channels) is reported as a
"financial conditions" dial. Research caveat baked into the design:
quintile asymmetry was economically real but t-stats were ~2 with
overlapping windows and n=4 turning points — so this module only
informs the operator and accumulates a scoreable history in
``state/macro_monitor.json``. It never touches the order path.

Pure computation is separated from fetching so tests stay hermetic.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger

STATE_FILENAME = "macro_monitor.json"

TICKERS = {
    "^FVX": "y_5y",
    "DX-Y.NYB": "dxy",
    "NG=F": "natgas",
    "CL=F": "wti",
    "BTC-USD": "btc",
}


@dataclass(frozen=True)
class MacroReadings:
    asof: datetime
    rates_shock_z: float  # +ve = yields rising fast (equity-negative)
    dollar_z: float  # +ve = dollar strengthening (equity-negative)
    energy_z: float  # +ve = energy rallying (slow equity-negative)
    btc_confirm_z: float  # +ve = risk-ON confirm
    composite: float  # mean of the three risk-off channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof.isoformat(),
            "rates_shock_z": self.rates_shock_z,
            "dollar_z": self.dollar_z,
            "energy_z": self.energy_z,
            "btc_confirm_z": self.btc_confirm_z,
            "composite": self.composite,
        }


def _trail_z(moves: pd.Series, look: int, hist: int = 252) -> float:
    """z of the latest `look`-day cumulative move vs `hist`-day history."""
    m = moves.rolling(look).sum()
    mu = m.rolling(hist).mean()
    sd = m.rolling(hist).std()
    z = (m - mu) / sd
    out = z.dropna()
    return float(out.iloc[-1]) if len(out) else 0.0


def compute_readings(closes: pd.DataFrame, asof: datetime | None = None) -> MacroReadings:
    """Pure path. ``closes``: columns y_5y, dxy, natgas, wti, btc —
    ~2 years of daily closes (yields in % points)."""
    rates = _trail_z(closes["y_5y"].dropna().diff(), 5)
    dollar = _trail_z(np.log(closes["dxy"].dropna()).diff(), 5)
    energy_parts = [
        _trail_z(np.log(closes[c].dropna()).diff(), 21)
        for c in ("natgas", "wti")
        if c in closes
    ]
    energy = float(np.mean(energy_parts)) if energy_parts else 0.0
    btc = _trail_z(np.log(closes["btc"].dropna()).diff(), 21) if "btc" in closes else 0.0
    composite = float(np.mean([rates, dollar, energy]))
    return MacroReadings(
        asof=asof or datetime.now(tz=timezone.utc),
        rates_shock_z=rates,
        dollar_z=dollar,
        energy_z=energy,
        btc_confirm_z=btc,
        composite=composite,
    )


def evaluate(r: MacroReadings, threshold: float = 1.5) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if r.rates_shock_z >= threshold:
        out.append(
            ("RATES_SHOCK", f"5y yield 5d move at {r.rates_shock_z:+.1f}σ — duration risk for tech")
        )
    if r.dollar_z >= threshold:
        out.append(
            ("DOLLAR_SQUEEZE", f"DXY 5d move at {r.dollar_z:+.1f}σ — tightening financial conditions")
        )
    if r.energy_z >= threshold:
        out.append(
            ("ENERGY_SHOCK", f"energy 21d momentum at {r.energy_z:+.1f}σ — slow-burn squeeze channel")
        )
    if r.composite >= threshold:
        out.append(("MACRO_COMPOSITE", f"composite at {r.composite:+.1f}σ — multiple channels stressed"))
    return out


def fetch_closes(lookback_days: int = 700) -> pd.DataFrame | None:
    """Network path — None on any failure (must never break the runner)."""
    try:
        import yfinance as yf

        raw = yf.download(
            " ".join(TICKERS),
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        cols = {}
        for tkr, name in TICKERS.items():
            try:
                cols[name] = raw[tkr]["Close"].dropna()
            except Exception:
                continue
        if "y_5y" not in cols or "dxy" not in cols:
            return None
        return pd.DataFrame(cols).sort_index().ffill(limit=3)
    except Exception as e:
        logger.bind(component="macro_monitor").info(
            f"macro fetch failed: {type(e).__name__}: {e}"
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


def _format_alert(active: list[tuple[str, str]], r: MacroReadings) -> str:
    snap = (
        f"_Dial: composite {r.composite:+.1f}σ | rates {r.rates_shock_z:+.1f} | "
        f"dollar {r.dollar_z:+.1f} | energy {r.energy_z:+.1f} | "
        f"BTC-confirm {r.btc_confirm_z:+.1f}._"
    )
    if not active:
        return f"✅ *Macro conditions normalized* — all channels back inside ±1.5σ.\n{snap}"
    lines = ["🌍 *Macro conditions signal*:"]
    for name, detail in active:
        lines.append(f"  • `{name}`: {detail}")
    lines.append("")
    lines.append(snap)
    lines.append(
        "_Advisory only (see docs/research_macro_leadlag.md — suggestive, "
        "not proven). Stacks with the vol-surface and SPY/VIX advisors._"
    )
    return "\n".join(lines)


async def poll_and_alert(
    *,
    readings: MacroReadings | None = None,
    state_path: Path | None = None,
    threshold: float = 1.5,
) -> dict[str, Any]:
    """One poll; ``readings`` injectable for tests."""
    state_path = state_path or (settings.state_dir / STATE_FILENAME)
    if readings is None:
        closes = fetch_closes()
        if closes is None or len(closes) < 300:
            return {"polled": False, "alert_sent": False}
        readings = compute_readings(closes)

    active = evaluate(readings, threshold)
    now_active = {n for n, _ in active}
    prior = _read_state(state_path)
    prior_active: set[str] = set(prior.get("active", []))

    new = now_active - prior_active
    cleared = bool(prior_active) and not now_active
    sent = False
    if new:
        sent = await _send_telegram(_format_alert(active, readings))
    elif cleared:
        sent = await _send_telegram(_format_alert([], readings))

    # Append to a bounded history so the advisor can be SCORED later
    # (composite vs realized NDX drawdowns) — the research note's
    # condition for ever promoting this beyond advisory.
    history = list(prior.get("history", []))[-499:]
    history.append(readings.to_dict())

    _write_state(
        state_path,
        {
            "active": sorted(now_active),
            "readings": readings.to_dict(),
            "history": history,
            "last_polled_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return {
        "polled": True,
        "readings": readings.to_dict(),
        "active": sorted(now_active),
        "new": sorted(new),
        "cleared": cleared,
        "alert_sent": bool(sent),
    }


async def _send_telegram(text: str) -> bool:
    try:
        from trading.bot.notifier import send_message
    except Exception:
        logger.warning("macro_monitor: cannot import telegram notifier; alert dropped")
        return False
    return await send_message(text)
