r"""Operator-declared capital flows: deposits and withdrawals.

Why (2026-09-23). Every return on the dashboard was "flow-adjusted" by a
heuristic: a day whose equity moved more than 25% was treated as a deposit
and contributed 0%. That caught the 5x paper top-up it was written for and
nothing else. The live account went ~CHF 88k → 91k by a top-up of about 3k
(3.4%), which the heuristic read as a 3.4% gain, so a real loss of ~6% was
shown as ~3%. A deposit is not a return, however small.

The broker API gives us NetLiq and cash but no clean cash-transfer feed, so
the operator states flows once (``/deposit``, ``/withdraw``) and every
return reads them: time-weighted returns neutralise the flow on its day, and
"P&L" becomes equity minus capital actually contributed.

The ledger is append-only JSON with a lock-free atomic replace; each entry
records when it was entered and by whom, so a correction is a new entry
(a negative deposit), never an edit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

FILENAME = "capital_flows.json"


@dataclass(frozen=True)
class CapitalFlow:
    day: date
    amount: float  # + deposit, - withdrawal, in ``currency``
    currency: str
    note: str
    recorded_at: str
    recorded_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "amount": self.amount,
            "currency": self.currency,
            "note": self.note,
            "recorded_at": self.recorded_at,
            "recorded_by": self.recorded_by,
        }


def _path(state_dir: Path) -> Path:
    return Path(state_dir) / FILENAME


def load_flows(state_dir: Path) -> list[CapitalFlow]:
    """All recorded flows, oldest first. A corrupt ledger raises.

    Silently returning [] would quietly turn every deposit back into a
    gain — the exact defect this ledger exists to remove.
    """
    p = _path(state_dir)
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    if not isinstance(raw, list):
        raise ValueError("capital_flows.json must be a list")
    out = [
        CapitalFlow(
            day=date.fromisoformat(str(r["day"])),
            amount=float(r["amount"]),
            currency=str(r.get("currency") or "CHF").upper(),
            note=str(r.get("note") or ""),
            recorded_at=str(r.get("recorded_at") or ""),
            recorded_by=str(r.get("recorded_by") or ""),
        )
        for r in raw
    ]
    return sorted(out, key=lambda f: (f.day, f.recorded_at))


def record_flow(
    state_dir: Path,
    *,
    amount: float,
    currency: str,
    day: date,
    note: str = "",
    recorded_by: str = "operator",
    now: datetime | None = None,
) -> CapitalFlow:
    if amount == 0:
        raise ValueError("a capital flow of 0 records nothing")
    now = now or datetime.now(tz=timezone.utc)
    if day > now.date():
        raise ValueError("a flow cannot be dated in the future")
    flow = CapitalFlow(
        day=day,
        amount=float(amount),
        currency=currency.upper(),
        note=note.strip()[:200],
        recorded_at=now.isoformat(),
        recorded_by=recorded_by,
    )
    existing = load_flows(state_dir)
    p = _path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([f.to_dict() for f in [*existing, flow]], indent=1))
    os.replace(tmp, p)
    return flow


def flows_in_base(
    flows: list[CapitalFlow], base_currency: str, fx_rates: dict[str, float]
) -> dict[str, float | None]:
    """Net flow per ISO day in the account's base currency.

    ``fx_rates`` maps currency -> base units per one unit (the managed
    view's convention). An unconvertible flow maps to None for its day so
    the caller can say so instead of guessing a rate.
    """
    base = (base_currency or "USD").upper()
    out: dict[str, float | None] = {}
    for f in flows:
        k = f.day.isoformat()
        rate = 1.0 if f.currency == base else fx_rates.get(f.currency)
        if rate is None or out.get(k, 0.0) is None:
            out[k] = None
            continue
        out[k] = (out.get(k) or 0.0) + f.amount * float(rate)
    return out
