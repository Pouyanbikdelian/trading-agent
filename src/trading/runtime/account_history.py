r"""Account history before the bot existed: IBKR Flex statements.

Why (2026-09-23). The dashboard's equity curve starts at the runner's first
snapshot (the live state dir began 2026-08-07), because the TWS API only
reports NetLiq *now*. The account is older than that, and its deposits
(the last one before May 2026) happened before any snapshot, so "since
the beginning" and "money put in" were both unknowable from state/.

IBKR's Flex reports carry exactly that: daily NAV in base currency
(``EquitySummaryByReportDateInBase``, section "Net Asset Value (NAV) in
Base") and every cash transfer (``CashTransaction`` of type
"Deposits/Withdrawals", section "Cash Transactions"). They are free,
official, and read-only.

Two ways in, same parser:

* a statement XML downloaded from Client Portal (``trading history
  import-flex FILE``), or
* the Flex Web Service with a read-only token (``trading history
  fetch-flex``; FLEX_TOKEN and FLEX_QUERY_ID in .env), optionally daily.

The result is ``state/account_history.json``: NAV by day and flows by
day, merged across imports (a later import wins for the same day). The
dashboard prepends it to runner.db's curve and treats its flows as
capital, so deposits no longer need typing in by hand.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

FILENAME = "account_history.json"
#: IBKR's current Flex Web Service host. Overridable because IBKR has moved
#: it before (the old /Universal/servlet/FlexStatementService.* path).
DEFAULT_SEND_URL = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
)
_TRANSFER_TYPES = ("deposit", "withdraw")  # "Deposits/Withdrawals" and variants
_RETRY_CODES = {"1001", "1004", "1005", "1006", "1007", "1008", "1009", "1018", "1019", "1021"}


class FlexError(RuntimeError):
    """A Flex statement could not be read or fetched."""


@dataclass
class FlexHistory:
    account: str | None = None
    base_currency: str | None = None
    nav: dict[str, float] = field(default_factory=dict)  # ISO day -> NAV in base
    flows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _parse_day(raw: str | None) -> date | None:
    """Flex dates: 20260807, 2026-08-07, 20260807;103000, 08/07/2026."""
    if not raw:
        return None
    s = str(raw).strip().split(";")[0].split(",")[0].split(" ")[0]
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _num(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).replace(",", ""))
    except ValueError:
        return None


def _rate_to_base(
    el: Any, currency: str, base: str, day: date, usdchf: dict[str, float] | None
) -> tuple[float | None, str]:
    """IBKR's own rate when the query carries it; never a silent 1.0.

    The first version fell back to 1.0 for a missing ``fxRateToBase``,
    which would have booked a 68,088-dollar deposit as 68,088 francs.
    """
    rate = _num(el.get("fxRateToBase"))
    if rate:
        return rate, "fxRateToBase"
    if not base or currency == base:
        return 1.0, "same currency"
    if usdchf and {currency, base} == {"USD", "CHF"}:
        known = [d for d in usdchf if d <= day.isoformat()]
        if known:
            r = float(usdchf[max(known)])
            return (r if currency == "USD" else 1.0 / r), "USDCHF close"
    return None, "none"


def _plausible(amount: float, rate: float | None, jump: float) -> bool:
    if rate is not None:
        est = amount * rate
        return est != 0 and 0.6 <= jump / est <= 1.4
    # Unknown rate: any real FX rate is within a factor of three of 1.
    return 0.3 <= jump / amount <= 3.0


def _as_transfer(
    amount: float, rate: float | None, jump: float | None, nav_before: float
) -> float | None:
    """Base-currency amount if this untyped cash row is a transfer, else None."""
    if jump is None:
        return None
    size = abs(amount * rate) if rate is not None else abs(jump)
    if size < max(500.0, 0.01 * max(nav_before, 0.0)):
        return None
    if not _plausible(amount, rate, jump):
        return None
    return round(amount * rate, 2) if rate is not None else round(jump, 2)


def parse_flex(xml: bytes | str, *, usdchf: dict[str, float] | None = None) -> FlexHistory:
    """Read NAV-by-day and deposits/withdrawals from one Flex statement.

    Tolerant by design: unknown sections are ignored, and a statement with
    neither section is an error (a query configured without them would
    otherwise import "nothing" and look like an empty account).
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise FlexError(f"not a Flex XML statement: {e}") from e
    if root.tag == "FlexStatementResponse":
        msg = root.findtext("ErrorMessage") or root.findtext("Status") or "error response"
        raise FlexError(f"IBKR returned an error instead of a statement: {msg}")
    out = FlexHistory()
    stmt = root.find(".//FlexStatement")
    if stmt is not None:
        out.account = stmt.get("accountId")
    seen_nav = seen_cash = False
    for el in root.iter("EquitySummaryByReportDateInBase"):
        seen_nav = True
        day, total = _parse_day(el.get("reportDate")), _num(el.get("total"))
        if day is None or total is None:
            continue
        out.nav[day.isoformat()] = total
        out.base_currency = out.base_currency or el.get("currency")
        out.account = out.account or el.get("accountId")
    base = str(out.base_currency or "").upper()
    days_sorted = sorted(out.nav)

    def nav_jump(day: date) -> tuple[float | None, float]:
        """NAV change into ``day`` and the NAV before it (base currency)."""
        key = day.isoformat()
        prior = [d for d in days_sorted if d < key]
        on_or_after = [d for d in days_sorted if d >= key]
        if not prior or not on_or_after:
            return None, 0.0
        before = out.nav[prior[-1]]
        return out.nav[on_or_after[0]] - before, before

    for el in root.iter("CashTransaction"):
        seen_cash = True
        amount = _num(el.get("amount"))
        day = _parse_day(el.get("dateTime") or el.get("reportDate") or el.get("settleDate"))
        if amount is None or day is None or amount == 0:
            continue
        currency = str(el.get("currency") or "").upper()
        kind = str(el.get("type") or "").lower()
        rate, basis = _rate_to_base(el, currency, base, day, usdchf)
        jump, before = nav_jump(_parse_day(el.get("reportDate")) or day)
        if kind:
            if not any(t in kind for t in _TRANSFER_TYPES):
                continue
            inferred = False
            if rate is not None:
                amount_base = round(amount * rate, 2)
            elif jump is not None and _plausible(amount, None, jump):
                amount_base, basis = round(jump, 2), "NAV jump"
            else:
                out.notes.append(
                    f"{day} {amount:,.2f} {currency}: no rate to {base or 'base'} — skipped"
                )
                continue
        else:
            # The query was built without the "Type" column, so interest,
            # dividends, withholding tax and transfers all look alike. A
            # transfer is the one that moves NAV by (about) its own size
            # on its own day, and is not small: that separates a deposit
            # from a 107-dollar interest credit without guessing.
            amount_base = _as_transfer(amount, rate, jump, before)
            if amount_base is None:
                continue
            if rate is None:
                basis = "NAV jump"
            inferred = True
        out.flows.append(
            {
                "day": day.isoformat(),
                "amount": amount,
                "currency": currency,
                "amount_base": amount_base,
                "basis": basis,
                "inferred": inferred,
                "description": str(el.get("description") or "")[:120],
                "id": el.get("transactionID") or f"{day.isoformat()}|{amount}|{currency}",
            }
        )
    if seen_cash and any(f["inferred"] for f in out.flows):
        out.notes.append(
            "the Cash Transactions section has no Type column — transfers were inferred "
            "from NAV jumps; add Type to the Flex query to make this exact"
        )
    if not seen_nav and not seen_cash:
        raise FlexError(
            "the statement has neither 'Net Asset Value (NAV) in Base' nor 'Cash Transactions' "
            "— add both sections to the Flex query"
        )
    return out


def load_history(state_dir: Path) -> dict[str, Any]:
    p = Path(state_dir) / FILENAME
    if not p.exists():
        return {"nav": {}, "flows": [], "imports": []}
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("account_history.json must be an object")
    raw.setdefault("nav", {})
    raw.setdefault("flows", [])
    raw.setdefault("imports", [])
    return raw


def merge_and_save(
    state_dir: Path, parsed: Iterable[FlexHistory], *, source: str, now: datetime | None = None
) -> dict[str, Any]:
    """Union with what is stored: later imports win per day; flows dedupe by id."""
    now = now or datetime.now(tz=timezone.utc)
    hist = load_history(state_dir)
    nav: dict[str, float] = dict(hist["nav"])
    flows = {str(f.get("id")): f for f in hist["flows"]}
    added_days = added_flows = 0
    for h in parsed:
        for d, v in h.nav.items():
            added_days += d not in nav
            nav[d] = v
        for f in h.flows:
            added_flows += f["id"] not in flows
            flows[f["id"]] = f
        hist["account"] = h.account or hist.get("account")
        hist["base_currency"] = h.base_currency or hist.get("base_currency")
    hist["nav"] = dict(sorted(nav.items()))
    hist["flows"] = sorted(flows.values(), key=lambda f: (f["day"], str(f["id"])))
    hist["imports"] = [
        *hist["imports"],
        {"at": now.isoformat(), "source": source, "new_days": added_days, "new_flows": added_flows},
    ][-50:]
    p = Path(state_dir) / FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hist, indent=1))
    os.replace(tmp, p)
    return {
        "days": len(nav),
        "new_days": added_days,
        "flows": len(flows),
        "new_flows": added_flows,
        "first": next(iter(hist["nav"]), None),
        "last": next(reversed(hist["nav"]), None),
    }


def fetch_flex(
    token: str,
    query_id: str,
    *,
    send_url: str = DEFAULT_SEND_URL,
    get: Callable[[str, dict[str, str]], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 10,
) -> bytes:
    """Two-step Flex Web Service fetch: SendRequest, then poll GetStatement.

    ``get(url, params)`` returns an object with ``.content`` (bytes);
    injectable for tests. Retries only the codes IBKR documents as
    "try again shortly", at most ``attempts`` times, well inside the
    per-token limit of 10 requests per minute.
    """
    if get is None:
        import httpx

        def get(url: str, params: dict[str, str]) -> Any:
            return httpx.get(
                url, params=params, headers={"User-Agent": "trading-agent/1"}, timeout=30.0
            )

    resp = ET.fromstring(get(send_url, {"t": token, "q": query_id, "v": "3"}).content)
    if (resp.findtext("Status") or "").lower() != "success":
        raise FlexError(
            f"SendRequest failed ({resp.findtext('ErrorCode')}): {resp.findtext('ErrorMessage')}"
        )
    ref = resp.findtext("ReferenceCode") or ""
    get_url = resp.findtext("Url") or send_url.replace("SendRequest", "GetStatement")
    for _ in range(attempts):
        sleep(6.0)
        body = get(get_url, {"t": token, "q": ref, "v": "3"}).content
        root = ET.fromstring(body)
        if root.tag != "FlexStatementResponse":
            return bytes(body)
        code = root.findtext("ErrorCode") or ""
        if code not in _RETRY_CODES:
            raise FlexError(f"GetStatement failed ({code}): {root.findtext('ErrorMessage')}")
    raise FlexError(
        "the statement was still being generated after several tries; run it again later"
    )


def history_days(
    state_dir: Path, *, before: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """(NAV days, flows, base currency) for the dashboard; NAV cut at ``before``."""
    try:
        hist = load_history(state_dir)
    except Exception:
        return [], [], None
    days = [
        {"t": d, "account": float(v)}
        for d, v in sorted(hist["nav"].items())
        if before is None or d < before
    ]
    return days, list(hist["flows"]), hist.get("base_currency")
