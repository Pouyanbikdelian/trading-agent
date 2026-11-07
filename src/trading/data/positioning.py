"""Free positioning adapters. No broker connection, trading action or invented inventory."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from trading.core.types import PositioningObservation

# Explicit market-code mapping; futures positions are macro context, not stock flows.
COT_MARKETS = {
    "13874A": ("gpe5-46if", ("SPY",), "S&P 500 futures"),
    "209742": ("gpe5-46if", ("QQQ",), "Nasdaq 100 futures"),
    "099741": ("gpe5-46if", ("EURUSD",), "EUR futures; EURUSD direction"),
    "092741": ("gpe5-46if", (), "CHF futures; inverse direction to USDCHF"),
    "043602": ("gpe5-46if", (), "10-year Treasury futures; basis trades can dominate"),
    "067651": ("72hh-3qpy", ("USO",), "WTI futures; not refiner stock flows"),
    "088691": ("72hh-3qpy", ("GLD",), "Gold futures"),
}
NY = ZoneInfo("America/New_York")


def _identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _finite(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError("nonfinite measurement")
    return result


def expected_cot_date(now: datetime) -> str:
    """Normal release expectation only. A holiday delay is a gap, never a fake release."""
    local = now.astimezone(NY)
    friday = (local - timedelta(days=(local.weekday() - 4) % 7)).replace(
        hour=15, minute=30, second=0, microsecond=0
    )
    if local < friday:
        friday -= timedelta(days=7)
    return (friday - timedelta(days=3)).date().isoformat()


def cot_features(
    rows: list[dict[str, Any]], code: str, now: datetime
) -> list[PositioningObservation]:
    dataset, symbols, scope = COT_MARKETS[code]
    categories = (
        {
            "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
            "leveraged_money": ("lev_money_positions_long", "lev_money_positions_short"),
        }
        if dataset == "gpe5-46if"
        else {"managed_money": ("m_money_positions_long_all", "m_money_positions_short_all")}
    )
    by_date = {}
    for row in rows:
        if row.get("cftc_contract_market_code") != code:
            raise ValueError("mixed COT markets")
        stamp = str(row["report_date_as_yyyy_mm_dd"])[:10]
        if stamp in by_date:
            raise ValueError("duplicate COT report date")
        by_date[stamp] = row
    ordered = sorted(by_date, reverse=True)[:156]
    if not ordered:
        raise ValueError("empty COT market")
    observed = datetime.fromisoformat(ordered[0]).replace(tzinfo=timezone.utc)
    if observed > now:
        raise ValueError("future COT observation")
    result = []
    for category, (long_key, short_key) in categories.items():
        values = {}
        for day in ordered:
            row = by_date[day]
            oi, long, short = map(
                _finite, (row["open_interest_all"], row[long_key], row[short_key])
            )
            if oi <= 0 or min(long, short) < 0 or max(long, short) > oi:
                raise ValueError("invalid COT positions")
            values[day] = ((long - short) / oi, long, short)
        net, long, short = values[ordered[0]]
        prior = values.get((observed - timedelta(days=7)).date().isoformat())
        four = values.get((observed - timedelta(days=28)).date().isoformat())
        history = [v[0] for d, v in values.items() if d != ordered[0]]
        facts = {
            "net_oi": net,
            "delta_1w": net - prior[0] if prior else None,
            "delta_4w": net - four[0] if four else None,
            "long_change_1w": long - prior[1] if prior else None,
            "short_change_1w": short - prior[2] if prior else None,
            "percentile": sum(v <= net for v in history) / len(history)
            if len(history) >= 26
            else None,
            "history_weeks": float(len(history)),
        }
        result.append(
            PositioningObservation(
                id="cot:" + _identity([dataset, code, category, ordered[0], facts]),
                source="cot",
                kind="measured",
                scope=f"{scope}; {category}; futures only",
                symbols=symbols,
                observed_at=observed,
                available_at=now,
                expires_at=observed + timedelta(days=11),
                source_url=f"https://publicreporting.cftc.gov/resource/{dataset}.json",
                facts=facts,
                note="First seen now; historical percentiles use revised history, not archived release vintages. Crowding is not a reversal forecast.",
            )
        )
    return result


def fetch_cot(now: datetime) -> list[PositioningObservation]:
    observations = []
    with httpx.Client(timeout=15) as client:
        for code, (dataset, _, _) in COT_MARKETS.items():
            response = client.get(
                f"https://publicreporting.cftc.gov/resource/{dataset}.json",
                params={
                    "$where": f"cftc_contract_market_code='{code}'",
                    "$order": "report_date_as_yyyy_mm_dd DESC",
                    "$limit": 156,
                },
            )
            response.raise_for_status()
            observations.extend(cot_features(response.json(), code, now))
    return observations


def gamma_features(
    symbol: str, spot: float, rows: list[dict[str, Any]], now: datetime
) -> PositioningObservation:
    """Black-Scholes approximation; unsigned OI sensitivity, not dealer inventory."""
    spot = _finite(spot)
    if spot <= 0:
        raise ValueError("invalid underlying")
    totals = {"call": 0.0, "put": 0.0}
    strikes: dict[float, float] = {}
    valid = 0
    for row in rows:
        try:
            strike, iv, oi = map(
                _finite, (row["strike"], row["impliedVolatility"], row["openInterest"])
            )
            expiry = datetime.fromisoformat(str(row["expiry"])).replace(hour=16, tzinfo=NY)
            days = (expiry - now).total_seconds() / 86400
            # 0DTE is deliberately excluded; tiny time-to-expiry amplifies stale IV errors.
            if expiry.date() == now.astimezone(NY).date() or not 0 < days <= 45:
                continue
            if not 0.8 * spot <= strike <= 1.2 * spot or not 0.01 <= iv <= 3 or oi < 0:
                continue
            if row.get("contractSize") != "REGULAR" or row.get("side") not in totals:
                continue
            quote = datetime.fromisoformat(str(row["lastTradeDate"]))
            if quote.tzinfo is None or not 0 <= (now - quote).total_seconds() <= 86400:
                continue
            t = days / 365.25
            # r=q=0 explicitly; sensitivity only, not an option valuation service.
            d1 = (math.log(spot / strike) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
            gamma = math.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * spot * iv * math.sqrt(t))
            gex = gamma * oi * 100 * spot * spot * 0.01
            totals[row["side"]] += gex
            strikes[strike] = strikes.get(strike, 0) + gex
            valid += 1
        except (ValueError, TypeError, KeyError):
            continue
    unsigned = sum(totals.values())
    if valid < 10 or not rows or valid / len(rows) < 0.5 or min(totals.values()) <= 0:
        raise ValueError("insufficient fresh two-sided options coverage")
    return PositioningObservation(
        id="gamma:" + _identity([symbol, now, totals]),
        source="gamma_estimate",
        kind="estimated",
        scope=f"{symbol} sampled standard options, 1–45 days; ±20% strikes",
        symbols=(symbol,),
        observed_at=now,
        available_at=now,
        expires_at=now + timedelta(minutes=30),
        source_url=f"https://finance.yahoo.com/quote/{symbol}/options/",
        facts={
            "spot": spot,
            "unsigned_gex_per_1pct": unsigned,
            "call_positive_put_negative": totals["call"] - totals["put"],
            "all_dealers_short": -unsigned,
            "all_dealers_long": unsigned,
            "largest_strike": max(strikes, key=lambda k: strikes[k]),
            "largest_strike_share": max(strikes.values()) / unsigned,
            "coverage": valid / len(rows),
            "contracts_used": float(valid),
        },
        note="MODELLED, not observed dealer gamma. OI as-of unknown; lastTradeDate is not a quote timestamp. Partial chain; no 0DTE. European Black-Scholes r=q=0 on American options. Opposite dealer-side assumptions reverse sign. Timing research only, never a directional measurement.",
    )


def fetch_gamma(symbol: str, now: datetime) -> PositioningObservation:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1d", interval="1m", timeout=10)
    if history.empty:
        raise ValueError("missing underlying bars")
    stamp = history.index[-1].to_pydatetime()
    if stamp.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= 1800:
        raise ValueError("stale underlying bar")
    spot = _finite(history["Close"].iloc[-1])
    expiries = [
        e
        for e in (ticker.options or ())
        if 0 < (datetime.fromisoformat(e).date() - now.astimezone(NY).date()).days <= 45
    ][:3]
    if not expiries:
        raise ValueError("options provider returned no usable expirations")
    rows: list[dict[str, Any]] = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        if chain is None:
            raise ValueError("options provider returned no chain")
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if (
                frame is None
                or frame.empty
                or not {
                    "strike",
                    "impliedVolatility",
                    "openInterest",
                    "lastTradeDate",
                    "contractSize",
                }
                <= set(frame.columns)
            ):
                raise ValueError(f"options provider returned incomplete {side} chain")
            selected = frame[(frame["strike"] >= 0.8 * spot) & (frame["strike"] <= 1.2 * spot)]
            rows.extend({**r, "expiry": expiry, "side": side} for r in selected.to_dict("records"))
    return gamma_features(symbol, spot, rows, now)
