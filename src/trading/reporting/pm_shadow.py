"""Matched prospective decision cohorts. No broker, order store or live state mutation.

Every arm starts with identical base-currency cash on the first observation
strictly after the decision. These overlapping cohorts are NOT a continuous
portfolio backtest, and are never summed into a fictitious track record.
"""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position, Signal
from trading.risk.limits import RiskLimits


def _limit_gross(weights: dict[str, float], maximum: float) -> dict[str, float]:
    gross = sum(weights.values())
    scale = min(1.0, maximum / gross) if gross else 1.0
    return {s: w * scale for s, w in weights.items() if w > 0}


def _shadow_plan(
    payload: dict[str, Any], result: dict[str, Any], cost_bps: float
) -> dict[str, Any]:
    context = payload.get("today_context") or {}
    ladder = context.get("candidate_ladder") or {}
    mechanical = ladder.get("mechanical_weights")
    if not isinstance(mechanical, dict) or not ladder.get("shadow_risk_limits"):
        return {
            "status": "unavailable",
            "reason": "exact mechanical weights or resolved risk limits missing",
        }
    forbidden = set(payload.get("operator_held_do_not_trade") or []) | set(
        payload.get("operator_excluded_never_buy") or []
    )
    mechanical = {s: float(w) for s, w in mechanical.items() if s not in forbidden and float(w) > 0}
    pm = {s: float(w) for s, w in result["weights"].items() if s not in forbidden}
    if any(not isfinite(w) or w < 0 for w in [*mechanical.values(), *pm.values()]):
        return {"status": "unavailable", "reason": "invalid long-only cohort weights"}
    scope = result["sentinel_scope_review"]
    cap = scope["live_cap_unchanged"]
    mechanical = _limit_gross(mechanical, 1.0)
    gross_pm = sum(pm.values())
    gross_mech = sum(mechanical.values())
    equal_pm = {s: gross_pm / len(pm) for s in pm}
    cash_matched = (
        {s: w * gross_pm / gross_mech for s, w in mechanical.items()} if gross_mech else {}
    )
    passed = {
        r["symbol"]
        for r in result["reasoning_review"]["candidate_reviews"]
        if r.get("decision") == "pass"
    }
    arms = {
        "pm": _limit_gross(pm, cap),
        "mechanical": _limit_gross(mechanical, cap),
        "mechanical_at_pm_gross": _limit_gross(cash_matched, cap),
        "pm_equal_weight": _limit_gross(equal_pm, cap),
        "mechanical_with_pm_veto": _limit_gross(
            {s: w for s, w in mechanical.items() if s not in passed}, cap
        ),
        "cash": {},
        "experimental_scoped_sentinel_mechanical": _limit_gross(
            mechanical, scope["experimental_market_only_cap"]
        ),
    }
    symbols = sorted({s for w in arms.values() for s in w})
    currencies = ladder.get("quote_currencies") or {}
    sectors = ladder.get("shadow_sector_map") or {}
    missing = [s for s in symbols if s not in currencies or s not in sectors]
    if missing:
        return {
            "status": "unavailable",
            "reason": "unmatched instrument/sector metadata: " + ", ".join(missing),
        }
    return {
        "status": "awaiting_forward_prices",
        "decision_at": result["ts"],
        "arms": arms,
        "symbols": symbols,
        "risk_limits": ladder["shadow_risk_limits"],
        "sectors": sectors,
        "quote_currencies": currencies,
        "base_currency": (context.get("operator_live_account_equity_NOT_YOUR_BOOK") or {}).get(
            "base_currency", "USD"
        ),
        "cost_bps_per_side": cost_bps,
        "initial_equity": 100000.0,
        "matched_requested_gross_available": gross_mech > 0 or gross_pm == 0,
        "scope": "Matched independent decision cohorts, not compounded portfolios or live fills. The scoped-Sentinel arm is experimental only.",
    }


def _evaluate_shadow(
    plan: dict[str, Any], prices: pd.DataFrame, fx: pd.DataFrame, *, horizon: int, now: datetime
) -> dict[str, Any]:
    from trading.risk.manager import RiskManager

    if horizon not in (5, 21, 63):
        raise ValueError("horizon must be 5, 21 or 63 observations")
    if plan.get("status") != "awaiting_forward_prices":
        return {"status": "unavailable", "reason": plan.get("reason", "shadow plan missing")}
    decision = pd.Timestamp(plan["decision_at"])
    if now.tzinfo is None or decision.tzinfo is None:
        raise ValueError("aware timestamps required")
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.tz is None
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
    ):
        return {
            "status": "unavailable",
            "reason": "prices require unique increasing aware observations",
        }
    # Daily caches may contain today's unfinished candle timestamped at
    # midnight. Wait until the next UTC date before grading that observation.
    completed_before = pd.Timestamp(now).tz_convert("UTC").normalize()
    future = prices.loc[(prices.index > decision) & (prices.index < completed_before)]
    if len(future) < horizon + 1:
        return {
            "status": "pending",
            "reason": "forward horizon not mature",
            "observations": len(future),
        }
    window = future.iloc[: horizon + 1]
    symbols = plan["symbols"]
    missing = sorted(set(symbols) - set(window.columns))
    if missing:
        return {"status": "unavailable", "reason": "missing price symbols: " + ", ".join(missing)}
    base = plan["base_currency"]
    converted = window[symbols].copy().astype(float)
    for symbol in symbols:
        quote = plan["quote_currencies"][symbol]
        if quote != base:
            key = quote + base
            if (
                key not in fx.columns
                or not isinstance(fx.index, pd.DatetimeIndex)
                or fx.index.tz is None
                or not fx.index.is_unique
            ):
                return {
                    "status": "unavailable",
                    "reason": f"dated FX {key} missing; no 1:1 fallback",
                }
            converted[symbol] *= fx[key].reindex(window.index)
    if any(not isfinite(float(v)) or float(v) <= 0 for v in converted.to_numpy().ravel()):
        return {
            "status": "unavailable",
            "reason": "missing/nonpositive price or same-date FX; no forward fill",
        }
    limits = RiskLimits.model_validate(plan["risk_limits"])
    capital = float(plan["initial_equity"])
    cost_rate = float(plan["cost_bps_per_side"]) / 10000
    if not isfinite(capital) or capital <= 0 or not isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("invalid cohort capital/costs")
    instruments = {
        s: Instrument(symbol=s, asset_class=AssetClass.EQUITY, currency=base) for s in symbols
    }
    results: dict[str, dict[str, Any]] = {}
    for name, targets in plan["arms"].items():
        risk = RiskManager(limits)  # purely in memory, no live halt file
        ts = window.index[0].to_pydatetime()
        account = AccountSnapshot(
            ts=ts,
            cash=capital,
            equity=capital,
            positions={},
            base_currency=base,
            cash_by_currency={base: capital},
        )
        # Reserve the same cost allowance for every arm; never introduce a cash loan.
        targets = _limit_gross(targets, 1 / (1 + cost_rate))
        orders, decisions = risk.signal_to_orders(
            Signal(
                ts=ts,
                strategy="shadow_" + name,
                target_weights={instruments[s].key: w for s, w in targets.items()},
            ),
            account=account,
            instruments={i.key: i for i in instruments.values()},
            last_prices={instruments[s].key: float(converted[s].iloc[0]) for s in symbols},
            sector_map={instruments[s].key: plan["sectors"][s] for s in symbols},
            order_id_factory=lambda: "shadow-never-submitted",
        )
        quantities = {o.instrument.symbol: o.quantity for o in orders}
        notional = sum(q * float(converted[s].iloc[0]) for s, q in quantities.items())
        fees = notional * cost_rate
        cash = capital - notional - fees
        equity_curve = [capital - fees]
        turnover = notional / capital
        exit_reason = None
        for offset in range(1, len(window)):
            ts = window.index[offset].to_pydatetime()
            positions = {
                instruments[s].key: Position(
                    instrument=instruments[s], quantity=q, avg_price=float(converted[s].iloc[0])
                )
                for s, q in quantities.items()
            }
            opening = AccountSnapshot(
                ts=ts, cash=cash, equity=equity_curve[-1], positions=positions, base_currency=base
            )
            risk.start_of_day(opening)
            value = sum(q * float(converted[s].iloc[offset]) for s, q in quantities.items())
            marked = opening.model_copy(update={"equity": cash + value})
            decision_risk = risk.evaluate_intraday(marked)
            if quantities and (decision_risk.action == "halt" or offset == len(window) - 1):
                closes = risk.force_flatten_orders(
                    list(positions.values()),
                    ts=ts,
                    order_id_factory=lambda: "shadow-close-never-submitted",
                )
                proceeds = sum(
                    o.quantity * float(converted[o.instrument.symbol].iloc[offset]) for o in closes
                )
                closing_fee = proceeds * cost_rate
                fees += closing_fee
                turnover += proceeds / capital
                cash += proceeds - closing_fee
                quantities = {}
                value = 0.0
                if decision_risk.action == "halt":
                    exit_reason = decision_risk.reason
            equity_curve.append(cash + value)
        peak = capital
        drawdown = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = max(drawdown, 1 - equity / peak)
        results[name] = {
            "net_return": equity_curve[-1] / capital - 1,
            "cost_fraction": fees / capital,
            "turnover": turnover,
            "max_drawdown": drawdown,
            "entry_gross": notional / capital,
            "risk_exit": exit_reason,
            "risk_decisions": [d.model_dump() for d in decisions],
        }
    returns = {k: float(v["net_return"]) for k, v in results.items()}
    return {
        "status": "graded",
        "horizon": horizon,
        "entry_at": window.index[0].isoformat(),
        "exit_at": window.index[-1].isoformat(),
        "base_currency": base,
        "arms": results,
        "attribution": {
            "pm_minus_mechanical": returns["pm"] - returns["mechanical"],
            "selection_and_sizing_at_matched_requested_gross": (
                returns["pm"] - returns["mechanical_at_pm_gross"]
            )
            if plan.get("matched_requested_gross_available", True)
            else None,
            "cash_allocation_effect": (returns["mechanical_at_pm_gross"] - returns["mechanical"])
            if plan.get("matched_requested_gross_available", True)
            else None,
            "sizing_effect": returns["pm"] - returns["pm_equal_weight"],
            "sentinel_scope_effect": returns["experimental_scoped_sentinel_mechanical"]
            - returns["mechanical"],
        },
        "assumptions": [
            "identical 100k starting base cash",
            "same daily risk and concentration checks",
            "daily-close execution with stated per-side costs",
            "FX pre-funded at observed rates; cash earns zero",
            "cohorts overlap; do not compound or treat as independent samples",
            "attribution uses matched requested gross; risk caps, rounding and costs can change achieved gross",
            "no broker fills, intraday guards, dividends, tax or market impact simulated",
        ],
    }


def _grade_saved_shadows(state_dir: Path, data_dir: Path, now: datetime) -> dict[str, Any]:
    """Run from the existing PM cadence or the offline script; gaps remain visible."""
    from trading.agents.pm_audit import _read_record, _verify_decision, _write_record
    from trading.core.file_lock import file_lock
    from trading.runtime.portfolio_stats import _read_close

    counts = {"graded": 0, "pending": 0, "unavailable": 0, "invalid_audit": 0}
    records = []
    cache: dict[str, pd.Series | None] = {}

    def series(symbol: str) -> pd.Series | None:
        if symbol not in cache:
            cache[symbol] = _read_close(data_dir, symbol)
        return cache[symbol]

    root = state_dir / "agent_pm" / "decisions"
    with file_lock(state_dir / "agent_pm" / "shadow_grading"):
        for directory in sorted(root.glob("pm-*")):
            try:
                raw, digest = _read_record(directory, "decision.json")
                _verify_decision(state_dir, {**raw, "audit_sha256": digest})
            except (OSError, ValueError, KeyError, TypeError):
                counts["invalid_audit"] += 1
                continue
            plan = raw.get("shadow_plan") or {
                "status": "unavailable",
                "reason": "legacy snapshot has no matched plan",
            }
            prices = pd.DataFrame()
            rates = pd.DataFrame()
            if plan.get("status") == "awaiting_forward_prices":
                calendar = series("SPY")
                if calendar is not None:
                    prices = pd.DataFrame(index=calendar.index)
                    for symbol in plan["symbols"]:
                        values = series(symbol)
                        if values is not None:
                            prices[symbol] = values.reindex(calendar.index)
                for quote in set(plan["quote_currencies"].values()):
                    if quote == plan["base_currency"]:
                        continue
                    key = quote + plan["base_currency"]
                    for name in ("1D.parquet", "1d.parquet"):
                        path = data_dir / "fx" / key / name
                        if path.exists():
                            values = pd.read_parquet(path)["close"]
                            rates[key] = values
                            break
            for horizon in (5, 21, 63):
                filename = f"shadow-{horizon}.json"
                if (directory / filename).exists():
                    try:
                        report, _ = _read_record(directory, filename)
                        if (
                            report.get("decision_sha256") != digest
                            or report.get("audit_id") != raw["audit_id"]
                            or report.get("horizon") != horizon
                        ):
                            raise ValueError(
                                "shadow grade does not match this decision and horizon"
                            )
                    except (OSError, ValueError, KeyError, TypeError):
                        counts["invalid_audit"] += 1
                        continue
                else:
                    try:
                        report = _evaluate_shadow(plan, prices, rates, horizon=horizon, now=now)
                    except (ValueError, TypeError, KeyError):
                        report = {
                            "status": "unavailable",
                            "reason": "invalid price/FX or cohort metadata",
                        }
                    if report["status"] == "graded":
                        report.update(
                            {
                                "audit_id": raw["audit_id"],
                                "decision_sha256": digest,
                                "graded_at": now.isoformat(),
                            }
                        )
                        # Preserve the exact future observations used in this grade.
                        report["price_observations"] = prices.loc[
                            report["entry_at"] : report["exit_at"]
                        ].to_json(orient="split", date_format="iso")
                        report["fx_observations"] = rates.reindex(
                            prices.loc[report["entry_at"] : report["exit_at"]].index
                        ).to_json(orient="split", date_format="iso")
                        _write_record(directory, filename, report)
                counts[report["status"]] += 1
                records.append({"audit_id": raw["audit_id"], "horizon": horizon, **report})
    return {
        "checked_at": now.isoformat(),
        "counts": counts,
        "records": records,
        "note": "Prospective matched cohorts only. Missing history/FX is unavailable, not zero performance. Overlapping cohorts are not independent.",
    }


def _format_shadow_status(report: dict[str, Any]) -> str:
    from trading.agents.pm_reasoning import _age_hours
    from trading.core.clock import UtcClock

    age = _age_hours(report.get("checked_at"), UtcClock().now())
    lines = [f"*PM matched shadow review* · {report.get('checked_at', 'unknown time')}"]
    if age is None or age > 24 * 16:
        lines.append(
            "⚠️ Shadow grader overdue or timestamp unknown; do not treat this report as current."
        )
    if report.get("status") == "failed":
        lines.append("⚠️ Grading failed: " + str(report.get("reason", "unknown")))
    counts = report.get("counts") or {}
    lines.append(
        " · ".join(
            f"{k}: {counts.get(k, 0)}"
            for k in ("graded", "pending", "unavailable", "invalid_audit")
        )
    )
    for horizon in (5, 21, 63):
        rows = [
            r
            for r in report.get("records", [])
            if r.get("status") == "graded" and r.get("horizon") == horizon
        ]
        if rows:
            delta = sum(r["attribution"]["pm_minus_mechanical"] for r in rows) / len(rows)
            lines.append(
                f"{horizon} observations: mean PM minus mechanical {delta * 100:+.2f}pp; {len(rows)} overlapping cohorts"
            )
    reasons = list(
        dict.fromkeys(
            str(r.get("reason"))
            for r in report.get("records", [])
            if r.get("status") == "unavailable"
        )
    )
    if reasons:
        lines.append("Coverage gaps: " + "; ".join(reasons[:3])[:350])
    lines.append(
        "Same capital, dates, FX, costs and daily risk model. Cohorts are not independent, compounded returns or live fills. Scoped-Sentinel results are experimental."
    )
    return "\n".join(lines)
