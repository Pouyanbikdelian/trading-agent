"""Deterministic pre-digest guards — mechanical checks, no LLM, no spend.

The committee's personas catch judgement errors; these catch the *mechanical*
ones an LLM reviewer tends to miss or rationalise away: a take that reads a
52-week-high percentile as bullish confirmation, a lone bull against a bearish
majority, book concentration at the highs, or a stance that contradicts its own
prediction. Pure functions over the committee's structured output — the flags
are advisory context for the manager and the digest, and NEVER an order gate
(same isolation contract as every agent).

Why deterministic and not a "senior reviewer" LLM: these errors are
rule-shaped, so a check catches them every time, for $0, and a confident take
cannot argue it down. Judgement-shaped errors stay the Challenger's job.
"""

from __future__ import annotations

from typing import Any

NEAR_HIGH = 0.95  # now_pctile_52w at/above this == "at the top of the range"
SECTOR_CLUSTER = 3  # this many held names in one sector == one bet, not N
CROWDED_AT_HIGHS = 3  # this many holdings at their highs == crowded long


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def run_guards(
    takes: dict[str, dict[str, Any]],
    context: dict[str, Any],
    *,
    near_high: float = NEAR_HIGH,
) -> list[str]:
    """Mechanical checks over committee takes + context. Returns short,
    human-readable flag strings (possibly empty). Pure and side-effect free."""
    flags: list[str] = []
    positions = context.get("positions") or []
    _stance_vs_prediction(takes, flags)
    _lone_voice(takes, flags)
    _bullish_at_high(takes, positions, near_high, flags)
    _crowded_at_highs(positions, near_high, flags)
    _sector_concentration(positions, flags)
    _correlated_book(context, flags)
    return flags


def _stance_vs_prediction(takes: dict[str, dict[str, Any]], flags: list[str]) -> None:
    for name, t in takes.items():
        stance = t.get("stance")
        direction = (t.get("prediction") or {}).get("direction")
        if stance == "bullish" and direction == "down":
            flags.append(f"{name}: bullish stance but predicts {direction} — self-contradiction.")
        elif stance == "bearish" and direction == "up":
            flags.append(f"{name}: bearish stance but predicts {direction} — self-contradiction.")


def _lone_voice(takes: dict[str, dict[str, Any]], flags: list[str]) -> None:
    bulls = [n for n, t in takes.items() if t.get("stance") == "bullish"]
    bears = [n for n, t in takes.items() if t.get("stance") == "bearish"]
    if len(bulls) == 1 and len(bears) >= 3:
        flags.append(
            f"lone-bull tell: only {bulls[0]} is bullish vs {len(bears)} bearish — "
            "a lone trend-follower usually means 'price went up', not an edge."
        )
    elif len(bears) == 1 and len(bulls) >= 3:
        flags.append(
            f"lone-bear tell: only {bears[0]} is bearish vs {len(bulls)} bullish — "
            "confirm it is a real risk, not a reflexive permabear."
        )


def _bullish_at_high(
    takes: dict[str, dict[str, Any]],
    positions: list[dict[str, Any]],
    near_high: float,
    flags: list[str],
) -> None:
    pctile = {
        p.get("symbol"): _as_float(p.get("now_pctile_52w")) for p in positions if p.get("symbol")
    }
    for name, t in takes.items():
        if t.get("stance") != "bullish":
            continue
        subj = (t.get("prediction") or {}).get("subject")
        pc = pctile.get(subj)
        if pc is not None and pc >= near_high:
            flags.append(
                f"{name} bullish on {subj} at {pc:.0%} of its 52w range — "
                "'at the high' is where price is, not whether reward/risk is good."
            )


def _crowded_at_highs(positions: list[dict[str, Any]], near_high: float, flags: list[str]) -> None:
    at_high = [
        str(p.get("symbol"))
        for p in positions
        if (pc := _as_float(p.get("now_pctile_52w"))) is not None and pc >= near_high
    ]
    if len(at_high) >= CROWDED_AT_HIGHS:
        flags.append(
            f"{len(at_high)} holdings at/above {near_high:.0%} of their 52w range "
            f"({', '.join(at_high)}) — crowded long, maximum give-back if it turns."
        )


def _sector_concentration(positions: list[dict[str, Any]], flags: list[str]) -> None:
    by_sector: dict[str, list[str]] = {}
    for p in positions:
        sec = p.get("sector")
        sym = p.get("symbol")
        if sec and sym:
            by_sector.setdefault(str(sec), []).append(str(sym))
    for sec, syms in by_sector.items():
        if len(syms) >= SECTOR_CLUSTER:
            flags.append(
                f"{len(syms)} holdings in {sec} ({', '.join(syms)}) — one correlated "
                f"bet, not {len(syms)} independent confirmations."
            )


def _correlated_book(context: dict[str, Any], flags: list[str]) -> None:
    """Cross-sector concentration the sector tag misses: if the book's effective
    number of bets (computed upstream from the correlation matrix) is well below
    the holding count, several names are really one bet. Reads the precomputed
    ``book_concentration`` number — no math here, so the guard stays pure."""
    bc = context.get("book_concentration")
    if not isinstance(bc, dict):
        return
    n = bc.get("n") or 0
    enb = bc.get("effective_bets")
    avg = bc.get("avg_corr")
    if enb is None or n < 3:
        return
    if enb < max(2.0, 0.5 * n):
        flags.append(
            f"book acts like ~{enb} effective bets across {n} holdings "
            f"(avg corr {avg}) — correlated names are one bet, not {n} confirmations."
        )
