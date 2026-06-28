"""Deterministic committee guards — pure functions, no LLM, no network."""

from __future__ import annotations

from typing import Any

from trading.agents.guards import run_guards


def _take(stance: str, subject: str = "SPY", direction: str | None = None) -> dict[str, Any]:
    return {
        "stance": stance,
        "take": f"{stance} on {subject}",
        "prediction": {
            "subject": subject,
            "direction": direction or ("up" if stance == "bullish" else "down"),
            "horizon_days": 5,
            "confidence": 0.7,
        },
    }


def test_no_flags_on_clean_input() -> None:
    takes = {"quant": _take("neutral"), "trader": _take("bullish")}
    assert run_guards(takes, {"positions": []}) == []


def test_lone_bull_tell() -> None:
    takes = {
        "quant": _take("bullish"),
        "trader": _take("bearish"),
        "risk_officer": _take("bearish"),
        "narrator": _take("bearish"),
    }
    flags = run_guards(takes, {"positions": []})
    assert any("lone-bull" in f for f in flags)


def test_bullish_at_52w_high_is_flagged() -> None:
    takes = {"quant": _take("bullish", subject="MU")}
    ctx = {"positions": [{"symbol": "MU", "now_pctile_52w": 1.0}]}
    flags = run_guards(takes, ctx)
    assert any("MU" in f and "52w range" in f for f in flags)


def test_crowded_at_highs() -> None:
    ctx = {
        "positions": [
            {"symbol": "SNDK", "now_pctile_52w": 1.0},
            {"symbol": "INTC", "now_pctile_52w": 0.99},
            {"symbol": "MU", "now_pctile_52w": 0.97},
        ]
    }
    flags = run_guards({}, ctx)
    assert any("crowded long" in f for f in flags)


def test_sector_concentration_when_tagged() -> None:
    ctx = {
        "positions": [
            {"symbol": "SNDK", "now_pctile_52w": 0.5, "sector": "Technology"},
            {"symbol": "INTC", "now_pctile_52w": 0.5, "sector": "Technology"},
            {"symbol": "MU", "now_pctile_52w": 0.5, "sector": "Technology"},
        ]
    }
    flags = run_guards({}, ctx)
    assert any("one correlated bet" in f for f in flags)


def test_stance_contradicts_prediction() -> None:
    takes = {"quant": _take("bullish", direction="down")}
    flags = run_guards(takes, {"positions": []})
    assert any("self-contradiction" in f for f in flags)


def test_robust_to_missing_fields() -> None:
    # None pctiles, missing symbols, empty prediction — must not raise.
    takes = {"x": {"stance": "bullish", "prediction": {}}}
    ctx = {"positions": [{"symbol": None, "now_pctile_52w": None}, {}]}
    assert isinstance(run_guards(takes, ctx), list)


def test_correlated_book_flagged() -> None:
    ctx = {"positions": [], "book_concentration": {"n": 6, "effective_bets": 1.6, "avg_corr": 0.82}}
    assert any("effective bets" in f for f in run_guards({}, ctx))


def test_diversified_book_not_flagged() -> None:
    ctx = {"positions": [], "book_concentration": {"n": 6, "effective_bets": 5.1, "avg_corr": 0.15}}
    assert not any("effective bets" in f for f in run_guards({}, ctx))
