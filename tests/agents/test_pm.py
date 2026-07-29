"""Agent PM — hermetic tests: injected LLM, injected prices, tmp state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trading.agents.pm import (
    START_EQUITY,
    UNIVERSE,
    _clamp_weights,
    format_pm_digest,
    run_agent_pm,
)
from trading.memory import MemoryStore

PRICES = {"SMH": 100.0, "XLE": 50.0, "SPY": 500.0, "TLT": 90.0}


@pytest.fixture
def mem(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def _pm_llm(weights: dict[str, Any]):
    def llm(system: str, prompt: str) -> dict[str, Any]:
        assert "Portfolio Manager" in system
        return {
            "target_weights": weights,
            "rationale": "Scout theme confirmed by 3m relative momentum; risk officer quiet.",
            "watch": "SMH closing below its 20dma",
        }

    return llm


def test_clamp_enforces_whitelist_cap_and_gross() -> None:
    w = _clamp_weights(
        {"SMH": 0.9, "FAKE": 0.5, "TLT": -0.2, "XLE": 0.25, "SPY": "0.25", "QQQ": 0.25}
    )
    assert "FAKE" not in w and "TLT" not in w  # off-universe / short dropped
    # SMH (0.9 -> per-name 0.25) + QQQ (0.25) are both tech_complex: the
    # 0.50 cluster cap; combined = 0.50 exactly so no scaling needed.
    assert w["SMH"] == 0.25 and w["QQQ"] == 0.25
    assert w["XLE"] == 0.25 and w["SPY"] == 0.25
    assert sum(w.values()) <= 1.0 + 1e-9  # gross cap
    assert all(s in UNIVERSE for s in w)


def test_clamp_drops_operator_held_symbols() -> None:
    w = _clamp_weights({"SMH": 0.2, "XLE": 0.2}, blocked=frozenset({"SMH"}))
    assert "SMH" not in w and w["XLE"] == 0.2  # pinned name cut to cash


def test_run_respects_holds_file(mem: MemoryStore, tmp_path: Path) -> None:
    """A /hold-pinned symbol must never receive PM allocation, and the PM
    prompt must disclose the pin (belt and braces)."""
    from trading.runner.holds import save_holds

    save_holds(tmp_path, {"SMH"})
    seen: dict[str, str] = {}

    def llm(system: str, prompt: str) -> dict[str, Any]:
        seen["prompt"] = prompt
        return {"target_weights": {"SMH": 0.25, "XLE": 0.2}, "rationale": "r", "watch": "w"}

    res = run_agent_pm({}, mem, tmp_path, llm=llm, prices=PRICES)
    assert res["ok"] is True
    assert res["weights"] == {"XLE": 0.2}
    assert "SMH" in json.loads(seen["prompt"])["operator_held_do_not_trade"]
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert "SMH" not in book["holdings"]


def test_first_run_builds_book_and_journals(mem: MemoryStore, tmp_path: Path) -> None:
    llm = _pm_llm({"SMH": 0.25, "XLE": 0.2, "FAKE": 0.5})
    res = run_agent_pm({}, mem, tmp_path, llm=llm, prices=PRICES)

    assert res["ok"] is True
    assert res["weights"] == {"SMH": 0.25, "XLE": 0.2}
    assert "FAKE" in res["dropped"]

    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert book["holdings"]["SMH"] == pytest.approx(0.25 * START_EQUITY / 100.0)
    assert book["holdings"]["XLE"] == pytest.approx(0.2 * START_EQUITY / 50.0)
    # cash = equity - invested - costs(10bps on turnover)
    invested = 0.45 * START_EQUITY
    assert book["cash"] == pytest.approx(START_EQUITY - invested - invested * 0.001, abs=0.01)
    assert any(e["kind"] == "agent_pm" for e in mem.journal_tail(5))


def test_rebalance_marks_to_market(mem: MemoryStore, tmp_path: Path) -> None:
    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    up = dict(PRICES, SMH=120.0)  # +20% on the holding
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SPY": 0.5}), prices=up)
    assert res["ok"] is True
    assert res["equity"] > START_EQUITY  # gain realized in the mark
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert "SMH" not in book["holdings"] and "SPY" in book["holdings"]


def test_missing_price_for_held_position_skips_run(mem: MemoryStore, tmp_path: Path) -> None:
    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SPY": 0.5}), prices={"SPY": 500.0})
    assert res["ok"] is False  # never guess a mark


def test_llm_failure_is_reported_not_raised(mem: MemoryStore, tmp_path: Path) -> None:
    def boom(s: str, p: str) -> dict[str, Any]:
        raise RuntimeError("LLM API 400: nope")

    res = run_agent_pm({}, mem, tmp_path, llm=boom, prices=PRICES)
    assert res["ok"] is False
    assert "did not trade" in format_pm_digest(res)


def test_digest_renders(mem: MemoryStore, tmp_path: Path) -> None:
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    text = format_pm_digest(res)
    assert "Agent PM" in text and "`SMH` 25%" in text and len(text) < 2000


def test_digest_names_the_book_and_currency(mem: MemoryStore, tmp_path: Path) -> None:
    """The sim posts USD into the same chat as a CHF-reporting real
    account. An unlabelled equity figure reads as the other book."""
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    text = format_pm_digest(res)
    assert "simulated" in text.lower()
    assert "USD" in text
    assert "not the trading account" in text


def test_digest_leads_with_what_changed(mem: MemoryStore, tmp_path: Path) -> None:
    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"XLE": 0.2}), prices=PRICES)
    assert res["closed"] == ["SMH"] and res["opened"] == ["XLE"]
    text = format_pm_digest(res)
    assert "Changed:" in text and "exited" in text and "opened" in text


def test_digest_embeds_the_book_with_units(mem: MemoryStore, tmp_path: Path) -> None:
    """One message per event: the digest carries the resulting holdings so
    a second share-count-only message is not needed."""
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    text = format_pm_digest(res, book)
    assert "Book now:" in text
    assert "sh ·" in text  # share count carries its unit
    assert "cash" in text


def test_rebalance_persists_marks(mem: MemoryStore, tmp_path: Path) -> None:
    """/pm must render value and weight without a network call."""
    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert book["marks"]["SMH"] == pytest.approx(100.0)
    assert book["marks_ts"]


def test_mark_to_market_refreshes_marks(mem: MemoryStore, tmp_path: Path) -> None:
    from trading.agents.pm import mark_to_market

    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    mark_to_market(tmp_path, prices={"SMH": 110.0, "SPY": 510.0})
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert book["marks"]["SMH"] == pytest.approx(110.0)


class TestFormatHoldings:
    def test_holdings_carry_units_value_and_weight(self) -> None:
        from trading.agents.pm import format_holdings

        lines = format_holdings(
            {"holdings": {"GLD": 285.839}, "cash": 100.0, "marks": {"GLD": 285.6}}
        )
        body = "\n".join(lines)
        assert "285.8 sh" in body  # shares, labelled
        assert "$81.6k" in body  # value
        assert "%" in body  # weight
        assert "cash" in body

    def test_unmarked_holding_is_flagged_not_guessed(self) -> None:
        from trading.agents.pm import format_holdings

        body = "\n".join(format_holdings({"holdings": {"GLD": 10.0}, "cash": 0.0, "marks": {}}))
        assert "unmarked" in body
        assert "sh" in body

    def test_empty_book_says_all_cash(self) -> None:
        from trading.agents.pm import format_holdings

        assert "all cash" in "\n".join(format_holdings({"holdings": {}, "cash": 1.0}))

    def test_stale_marks_are_disclosed(self) -> None:
        from trading.agents.pm import _marks_age_note

        old = (datetime.now(tz=timezone.utc) - timedelta(days=3)).isoformat()
        assert "3d old" in _marks_age_note({"marks_ts": old})
        assert _marks_age_note({"marks_ts": datetime.now(tz=timezone.utc).isoformat()}) == ""
        assert _marks_age_note({}) == ""


def test_daily_mark_and_performance(mem: MemoryStore, tmp_path: Path) -> None:
    from trading.agents.pm import mark_to_market, performance

    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    res = mark_to_market(tmp_path, prices={"SMH": 110.0, "SPY": 510.0})
    assert res["ok"] is True
    # idempotent per day: a re-mark replaces, not appends
    n_before = performance(tmp_path)["points"]
    mark_to_market(tmp_path, prices={"SMH": 111.0, "SPY": 511.0})
    perf = performance(tmp_path)
    assert perf["points"] == n_before
    assert perf["return_pct"] > 0  # SMH 100 -> 111 on a 25% position
    assert perf["max_drawdown_pct"] >= 0
    # missing price refuses to guess
    assert mark_to_market(tmp_path, prices={"SPY": 500.0})["ok"] is False


def test_news_load_drops_stale(tmp_path: Path) -> None:
    from trading.runtime.news_watch import load

    p = tmp_path / "news.json"
    fresh = datetime.now(tz=timezone.utc).isoformat()
    stale = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
    p.write_text(json.dumps({"t": fresh, "headlines": [{"title": "x"}]}))
    assert load(tmp_path)["headlines"]
    p.write_text(json.dumps({"t": stale, "headlines": [{"title": "x"}]}))
    assert load(tmp_path) == {}
