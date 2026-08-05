"""Agent PM — hermetic tests: injected LLM, injected prices, tmp state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trading.agents.pm import (
    MAX_ETF_POSITIONS,
    PM_CHARTER,
    PROMPT_BUDGET,
    START_EQUITY,
    UNIVERSE,
    _budgeted_prompt,
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
    w = _clamp_weights({"SMH": 0.9, "FAKE": 0.5, "TLT": -0.2, "XLE": 0.25, "QQQ": 0.25})
    assert "FAKE" not in w and "TLT" not in w  # off-universe / short dropped
    # SMH (0.9 -> per-name 0.25) + QQQ (0.25) are both tech_complex: the
    # 0.50 cluster cap; combined = 0.50 exactly so no scaling needed.
    assert w["SMH"] == 0.25 and w["QQQ"] == 0.25
    assert w["XLE"] == 0.25
    assert len(w) <= MAX_ETF_POSITIONS  # ETF-count cap, see TestAntiFixation
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


def test_unfetched_holding_falls_back_to_stored_mark(mem: MemoryStore, tmp_path: Path) -> None:
    """A transient price-fetch gap must not freeze the book.

    Aborting the cycle on any unpriced holding turned a routine yfinance
    miss into a silent no-trade week; a run of them into a book that had
    not moved in a month.
    """
    run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SPY": 0.5}), prices={"SPY": 500.0})
    assert res["ok"] is True
    assert res["stale_marks"] == ["SMH"]  # disclosed, not silently folded in
    book = json.loads((tmp_path / "agent_pm" / "portfolio.json").read_text())
    assert "SPY" in book["holdings"] and "SMH" not in book["holdings"]


def test_holding_with_no_mark_anywhere_still_skips_run(mem: MemoryStore, tmp_path: Path) -> None:
    """The fallback is a stored mark, never a guess: a name with no price
    in the fetch AND none on file still aborts."""
    pm_dir = tmp_path / "agent_pm"
    pm_dir.mkdir(parents=True)
    (pm_dir / "portfolio.json").write_text(
        json.dumps({"cash": 1000.0, "holdings": {"SMH": 10.0}, "history": []})
    )
    res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SPY": 0.5}), prices={"SPY": 500.0})
    assert res["ok"] is False
    assert "SMH" in res["reason"]


class TestAntiFixation:
    """The 2026-08-05 regression suite: the simulated book held the same
    names for weeks with healthy turnover, because nothing in the prompt
    ever named a stock the PM did not already own, and nothing in the
    output measured 'the name set did not move'."""

    def test_etf_count_cap_is_enforced_not_merely_asked_for(self) -> None:
        w = _clamp_weights({"XLE": 0.2, "XLV": 0.19, "XLU": 0.18, "XLP": 0.17, "XLB": 0.16})
        assert len(w) == MAX_ETF_POSITIONS
        assert set(w) == {"XLE", "XLV", "XLU"}  # heaviest kept, deterministic

    def test_etf_cap_does_not_touch_single_stocks(self) -> None:
        w = _clamp_weights(
            {"XLE": 0.2, "XLV": 0.1, "XLU": 0.1, "XLP": 0.1, "JPM": 0.1, "LMT": 0.1},
            stocks=("JPM", "LMT"),
        )
        assert sum(1 for s in w if s in UNIVERSE) == MAX_ETF_POSITIONS
        assert w["JPM"] == 0.1 and w["LMT"] == 0.1

    def test_agent_takes_reach_the_prompt(self, mem: MemoryStore, tmp_path: Path) -> None:
        """The CREATIVE SCOUT RULE keys on individual takes; the committee
        journal row carries only the manager's synthesis, so the takes have
        to be read from their own journal kind."""
        mem.journal(
            "take",
            {
                "agent": "scout",
                "stance": "bullish",
                "take": "Uranium restart cycle is under-owned.",
                "prediction": {
                    "subject": "URA",
                    "direction": "up",
                    "horizon_days": 30,
                    "confidence": 0.8,
                },
            },
            actor="scout",
        )
        seen: dict[str, str] = {}

        def llm(system: str, prompt: str) -> dict[str, Any]:
            seen["prompt"] = prompt
            return {"target_weights": {"XLE": 0.2}, "rationale": "r", "watch": "w"}

        run_agent_pm({}, mem, tmp_path, llm=llm, prices=PRICES)
        takes = json.loads(seen["prompt"])["agent_takes"]
        assert [t["agent"] for t in takes] == ["scout"]
        assert takes[0]["prediction"]["confidence"] == 0.8

    def test_candidate_ladder_reaches_the_prompt(self, mem: MemoryStore, tmp_path: Path) -> None:
        """The ladder is the PM's only source of names it does not own."""
        ladder = {"strategy": "top_k_momentum", "ranked": [{"rank": 1, "symbol": "NVDA"}]}
        seen: dict[str, str] = {}

        def llm(system: str, prompt: str) -> dict[str, Any]:
            seen["prompt"] = prompt
            return {"target_weights": {"XLE": 0.2}, "rationale": "r", "watch": "w"}

        res = run_agent_pm({"candidate_ladder": ladder}, mem, tmp_path, llm=llm, prices=PRICES)
        assert json.loads(seen["prompt"])["today_context"]["candidate_ladder"] == ladder
        assert res["candidate_ladder_seen"] is True

    def test_charter_names_no_example_tickers(self) -> None:
        """Exemplar tickers in the instructions became the candidate list:
        with no ranked feed, the model picked what the prompt named."""
        for ticker in ("JPM", "LMT", "RTX", "AMAT", "ASML", "UNH", "LLY"):
            assert ticker not in PM_CHARTER

    def test_staleness_counts_cycles_with_no_name_change(
        self, mem: MemoryStore, tmp_path: Path
    ) -> None:
        """Weight churn reads as activity; an unchanged name set does not.
        Turnover was healthy the whole time the book was frozen."""
        run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
        for expected in (1, 2, 3):
            # Same name, different weight every cycle: real turnover, no
            # evolution — exactly the observed failure.
            res = run_agent_pm(
                {}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25 - expected * 0.02}), prices=PRICES
            )
            assert res["cycles_since_name_change"] == expected
        assert "same names for 3 cycles" in format_pm_digest(res)

        fresh = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"XLE": 0.2}), prices=PRICES)
        assert fresh["cycles_since_name_change"] == 0
        assert "same names" not in format_pm_digest(fresh)

    def test_digest_flags_a_missing_ladder(self, mem: MemoryStore, tmp_path: Path) -> None:
        res = run_agent_pm({}, mem, tmp_path, llm=_pm_llm({"SMH": 0.25}), prices=PRICES)
        assert res["candidate_ladder_seen"] is False
        assert "no candidate ladder" in format_pm_digest(res)


class TestPromptBudget:
    """``json.dumps(payload)[:24000]`` put today_context last, so the first
    casualty of an overflow was the market data the decision is about, and
    the model received JSON cut off mid-string."""

    @staticmethod
    def _payload(headlines: int = 400, rulings: int = 6) -> dict[str, Any]:
        return {
            "sim_portfolio": {"equity": 1_000_000, "holdings": {"SMH": 10}},
            "agent_takes": [{"agent": f"a{i}", "take": "x" * 300} for i in range(16)],
            "today_context": {
                "candidate_ladder": {"ranked": [{"rank": 1, "symbol": "NVDA"}]},
                "macro_dial": {"vix": 17.2},
                "headlines": [{"title": "h" * 120} for _ in range(headlines)],
            },
            "recent_committee_rulings": [
                {"payload": {"ruling": {"proposal": "p" * 2000}}} for _ in range(rulings)
            ],
        }

    def test_small_payload_is_untouched(self) -> None:
        p = {"a": 1, "today_context": {"macro_dial": {"vix": 17.2}}}
        assert json.loads(_budgeted_prompt(p)) == p

    def test_overflow_stays_valid_json(self) -> None:
        out = _budgeted_prompt(self._payload())
        assert len(out) <= PROMPT_BUDGET
        json.loads(out)  # would raise on a mid-string slice

    def test_overflow_sacrifices_gossip_before_the_decision_inputs(self) -> None:
        out = json.loads(_budgeted_prompt(self._payload()))
        assert out["today_context"]["candidate_ladder"]  # the new-name feed survives
        assert out["today_context"]["macro_dial"]  # today's market survives
        assert out["sim_portfolio"]["holdings"]  # the book survives
        assert len(out["today_context"].get("headlines", [])) < 400  # gossip paid

    def test_history_is_trimmed_newest_first(self) -> None:
        # Headlines alone cannot close a very large gap, so rulings must
        # give way too — and the tail (oldest) goes first.
        out = json.loads(_budgeted_prompt(self._payload(headlines=0, rulings=40)))
        assert 0 < len(out["recent_committee_rulings"]) < 40
        assert out["today_context"]["candidate_ladder"]


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
