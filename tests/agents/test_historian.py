"""Historian — hermetic tests: the LLM proposes, the store's mechanics dispose."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from trading.agents.historian import format_historian_digest, run_historian
from trading.memory import MemoryStore


@pytest.fixture
def mem(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def test_creates_capped_lessons_and_votes(mem: MemoryStore) -> None:
    existing = mem.add_lesson("Sharp corrections inside uptrends resolve upward within 10 days")
    evidence_id = mem.add_prediction(
        agent="quant",
        subject="SMH",
        direction="up",
        horizon_days=5,
        confidence=0.7,
        statement="semis held through the correction",
    )
    mem.grade_prediction(evidence_id, realized_move=0.03)

    def llm(system: str, prompt: str) -> dict[str, Any]:
        if "reviewer" in system:  # the separate, independent voting call
            return {
                "votes": [
                    {
                        "lesson_id": existing,
                        "supports": True,
                        "evidence_id": evidence_id,
                        "why": "SMH rose 3% after the correction.",
                    },
                    {"lesson_id": "ls-hallucinated", "supports": True, "why": "n/a"},
                ]
            }
        assert "Historian" in system
        assert existing in prompt  # sees the lesson book
        return {
            "new_lessons": [
                {"statement": "Gold breaking down while equities hold flags risk-on rotation"},
                {"statement": "Crowded sector momentum unwinds fastest in the first 2 days"},
                {"statement": "A third lesson beyond the cap should be ignored entirely"},
            ],
            "retire": [{"lesson_id": existing, "why": "not established; must be ignored"}],
        }

    digest = run_historian(mem, llm=llm)
    assert digest["ok"] is True
    assert len(digest["created"]) == 2  # cap enforced
    assert digest["voted"] == 1  # hallucinated id ignored
    assert digest["retired"] == 0  # candidates cannot be retired
    rows = mem.lessons()
    assert len(rows) == 3
    assert all(r["status"] == "candidate" for r in rows)  # nothing auto-established


def test_garbage_statements_skipped_and_empty_week_ok(mem: MemoryStore) -> None:
    digest = run_historian(
        mem, llm=lambda s, p: {"new_lessons": [{"statement": "be careful"}], "votes": []}
    )
    assert digest["created"] == []  # too short -> garbage guard
    assert "no new lessons" in format_historian_digest(digest)


def test_promotion_needs_three_measured_outcomes(mem: MemoryStore) -> None:
    lid = mem.add_lesson("In low-IV uptrends, dip buys recover within five sessions")
    outcome_ids = []
    for i in range(3):
        pid = mem.add_prediction(
            agent="quant",
            subject=f"S{i}",
            direction="up",
            horizon_days=5,
            confidence=0.7,
            statement="recovery",
        )
        mem.grade_prediction(pid, realized_move=0.02)
        outcome_ids.append(pid)

    mem.add_evidence(lid, outcome_ids[0], supports=True, reason="first measured recovery")
    mem.add_evidence(lid, outcome_ids[1], supports=True, reason="second measured recovery")
    assert mem.lessons(status="candidate")[0]["id"] == lid
    mem.add_evidence(lid, outcome_ids[2], supports=True, reason="third measured recovery")
    assert mem.lessons(status="established")[0]["id"] == lid
    assert mem.lessons(status="established")[0]["support"] == 3


def test_tuesday_and_friday_cannot_double_count_one_outcome(mem: MemoryStore) -> None:
    """Twice-weekly review can interpret an outcome once, never manufacture two."""
    lid = mem.add_lesson("Semi clusters become one risk factor in a broad selloff")
    evidence_id = mem.add_prediction(
        agent="quant",
        subject="SMH",
        direction="down",
        horizon_days=5,
        confidence=0.7,
        statement="semi risk factor",
    )
    mem.grade_prediction(evidence_id, realized_move=-0.02)

    def llm(system: str, prompt: str) -> dict[str, Any]:
        if "reviewer" in system:
            return {"votes": [{"lesson_id": lid, "supports": True, "evidence_id": evidence_id}]}
        return {"new_lessons": [], "retire": []}

    tuesday = run_historian(mem, llm=llm, now=datetime(2026, 8, 11, 22, 45, tzinfo=timezone.utc))
    friday = run_historian(mem, llm=llm, now=datetime(2026, 8, 14, 22, 45, tzinfo=timezone.utc))

    assert tuesday["voted"] == 1
    assert friday["voted"] == 0
    assert mem.lessons()[0]["support"] == 1


class TestWeekEvidence:
    """The historian's charter promised it a week. Before 2026-07-30 it
    got `journal_tail(80)` — the newest 80 rows of anything, which in a
    busy week is three days and in a quiet one is a fortnight."""

    def test_the_window_is_days_not_a_row_count(self, mem: MemoryStore) -> None:
        from trading.agents.historian import build_week_evidence

        for i in range(200):
            mem.journal("cycle", {"i": i})
        mem.conn.execute("UPDATE journal SET ts = ts - ? WHERE id <= 150", (30 * 86400.0,))
        rows = build_week_evidence(mem)["week_journal"].get("cycle", [])
        assert all(r["payload"]["i"] >= 150 for r in rows), "rows from a month ago leaked in"

    def test_graded_outcomes_survive_a_noisy_week(self, mem: MemoryStore) -> None:
        """The failure the per-kind budget exists to prevent: outcomes are
        the only rows carrying measured truth, and a flat newest-N list
        lets chatter push every one of them out of view."""
        from trading.agents.historian import build_week_evidence

        for i in range(300):
            mem.journal("take", {"i": i})  # noisy, low-information
        mem.journal("prediction_graded", {"id": "pr-1", "outcome": "miss"})
        for i in range(300):
            mem.journal("take", {"i": 1000 + i})

        journal = build_week_evidence(mem)["week_journal"]
        assert len(journal.get("prediction_graded", [])) == 1
        assert len(journal.get("take", [])) <= 15  # budgeted, not unbounded

    def test_measured_edge_is_included(self, mem: MemoryStore) -> None:
        """The historian could not previously see the one table that
        measures anything."""
        from trading.agents.historian import build_week_evidence

        assert "measured_edge" in build_week_evidence(mem)

    def test_historian_gets_retired_lesson_and_long_horizon_scorecard(
        self, mem: MemoryStore
    ) -> None:
        active = mem.add_lesson("Semiconductor breadth confirms breakouts", tags="semis breadth")
        retired = mem.add_lesson(
            "Semiconductor breakouts failed in thin breadth", tags="semis breadth"
        )
        mem.retire_lesson(retired, "repeated thin-breadth failures")
        pid = mem.add_prediction(
            agent="quant",
            subject="SMH",
            direction="up",
            horizon_days=5,
            confidence=0.7,
            statement="semiconductor breakout",
        )
        mem.grade_prediction(pid, realized_move=0.03)
        seen: dict[str, str] = {}

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                return {"votes": []}
            seen["historian"] = prompt
            return {"new_lessons": [], "retire": []}

        run_historian(mem, llm=llm)

        assert active in seen["historian"]
        assert retired in seen["historian"]
        assert pid in seen["historian"]
        assert "historical_dossier" in seen["historian"]

    def test_protected_memory_is_never_silently_dropped(self) -> None:
        from trading.agents.historian import _budgeted_evidence

        with pytest.raises(ValueError, match="protected memory"):
            _budgeted_evidence(
                {"lesson_book": [{"statement": "x" * 500}], "historical_dossier": {}}, budget=100
            )


class TestVoterIndependence:
    """The +3-net-support promotion mechanic is only a chaos filter if the
    votes approximate independent evidence. Asked in one breath to propose
    lessons and judge them, a model leans towards confirming its own."""

    def test_the_voter_never_sees_lesson_ids_or_support_counts(self, mem: MemoryStore) -> None:
        lid = mem.add_lesson("In low-IV uptrends, dip buys recover within five sessions")
        mem.add_evidence(lid, "wk-1", supports=True)
        seen: dict[str, str] = {}

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                seen["voter"] = prompt
                return {"votes": []}
            return {"new_lessons": [], "retire": []}

        run_historian(mem, llm=llm)
        assert lid not in seen["voter"], "voter can see which lesson is which"
        assert "support" not in seen["voter"], "voter can see which way the desk leans"

    def test_lessons_created_this_run_cannot_vote_for_themselves(self, mem: MemoryStore) -> None:
        """A candidate written ten seconds ago has no week of evidence
        behind it; letting it collect support immediately would start it a
        third of the way to promotion for free."""

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                # Vote yes on every claim the reviewer was shown.
                import json as _json

                claims = _json.loads(prompt).get("claims", [])
                return {"votes": [{"lesson_id": c["n"], "supports": True} for c in claims]}
            return {
                "new_lessons": [
                    {"statement": "Semis lead the tape out of momentum drawdowns by two days"}
                ],
                "retire": [],
            }

        digest = run_historian(mem, llm=llm)
        assert len(digest["created"]) == 1
        assert digest["voted"] == 0  # nothing else in the book to vote on
        assert mem.lessons()[0]["support"] == 0

    def test_one_review_pass_allows_one_outcome_per_lesson(self, mem: MemoryStore) -> None:
        lid = mem.add_lesson("Semiconductor breadth confirms breakouts")
        outcomes = []
        for i in range(2):
            pid = mem.add_prediction(
                agent="quant",
                subject=f"S{i}",
                direction="up",
                horizon_days=5,
                confidence=0.6,
                statement="breakout",
            )
            mem.grade_prediction(pid, realized_move=0.02)
            outcomes.append(pid)

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                return {
                    "votes": [
                        {"lesson_id": lid, "supports": True, "evidence_id": outcomes[0]},
                        {"lesson_id": lid, "supports": True, "evidence_id": outcomes[1]},
                    ]
                }
            return {"new_lessons": [], "retire": []}

        digest = run_historian(mem, llm=llm)
        assert digest["voted"] == 1
        assert mem.lessons()[0]["support"] == 1

    def test_old_dossier_outcome_cannot_count_as_this_weeks_evidence(
        self, mem: MemoryStore
    ) -> None:
        """Historical context informs a review but cannot manufacture support."""
        lid = mem.add_lesson("Semiconductor breadth confirms breakouts")
        old = mem.add_prediction(
            agent="quant",
            subject="SMH",
            direction="up",
            horizon_days=5,
            confidence=0.6,
            statement="old breakout outcome",
        )
        mem.grade_prediction(old, realized_move=0.02)
        old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=30)).timestamp()
        mem.conn.execute("UPDATE predictions SET graded_ts = ? WHERE id = ?", (old_ts, old))
        mem.conn.execute("UPDATE journal SET ts = ? WHERE kind = 'prediction_graded'", (old_ts,))

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                return {"votes": [{"lesson_id": lid, "supports": True, "evidence_id": old}]}
            return {"new_lessons": [], "retire": []}

        digest = run_historian(mem, llm=llm)
        assert (
            mem.historian_dossier(focus_statements=[])["scorecard"]["recent_outcomes"][0]["id"]
            == old
        )
        assert digest["voted"] == 0
        assert mem.lessons()[0]["support"] == 0

    def test_a_failed_vote_leaves_the_lesson_book_untouched(self, mem: MemoryStore) -> None:
        lid = mem.add_lesson("Crowded sector momentum unwinds fastest in the first two days")

        def llm(system: str, prompt: str) -> dict[str, Any]:
            if "reviewer" in system:
                raise RuntimeError("LLM API 500")
            return {"new_lessons": [], "retire": []}

        digest = run_historian(mem, llm=llm)
        assert digest["ok"] is True  # distillation still succeeded
        assert digest["voted"] == 0
        assert mem.lessons()[0]["id"] == lid
        assert mem.lessons()[0]["support"] == 0


def test_llm_failure_reported(mem: MemoryStore) -> None:
    def boom(s: str, p: str) -> dict[str, Any]:
        raise RuntimeError("LLM API 500")

    digest = run_historian(mem, llm=boom)
    assert digest["ok"] is False
    assert "skipped" in format_historian_digest(digest)


def test_stock_universe_clamp() -> None:
    from trading.agents.pm import _clamp_weights

    w = _clamp_weights({"NVDA": 0.3, "SMH": 0.3, "FAKE": 0.2}, stocks=("NVDA", "AAPL"))
    assert w["NVDA"] == 0.10  # single-stock cap
    assert w["SMH"] == 0.25  # ETF cap
    assert "FAKE" not in w
    # NVDA + SMH are both tech_complex: 0.35 combined > 0.40? No — under
    # the cluster cap, so untouched.
    assert abs(sum(w.values()) - 0.35) < 1e-9


def test_sector_cluster_cap() -> None:
    from trading.agents.pm import MAX_CLUSTER, _clamp_weights

    # Six correlated tech names summing to 0.71 must be scaled to 0.40;
    # the uncorrelated names are untouched; freed weight goes to cash.
    raw = {"SMH": 0.25, "XLK": 0.2, "MU": 0.1, "NVDA": 0.08, "INTC": 0.08, "GLD": 0.1}
    w = _clamp_weights(raw, stocks=("MU", "NVDA", "INTC"))
    tech = w["SMH"] + w["XLK"] + w["MU"] + w["NVDA"] + w["INTC"]
    assert abs(tech - MAX_CLUSTER) < 0.01
    assert w["GLD"] == 0.1  # not in the offending cluster
    # proportional scaling preserves the PM's relative preferences
    assert w["SMH"] > w["MU"] > 0
