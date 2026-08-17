"""Memory spine — hermetic tests over a temp vault."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trading.memory import MemoryStore
from trading.memory.store import lesson_condition_fingerprint


@pytest.fixture
def mem(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def test_journal_is_append_only_and_readable(mem: MemoryStore) -> None:
    mem.journal("cycle", {"status": "ok", "orders": 8})
    mem.journal("halt", {"reason": "daily loss"}, actor="risk_manager")
    tail = mem.journal_tail(5)
    assert len(tail) == 2
    assert tail[0]["kind"] == "halt" and tail[0]["actor"] == "risk_manager"
    assert tail[1]["payload"]["orders"] == 8
    only_halts = mem.journal_tail(5, kind="halt")
    assert len(only_halts) == 1


def test_episode_roundtrip_with_entry_percentile(mem: MemoryStore) -> None:
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    eid = mem.add_episode(
        symbol="mu",
        ts_open=t0,
        ts_close=t0 + timedelta(days=30),
        entry_px=935.0,
        exit_px=1010.0,
        pnl_pct=0.08,
        entry_pctile_52w=0.91,  # bought near the top
        context={"regime": "bull", "vix": 14.2},
        tags="momentum semis",
    )
    rows = mem.episodes_for("MU")
    assert len(rows) == 1
    assert rows[0]["id"] == eid
    assert rows[0]["entry_pctile_52w"] == pytest.approx(0.91)


def test_lesson_lifecycle_promote_and_retire(mem: MemoryStore) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eps = [
        mem.add_episode(
            symbol=f"S{i}",
            ts_open=t0,
            ts_close=t0 + timedelta(days=10),
            entry_px=100,
            exit_px=90,
            pnl_pct=-0.10,
            entry_pctile_52w=0.95,
        )
        for i in range(4)
    ]
    lid = mem.add_lesson(
        "Buying >90th pctile of 52w range in a rates shock is -EV",
        origin_episodes=[eps[0]],
        tags="entry-timing",
    )
    assert mem.lessons(status="candidate")[0]["id"] == lid

    for e in eps[1:]:
        mem.add_evidence(lid, e, supports=True)
    assert mem.lessons(status="established")[0]["id"] == lid

    # Card exists, is Obsidian-flavored markdown, and survives retirement.
    card = (mem.lessons_dir / f"{lid}.md").read_text()
    assert "status: established" in card and "[[" in card

    mem.retire_lesson(lid, "regime changed; contradicted 2024-2026")
    card = (mem.lessons_dir / f"{lid}.md").read_text()
    assert "Retired" in card and "regime changed" in card
    # Never deleted:
    assert mem.lessons(status="retired")[0]["id"] == lid


def test_conditioned_retrieval_mixes_matches_with_counterweights(mem: MemoryStore) -> None:
    """Today's similarity must not hide durable lessons from other regimes."""
    current = {
        "macro_bucket": "stress",
        "vol_bucket": "elevated",
        "triggers": ["vix_spike"],
    }
    matching = mem.add_lesson(
        "Stress-vol selloffs need breadth confirmation before adding risk.",
        status="established",
        conditions={"snapshot": current},
    )
    partial = mem.add_lesson(
        "Stress macro still needs low-vol confirmation before exposure rises.",
        status="established",
        conditions={"snapshot": {"macro_bucket": "stress", "vol_bucket": "normal"}},
    )
    broad = mem.add_lesson(
        "Never let a single sector become the whole risk budget.", status="established"
    )
    second_broad = mem.add_lesson(
        "Keep a cash buffer even when the tape feels straightforward.", status="established"
    )
    different = mem.add_lesson(
        "Easing-regime rallies can tolerate wider cyclical exposure.",
        status="established",
        conditions={"snapshot": {"macro_bucket": "easing", "vol_bucket": "low"}},
    )

    selected = mem.retrieve_lessons(current, max_relevant=3, max_diversifiers=2)
    by_id = {row["id"]: row for row in selected}

    assert by_id[matching]["retrieval_role"] == "regime_match"
    assert by_id[partial]["retrieval_role"] == "partial_match"
    assert by_id[partial]["different_conditions"] == ["vol_bucket"]
    assert by_id[broad]["retrieval_role"] == "broad_prior"
    assert by_id[second_broad]["retrieval_role"] == "broad_prior"
    assert by_id[different]["retrieval_role"] == "diversifier"
    assert len(selected) <= 5

    legacy_mem = MemoryStore(mem.root.parent / "legacy-memory")
    legacy_first = legacy_mem.add_lesson("First unsnapshotted prior.", status="established")
    legacy_second = legacy_mem.add_lesson("Second unsnapshotted prior.", status="established")
    legacy_only = legacy_mem.retrieve_lessons({}, max_relevant=0, max_diversifiers=2)
    assert {row["id"] for row in legacy_only if row["retrieval_role"] == "broad_prior"} == {
        legacy_first,
        legacy_second,
    }


def test_review_rotation_never_expires_a_quiet_regime_candidate(mem: MemoryStore) -> None:
    """Review timestamps allocate scarce attention; they are not validity dates."""
    lid = mem.add_lesson(
        "A quiet easing regime may still reward cyclicals after a long pause.",
        conditions={"snapshot": {"macro_bucket": "easing"}},
    )

    queued = mem.candidate_review_queue({"macro_bucket": "stress"}, limit=1)
    assert [row["id"] for row in queued] == [lid]
    mem.mark_lessons_reviewed([lid])

    row = mem.lessons(status="candidate")[0]
    assert row["id"] == lid
    assert row["last_reviewed_ts"] is not None


def test_lesson_condition_fingerprint_omits_unknowns_not_false_neutrality() -> None:
    assert lesson_condition_fingerprint({"macro_dial": {}, "vol_surface": {}}) == {}
    assert lesson_condition_fingerprint(
        {
            "macro_dial": {"composite": 2.1},
            "vol_surface": {"atm_iv": 0.31},
            "style_leader": "defensive",
            "spy_vix_triggers": [{"name": "vix_spike"}],
        }
    ) == {
        "macro_bucket": "stress",
        "vol_bucket": "elevated",
        "style_leader": "defensive",
        "triggers": ["vix_spike"],
    }


def test_repeated_evidence_does_not_count_twice(mem: MemoryStore) -> None:
    """A retry is not a second observation supporting the lesson."""
    lid = mem.add_lesson("Semi clusters become one risk factor in a broad selloff")

    assert mem.add_evidence(lid, "wk-2026-W33", supports=True) is True
    assert mem.add_evidence(lid, "wk-2026-W33", supports=True) is False

    row = mem.lessons()[0]
    assert row["support"] == 1


def test_weekly_review_cannot_promote_a_lesson(mem: MemoryStore) -> None:
    """A fluent weekly review is provenance, not a measured result."""
    lid = mem.add_lesson("Semis lead the tape only when breadth is improving")
    for week in ("wk-2026-W31", "wk-2026-W32", "wk-2026-W33"):
        assert mem.add_evidence(lid, week, supports=True, reason="historian review") is True

    assert mem.lessons(status="candidate")[0]["id"] == lid
    evidence = mem.lesson_evidence(lid)
    assert {row["kind"] for row in evidence} == {"review"}
    assert all(row["reason"] == "historian review" for row in evidence)


def test_contradictory_retry_for_one_outcome_is_ignored(mem: MemoryStore) -> None:
    lid = mem.add_lesson("Quality rebounds after breadth washes out")
    pid = mem.add_prediction(
        agent="quant",
        subject="SPY",
        direction="up",
        horizon_days=5,
        confidence=0.6,
        statement="washout rebound",
    )
    mem.grade_prediction(pid, realized_move=0.02)

    assert mem.add_evidence(lid, pid, supports=True, reason="first reading") is True
    assert mem.add_evidence(lid, pid, supports=False, reason="flipped retry") is False
    row = mem.lessons()[0]
    assert row["support"] == 1 and row["contradict"] == 0


def test_fabricated_or_ungraded_outcome_cannot_count_as_evidence(mem: MemoryStore) -> None:
    lid = mem.add_lesson("Breadth needs confirmation before a breakout")
    ungraded = mem.add_prediction(
        agent="quant",
        subject="SMH",
        direction="up",
        horizon_days=5,
        confidence=0.6,
        statement="breakout holds",
    )

    assert mem.add_evidence(lid, "pr-does-not-exist", supports=True) is False
    assert mem.add_evidence(lid, ungraded, supports=True) is False
    assert mem.lessons()[0]["support"] == 0


def test_measured_contradictions_challenge_an_established_lesson(mem: MemoryStore) -> None:
    lid = mem.add_lesson("A narrow semiconductor rally is safe to chase")
    supporting: list[str] = []
    contradicting: list[str] = []
    for i in range(6):
        pid = mem.add_prediction(
            agent="quant",
            subject=f"S{i}",
            direction="up",
            horizon_days=5,
            confidence=0.6,
            statement="semiconductor breadth",
        )
        mem.grade_prediction(pid, realized_move=0.02)
        (supporting if i < 3 else contradicting).append(pid)

    for pid in supporting:
        mem.add_evidence(lid, pid, supports=True, reason="supporting outcome")
    assert mem.lessons(status="established")[0]["id"] == lid
    for pid in contradicting:
        mem.add_evidence(lid, pid, supports=False, reason="contradicting outcome")

    assert mem.lessons(status="challenged")[0]["id"] == lid
    assert "status: challenged" in (mem.lessons_dir / f"{lid}.md").read_text()


def test_historian_dossier_retrieves_measured_history_and_related_retirements(
    mem: MemoryStore,
) -> None:
    active = mem.add_lesson("Semiconductor breadth must confirm a breakout", tags="semis breadth")
    retired = mem.add_lesson("Semiconductor breakouts failed in thin breadth", tags="semis breadth")
    mem.retire_lesson(retired, "three measured failures in thin breadth")
    pid = mem.add_prediction(
        agent="quant",
        subject="SMH",
        direction="up",
        horizon_days=5,
        confidence=0.7,
        statement="semiconductor breakout",
    )
    mem.grade_prediction(pid, realized_move=0.03)

    dossier = mem.historian_dossier(focus_statements=[mem.lessons()[0]["statement"]])
    assert dossier["scorecard"]["graded"] == 1
    assert dossier["scorecard"]["recent_outcomes"][0]["id"] == pid
    assert dossier["related_retired_lessons"][0]["id"] == retired
    assert active in {row["id"] for row in mem.lessons(status="candidate")}


def test_historian_dossier_separates_current_session_pending_predictions(
    mem: MemoryStore,
) -> None:
    """A valid weekend wait is not evidence that scorecard calibration failed."""
    prediction_id = mem.add_prediction(
        agent="quant",
        subject="SPY",
        direction="up",
        horizon_days=14,
        confidence=0.7,
        statement="weekend timing test",
    )
    made = datetime(2026, 8, 21, 12, tzinfo=timezone.utc)
    due = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
    mem.conn.execute(
        "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
        (made.timestamp(), due.timestamp(), prediction_id),
    )
    # Labor Day makes this a >48-hour wait, which must remain pending until
    # Tuesday's actual session has settled.
    journal_at = datetime(2026, 9, 4, 22, 30, tzinfo=timezone.utc)
    mem.conn.execute(
        "INSERT INTO journal (ts, kind, actor, payload) VALUES (?, 'daily', 'memory', ?)",
        (
            journal_at.timestamp(),
            json.dumps({"awaiting_next_daily_bar_prediction_ids": [prediction_id]}),
        ),
    )

    pending = mem.historian_dossier(
        focus_statements=[], asof=datetime(2026, 9, 7, 19, tzinfo=timezone.utc)
    )["scorecard"]
    expired = mem.historian_dossier(
        focus_statements=[], asof=datetime(2026, 9, 8, 23, 1, tzinfo=timezone.utc)
    )["scorecard"]

    assert pending["overdue_ungraded"] == 0
    assert pending["session_pending_ungraded"] == 1
    assert expired["overdue_ungraded"] == 1
    assert expired["session_pending_ungraded"] == 0


def test_scorecard_backfill_targets_group_blocked_subjects(mem: MemoryStore) -> None:
    first = mem.add_prediction(
        agent="quant",
        subject="SMH",
        direction="up",
        horizon_days=5,
        confidence=0.6,
        statement="semis recover",
    )
    second = mem.add_prediction(
        agent="scout",
        subject="SMH",
        direction="down",
        horizon_days=5,
        confidence=0.6,
        statement="semis weaken",
    )
    past = (datetime.now(tz=timezone.utc) - timedelta(days=20)).timestamp()
    mem.conn.execute(
        "UPDATE predictions SET ts = ?, due_ts = ? WHERE id IN (?, ?)",
        (past, past + 5 * 86400, first, second),
    )

    targets = mem.scorecard_backfill_targets()
    assert targets == [{"subject": "SMH", "earliest_ts": past, "n": 2}]


def test_dossier_appends_keep_history(mem: MemoryStore) -> None:
    p = mem.update_dossier(
        "fomc_chair",
        "Chair X leans dovish; first speech Friday.",
        expects="cut in September is 80% priced",
    )
    mem.update_dossier("fomc_chair", "Speech delivered: more hawkish than priced.")
    text = p.read_text()
    assert text.count("###") == 2  # both timestamped entries kept
    assert "Crowd expects" in text
    assert "fomc_chair" in mem.dossiers()


def test_prediction_grading_and_trust_flow(mem: MemoryStore) -> None:
    pid = mem.add_prediction(
        agent="narrator",
        subject="NDX",
        direction="down",
        horizon_days=5,
        confidence=0.7,
        statement="rates shock bites within a week",
        sources=["reuters", "fintwit_anon"],
    )
    # Not due yet relative to creation time? due = now + 5d, so nothing due now.
    assert mem.due_predictions() == []
    # Pretend the horizon passed:
    future = datetime.now(tz=timezone.utc) + timedelta(days=6)
    due = mem.due_predictions(asof=future)
    assert [r["id"] for r in due] == [pid]

    outcome = mem.grade_prediction(pid, realized_move=-0.03)
    assert outcome == "hit"
    # Double grading is a no-op:
    assert mem.grade_prediction(pid, realized_move=0.5) == "skipped"

    cal = mem.calibration()
    assert cal[0]["agent"] == "narrator" and cal[0]["hit_rate"] == pytest.approx(1.0)

    # Both cited sources got credit; unknown source stays neutral.
    assert mem.trust("reuters") > 0.5
    assert mem.trust("fintwit_anon") > 0.5
    assert mem.trust("never_seen") == pytest.approx(0.5)
    table = mem.trust_table()
    assert {t["source"] for t in table} == {"reuters", "fintwit_anon"}


def test_flat_band_and_miss_path(mem: MemoryStore) -> None:
    pid = mem.add_prediction(
        agent="trader",
        subject="AAPL",
        direction="up",
        horizon_days=1,
        confidence=0.9,
        statement="bounce",
        sources=["gossip_guy"],
    )
    assert mem.grade_prediction(pid, realized_move=0.001) == "miss"  # flat != up
    assert mem.trust("gossip_guy") < 0.5


def test_stats_counts(mem: MemoryStore) -> None:
    mem.journal("note", {"x": 1})
    mem.update_dossier("iran_conflict", "ceasefire talks stalled")
    s = mem.stats()
    assert s["journal"] >= 2  # note + dossier_update journal entry
    assert s["dossiers"] == 1
