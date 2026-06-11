"""Memory spine — hermetic tests over a temp vault."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading.memory import MemoryStore


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
