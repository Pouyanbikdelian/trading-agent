"""A measured id is not necessarily a fresh independent validation sample."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.memory import MemoryStore

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr("trading.memory.store._now", lambda: BASE.timestamp())
    with_store = MemoryStore(tmp_path / "memory")
    yield with_store
    with_store.close()


def _prediction(mem, monkeypatch, day, *, symbol="SPY", horizon=5, agent="quant"):
    start = BASE + timedelta(days=day)
    monkeypatch.setattr("trading.memory.store._now", lambda: start.timestamp())
    pid = mem.add_prediction(
        agent=agent,
        subject=symbol,
        direction="up",
        horizon_days=horizon,
        confidence=0.7,
        statement="Five-session breadth recovery prediction",
    )
    end = start + timedelta(days=horizon)
    monkeypatch.setattr("trading.memory.store._now", lambda: end.timestamp())
    mem.grade_prediction(pid, realized_move=0.03)
    return pid


def _card(mem, lesson_id):
    return next(card for card in mem.lesson_review({})["queue"] if card["id"] == lesson_id)


def test_overlapping_agents_and_horizons_cannot_manufacture_promotion(mem, monkeypatch):
    lesson_id = mem.add_lesson("Breadth recoveries should persist for five sessions.")
    for day, horizon, agent in ((1, 5, "quant"), (2, 3, "macro"), (3, 10, "scout")):
        prediction = _prediction(mem, monkeypatch, day, symbol="spy", horizon=horizon, agent=agent)
        assert mem.add_evidence(lesson_id, prediction, supports=True)

    card = _card(mem, lesson_id)
    assert card["status"] == "candidate"
    assert card["outcome_support"] == 3  # all measurements are still auditable
    assert card["validation_support"] == 1
    assert card["validation_excluded"] == 2


def test_three_non_overlapping_prospective_windows_can_promote(mem, monkeypatch):
    lesson_id = mem.add_lesson("Breadth recoveries should persist for five sessions.")
    for day in (1, 7, 13):
        assert mem.add_evidence(lesson_id, _prediction(mem, monkeypatch, day), supports=True)
    card = _card(mem, lesson_id)
    assert card["status"] == "established"
    assert card["validation_support"] == 3


def test_old_observations_graded_after_proposal_are_still_discovery(mem, monkeypatch):
    predictions = [_prediction(mem, monkeypatch, day, symbol=f"S{day}") for day in (1, 2, 3)]
    mem.conn.execute("UPDATE predictions SET graded_ts = NULL")
    lesson_id = mem.add_lesson("A thesis cannot validate itself from its discovery sample.")
    for prediction in predictions:
        mem.grade_prediction(prediction, realized_move=0.03)
        assert mem.add_evidence(lesson_id, prediction, supports=True)
    card = _card(mem, lesson_id)
    assert card["outcome_support"] == 3
    assert card["validation_support"] == 0
    assert card["status"] == "candidate"


def test_new_prediction_that_overlaps_origin_cannot_validate_the_claim(mem, monkeypatch):
    origin = _prediction(mem, monkeypatch, 1, horizon=10)
    # A discovery source may cover a longer window than a later forecast.
    monkeypatch.setattr("trading.memory.store._now", lambda: (BASE + timedelta(days=5)).timestamp())
    lesson_id = mem.add_lesson(
        "Breadth recovery applies outside its discovery period.", origin_episodes=[origin]
    )
    overlapping = _prediction(mem, monkeypatch, 6, horizon=10)
    assert mem.add_evidence(lesson_id, overlapping, supports=True)
    assert _card(mem, lesson_id)["validation_support"] == 0


def test_predictions_and_episodes_share_the_same_overlap_boundary(mem, monkeypatch):
    lesson_id = mem.add_lesson("A forecast and the trade it describes are one sample.")
    prediction = _prediction(mem, monkeypatch, 1)
    episode = mem.add_episode(
        symbol="SPY",
        ts_open=BASE + timedelta(days=2),
        ts_close=BASE + timedelta(days=7),
        entry_px=100,
        exit_px=103,
        pnl_pct=0.03,
        entry_pctile_52w=None,
    )
    assert mem.add_evidence(lesson_id, prediction, supports=True)
    assert mem.add_evidence(lesson_id, episode, supports=True)
    card = _card(mem, lesson_id)
    assert card["outcome_support"] == 2 and card["validation_support"] == 1


def test_overlapping_contradiction_cannot_be_drowned_out_by_support(mem, monkeypatch):
    lesson_id = mem.add_lesson("Review disagreement is not three supporting market events.")
    for symbol in ("SPY", "QQQ"):
        assert mem.add_evidence(
            lesson_id, _prediction(mem, monkeypatch, 1, symbol=symbol), supports=True
        )
    assert mem.add_evidence(
        lesson_id, _prediction(mem, monkeypatch, 2, symbol="SPY"), supports=False
    )
    card = _card(mem, lesson_id)
    assert card["validation_support"] == 1
    assert card["validation_contradict"] == 1


def test_restoration_requires_observations_started_after_restoration(mem, monkeypatch):
    lesson_id = mem.add_lesson("A restored belief must earn a new prospective sample.")
    old_ids = [_prediction(mem, monkeypatch, day, symbol=f"S{day}") for day in (1, 2, 3)]
    mem.retire_lesson(lesson_id, "operator archived an unproven hypothesis", actor="operator")
    assert mem.restore_retired_lesson(lesson_id, "new evidence may test it", actor="operator")
    for prediction in old_ids:
        mem.add_evidence(lesson_id, prediction, supports=True)
    assert _card(mem, lesson_id)["validation_support"] == 0
    for day in (10, 16, 22):
        mem.add_evidence(lesson_id, _prediction(mem, monkeypatch, day), supports=True)
    assert _card(mem, lesson_id)["status"] == "established"


@pytest.mark.parametrize(
    "status,capacity", [("candidate", 12), ("established", 5), ("challenged", 3)]
)
def test_reserved_oldest_slots_cover_other_regimes_without_changing_status(mem, status, capacity):
    quiet_ids = []
    for index in range(capacity + 3):
        lesson_id = mem.add_lesson(
            f"Quiet-regime claim {index} remains eligible for review.",
            status="established" if status == "challenged" else status,
            conditions={"snapshot": {"macro_bucket": "easing"}},
        )
        if status == "challenged":
            mem.set_lesson_status(lesson_id, "challenged")
        quiet_ids.append(lesson_id)
    for index in range(capacity):
        lesson_id = mem.add_lesson(
            f"Current-regime claim {index} wins the relevance ranking.",
            status="established" if status == "challenged" else status,
            conditions={"snapshot": {"macro_bucket": "stress"}},
        )
        if status == "challenged":
            mem.set_lesson_status(lesson_id, "challenged")
        mem.conn.execute(
            "UPDATE lessons SET last_reviewed_ts = ? WHERE id = ?", (BASE.timestamp(), lesson_id)
        )

    seen = set()
    for _ in range(capacity + 3):
        queue = [
            card
            for card in mem.lesson_review({"macro_bucket": "stress"})["queue"]
            if card["status"] == status
        ]
        assert len(queue) <= capacity
        assert len({card["id"] for card in queue}) == len(queue)
        seen.update(card["id"] for card in queue)
        mem.mark_lessons_reviewed([card["id"] for card in queue])
    assert set(quiet_ids) <= seen
    assert len(mem.lessons(status=status)) == 2 * capacity + 3


def test_zero_capacity_review_is_empty(mem):
    mem.add_lesson("A bounded review with zero capacity must remain empty.")
    assert mem.candidate_review_queue({}, limit=0) == []
