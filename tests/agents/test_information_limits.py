"""Limits that cut what one agent hands another (audit of 2026-09-26).

Each test pins one place where information used to be lost silently on
its way between agents: a news source that never survived the cap, topics
that never reached the prompt, a weekly reading judged stale every Friday,
a stale news file that vanished without a word.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.agents.context import STYLE_MAX_AGE_H, balanced_headlines, build_context
from trading.runtime import news_watch

NOW = datetime.now(tz=timezone.utc)


def test_every_topic_gets_a_turn_before_any_gets_a_second() -> None:
    items = [{"topic": "market", "title": f"m{i}"} for i in range(60)]
    items += [{"topic": "quantum", "title": "q1"}, {"topic": "ai_capex", "title": "a1"}]
    items += [{"topic": "reddit_stocks", "title": f"r{i}"} for i in range(5)]
    out = balanced_headlines(items, 10)
    topics = [h["topic"] for h in out]
    assert {"market", "quantum", "ai_capex", "reddit_stocks"} <= set(topics)
    assert len(out) == 10
    assert balanced_headlines(items, 1000) and len(balanced_headlines(items, 1000)) == len(items)


def test_reddit_is_no_longer_crowded_out_by_rss(tmp_path: Path, monkeypatch) -> None:
    """(rss + reddit)[:100] with 100+ RSS items dropped every Reddit post."""
    monkeypatch.setattr(
        news_watch,
        "fetch_headlines",
        lambda: [{"topic": f"t{i % 24}", "title": str(i)} for i in range(120)],
    )
    monkeypatch.setattr(
        news_watch,
        "fetch_reddit_signals",
        lambda: [
            {"topic": "reddit_stocks", "title": f"r{i}", "source": "reddit:u/x"} for i in range(12)
        ],
    )
    monkeypatch.setattr(news_watch, "fetch_sector_momentum", lambda: {})
    reading = news_watch.collect(tmp_path)
    assert len(reading["headlines"]) == 132
    assert sum(1 for h in reading["headlines"] if h["topic"] == "reddit_stocks") == 12


def _stamp(path: Path, age_h: float, **body) -> None:
    path.write_text(
        json.dumps({"last_polled_at": (NOW - timedelta(hours=age_h)).isoformat(), **body})
    )


def test_the_weekly_style_reading_counts_on_friday(tmp_path: Path) -> None:
    """style_advisor runs Sundays; 36h dropped it from every Friday committee."""
    _stamp(tmp_path / "style_advisor.json", age_h=5 * 24, leader="momentum")
    ctx = build_context(tmp_path, tmp_path, include_candidate_ladder=False)
    assert ctx["style_leader"] == "momentum"
    assert STYLE_MAX_AGE_H >= 7 * 24


def test_a_stale_news_file_is_named_not_silently_dropped(tmp_path: Path) -> None:
    (tmp_path / news_watch.STATE_FILENAME).write_text(
        json.dumps({"t": (NOW - timedelta(hours=72)).isoformat(), "headlines": [{"title": "x"}]})
    )
    ctx = build_context(tmp_path, tmp_path, include_candidate_ladder=False)
    assert "headlines" not in ctx
    assert any("news collection older than 36h" in g for g in ctx.get("_data_gaps", []))
