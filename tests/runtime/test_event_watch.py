"""Event watch — hermetic: injected fetch, injected LLM, tmp state, fixed clock."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trading.runtime.event_watch import (
    cluster,
    format_event_alert,
    format_recent,
    run_event_watch,
    score_title,
    title_key,
)

# 15:00 Zurich on a weekday — outside the default 23-7 quiet hours.
DAY = datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
# 02:00 Zurich — inside quiet hours.
NIGHT = datetime(2026, 11, 21, 1, 0, tzinfo=timezone.utc)

STRIKE = [
    {
        "topic": "war",
        "title": "US airstrike hits Iranian oil terminal, Brent jumps",
        "source": "Reuters",
    },
    {
        "topic": "war",
        "title": "Brent jumps after US airstrike hits Iranian oil terminal",
        "source": "Bloomberg",
    },
    {
        "topic": "war",
        "title": "US airstrike on Iranian oil terminal sends Brent higher",
        "source": "AP",
    },
]
NOISE = [
    {"topic": "fed", "title": "Fed week ahead: what to know", "source": "Blog"},
    {"topic": "fed", "title": "3 stocks to buy before the Fed", "source": "Fool"},
]


def _llm_major(_s: str, prompt: str) -> dict[str, Any]:
    p = json.loads(prompt)
    assert p["candidates"], "LLM must only be called with candidates"
    return {
        "events": [
            {
                "ids": [0],
                "headline": "US strikes Iranian oil terminal",
                "category": "geopolitical",
                "severity": 5,
                "is_new_development": True,
                "why_it_matters": "Gulf supply risk; oil and defense reprice.",
                "market_read": "risk_off",
                "exposed": ["XLE"],
                "watch": "Brent, Hormuz shipping",
            }
        ]
    }


def _llm_sev(sev: int):
    def f(_s: str, _p: str) -> dict[str, Any]:
        return {
            "events": [
                {
                    "ids": [0],
                    "headline": "ECB surprises with a cut",
                    "category": "central_bank",
                    "severity": sev,
                    "is_new_development": True,
                    "why_it_matters": "EUR lower.",
                    "market_read": "mixed",
                }
            ]
        }

    return f


def _ecb() -> list[dict[str, str]]:
    return [{"topic": "global_cb", "title": "ECB surprise rate cut stuns markets", "source": "FT"}]


def _boom(_s: str, _p: str) -> dict[str, Any]:
    raise AssertionError("LLM must not be called")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "EVENT_WATCH_QUIET_HOURS",
        "EVENT_WATCH_MIN_SEVERITY",
        "EVENT_WATCH_MAX_ALERTS_PER_DAY",
        "EVENT_WATCH_MIN_SCORE",
    ):
        monkeypatch.delenv(k, raising=False)


def test_scoring_prefers_hard_events_over_noise() -> None:
    assert score_title("Fed delivers surprise emergency rate cut") >= 6
    assert score_title("Fed week ahead: what to know") < 3
    assert score_title("3 stocks to buy before the Fed") < 3


def test_cluster_counts_outlets_and_boosts() -> None:
    items = [{**h, "score": score_title(h["title"])} for h in STRIKE]
    cs = cluster(items)
    assert len(cs) == 1
    assert cs[0]["sources"] == {"Reuters", "Bloomberg", "AP"}
    assert cs[0]["score"] >= max(i["score"] for i in items) + 2


def test_noise_only_never_calls_llm(tmp_path: Path) -> None:
    r = run_event_watch(tmp_path, fetch=lambda: NOISE, llm=_boom, holdings=[], now=DAY)
    assert r["quiet"] is True


def test_major_event_sends_once_and_dedups(tmp_path: Path) -> None:
    r = run_event_watch(tmp_path, fetch=lambda: STRIKE, llm=_llm_major, holdings=["XLE"], now=DAY)
    assert r["quiet"] is False
    ev = r["send"][0]
    assert ev["severity"] == 5 and ev["review_suggested"] is True
    assert ev["n_outlets"] == 3
    # Same headlines next run: already seen -> no LLM, no message.
    r2 = run_event_watch(
        tmp_path, fetch=lambda: STRIKE, llm=_boom, holdings=[], now=DAY + timedelta(minutes=30)
    )
    assert r2["quiet"] is True
    st = json.loads((tmp_path / "event_watch.json").read_text())
    assert st["last_major"]["headline"].startswith("US strikes")


def test_below_threshold_not_sent(tmp_path: Path) -> None:
    fetch = _ecb
    r = run_event_watch(tmp_path, fetch=fetch, llm=_llm_sev(3), holdings=[], now=DAY)
    assert r["quiet"] is True


def test_not_new_development_suppressed(tmp_path: Path) -> None:
    def llm(_s: str, _p: str) -> dict[str, Any]:
        out = _llm_sev(4)(_s, _p)
        out["events"][0]["is_new_development"] = False
        return out

    fetch = _ecb
    assert run_event_watch(tmp_path, fetch=fetch, llm=llm, holdings=[], now=DAY)["quiet"] is True


def test_quiet_hours_hold_sev4_and_release_as_digest(tmp_path: Path) -> None:
    fetch = _ecb
    r = run_event_watch(tmp_path, fetch=fetch, llm=_llm_sev(4), holdings=[], now=NIGHT)
    assert r["quiet"] is True and r["held_for_digest"] == 1
    morning = NIGHT + timedelta(hours=7)  # 09:00 Zurich
    r2 = run_event_watch(tmp_path, fetch=lambda: [], llm=_boom, holdings=[], now=morning)
    assert r2["quiet"] is False and r2["digest"] is True
    assert "overnight digest" in format_event_alert(r2["send"], digest=True)


def test_quiet_hours_do_not_hold_major(tmp_path: Path) -> None:
    r = run_event_watch(tmp_path, fetch=lambda: STRIKE, llm=_llm_major, holdings=[], now=NIGHT)
    assert r["quiet"] is False


def test_daily_cap_spares_major(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVENT_WATCH_MAX_ALERTS_PER_DAY", "0")
    fetch = _ecb
    assert (
        run_event_watch(tmp_path, fetch=fetch, llm=_llm_sev(4), holdings=[], now=DAY)["quiet"]
        is True
    )
    r = run_event_watch(tmp_path, fetch=lambda: STRIKE, llm=_llm_major, holdings=[], now=DAY)
    assert r["quiet"] is False


def test_llm_failure_is_silent(tmp_path: Path) -> None:
    def bad(_s: str, _p: str) -> dict[str, Any]:
        raise RuntimeError("api down")

    assert (
        run_event_watch(tmp_path, fetch=lambda: STRIKE, llm=bad, holdings=[], now=DAY)["quiet"]
        is True
    )


def test_format_escapes_markdown_and_recent(tmp_path: Path) -> None:
    ev = {
        "severity": 4,
        "headline": "S&P_500 *plunges*",
        "category": "market",
        "why_it_matters": "x",
        "market_read": "risk_off",
        "exposed": ["BRK_B"],
        "watch": "",
        "outlets": ["Reuters"],
        "n_outlets": 1,
        "review_suggested": False,
        "t": datetime.now(tz=timezone.utc).isoformat(),
    }
    msg = format_event_alert([ev])
    assert "S&P\\_500 \\*plunges\\*" in msg and "BRK\\_B" in msg
    assert "Nothing" in format_recent(tmp_path)
    (tmp_path / "event_watch.json").write_text(json.dumps({"alerted": [ev]}))
    assert "Last 24h" in format_recent(tmp_path)


def test_title_key_is_order_insensitive() -> None:
    assert title_key("Iran strikes oil terminal") == title_key("Oil terminal: Iran strikes")
