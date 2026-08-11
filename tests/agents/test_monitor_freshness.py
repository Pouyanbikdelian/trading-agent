"""The desk must not read a four-day-old VIX as today's VIX.

macro_monitor, options_monitor, advisor and style_advisor are the ONLY
view the PM has of regime and volatility, and all four are written by
scheduled jobs inside the runner. A stopped runner does not delete them
— it just stops changing them. ``_read_json`` returned the last reading
forever, and the PM reasoned over it as current.

That is exactly what happened 2026-08-07..08-11: the runner was stopped
for four days and no part of the system said the risk picture was stale.

Dropped rather than labelled: a stale VIX presented as today's VIX is
worse than no VIX, because the desk cannot tell that it is blind.
``_data_gaps`` is how it is told.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.agents.context import (
    MONITOR_MAX_AGE_H,
    _monitor_max_age_h,
    _read_fresh_json,
)

NOW = datetime.now(tz=timezone.utc)


def _write(path: Path, *, age_h: float, key: str = "last_polled_at", **body) -> Path:
    stamp = (NOW - timedelta(hours=age_h)).isoformat()
    path.write_text(json.dumps({key: stamp, **body}))
    return path


class TestFreshReadingsPassThrough:
    def test_a_recent_reading_is_returned(self, tmp_path) -> None:
        gaps: list[str] = []
        p = _write(tmp_path / "m.json", age_h=2, readings={"vix": 18.0})

        out = _read_fresh_json(p, "macro_dial", gaps=gaps, max_age_h=36.0)

        assert out["readings"] == {"vix": 18.0}
        assert gaps == []

    @pytest.mark.parametrize("key", ["last_polled_at", "asof", "t"])
    def test_every_timestamp_key_the_monitors_write_is_understood(self, tmp_path, key) -> None:
        """macro/options write 'asof', advisor/style write 'last_polled_at',
        news writes 't'. Missing one silently blinds that monitor."""
        gaps: list[str] = []
        p = _write(tmp_path / f"{key}.json", age_h=1, key=key, metrics={"iv": 0.2})

        assert _read_fresh_json(p, "vol_surface", gaps=gaps, max_age_h=36.0)["metrics"]
        assert gaps == []


class TestStaleReadingsAreDropped:
    def test_the_four_day_outage_is_caught(self, tmp_path) -> None:
        """The regression, in the exact shape it occurred."""
        gaps: list[str] = []
        p = _write(tmp_path / "advisor.json", age_h=96, active=[{"name": "slow_grind"}])

        out = _read_fresh_json(p, "spy_vix_triggers", gaps=gaps, max_age_h=36.0)

        assert out == {}
        assert len(gaps) == 1
        assert "96h old" in gaps[0] and "DROPPED" in gaps[0]

    def test_just_inside_the_window_survives(self, tmp_path) -> None:
        gaps: list[str] = []
        p = _write(tmp_path / "m.json", age_h=35, readings={"vix": 20.0})

        assert _read_fresh_json(p, "macro_dial", gaps=gaps, max_age_h=36.0) != {}

    def test_just_outside_the_window_is_dropped(self, tmp_path) -> None:
        gaps: list[str] = []
        p = _write(tmp_path / "m.json", age_h=37, readings={"vix": 20.0})

        assert _read_fresh_json(p, "macro_dial", gaps=gaps, max_age_h=36.0) == {}


class TestAbsentIsNotZero:
    def test_a_missing_file_is_reported_as_a_gap(self, tmp_path) -> None:
        gaps: list[str] = []

        out = _read_fresh_json(tmp_path / "nope.json", "macro_dial", gaps=gaps, max_age_h=36.0)

        assert out == {}
        assert "no reading on disk" in gaps[0]

    def test_an_undated_reading_is_refused(self, tmp_path) -> None:
        """Cannot be verified, so it cannot be trusted."""
        gaps: list[str] = []
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"readings": {"vix": 18.0}}))

        assert _read_fresh_json(p, "macro_dial", gaps=gaps, max_age_h=36.0) == {}
        assert "no timestamp" in gaps[0]

    def test_a_corrupt_timestamp_is_refused(self, tmp_path) -> None:
        gaps: list[str] = []
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"last_polled_at": "not-a-date", "readings": {"vix": 1}}))

        assert _read_fresh_json(p, "macro_dial", gaps=gaps, max_age_h=36.0) == {}
        assert "unreadable timestamp" in gaps[0]


class TestTheThresholdIsResolvedAtCallTime:
    def test_the_default_matches_news_watch(self) -> None:
        assert _monitor_max_age_h() == MONITOR_MAX_AGE_H == 36.0

    def test_the_env_override_is_read_live(self, monkeypatch) -> None:
        """Frozen-at-import is the trap that bit pm_signal."""
        monkeypatch.setenv("MONITOR_MAX_AGE_H", "6")

        assert _monitor_max_age_h() == 6.0

    def test_a_nonsense_value_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("MONITOR_MAX_AGE_H", "banana")

        assert _monitor_max_age_h() == MONITOR_MAX_AGE_H

    def test_zero_cannot_blind_the_desk(self, monkeypatch) -> None:
        """0 would drop every reading on every run, silently."""
        monkeypatch.setenv("MONITOR_MAX_AGE_H", "0")

        assert _monitor_max_age_h() == MONITOR_MAX_AGE_H


class TestTheDeskIsToldItIsBlind:
    def test_build_context_reports_gaps(self, tmp_path) -> None:
        """The whole point: the PM must be able to see that it cannot see."""
        from trading.agents.context import build_context

        _write(tmp_path / "macro_monitor.json", age_h=200, readings={"vix": 18.0})

        ctx = build_context(state_dir=tmp_path, data_dir=tmp_path)

        assert any("macro_dial" in g for g in ctx.get("_data_gaps", []))
        assert ctx["macro_dial"] == {}
