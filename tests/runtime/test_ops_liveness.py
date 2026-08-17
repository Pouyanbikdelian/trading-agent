"""Outcome-based watchdog checks.

On 2026-08-06 three loops — the nightly prediction grader, the shadow
ledger and the weekly historian — were found to have been running and
achieving nothing for weeks. Every existing ops check was green
throughout, because they all asked whether a FILE was fresh. A file gets
touched whether or not the work inside it succeeded.

These tests pin checks that ask a different question: did the loop
produce anything?
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.runtime.ops_watch import check_learning_loops, check_recent_errors

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def memdb(tmp_path: Path) -> Path:
    (tmp_path / "memory").mkdir()
    conn = sqlite3.connect(tmp_path / "memory" / "memory.db")
    conn.executescript(
        """CREATE TABLE journal (id INTEGER PRIMARY KEY, ts REAL, kind TEXT,
                                 actor TEXT, payload TEXT);
           CREATE TABLE predictions (id TEXT PRIMARY KEY, ts REAL, due_ts REAL,
                                     graded_ts REAL);
           CREATE TABLE shadow (id TEXT PRIMARY KEY, ts REAL, ret_21d REAL);"""
    )
    conn.commit()
    conn.close()
    return tmp_path


def _journal(state_dir: Path, kind: str, *, days_ago: float, payload: str = "{}") -> None:
    _journal_at(state_dir, kind, at=NOW - timedelta(days=days_ago), payload=payload)


def _journal_at(state_dir: Path, kind: str, *, at: datetime, payload: str = "{}") -> None:
    conn = sqlite3.connect(state_dir / "memory" / "memory.db")
    conn.execute(
        "INSERT INTO journal (ts, kind, actor, payload) VALUES (?, ?, 'x', ?)",
        (at.timestamp(), kind, payload),
    )
    conn.commit()
    conn.close()


class TestLoopLiveness:
    def test_all_loops_recent_is_silent(self, memdb: Path) -> None:
        for kind, age in (("committee", 1), ("agent_pm", 2), ("historian", 3), ("daily", 0.5)):
            _journal(memdb, kind, days_ago=age)
        assert check_learning_loops(memdb, now=NOW) == []

    def test_a_loop_that_never_ran_is_named(self, memdb: Path) -> None:
        """The historian was scheduled inside `if _guards_enabled():`, a
        flag defaulting to false — so on some boxes it had never run at
        all, and nothing said so."""
        for kind in ("committee", "agent_pm", "daily"):
            _journal(memdb, kind, days_ago=1)
        issues = check_learning_loops(memdb, now=NOW)
        assert any("historian" in i and "NEVER" in i for i in issues)

    def test_a_stalled_loop_is_named_with_its_age(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        _journal(memdb, "daily", days_ago=9)
        issues = check_learning_loops(memdb, now=NOW)
        assert any("nightly memory pass" in i and "9.0d ago" in i for i in issues)

    def test_twice_weekly_historian_alerts_after_five_days(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "daily"):
            _journal(memdb, kind, days_ago=1)
        _journal(memdb, "historian", days_ago=5.1)

        issues = check_learning_loops(memdb, now=NOW)

        assert any("historian distillation" in i and "5.1d ago" in i for i in issues)

    def test_missing_database_is_not_an_issue(self, tmp_path: Path) -> None:
        """Before the first run there is nothing to complain about."""
        assert check_learning_loops(tmp_path, now=NOW) == []


class TestScorecardHealth:
    def _preds(self, state_dir: Path, n: int, *, graded: bool, due_days_ago: float) -> None:
        conn = sqlite3.connect(state_dir / "memory" / "memory.db")
        for i in range(n):
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                (
                    f"pr-{graded}-{i}",
                    (NOW - timedelta(days=due_days_ago + 14)).timestamp(),
                    (NOW - timedelta(days=due_days_ago)).timestamp(),
                    NOW.timestamp() if graded else None,
                ),
            )
        conn.commit()
        conn.close()

    def test_overdue_ungraded_predictions_alert(self, memdb: Path) -> None:
        """The exact signature of the tz bug: they pile up forever while
        the grader logs a cheerful nothing every night."""
        for kind in ("committee", "agent_pm", "historian", "daily"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 8, graded=False, due_days_ago=5)
        issues = check_learning_loops(memdb, now=NOW)
        assert any("overdue and ungraded" in i for i in issues)

    def test_expected_next_session_rows_are_exempt_from_overdue_alarm(self, memdb: Path) -> None:
        """A Friday/Sunday expiry can legitimately remain ungraded until
        Monday's bar arrives. Only exact IDs from a fresh daily pass receive
        that exemption; a stale payload or a subject-level match is unsafe."""
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 8, graded=False, due_days_ago=5)
        _journal(
            memdb,
            "daily",
            days_ago=0,
            payload=json.dumps(
                {"awaiting_next_daily_bar_prediction_ids": [f"pr-False-{i}" for i in range(8)]}
            ),
        )

        issues = check_learning_loops(memdb, now=NOW)

        assert not any("overdue and ungraded" in issue for issue in issues)

    def test_only_exact_pending_ids_are_exempt_from_overdue_alarm(self, memdb: Path) -> None:
        """An unrelated old prediction must not hide behind another
        subject's normal next-session wait."""
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 10, graded=False, due_days_ago=5)
        _journal(
            memdb,
            "daily",
            days_ago=0,
            payload=json.dumps(
                {"awaiting_next_daily_bar_prediction_ids": [f"pr-False-{i}" for i in range(5)]}
            ),
        )

        issues = check_learning_loops(memdb, now=NOW)

        assert any("5 prediction(s) overdue and ungraded" in issue for issue in issues)

    def test_calendar_failure_fails_closed_to_an_overdue_alert(
        self, memdb: Path, monkeypatch
    ) -> None:
        """Health checks must not die or suppress an alert if calendar data breaks."""
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 8, graded=False, due_days_ago=5)
        _journal(
            memdb,
            "daily",
            days_ago=0,
            payload=json.dumps(
                {"awaiting_next_daily_bar_prediction_ids": [f"pr-False-{i}" for i in range(8)]}
            ),
        )

        def unavailable_calendar(*_args, **_kwargs) -> bool:
            raise RuntimeError("calendar unavailable")

        monkeypatch.setattr(
            "trading.runtime.portfolio_stats.nyse_session_settled_since", unavailable_calendar
        )
        issues = check_learning_loops(memdb, now=NOW)

        assert any("8 prediction(s) overdue and ungraded" in issue for issue in issues)

    def test_awaiting_ids_expire_after_the_next_session_settles(self, memdb: Path) -> None:
        """A Sunday journal must not mute a Monday cache/grader failure.

        The grader normally replaces the row on Monday night.  If it does
        not, exact IDs from Sunday are historical evidence only once the
        Monday's cache window and scheduled daily grader have both passed.
        """
        pre_grader = datetime(2026, 8, 17, 22, 44, tzinfo=timezone.utc)
        now = datetime(2026, 8, 17, 23, 1, tzinfo=timezone.utc)
        for kind in ("committee", "agent_pm", "historian"):
            _journal_at(memdb, kind, at=now - timedelta(days=1))
        ids = [f"pr-weekend-{i}" for i in range(8)]
        _journal_at(
            memdb,
            "daily",
            at=datetime(2026, 8, 16, 22, 30, tzinfo=timezone.utc),
            payload=json.dumps({"awaiting_next_daily_bar_prediction_ids": ids}),
        )
        conn = sqlite3.connect(memdb / "memory" / "memory.db")
        for prediction_id in ids:
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, NULL)",
                (
                    prediction_id,
                    datetime(2026, 7, 31, tzinfo=timezone.utc).timestamp(),
                    datetime(2026, 8, 14, 12, tzinfo=timezone.utc).timestamp(),
                ),
            )
        conn.commit()
        conn.close()

        before_grader = check_learning_loops(memdb, now=pre_grader)
        issues = check_learning_loops(memdb, now=now)

        assert not any("overdue and ungraded" in issue for issue in before_grader)
        assert any("8 prediction(s) overdue and ungraded" in issue for issue in issues)

    def test_awaiting_ids_survive_a_long_market_holiday(self, memdb: Path) -> None:
        """Wall-clock age is not evidence of failure while NYSE is shut.

        Labor Day follows this Friday, so the historical awaiting IDs are
        still correct after more than 48 hours. Tuesday's settled close is
        what finally makes them stale.
        """
        holiday = datetime(2026, 9, 7, 23, tzinfo=timezone.utc)
        ids = [f"pr-holiday-{i}" for i in range(8)]
        for kind in ("committee", "agent_pm", "historian"):
            _journal_at(memdb, kind, at=holiday - timedelta(days=1))
        _journal_at(
            memdb,
            "daily",
            at=datetime(2026, 9, 4, 22, 30, tzinfo=timezone.utc),
            payload=json.dumps({"awaiting_next_daily_bar_prediction_ids": ids}),
        )
        conn = sqlite3.connect(memdb / "memory" / "memory.db")
        for prediction_id in ids:
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, NULL)",
                (
                    prediction_id,
                    datetime(2026, 8, 21, tzinfo=timezone.utc).timestamp(),
                    datetime(2026, 9, 4, 12, tzinfo=timezone.utc).timestamp(),
                ),
            )
        conn.commit()
        conn.close()

        holiday_issues = check_learning_loops(memdb, now=holiday)
        settled_issues = check_learning_loops(
            memdb, now=datetime(2026, 9, 8, 23, 1, tzinfo=timezone.utc)
        )

        assert not any("overdue and ungraded" in issue for issue in holiday_issues)
        assert any("8 prediction(s) overdue and ungraded" in issue for issue in settled_issues)

    def test_many_predictions_none_ever_graded_alerts(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian", "daily"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 25, graded=False, due_days_ago=0.5)  # not yet overdue
        issues = check_learning_loops(memdb, now=NOW)
        assert any("ZERO ever graded" in i for i in issues)

    def test_a_healthy_scorecard_is_silent(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian", "daily"):
            _journal(memdb, kind, days_ago=1)
        self._preds(memdb, 30, graded=True, due_days_ago=5)
        assert check_learning_loops(memdb, now=NOW) == []

    def test_scorecard_blocker_names_missing_subjects(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        _journal(
            memdb,
            "daily",
            days_ago=0,
            payload=json.dumps({"unpriced_subjects": ["SMH"], "cache_behind_subjects": ["QQQ"]}),
        )

        issues = check_learning_loops(memdb, now=NOW)
        assert any(
            "missing prices: SMH" in issue and "cache behind: QQQ" in issue for issue in issues
        )

    def test_scorecard_waiting_for_tomorrows_bar_is_not_an_ops_alert(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        _journal(
            memdb,
            "daily",
            days_ago=0,
            payload=json.dumps({"awaiting_next_daily_bar_subjects": ["INTC", "SMH"]}),
        )

        issues = check_learning_loops(memdb, now=NOW)

        assert not any("scorecard" in issue for issue in issues)

    def test_clean_later_daily_pass_clears_an_old_blocker(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "historian"):
            _journal(memdb, kind, days_ago=1)
        _journal(
            memdb,
            "scorecard_blocked",
            days_ago=0,
            payload=json.dumps({"cache_behind_subjects": ["STALE"]}),
        )
        _journal(memdb, "daily", days_ago=0, payload=json.dumps({"graded_today": 5}))

        issues = check_learning_loops(memdb, now=NOW)

        assert not any("scorecard data" in issue for issue in issues)

    def test_latest_failed_historian_is_not_counted_as_healthy(self, memdb: Path) -> None:
        for kind in ("committee", "agent_pm", "daily"):
            _journal(memdb, kind, days_ago=1)
        conn = sqlite3.connect(memdb / "memory" / "memory.db")
        conn.execute(
            "INSERT INTO journal (ts, kind, actor, payload) VALUES (?, ?, ?, ?)",
            (
                NOW.timestamp(),
                "historian",
                "historian",
                json.dumps({"ok": False, "reason": "budget"}),
            ),
        )
        conn.commit()
        conn.close()

        issues = check_learning_loops(memdb, now=NOW)
        assert any("historian distillation failed: budget" in issue for issue in issues)


class TestErrorLogScan:
    def _log(self, tmp_path: Path, lines: list[str]) -> Path:
        d = tmp_path / "logs"
        d.mkdir(exist_ok=True)
        today = datetime.now(tz=timezone.utc)
        stamp = today.strftime("%Y-%m-%d %H:%M:%S")
        (d / f"trading.{today.date().isoformat()}.log").write_text(
            "\n".join(f"{stamp}.000 | {ln}" for ln in lines)
        )
        return d

    def test_errors_are_surfaced(self, tmp_path: Path) -> None:
        d = self._log(
            tmp_path,
            [
                "INFO     | trading.runner.runner:cycle:100 - all good",
                "ERROR    | trading.memory:grade:1200 - memory grader failed",
            ],
        )
        out = check_recent_errors(d)
        assert len(out) == 1 and "memory grader failed" in out[0]

    def test_repeats_are_grouped_not_repeated(self, tmp_path: Path) -> None:
        """A loop throwing two hundred times is one alert."""
        d = self._log(tmp_path, ["ERROR    | trading.x:y:1 - boom"] * 200)
        out = check_recent_errors(d)
        assert len(out) == 1 and "x200" in out[0]

    def test_clean_log_is_silent(self, tmp_path: Path) -> None:
        d = self._log(tmp_path, ["INFO     | trading.a:b:1 - fine"] * 5)
        assert check_recent_errors(d) == []

    def test_missing_log_is_silent(self, tmp_path: Path) -> None:
        assert check_recent_errors(tmp_path / "nope") == []
