"""The dashboard must not tell a live operator they are on paper.

Three strings on the page were hardcoded and, on 2026-08-11, all three
were wrong on a live account:

* "Equity (paper)" and "traded book (paper)" — over a real CHF book.
* "rebalance (paper) 21:05" — the runner's cron had moved to 19:00, and
  the PM slot said Mon 14:30 when it actually fires 45 min before the
  cycle.
* "broker snapshot 90h ago" on a runner writing a snapshot every 60s —
  ``_age`` stat'd ``runner.db``, but in WAL mode SQLite writes to
  ``runner.db-wal`` and only touches the main file on checkpoint.

Same defect three times: the display describing a different system than
the one running. On a page an operator reads before deciding whether to
resume live trading, that is not cosmetic.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

APP = Path("src/trading/dashboard/app.py").read_text()
#: Since the 2026-09-23 redesign the page lives in its own file.
PAGE = Path("src/trading/dashboard/static/index.html").read_text()


class TestNoHardcodedPaperLabels:
    def test_no_label_says_paper_in_plain_text(self) -> None:
        assert "(paper)" not in PAGE
        assert "Equity (paper)" not in PAGE

    def test_the_equity_card_label_is_filled_from_the_environment(self) -> None:
        assert 'Equity (<span class="envlbl">paper</span>)' in PAGE
        assert 'Account NetLiq <span class="envlbl">paper</span>' in PAGE

    def test_the_strategy_race_series_follows_the_environment(self) -> None:
        """The race's account series names its environment AND its book.

        It was called "momentum top-k": at STRATEGY_SLEEVE_PCT=0.0 the
        momentum book cannot place an order, so the line is the whole
        account's NetLiq including the operator's own positions. See
        tests/dashboard/test_race_chart.py.
        """
        assert "'momentum top-k (paper)'" not in PAGE
        assert "'account NetLiq ('+(D.env||'paper')+' · '+ccy+')'" in PAGE

    def test_every_env_label_is_filled_from_the_payload(self) -> None:
        assert "document.querySelectorAll('.envlbl')" in PAGE
        assert "e.textContent=D.env||'paper'" in PAGE

    def test_live_is_visually_flagged(self) -> None:
        """Reading 'live' should not require looking for it."""
        assert "env==='live'" in PAGE

    def test_the_server_supplies_the_environment(self) -> None:
        assert 'out["env"] = getattr(_s, "trading_env", "") or ""' in APP


class TestScheduleComesFromTheRealCron:
    """The page used to parse the cron in JavaScript, as UTC, and hardcode
    six other job times; each was wrong the day the schedule moved. The
    server now computes every next fire time with APScheduler's own
    triggers (dashboard/cockpit.schedule_block)."""

    def test_the_page_no_longer_parses_cron(self) -> None:
        assert "parseCron" not in PAGE
        assert "D.schedule" in PAGE

    def test_the_server_supplies_the_cron_and_its_timezone(self) -> None:
        assert 'out["cycle_cron"] = os.getenv("CRON", "")' in APP
        assert 'out["cycle_tz"] = os.getenv("SCHEDULE_TZ", "") or "UTC"' in APP

    def test_next_fire_times_follow_new_york(self) -> None:
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from zoneinfo import ZoneInfo

        from trading.dashboard.cockpit import schedule_block

        now = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)  # after the DST change
        jobs = {
            j["key"]: datetime.fromisoformat(j["at"]).astimezone(ZoneInfo("America/New_York"))
            for j in schedule_block(
                SimpleNamespace(pm_pre_cycle_lead_minutes=45),
                cron="0 15 * * FRI",
                tz="America/New_York",
                now=now,
            )
        }
        assert jobs["cycle"].strftime("%a %H:%M") == "Fri 15:00"
        assert jobs["pm"].strftime("%a %H:%M") == "Fri 14:15"
        assert jobs["broker_ready"].strftime("%a %H:%M") == "Fri 14:00"
        assert jobs["curator"].strftime("%a %H:%M") == "Fri 19:00"
        assert jobs["reconcile"].strftime("%H:%M") == "16:30"

    def test_no_cron_still_lists_the_fixed_jobs(self) -> None:
        from types import SimpleNamespace

        from trading.dashboard.cockpit import schedule_block

        keys = {j["key"] for j in schedule_block(SimpleNamespace(), cron="", tz="UTC")}
        assert "cycle" not in keys and {"committee", "curator", "reconcile"} <= keys


class TestSnapshotAgeSeesWalWrites:
    def test_the_age_helper_is_wal_aware(self) -> None:
        assert '"snapshot": _db_age(state_dir / "runner.db")' in APP
        assert "artifact_age_seconds(p, now=now)" in APP
        assert '"snapshot": _age(state_dir / "runner.db")' not in APP

    def test_a_wal_write_counts_as_fresh(self, tmp_path) -> None:
        """The regression: main file old, WAL current -> must read fresh."""
        db = tmp_path / "runner.db"
        sqlite3.connect(db).close()
        old = time.time() - 90 * 3600
        import os

        os.utime(db, (old, old))
        (tmp_path / "runner.db-wal").write_bytes(b"x")  # written just now

        now = time.time()

        def db_age(p: Path) -> int | None:
            stamps = [
                q.stat().st_mtime
                for q in (p, p.with_suffix(p.suffix + "-wal"), p.with_suffix(p.suffix + "-shm"))
                if q.exists()
            ]
            return int((now - max(stamps)) / 60) if stamps else None

        assert db_age(db) is not None
        assert db_age(db) < 5  # minutes — not 5400

    def test_a_missing_database_is_still_none(self, tmp_path) -> None:
        def db_age(p: Path) -> int | None:
            stamps = [
                q.stat().st_mtime
                for q in (p, p.with_suffix(p.suffix + "-wal"), p.with_suffix(p.suffix + "-shm"))
                if q.exists()
            ]
            return int((time.time() - max(stamps)) / 60) if stamps else None

        assert db_age(tmp_path / "nope.db") is None


class TestOneImplementationOfArtifactAge:
    """The dashboard fix alone left the ops watchdog still telling the
    operator 'broker snapshot 91h old' about a runner writing every 60s.
    A fix applied to one path is not a fix to the invariant."""

    def test_the_watchdog_uses_the_shared_helper(self) -> None:
        src = Path("src/trading/runtime/ops_watch.py").read_text()

        assert "from trading.core.clock import artifact_age_seconds" in src
        assert "p.stat().st_mtime" not in src

    def test_the_dashboard_uses_the_shared_helper(self) -> None:
        assert "from trading.core.clock import artifact_age_seconds" in APP

    def test_the_helper_sees_a_wal_write(self, tmp_path) -> None:
        import os

        from trading.core.clock import artifact_age_seconds

        db = tmp_path / "runner.db"
        sqlite3.connect(db).close()
        old = time.time() - 91 * 3600
        os.utime(db, (old, old))
        (tmp_path / "runner.db-wal").write_bytes(b"x")

        assert artifact_age_seconds(db) < 60  # seconds, not 327600

    def test_the_helper_returns_none_when_absent(self, tmp_path) -> None:
        from trading.core.clock import artifact_age_seconds

        assert artifact_age_seconds(tmp_path / "nope.db") is None
