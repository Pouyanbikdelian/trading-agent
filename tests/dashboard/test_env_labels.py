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


class TestNoHardcodedPaperLabels:
    def test_the_equity_card_label_is_not_hardcoded(self) -> None:
        assert "Equity (paper)" not in APP
        assert 'Equity (<span class="envlbl">paper</span>)' in APP

    def test_the_account_card_label_is_not_hardcoded(self) -> None:
        assert "— traded book (paper)</span>" not in APP
        assert '— traded book (<span class="envlbl">paper</span>)' in APP

    def test_the_strategy_race_series_follows_the_environment(self) -> None:
        assert "'momentum top-k (paper)'" not in APP
        assert "'momentum top-k ('+(d.env||'paper')+')'" in APP

    def test_every_env_label_is_filled_from_the_payload(self) -> None:
        assert "document.querySelectorAll('.envlbl')" in APP
        assert "e.textContent=d.env||'paper'" in APP

    def test_live_is_visually_flagged(self) -> None:
        """Reading 'live' should not require looking for it."""
        assert "if((d.env||'')==='live')" in APP

    def test_the_server_supplies_the_environment(self) -> None:
        assert 'out["env"] = getattr(_s, "trading_env", "") or ""' in APP


class TestScheduleComesFromTheRealCron:
    def test_the_rebalance_slot_is_not_hardcoded(self) -> None:
        assert "{n:'⚖️ rebalance (paper)',dow:[5],h:21,m:5}" not in APP
        assert "'⚖️ rebalance ('+envl+')'" in APP

    def test_the_pm_slot_is_not_hardcoded_to_monday(self) -> None:
        assert "{n:'🧪 PM rebalance',dow:[1],h:14,m:30}" not in APP

    def test_the_pm_slot_is_derived_45_minutes_before_the_cycle(self) -> None:
        """Mirrors runner._precycle_trigger(lead_minutes=45)."""
        assert "shift(cyc,45)" in APP

    def test_the_server_supplies_the_cron(self) -> None:
        assert 'out["cycle_cron"] = os.getenv("CRON", "")' in APP


class TestCronParsing:
    """The JS parser, reimplemented in Python against the same shapes, so a
    regression in the accepted formats is caught here rather than by an
    operator noticing the wrong time on the page."""

    @staticmethod
    def _parse(t: str):
        dowmap = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}
        parts = (t or "").strip().split()
        if len(parts) < 5:
            return None
        try:
            m, h = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        f = parts[4].upper()
        dow: list[int] = []
        if f == "*":
            dow = [0, 1, 2, 3, 4, 5, 6]
        else:
            for part in f.split(","):
                r = [dowmap.get(x) if x in dowmap else int(x) for x in part.split("-")]
                if len(r) == 2:
                    dow.extend(i % 7 for i in range(r[0], r[1] + 1))
                else:
                    dow.append(r[0] % 7)
        return {"h": h, "m": m, "dow": dow} if dow else None

    def test_the_live_cron_parses(self) -> None:
        assert self._parse("0 19 * * FRI") == {"h": 19, "m": 0, "dow": [5]}

    def test_the_old_paper_cron_parses(self) -> None:
        assert self._parse("5 21 * * FRI") == {"h": 21, "m": 5, "dow": [5]}

    def test_a_weekday_range_parses(self) -> None:
        assert self._parse("0 16 * * MON-FRI")["dow"] == [1, 2, 3, 4, 5]

    def test_numeric_days_parse(self) -> None:
        assert self._parse("0 16 * * 1-5")["dow"] == [1, 2, 3, 4, 5]

    def test_a_star_means_every_day(self) -> None:
        assert len(self._parse("30 22 * * *")["dow"]) == 7

    def test_junk_yields_none_rather_than_a_wrong_time(self) -> None:
        for junk in ("", "nonsense", "0 19 *", "x y * * FRI"):
            assert self._parse(junk) is None

    def test_the_pm_slot_lands_45_minutes_earlier(self) -> None:
        c = self._parse("0 19 * * FRI")
        total = c["h"] * 60 + c["m"] - 45
        assert (total // 60, total % 60) == (18, 15)  # matches the runner's log


class TestSnapshotAgeSeesWalWrites:
    def test_the_age_helper_is_wal_aware(self) -> None:
        assert '"snapshot": _db_age(state_dir / "runner.db")' in APP
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
