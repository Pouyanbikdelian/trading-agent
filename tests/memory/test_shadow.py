"""The counterfactual ledger — what the desk considered and did not do.

The property under test throughout is that a shadow result is never
reported without its benchmark and its sample size. A ledger of passed
names shows handsome absolute returns in any rising market, and a spread
computed on a handful of names is an anecdote; both failure modes are
about presentation rather than arithmetic, so they are tested here rather
than left to the reader of the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.memory.store import MemoryStore

DAY = 86400.0


@pytest.fixture()
def mem(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory")
    yield store
    store.close()


def _ago(days: float) -> float:
    return datetime.now(tz=timezone.utc).timestamp() - days * DAY


class TestRecording:
    def test_a_passed_name_is_stored_with_its_rank_and_reason(self, mem: MemoryStore) -> None:
        sid = mem.add_shadow(
            symbol="gs",
            origin="ladder",
            disposition="passed",
            rank=23,
            score=4.1,
            why="rank 23 of 30",
            conditions={"vol_bucket": "normal"},
            px_at=500.0,
        )
        row = mem.conn.execute("SELECT * FROM shadow WHERE id = ?", (sid,)).fetchone()
        assert row["symbol"] == "GS"  # normalised
        assert row["disposition"] == "passed"
        assert row["rank"] == 23
        assert row["graded_ts"] is None

    def test_shadow_rows_show_up_in_stats(self, mem: MemoryStore) -> None:
        mem.add_shadow(symbol="JPM", origin="ladder", disposition="taken")
        assert mem.stats()["shadow"] == 1


class TestLegMaturity:
    """Legs fill independently. A 5-day read next week beats a complete
    row next quarter — an unprofitable selection step should be visible
    long before its 63-day leg matures."""

    def test_only_matured_rows_come_back(self, mem: MemoryStore) -> None:
        mem.add_shadow(symbol="OLD", origin="ladder", disposition="passed", ts=_ago(30))
        mem.add_shadow(symbol="NEW", origin="ladder", disposition="passed", ts=_ago(1))

        due5 = {r["symbol"] for r in mem.ungraded_shadow(5)}
        assert due5 == {"OLD"}
        assert mem.ungraded_shadow(63) == []

    def test_a_filled_leg_is_not_returned_again(self, mem: MemoryStore) -> None:
        sid = mem.add_shadow(symbol="OLD", origin="ladder", disposition="passed", ts=_ago(30))
        mem.grade_shadow_leg(sid, 5, ret=0.03, bench=0.01)
        assert mem.ungraded_shadow(5) == []
        assert len(mem.ungraded_shadow(21)) == 1

    def test_graded_ts_lands_only_on_the_final_leg(self, mem: MemoryStore) -> None:
        sid = mem.add_shadow(symbol="OLD", origin="ladder", disposition="passed", ts=_ago(90))
        mem.grade_shadow_leg(sid, 5, ret=0.03, bench=0.01)
        mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)
        assert mem.conn.execute("SELECT graded_ts FROM shadow").fetchone()[0] is None
        mem.grade_shadow_leg(sid, 63, ret=0.09, bench=0.04)
        assert mem.conn.execute("SELECT graded_ts FROM shadow").fetchone()[0] is not None


class TestEdgeReport:
    def _populate(self, mem: MemoryStore, *, taken: list[float], passed: list[float]) -> None:
        """Everything gets the same +2% benchmark, so excess return is the
        only thing that varies and the spread is checkable by hand."""
        for i, r in enumerate(taken):
            sid = mem.add_shadow(symbol=f"T{i}", origin="ladder", disposition="taken", ts=_ago(40))
            mem.grade_shadow_leg(sid, 21, ret=r, bench=0.02)
        for i, r in enumerate(passed):
            sid = mem.add_shadow(symbol=f"P{i}", origin="ladder", disposition="passed", ts=_ago(40))
            mem.grade_shadow_leg(sid, 21, ret=r, bench=0.02)

    def test_positive_spread_when_picks_beat_passes(self, mem: MemoryStore) -> None:
        self._populate(mem, taken=[0.10, 0.08], passed=[0.01, 0.03])
        row = mem.edge_report(leg_days=21)[0]
        assert row["n_taken"] == 2 and row["n_passed"] == 2
        assert row["spread"] == pytest.approx(0.07)

    def test_negative_spread_is_reported_not_hidden(self, mem: MemoryStore) -> None:
        """The finding this table exists to be able to make."""
        self._populate(mem, taken=[0.01], passed=[0.09])
        assert mem.edge_report(leg_days=21)[0]["spread"] == pytest.approx(-0.08)

    def test_returns_are_quoted_net_of_the_benchmark(self, mem: MemoryStore) -> None:
        """A +5% pick in a +2% market is +3% of edge, not +5%. Absolute
        return would flatter every side of this ledger in a bull market."""
        self._populate(mem, taken=[0.05], passed=[0.02])
        row = mem.edge_report(leg_days=21)[0]
        assert row["taken_excess"] == pytest.approx(0.03)
        assert row["passed_excess"] == pytest.approx(0.0)

    def test_an_origin_with_no_counterfactual_reports_none_not_zero(self, mem: MemoryStore) -> None:
        """A mandate the desk always honours has no control group. Zero
        would read as 'no edge measured'; None reads as 'not measurable',
        which is the truth."""
        sid = mem.add_shadow(symbol="GS", origin="mandate", disposition="taken", ts=_ago(40))
        mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)
        row = next(r for r in mem.edge_report(leg_days=21) if r["origin"] == "mandate")
        assert row["spread"] is None
        assert row["n_passed"] == 0

    def test_ungraded_rows_are_excluded(self, mem: MemoryStore) -> None:
        mem.add_shadow(symbol="X", origin="ladder", disposition="taken", ts=_ago(40))
        assert mem.edge_report(leg_days=21) == []

    def test_origins_are_reported_separately(self, mem: MemoryStore) -> None:
        """The ladder and the committee are different selection processes
        and one can have edge while the other destroys it."""
        for origin in ("ladder", "committee"):
            for disp in ("taken", "passed"):
                sid = mem.add_shadow(
                    symbol=f"{origin[:2]}{disp[:2]}",
                    origin=origin,
                    disposition=disp,
                    ts=_ago(40),
                )
                mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)
        assert {r["origin"] for r in mem.edge_report(leg_days=21)} == {"ladder", "committee"}

    def test_stale_rows_fall_out_of_the_window(self, mem: MemoryStore) -> None:
        sid = mem.add_shadow(symbol="OLD", origin="ladder", disposition="taken", ts=_ago(500))
        mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)
        assert mem.edge_report(leg_days=21, since_days=365) == []


class TestWhySlices:
    """/edge says whether the selection step worked. These say where the
    answer comes from — all SQL over columns stored at decision time, so
    no model is ever asked to speculate about a cause."""

    def _ladder(self, mem: MemoryStore) -> None:
        """A ladder where the top five genuinely outperform."""
        for i in range(1, 31):
            excess = 0.08 if i <= 5 else (0.02 if i <= 15 else -0.01)
            sid = mem.add_shadow(
                symbol=f"S{i}",
                origin="ladder",
                disposition="taken" if i <= 9 else "passed",
                rank=i,
                conditions={"vol_bucket": "normal" if i % 2 else "elevated"},
                pctile_52w=0.97 if i <= 5 else 0.60,
                ts=_ago(40),
            )
            mem.grade_shadow_leg(sid, 21, ret=excess + 0.02, bench=0.02)

    def test_rank_buckets_come_back_in_ladder_order(self, mem: MemoryStore) -> None:
        """Ordered by rank, not by sample size — reading it top to bottom
        is the whole point."""
        self._ladder(mem)
        rows = mem.edge_by_rank(leg_days=21)
        assert [r["rank_bucket"] for r in rows] == ["1-5", "6-15", "16-30"]

    def test_a_working_score_shows_a_gradient(self, mem: MemoryStore) -> None:
        """The discrimination test. A flat result here means the score
        ranks nothing and no amount of moving the cut will help."""
        self._ladder(mem)
        rows = mem.edge_by_rank(leg_days=21)
        assert rows[0]["excess"] > rows[-1]["excess"]

    def test_condition_slice_splits_picks_from_passes(self, mem: MemoryStore) -> None:
        self._ladder(mem)
        rows = mem.edge_by_condition("vol_bucket", leg_days=21)
        assert {r["condition"] for r in rows} == {"normal", "elevated"}
        assert all(r["n_taken"] and r["n_passed"] for r in rows)

    def test_entry_slice_separates_names_bought_near_the_high(self, mem: MemoryStore) -> None:
        self._ladder(mem)
        buckets = {r["entry_bucket"] for r in mem.edge_by_entry(leg_days=21)}
        assert "above 0.95" in buckets

    def test_a_condition_key_cannot_smuggle_sql(self, mem: MemoryStore) -> None:
        """The key is interpolated into a json_extract path, so it is the
        one caller-supplied string in this module that reaches SQL."""
        with pytest.raises(ValueError, match="unsafe condition key"):
            mem.edge_by_condition("x'; DROP TABLE shadow;--")

    def test_slices_are_empty_before_anything_matures(self, mem: MemoryStore) -> None:
        mem.add_shadow(symbol="X", origin="ladder", disposition="taken", rank=1, ts=_ago(40))
        assert mem.edge_by_rank(leg_days=21) == []


class TestOverlappingObservations:
    """Surfaced 2026-07-30, second live cycle: two runs seven minutes
    apart each wrote 30 rows for the same 30 names. The ladder re-ranks
    daily, so rows grow ~30/day over a universe that turns over slowly,
    and consecutive rows' 21-day forward windows overlap by twenty days.
    A raw n of 1200 can be forty independent observations in a convincing
    costume — which is precisely the overconfidence the thin-sample
    warning was built to prevent."""

    def _same_names_many_days(self, mem: MemoryStore, *, days: int = 20) -> None:
        for day in range(days):
            for i, sym in enumerate(("AAPL", "MSFT", "GS"), start=1):
                sid = mem.add_shadow(
                    symbol=sym,
                    origin="ladder",
                    disposition="taken" if i == 1 else "passed",
                    rank=i,
                    pctile_52w=0.6,
                    conditions={"vol_bucket": "normal"},
                    ts=_ago(40 + day),
                )
                mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)

    def test_distinct_names_are_reported_next_to_row_counts(self, mem: MemoryStore) -> None:
        self._same_names_many_days(mem)
        row = mem.edge_report(leg_days=21)[0]
        assert row["n_taken"] == 20 and row["n_taken_symbols"] == 1
        assert row["n_passed"] == 40 and row["n_passed_symbols"] == 2

    def test_slices_report_distinct_names_too(self, mem: MemoryStore) -> None:
        self._same_names_many_days(mem)
        by_rank = mem.edge_by_rank(leg_days=21)
        assert by_rank[0]["n"] == 60 and by_rank[0]["n_symbols"] == 3

    def test_a_group_with_no_rows_still_reports_zero_symbols(self, mem: MemoryStore) -> None:
        """A mandate origin with only taken rows must not KeyError when the
        caller reaches for the passed side."""
        sid = mem.add_shadow(symbol="GS", origin="mandate", disposition="taken", ts=_ago(40))
        mem.grade_shadow_leg(sid, 21, ret=0.05, bench=0.02)
        row = next(r for r in mem.edge_report(leg_days=21) if r["origin"] == "mandate")
        assert row["n_passed_symbols"] == 0


class TestWhichNamesCountAsTaken:
    """Regression, 2026-07-30 first live run: the ledger logged "recorded
    30 candidate(s), 501 taken". Membership in ``target_weights`` was read
    as "bought", but the signal carries a key for every instrument in the
    universe — so every ranked name was labelled taken, the ledger held no
    passed rows, and /edge would have reported no measurable spread
    forever while looking like it was working."""

    def test_only_non_zero_weights_count(self) -> None:
        from trading.runner.cycle import bought_symbols

        weights = {"EQUITY:AAPL": 0.10, "EQUITY:MSFT": 0.0, "EQUITY:GS": 0.0}
        assert bought_symbols(weights) == {"AAPL"}

    def test_a_universe_of_zeros_buys_nothing(self) -> None:
        from trading.runner.cycle import bought_symbols

        assert bought_symbols({f"EQUITY:S{i}": 0.0 for i in range(501)}) == set()

    def test_shorts_count_as_held(self) -> None:
        from trading.runner.cycle import bought_symbols

        assert bought_symbols({"EQUITY:SPY": -0.15}) == {"SPY"}

    def test_empty_and_none_are_safe(self) -> None:
        from trading.runner.cycle import bought_symbols

        assert bought_symbols({}) == set()
        assert bought_symbols(None) == set()


class TestEdgeCommandPresentation:
    def test_a_thin_sample_is_flagged_rather_than_stated_as_fact(self) -> None:
        """Nine names is a story. The report must say so on its face —
        this is the number most likely to be quoted out of context."""
        from trading.bot.telegram import _EDGE_MIN_N

        assert _EDGE_MIN_N >= 20
