"""Six stop-outs should arrive as one message, not six.

A single exit reads fine on its own line. Several arrive as several
separate Telegram bubbles the operator has to add up by hand, at exactly
the moment they are least inclined to — a fast tape. What decides the
next move is three numbers: how much went to cash, what it cost, and
what is still at risk.

The roll-up also carries the one thing the system cannot do for itself:
nothing redeploys after a guard exit, and `/cycle` alone would re-buy the
names that just stopped out, because neither the cycle nor the PM can see
that the guards sold anything.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trading.runtime.guards import format_exit_rollup

NOW = datetime(2026, 8, 11, 14, 35, tzinfo=timezone.utc)


def _exit(sym: str, *, price=100.0, entry=110.0, qty=100.0, reason="trailing_stop") -> dict:
    return {
        "symbol": sym,
        "reason": reason,
        "price": price,
        "qty": qty,
        "proceeds": price * qty,
        "entry": entry,
        "pnl_pct": (price / entry - 1.0) * 100.0,
        "pnl_usd": (price - entry) * qty,
    }


class TestWhenItFires:
    def test_a_single_exit_gets_no_rollup(self) -> None:
        """One line is already readable; a summary of it is noise."""
        assert format_exit_rollup([_exit("VST")], equity=88_000.0) is None

    def test_no_exits_gets_no_rollup(self) -> None:
        assert format_exit_rollup([], equity=88_000.0) is None

    def test_two_exits_is_enough(self) -> None:
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=88_000.0)

        assert out is not None and "2 positions closed" in out


class TestTheNumbersThatDecideTheNextMove:
    def test_proceeds_and_realized_are_totalled(self) -> None:
        out = format_exit_rollup(
            [_exit("VST", price=100, entry=110), _exit("NVDA", price=200, entry=180)],
            equity=88_000.0,
            now=NOW,
        )

        assert "*To cash* $30,000" in out  # 100*100 + 200*100
        assert "*Realized* $1,000" in out  # -1000 + 2000

    def test_a_loss_is_signed_with_a_real_minus(self) -> None:
        """A hyphen next to a dollar sign reads as a range in Telegram."""
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=88_000.0)

        assert "−$2,000" in out
        assert "-$2,000" not in out

    def test_what_is_still_at_risk_is_shown(self) -> None:
        out = format_exit_rollup(
            [_exit("VST"), _exit("NVDA")], equity=88_000.0, remaining_value=30_800.0
        )

        assert "*Still deployed* $30,800 of $88,000 (35%)" in out

    def test_equity_of_none_does_not_break_it(self) -> None:
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=None)

        assert out is not None and "Still deployed" not in out


class TestItReadsWell:
    def test_the_table_is_in_a_fenced_block(self) -> None:
        """Monospace so columns align — and Telegram does not parse
        entities inside a fence, so an odd symbol cannot break the post."""
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=88_000.0)

        assert out.count("```") == 2

    def test_columns_are_aligned(self) -> None:
        out = format_exit_rollup(
            [_exit("A", price=5.0), _exit("LONGSYM", price=1234.5)], equity=88_000.0
        )
        body = out.split("```")[1].strip().splitlines()

        # The percent column lands in the same place on every row.
        assert len({line.index("%") for line in body}) == 1

    def test_biggest_position_leads(self) -> None:
        out = format_exit_rollup(
            [_exit("SMALL", price=10), _exit("BIG", price=900)], equity=88_000.0
        )
        body = out.split("```")[1]

        assert body.index("BIG") < body.index("SMALL")

    def test_the_reason_mix_is_named(self) -> None:
        mixed = format_exit_rollup(
            [_exit("A"), _exit("B", reason="take_profit")], equity=88_000.0
        )
        allstops = format_exit_rollup([_exit("A"), _exit("B")], equity=88_000.0)

        assert "1 stop, 1 target" in mixed
        assert "trailing stops" in allstops


class TestTheWarningIsTheWholePoint:
    def test_it_says_nothing_redeploys(self) -> None:
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=88_000.0)

        assert "Nothing redeploys automatically" in out

    def test_it_warns_that_cycle_alone_would_rebuy(self) -> None:
        """The trap: the PM's standing decision still lists these names."""
        out = format_exit_rollup([_exit("VST"), _exit("NVDA")], equity=88_000.0)

        assert "re-buy the same names" in out
        assert "/pm run" in out and out.index("/pm run") < out.index("→ `/cycle`")


class TestWiring:
    def test_check_guards_returns_a_rollup_key(self, tmp_path) -> None:
        from trading.runtime.guards import check_guards

        out = check_guards(
            tmp_path,
            tmp_path,
            positions=[{"symbol": "VST", "qty": 10.0, "avg_price": 100.0, "mark": 100.0}],
            prices={"VST": 100.0},
            equity=88_000.0,
        )

        assert "rollup" in out and "exit_alerts" in out

    def test_the_runner_suppresses_per_name_alerts_when_rolling_up(self) -> None:
        """A roll-up PLUS six bubbles would be worse than either alone."""
        from pathlib import Path

        src = Path("src/trading/runner/runner.py").read_text()

        assert 'rollup = result.get("rollup")' in src
        assert "if rollup and msg in exit_msgs:" in src
