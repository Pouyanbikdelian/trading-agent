"""A standing "never buy this" that the code enforces, not a prompt.

The 2026-08-19 audit traced the operator's "I don't like PM (Philip
Morris), don't buy it in the future" and found it had become a 14-day
free-text mandate, graded by a matcher that scores "don't buy PM" exactly
like "buy PM", injected into an LLM prompt, read by no filter. The agent
PM went on holding PM at 7%.

So the contract these tests pin is deliberately narrow and mechanical:

  * an excluded name cannot be BOUGHT by anything automatic;
  * an excluded name can always be SOLD — banning a stock must never
    trap the position;
  * a manual `/buy` still works, because the operator typing an order now
    outranks a preference they expressed in the past;
  * the ban never expires.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from trading.runner.exclusions import (
    add_exclusion,
    filter_excluded_orders,
    load_exclusion_details,
    load_exclusions,
    normalize_symbol,
    remove_exclusion,
)

NOW = datetime(2026, 8, 19, 19, 0, tzinfo=timezone.utc)


def _order(symbol: str, side: Side, qty: float = 10.0) -> Order:
    return Order(
        client_order_id=f"{symbol}-{side.value}",
        instrument=Instrument(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            currency="USD",
            exchange="SMART",
        ),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW,
    )


class TestTheStore:
    def test_a_ban_round_trips_with_its_reason(self, tmp_path: Path) -> None:
        assert add_exclusion(tmp_path, "pm", reason="tobacco") is True

        assert load_exclusions(tmp_path) == {"PM"}
        assert load_exclusion_details(tmp_path)["PM"]["reason"] == "tobacco"

    def test_banning_twice_is_reported_not_duplicated(self, tmp_path: Path) -> None:
        add_exclusion(tmp_path, "PM", reason="first")

        assert add_exclusion(tmp_path, "PM", reason="second") is False
        # The original reason survives — a repeat is a no-op, not an edit.
        assert load_exclusion_details(tmp_path)["PM"]["reason"] == "first"

    def test_lifting_a_ban_that_was_never_set_is_reported(self, tmp_path: Path) -> None:
        assert remove_exclusion(tmp_path, "PM") is False

    def test_lifting_a_ban_removes_only_that_name(self, tmp_path: Path) -> None:
        add_exclusion(tmp_path, "PM")
        add_exclusion(tmp_path, "MO")

        assert remove_exclusion(tmp_path, "PM") is True
        assert load_exclusions(tmp_path) == {"MO"}

    def test_no_expiry_field_is_written(self, tmp_path: Path) -> None:
        """The mandate store's silent 14-day TTL is what let PM come back."""
        add_exclusion(tmp_path, "PM", reason="tobacco")

        raw = json.loads((tmp_path / "exclusions.json").read_text())
        serialized = json.dumps(raw).lower()
        assert "expire" not in serialized and "ttl" not in serialized

    def test_a_hand_edited_bare_list_still_loads(self, tmp_path: Path) -> None:
        (tmp_path / "exclusions.json").write_text(json.dumps({"symbols": ["pm", "mo"]}))

        assert load_exclusions(tmp_path) == {"PM", "MO"}

    def test_a_corrupt_file_blocks_nothing_and_says_so_to_the_reader(self, tmp_path: Path) -> None:
        """Fail OPEN, unlike halt.json.

        An unreadable preference file must not stop the desk trading; it
        degrades to the state the system was in before the file existed.
        The detailed reader raises so `/exclusions` can warn that nothing
        is currently being enforced.
        """
        (tmp_path / "exclusions.json").write_text("{not json")

        assert load_exclusions(tmp_path) == set()
        with pytest.raises(Exception):
            load_exclusion_details(tmp_path)

    @pytest.mark.parametrize("bad", ["", "   ", "A B", "*", "PM;DROP", "X" * 13])
    def test_a_non_ticker_is_refused_rather_than_stored(self, bad: str) -> None:
        with pytest.raises(ValueError):
            normalize_symbol(bad)

    @pytest.mark.parametrize("ok", ["BRK.B", "RDS-A", "pm", " mo "])
    def test_real_ticker_shapes_are_accepted(self, ok: str) -> None:
        assert normalize_symbol(ok) == ok.strip().upper()


class TestTheOrderFilter:
    def test_a_buy_in_an_excluded_name_is_dropped(self) -> None:
        kept, dropped = filter_excluded_orders([_order("PM", Side.BUY)], {"PM"})

        assert kept == []
        assert [o.instrument.symbol for o in dropped] == ["PM"]

    def test_a_sell_in_an_excluded_name_survives(self) -> None:
        """The asymmetry that makes this different from /hold.

        Banning a stock must never mean you cannot get out of it.
        """
        kept, dropped = filter_excluded_orders([_order("PM", Side.SELL)], {"PM"})

        assert [o.instrument.symbol for o in kept] == ["PM"]
        assert dropped == []

    def test_unrelated_names_pass_through_untouched(self) -> None:
        orders = [_order("GLD", Side.BUY), _order("AMD", Side.SELL)]

        kept, dropped = filter_excluded_orders(orders, {"PM"})

        assert kept == orders
        assert dropped == []

    def test_an_empty_ban_list_is_a_no_op(self) -> None:
        orders = [_order("PM", Side.BUY)]

        kept, dropped = filter_excluded_orders(orders, set())

        assert kept == orders and dropped == []

    def test_a_mixed_basket_keeps_the_exit_and_drops_the_entry(self) -> None:
        orders = [
            _order("PM", Side.BUY, 100),
            _order("PM", Side.SELL, 40),
            _order("GLD", Side.BUY, 5),
        ]

        kept, dropped = filter_excluded_orders(orders, {"PM"})

        assert [(o.instrument.symbol, o.side) for o in kept] == [
            ("PM", Side.SELL),
            ("GLD", Side.BUY),
        ]
        assert [o.side for o in dropped] == [Side.BUY]


class TestThePMAgentCannotAllocateToAnExcludedName:
    def test_an_excluded_symbol_is_clamped_to_cash(self) -> None:
        """The PM's prose is intent; `_clamp_weights` is the constraint.

        `run_agent_pm` passes ``blocked=held | excluded``, so this is the
        function that decides whether a banned name can survive into the
        target — regardless of what the model wrote.
        """
        from trading.agents.pm import _clamp_weights

        weights = _clamp_weights(
            {"PM": 0.07, "JPM": 0.08},
            stocks=("PM", "JPM"),
            blocked=frozenset({"PM"}),
        )

        assert "PM" not in weights
        assert weights["JPM"] == pytest.approx(0.08)

    def test_the_charter_tells_the_pm_about_exclusions_by_name(self) -> None:
        """A hard filter the model cannot see just wastes basket slots."""
        from trading.agents.pm import PM_CHARTER

        assert "operator_excluded_never_buy" in PM_CHARTER
        assert "NEVER allocate to them" in PM_CHARTER
