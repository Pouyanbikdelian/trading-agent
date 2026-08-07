"""The long-only invariant on the command path.

The cycle has netted working orders into its sizing since 2026-07-15,
when three unfilled batches stacked at the open and took MU and SNDK
short. ``/sell all``, ``/close`` and ``/flatten`` never got the same
treatment: they size off settled shares and submit directly.

The guards make that reachable rather than theoretical. They run every
15 minutes through 20:59 UTC — past the US close — so their exits queue
to the next open. An operator who sees the trailing-stop alert and
reaches for ``/close`` sells the position twice.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading.risk.presubmit import clamp_sell, sellable_quantity


def _pos(symbol: str, qty: float):
    return SimpleNamespace(instrument=SimpleNamespace(symbol=symbol), quantity=qty)


def _order(symbol: str, qty: float, side: str = "SELL"):
    return SimpleNamespace(
        instrument=SimpleNamespace(symbol=symbol),
        quantity=qty,
        side=SimpleNamespace(value=side),
    )


class _Broker:
    def __init__(self, positions=(), open_orders=(), orders_exc=None):
        self._positions = list(positions)
        self._open = list(open_orders)
        self._exc = orders_exc

    def get_positions(self):
        return self._positions

    def get_open_orders(self):
        if self._exc:
            raise self._exc
        return self._open


class TestSellableQuantity:
    def test_no_working_orders_means_the_whole_position(self):
        r = sellable_quantity(_Broker([_pos("VST", 63)]), "VST")
        assert r.sellable == 63 and r.working_sells == 0 and r.orders_known

    def test_a_working_sell_reduces_what_is_left(self):
        b = _Broker([_pos("VST", 63)], [_order("VST", 63)])
        assert sellable_quantity(b, "VST").sellable == 0

    def test_a_partial_working_sell_leaves_the_remainder(self):
        b = _Broker([_pos("VST", 63)], [_order("VST", 20)])
        assert sellable_quantity(b, "VST").sellable == 43

    def test_working_buys_do_not_count_as_owned(self):
        """An unfilled buy is not ours to sell — counting it is how a sell
        fires ahead of its own buy and ends short."""
        b = _Broker([_pos("VST", 10)], [_order("VST", 50, side="BUY")])
        assert sellable_quantity(b, "VST").sellable == 10

    def test_other_symbols_are_ignored(self):
        b = _Broker([_pos("VST", 63)], [_order("WMT", 63)])
        assert sellable_quantity(b, "VST").sellable == 63

    def test_unreadable_open_orders_degrade_to_settled_and_say_so(self):
        b = _Broker([_pos("VST", 63)], orders_exc=ConnectionError("gateway"))
        r = sellable_quantity(b, "VST")
        assert r.sellable == 63
        assert not r.orders_known


class TestClampSell:
    def test_the_2026_07_15_double_sell_is_refused(self):
        """Guard exit queued overnight, operator /close the next morning."""
        b = _Broker([_pos("VST", 63)], [_order("VST", 63)])
        with pytest.raises(ValueError) as exc:
            clamp_sell(b, "VST", 63)
        msg = str(exc.value)
        assert "already working" in msg
        assert "short" in msg

    def test_a_normal_sell_passes_through_untouched(self):
        qty, note = clamp_sell(_Broker([_pos("VST", 63)]), "VST", 63)
        assert qty == 63 and note is None

    def test_an_oversized_sell_is_clamped_not_refused(self):
        b = _Broker([_pos("VST", 63)], [_order("VST", 20)])
        qty, note = clamp_sell(b, "VST", 63)
        assert qty == 43
        assert note is not None and "63" in note and "43" in note

    def test_selling_what_is_not_held_is_refused(self):
        with pytest.raises(ValueError, match="no long position"):
            clamp_sell(_Broker([]), "VST", 10)

    def test_the_note_flags_when_working_orders_were_unknown(self):
        """Clamping on settled-only is weaker protection and must say so —
        otherwise the note reads like a complete picture."""
        b = _Broker([_pos("VST", 63)], orders_exc=ConnectionError("gateway"))
        qty, note = clamp_sell(b, "VST", 100)
        assert qty == 63
        assert note is not None and "UNAVAILABLE" in note

    def test_a_smaller_partial_sell_is_allowed_alongside_a_working_one(self):
        b = _Broker([_pos("VST", 63)], [_order("VST", 20)])
        qty, note = clamp_sell(b, "VST", 10)
        assert qty == 10 and note is None
