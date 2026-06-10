"""Operator holds — pin positions out of the cycle's reach."""

from __future__ import annotations

from datetime import datetime, timezone

from trading.core.types import AssetClass, Instrument, Order, Side
from trading.execution import new_client_order_id
from trading.runner.holds import filter_held_orders, load_holds, save_holds


def _order(sym: str, side: Side = Side.BUY) -> Order:
    return Order(
        client_order_id=new_client_order_id(),
        instrument=Instrument(symbol=sym, asset_class=AssetClass.EQUITY),
        side=side,
        quantity=10,
        created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )


def test_round_trip_and_case_normalization(tmp_path) -> None:
    assert load_holds(tmp_path) == set()
    save_holds(tmp_path, {"nvda", "AAPL"})
    assert load_holds(tmp_path) == {"NVDA", "AAPL"}
    save_holds(tmp_path, set())
    assert load_holds(tmp_path) == set()


def test_corrupt_file_means_no_holds(tmp_path) -> None:
    (tmp_path / "holds.json").write_text("{not json")
    assert load_holds(tmp_path) == set()


def test_filter_drops_both_buys_and_sells_on_held_symbols() -> None:
    orders = [_order("NVDA", Side.SELL), _order("AAPL", Side.BUY), _order("MSFT", Side.BUY)]
    kept, dropped = filter_held_orders(orders, {"NVDA", "MSFT"})
    assert [o.instrument.symbol for o in kept] == ["AAPL"]
    assert {o.instrument.symbol for o in dropped} == {"NVDA", "MSFT"}


def test_filter_no_holds_is_passthrough() -> None:
    orders = [_order("AAPL")]
    kept, dropped = filter_held_orders(orders, set())
    assert kept == orders and dropped == []


def test_apply_runtime_overrides_reserves_slots_and_k_override(tmp_path) -> None:
    from trading.runner.holds import apply_runtime_overrides, save_holds, save_k_override
    from trading.strategies.top_k_momentum import TopKMomentumParams

    params = TopKMomentumParams(k=8)
    # No state -> passthrough.
    out, notes = apply_runtime_overrides(params, tmp_path)
    assert out.k == 8 and notes == []

    # /k override wins over configured default.
    save_k_override(tmp_path, 12)
    out, notes = apply_runtime_overrides(params, tmp_path)
    assert out.k == 12 and any("overridden" in n for n in notes)

    # Holds reserve slots on top of the override: 12 - 2 = 10.
    save_holds(tmp_path, {"NVDA", "MU"})
    out, notes = apply_runtime_overrides(params, tmp_path)
    assert out.k == 10

    # Reservation never drives k below 1.
    save_k_override(tmp_path, 2)
    save_holds(tmp_path, {"A", "B", "C", "D"})
    out, _ = apply_runtime_overrides(params, tmp_path)
    assert out.k == 1


def test_apply_runtime_overrides_passthrough_without_k(tmp_path) -> None:
    from trading.runner.holds import apply_runtime_overrides, save_k_override

    class NoK:
        pass

    save_k_override(tmp_path, 12)
    obj = NoK()
    out, notes = apply_runtime_overrides(obj, tmp_path)
    assert out is obj and notes == []
