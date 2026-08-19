"""The halted review planner is informational, never an execution bypass."""

from __future__ import annotations

import pytest

from tests.risk.conftest import instruments_dict, signal_from


def test_halt_blocks_normal_orders_but_allows_ephemeral_risk_sized_preview(
    mgr, aapl, account_100k, t0
) -> None:
    """A review may show the exact basket without clearing the hard gate."""
    mgr.halt("daily loss limit")
    signal = signal_from(t0, {aapl.key: 0.05})
    prices = {aapl.key: 100.0}
    instruments = instruments_dict(aapl)

    executable, executable_decisions = mgr.signal_to_orders(
        signal,
        account=account_100k,
        last_prices=prices,
        instruments=instruments,
    )
    preview, preview_decisions = mgr.preview_signal_to_orders(
        signal,
        account=account_100k,
        last_prices=prices,
        instruments=instruments,
        order_id_factory=lambda: "review-only-order",
    )

    assert executable == []
    assert executable_decisions[0].action == "halt"
    assert mgr.is_halted()  # Previewing cannot resume or weaken the halt.

    assert len(preview) == 1
    assert preview[0].client_order_id == "review-only-order"
    assert preview[0].quantity == pytest.approx(50.0)
    assert not any(decision.action == "halt" for decision in preview_decisions)
