"""The per-cycle approval gate — what stands between a basket and the market.

`REQUIRE_CYCLE_APPROVAL` is the control the operator leans on for live
trading, and it had no tests at all.

The property pinned hardest here: a decision must belong to THIS cycle.
The bot stamps the pending cycle's id into every decision file, and
until 2026-08-07 nothing compared it — written and discarded, the same
shape as `flatten_on_next_cycle`. The inline keyboards already refuse a
tap minted for another basket; a typed `/approve` had no equivalent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)

TS = datetime(2026, 8, 7, 21, 5, tzinfo=timezone.utc)


class _Alerts:
    def __init__(self) -> None:
        self.msgs: list[tuple[str, str]] = []

    def info(self, m: str, **kw: Any) -> None:
        self.msgs.append(("info", m))

    def warning(self, m: str, **kw: Any) -> None:
        self.msgs.append(("warning", m))

    def error(self, m: str, **kw: Any) -> None:
        self.msgs.append(("error", m))

    def text(self) -> str:
        return "\n".join(m for _, m in self.msgs)


class _Cycle:
    from trading.runner.cycle import Cycle as _Real

    _request_cycle_approval = _Real._request_cycle_approval
    _build_cycle_id = _Real._build_cycle_id
    _rescale_orders = _Real._rescale_orders
    _working_orders = _Real._working_orders
    APPROVAL_PENDING_FILE = _Real.APPROVAL_PENDING_FILE
    APPROVAL_DECISION_FILE = _Real.APPROVAL_DECISION_FILE
    APPROVAL_POLL_INTERVAL_S = 0.01

    def __init__(self, risk_manager=None) -> None:
        self.alerts = _Alerts()
        self.risk_manager = risk_manager
        self.broker = SimpleNamespace(get_open_orders=lambda: [])


def _order(symbol: str, qty: float = 10.0) -> Order:
    return Order(
        client_order_id=f"o-{symbol}",
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        side=Side.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=TS,
    )


@pytest.fixture
def approval_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading.core.config.settings",
        SimpleNamespace(state_dir=tmp_path, cycle_approval_timeout_s=0.05),
    )
    return tmp_path


@pytest.fixture
def account():
    return SimpleNamespace(equity=88_000.0, base_currency="CHF", positions={}, ts=TS)


def _decide(state_dir, cycle: _Cycle, payload: dict) -> None:
    (state_dir / cycle.APPROVAL_DECISION_FILE).write_text(json.dumps(payload))


def _pending_id(state_dir, cycle: _Cycle) -> str:
    return json.loads((state_dir / cycle.APPROVAL_PENDING_FILE).read_text())["id"]


def _run(cycle, orders, account, state_dir, decision: dict | None):
    """Drive the gate, planting a decision the moment it starts polling."""
    import threading

    result: list[Any] = []

    def plant() -> None:
        import time

        for _ in range(200):
            if (state_dir / cycle.APPROVAL_PENDING_FILE).exists():
                if decision is not None:
                    payload = dict(decision)
                    if payload.get("id") == "__CURRENT__":
                        payload["id"] = _pending_id(state_dir, cycle)
                    _decide(state_dir, cycle, payload)
                return
            time.sleep(0.005)

    t = threading.Thread(target=plant)
    t.start()
    result = cycle._request_cycle_approval(orders, account, {"equity:NVDA": 100.0})
    t.join()
    return result


def test_a_plain_approval_submits_the_basket(approval_settings, account) -> None:
    cycle = _Cycle()
    orders = [_order("NVDA")]

    out = _run(
        cycle, orders, account, approval_settings, {"id": "__CURRENT__", "action": "approve"}
    )

    assert out == orders


def test_a_decision_for_another_cycle_is_refused(approval_settings, account) -> None:
    """The regression. An operator scrolling back to yesterday's prompt and
    typing /approve must not approve today's basket."""
    cycle = _Cycle()
    orders = [_order("NVDA")]

    out = _run(
        cycle, orders, account, approval_settings, {"id": "some-other-cycle", "action": "approve"}
    )

    assert out == []
    assert "ignoring an approval" in cycle.alerts.text()


def test_no_response_auto_rejects(approval_settings, account) -> None:
    cycle = _Cycle()

    out = _run(cycle, [_order("NVDA")], account, approval_settings, None)

    assert out == []
    assert "auto-rejected" in cycle.alerts.text()


def test_an_explicit_reject_submits_nothing(approval_settings, account) -> None:
    cycle = _Cycle()

    out = _run(
        cycle,
        [_order("NVDA")],
        account,
        approval_settings,
        {"id": "__CURRENT__", "action": "reject"},
    )

    assert out == []


def test_scaling_shrinks_the_basket(approval_settings, account) -> None:
    cycle = _Cycle()

    out = _run(
        cycle,
        [_order("NVDA", 10)],
        account,
        approval_settings,
        {"id": "__CURRENT__", "action": "scale", "scale_factor": 0.5},
    )

    assert len(out) == 1 and out[0].quantity == 5


def test_scaling_to_zero_is_a_rejection(approval_settings, account) -> None:
    cycle = _Cycle()

    out = _run(
        cycle,
        [_order("NVDA", 10)],
        account,
        approval_settings,
        {"id": "__CURRENT__", "action": "scale", "scale_factor": 0.0},
    )

    assert out == []


def test_the_coordination_files_are_always_cleaned_up(approval_settings, account) -> None:
    """A surviving pending file makes the bot offer to approve a basket
    that no longer exists."""
    cycle = _Cycle()

    _run(
        cycle,
        [_order("NVDA")],
        account,
        approval_settings,
        {"id": "__CURRENT__", "action": "approve"},
    )

    assert not (approval_settings / cycle.APPROVAL_PENDING_FILE).exists()
    assert not (approval_settings / cycle.APPROVAL_DECISION_FILE).exists()


def test_a_decision_with_no_id_is_still_honoured(approval_settings, account) -> None:
    """Backward compatibility: a decision file written by an older bot
    carries no id, and refusing those would break approvals mid-upgrade."""
    cycle = _Cycle()
    orders = [_order("NVDA")]

    out = _run(cycle, orders, account, approval_settings, {"action": "approve"})

    assert out == orders
