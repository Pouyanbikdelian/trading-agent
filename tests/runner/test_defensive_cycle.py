"""A halt blocks new exposure — it does not blind the desk.

These tests pin the seam added on 2026-08-19: a ``/cycle`` run while the
risk manager is halted still builds the real basket, strips it to the orders
that provably lower exposure, and asks the operator to approve it. The
guarantees worth breaking a build over are:

* buys never survive the filter, whatever the PM proposed;
* the operator's approval is mandatory, even with REQUIRE_CYCLE_APPROVAL off;
* the reduce-only proof is repeated against a *fresh* broker snapshot after
  the approval wait, because the book moves while a human is reading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading.core.config import settings
from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)
from trading.runner.cycle import Cycle

NOW = datetime(2026, 8, 19, 19, 0, tzinfo=timezone.utc)


class _Alerts:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, msg: str, **_kw: object) -> None:
        self.messages.append(msg)

    info = warning = error = critical = _record

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


def _inst(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        currency="USD",
        exchange="SMART",
    )


def _order(symbol: str, side: Side, qty: float) -> Order:
    return Order(
        client_order_id=f"{symbol}-{side.value}",
        instrument=_inst(symbol),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW,
    )


def _position(symbol: str, qty: float) -> Position:
    return Position(instrument=_inst(symbol), quantity=qty, avg_price=100.0)


def _account(*positions: Position) -> SimpleNamespace:
    return SimpleNamespace(
        ts=NOW,
        equity=86_087.26,
        cash=1_000.0,
        base_currency="CHF",
        positions={p.instrument.key: p for p in positions},
    )


@pytest.fixture
def cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cycle:
    monkeypatch.setattr(
        "trading.core.config.settings", settings.model_copy(update={"state_dir": tmp_path})
    )
    obj = Cycle.__new__(Cycle)
    obj.alerts = _Alerts()
    obj._working = []
    obj._working_orders = lambda: list(obj._working)  # type: ignore[method-assign]
    return obj


class TestThePlannerKeepsOnlyReductions:
    def test_a_rotation_basket_is_reduced_to_its_exits(self, cycle: Cycle) -> None:
        orders = [
            _order("GLD", Side.BUY, 200),
            _order("AMD", Side.SELL, 80),
            _order("SNDK", Side.SELL, 25),
        ]
        account = _account(_position("AMD", 160), _position("SNDK", 25))

        result, reason = cycle._plan_defensive_reduction(orders, account=account)

        assert result is not None
        assert [o.instrument.symbol for o in result.kept] == ["AMD", "SNDK"]
        assert result.dropped[0][0] == "GLD"
        assert reason == "reduce-only subset available"
        # The plan came from the review-only sizing path; those display ids
        # must never reach the broker or the ledger.
        assert not any(o.client_order_id.startswith("review-") for o in result.kept)

    def test_a_pure_buy_basket_yields_nothing_and_says_why(self, cycle: Cycle) -> None:
        result, reason = cycle._plan_defensive_reduction(
            [_order("GLD", Side.BUY, 200)], account=_account(_position("AMD", 160))
        )

        assert result is not None and not result.has_orders
        assert "opened or increased" in reason

    def test_a_flat_account_has_nothing_to_reduce(self, cycle: Cycle) -> None:
        result, reason = cycle._plan_defensive_reduction(
            [_order("GLD", Side.BUY, 200)], account=_account()
        )

        assert result is None
        assert "no positions" in reason

    def test_the_operator_can_switch_the_whole_path_off(
        self, cycle: Cycle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "trading.core.config.settings",
            settings.model_copy(update={"allow_defensive_cycle_when_halted": False}),
        )

        result, reason = cycle._plan_defensive_reduction(
            [_order("AMD", Side.SELL, 80)], account=_account(_position("AMD", 160))
        )

        assert result is None
        assert "disabled" in reason

    def test_an_unreadable_working_order_book_refuses_to_guess(self, cycle: Cycle) -> None:
        def _boom() -> list[Order]:
            raise RuntimeError("gateway is wedged")

        cycle._working_orders = _boom  # type: ignore[method-assign]

        result, reason = cycle._plan_defensive_reduction(
            [_order("AMD", Side.SELL, 80)], account=_account(_position("AMD", 160))
        )

        assert result is None
        assert "working orders could not be read" in reason


class TestTheFinalRecheckUsesAFreshBook:
    def test_a_position_closed_during_the_approval_wait_is_dropped(self, cycle: Cycle) -> None:
        """A trailing-stop guard can exit the name while the operator reads."""
        cycle._fetch_account = lambda _ts: _account(_position("SNDK", 25))  # type: ignore[method-assign]
        cycle._clock = lambda: NOW  # type: ignore[method-assign]

        final = cycle._reverify_reduce_only(
            [_order("AMD", Side.SELL, 80), _order("SNDK", Side.SELL, 25)]
        )

        assert [o.instrument.symbol for o in final] == ["SNDK"]
        assert "final reduce-only re-check" in cycle.alerts.text

    def test_a_partially_filled_position_resizes_the_order_down(self, cycle: Cycle) -> None:
        cycle._fetch_account = lambda _ts: _account(_position("AMD", 30))  # type: ignore[method-assign]
        cycle._clock = lambda: NOW  # type: ignore[method-assign]

        final = cycle._reverify_reduce_only([_order("AMD", Side.SELL, 80)])

        assert [(o.instrument.symbol, o.quantity) for o in final] == [("AMD", 30)]

    def test_an_unreachable_broker_submits_nothing(self, cycle: Cycle) -> None:
        def _boom(_ts: datetime) -> object:
            raise RuntimeError("no broker session")

        cycle._fetch_account = _boom  # type: ignore[method-assign]
        cycle._clock = lambda: NOW  # type: ignore[method-assign]

        assert cycle._reverify_reduce_only([_order("AMD", Side.SELL, 80)]) == []
        assert "nothing was sent" in cycle.alerts.text


class TestTheDefensiveCycleItself:
    """The submission contract: approval first, halted submit second, in that order."""

    def _rig(self, cycle: Cycle, *, approved: list[Order] | None, live: list[Position]):
        seen: dict[str, object] = {}

        def _approval(orders, account, last_prices, **kw):
            seen["approval_orders"] = list(orders)
            seen["defensive_flag"] = kw.get("defensive")
            seen["gate_label"] = kw.get("defensive_gate")
            seen["offered_picks"] = kw.get("candidates")
            return list(orders) if approved is None else approved

        def _submit(orders, prices, ts_start, *, allow_when_halted=False):
            seen["submitted"] = list(orders)
            seen["allow_when_halted"] = allow_when_halted
            return len(orders), [], False, False

        cycle.risk_manager = SimpleNamespace(is_halted=lambda: True)  # type: ignore[assignment]
        cycle._request_cycle_approval = _approval  # type: ignore[method-assign]
        cycle._submit_and_reconcile = _submit  # type: ignore[method-assign]
        cycle._fetch_account = lambda _ts: _account(*live)  # type: ignore[method-assign]
        cycle._clock = lambda: NOW  # type: ignore[method-assign]
        cycle._elapsed_ms = lambda _ts: 1000.0  # type: ignore[method-assign]
        cycle._announce_fills = lambda _fills: None  # type: ignore[method-assign]
        return seen

    def _run(self, cycle: Cycle, orders: list[Order], account):
        from trading.core.types import RiskDecision, Signal

        return cycle._run_defensive_cycle(
            ts_start=NOW,
            intraday=RiskDecision(action="halt", reason="daily loss -1.09% breaches limit -0.60%"),
            decisions=[],
            orders=orders,
            dropped=[("GLD", "no live position — this order would open new exposure")],
            clamped=[],
            account=account,
            prices=None,
            last_prices={},
            fx_rates={},
            signal=Signal(ts=NOW, strategy="agent_pm", target_weights={}),
            held_symbols=set(),
        )

    def test_an_approved_basket_submits_with_the_halted_exemption(self, cycle: Cycle) -> None:
        orders = [_order("AMD", Side.SELL, 80)]
        seen = self._rig(cycle, approved=None, live=[_position("AMD", 160)])

        report = self._run(cycle, orders, _account(_position("AMD", 160)))

        assert seen["defensive_flag"] is True
        assert seen["allow_when_halted"] is True
        assert [o.instrument.symbol for o in seen["submitted"]] == ["AMD"]
        assert report.status == "ok"
        assert report.orders_submitted == 1
        # The operator is told the halt is still in force after a reduction.
        assert "remains halted" in cycle.alerts.text

    def test_a_rejected_basket_sends_nothing(self, cycle: Cycle) -> None:
        seen = self._rig(cycle, approved=[], live=[_position("AMD", 160)])

        report = self._run(cycle, [_order("AMD", Side.SELL, 80)], _account(_position("AMD", 160)))

        assert "submitted" not in seen
        assert report.status == "no_orders"
        assert report.orders_submitted == 0

    def test_the_book_emptying_mid_approval_stands_the_cycle_down(self, cycle: Cycle) -> None:
        seen = self._rig(cycle, approved=None, live=[])

        report = self._run(cycle, [_order("AMD", Side.SELL, 80)], _account(_position("AMD", 160)))

        assert "submitted" not in seen
        assert report.status == "no_orders"
        assert "stood down" in cycle.alerts.text

    def test_no_alternate_basket_controls_are_offered_while_halted(self, cycle: Cycle) -> None:
        """`/pick` rebuilds through signal_to_orders, which a halt refuses anyway."""
        seen = self._rig(cycle, approved=None, live=[_position("AMD", 160)])

        self._run(cycle, [_order("AMD", Side.SELL, 80)], _account(_position("AMD", 160)))

        assert seen["offered_picks"] is None


class TestItReportsTheRealReasonForRefusing:
    """ "Halted" is a specific claim, not a synonym for "cannot trade".

    On 2026-08-19 the operator had already `/resume`d. The gate refused
    because no verified NYSE-open baseline existed yet — and the card
    told him execution was halted. Reporting the wrong reason for a
    refusal is the exact defect the rest of this work removed.
    """

    def _rig(self, cycle: Cycle, *, halted: bool):
        seen: dict[str, object] = {}
        cycle.risk_manager = SimpleNamespace(is_halted=lambda: halted)  # type: ignore[assignment]

        def _approval(orders, account, last_prices, **kw):
            seen["gate_label"] = kw.get("defensive_gate")
            return []

        cycle._request_cycle_approval = _approval  # type: ignore[method-assign]
        cycle._elapsed_ms = lambda _ts: 1000.0  # type: ignore[method-assign]
        cycle._clock = lambda: NOW  # type: ignore[method-assign]
        return seen

    def _run(self, cycle: Cycle, reason: str):
        from trading.core.types import RiskDecision, Signal

        return cycle._run_defensive_cycle(
            ts_start=NOW,
            intraday=RiskDecision(action="reject", reason=reason),
            decisions=[],
            orders=[_order("AMD", Side.SELL, 6)],
            dropped=[],
            clamped=[],
            account=_account(_position("AMD", 8)),
            prices=None,
            last_prices={},
            fx_rates={},
            signal=Signal(ts=NOW, strategy="agent_pm", target_weights={}),
            held_symbols=set(),
        )

    def test_a_gated_but_unhalted_account_is_not_called_halted(self, cycle: Cycle) -> None:
        seen = self._rig(cycle, halted=False)

        self._run(cycle, "daily-loss baseline unavailable for the current NYSE session")

        assert seen["gate_label"] == "execution is gated"
        assert "execution is halted" not in cycle.alerts.text
        # The actual reason still reaches the operator verbatim.
        assert "baseline unavailable" in cycle.alerts.text

    def test_a_genuinely_halted_account_still_says_halted(self, cycle: Cycle) -> None:
        seen = self._rig(cycle, halted=True)

        self._run(cycle, "daily loss -1.09% breaches limit -0.60%")

        assert seen["gate_label"] == "execution is halted"
        assert "execution is halted" in cycle.alerts.text
