r"""Daily broker reconciliation: make the local ledger agree with IBKR.

Why this exists (2026-09-23). Fills were reconciled only inside a cycle
that submitted orders, from ib-async's in-session ``fills()`` cache. The
gateway restarts every night, so the common case — an order queued Friday
after the close that fills Monday at the open — was seen by nobody. Five
August rows sat ``pending`` for over a month; realized P&L, the episode
reconstruction and the stale-order alarm all read from that ledger.

What one pass does, reading only (it never submits, cancels or amends):

1. Asks the gateway for the executions it still holds (``reqExecutions``;
   falls back to the session cache on brokers without it) and records any
   that belong to a still-open local order, then settles that order's
   status from the fills actually recorded.
2. Asks for completed orders (``reqCompletedOrders``). A local open order
   the broker reports cancelled / expired / inactive becomes ``cancelled``
   — broker evidence, not inference. The broker's permanent id is recorded
   for every matched order.
3. Reports what it could not settle: orders the broker calls filled whose
   executions are no longer visible (needs a statement, not a guess), and
   local open orders the broker neither works nor lists as completed.

It deliberately does NOT mark anything ``unreconciled``: retiring a row
whose outcome is unknown stays a human decision (``/orders resolve``).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from trading.core.logging import logger
from trading.core.types import BrokerOrderReport, Fill, OrderStatus
from trading.execution.store import OrderStore

#: Never ask further back than this. IBKR's own execution history is days,
#: so a wider window only costs time.
MAX_LOOKBACK = timedelta(days=14)


@dataclass
class ReconcileReport:
    fills_recorded: int = 0
    settled: dict[str, str] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    broker_ids_recorded: int = 0
    filled_without_executions: list[str] = field(default_factory=list)
    unseen_open: list[str] = field(default_factory=list)
    unmatched_executions: int = 0
    ambiguous: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.fills_recorded or self.settled or self.cancelled)

    @property
    def needs_attention(self) -> bool:
        return bool(self.filled_without_executions or self.unseen_open or self.ambiguous)

    def summary(self) -> str:
        lines = ["🧾 *Broker reconciliation*"]
        if self.fills_recorded:
            lines.append(f"recorded {self.fills_recorded} fill(s)")
        for coid, status in sorted(self.settled.items()):
            lines.append(f"`{coid}` → {status}")
        for coid in self.cancelled:
            lines.append(f"`{coid}` → cancelled (broker)")
        if self.filled_without_executions:
            lines.append(
                "⚠️ broker reports FILLED but the executions are no longer visible — "
                "check the IBKR statement: "
                + ", ".join(f"`{c}`" for c in self.filled_without_executions)
            )
        if self.unseen_open:
            lines.append(
                "⚠️ open locally, unknown to the broker (not working, not completed): "
                + ", ".join(f"`{c}`" for c in self.unseen_open)
                + ". `/orders resolve` retires them once you have checked."
            )
        if self.ambiguous:
            lines.append(
                "⚠️ several broker orders share one id; left untouched: "
                + ", ".join(f"`{c}`" for c in self.ambiguous)
            )
        if len(lines) == 1:
            lines.append("ledger already agrees with the broker")
        return "\n".join(lines)


def _call_optional(broker: Any, name: str, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
    fn: Callable[..., Any] | None = getattr(broker, name, None)
    if not callable(fn):
        return False, None
    return True, fn(*args, **kwargs)


def _filled_locally(store: OrderStore, coid: str) -> float:
    return float(sum(f.quantity for f in store.load_fills(client_order_id=coid)))


def reconcile_with_broker(
    store: OrderStore,
    broker: Any,
    *,
    now: datetime,
    lookback: timedelta = MAX_LOOKBACK,
    apply_lock: Callable[[], AbstractContextManager[Any]] = nullcontext,
) -> ReconcileReport:
    """One read-only reconciliation pass. Broker read errors propagate.

    All broker reads happen first, outside ``apply_lock``; only the ledger
    writes run inside it. The runner passes the execution lock, so a manual
    /close is never kept waiting behind a slow gateway. Nothing a concurrent
    submission creates can be mis-settled: fills are applied only to orders
    that were open before the reads, and a cancel only on broker evidence
    naming that exact order.

    A failed read fails the pass loudly rather than settle the ledger from
    a partial view of the broker.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    report = ReconcileReport()
    open_rows = store.open_orders()
    if not open_rows:
        return report
    open_by_id = {order.client_order_id: order for order, _status, _bid in open_rows}

    oldest = min(order.created_at for order in open_by_id.values())
    since = max(oldest - timedelta(minutes=5), now - lookback)

    # ---- reads (no lock)
    has_exec, fills = _call_optional(broker, "get_executions", since=since)
    if has_exec:
        report.sources.append("executions")
    else:
        fills = broker.get_fills(since=since)
        report.sources.append("session_fills")
    has_completed, completed = _call_optional(broker, "get_completed_orders")
    completed_by_id: dict[str, BrokerOrderReport] = {}
    duplicated: set[str] = set()
    if has_completed:
        report.sources.append("completed_orders")
        for rep in completed or []:
            if not isinstance(rep, BrokerOrderReport):
                continue
            if rep.client_order_id in completed_by_id:
                duplicated.add(rep.client_order_id)
            completed_by_id[rep.client_order_id] = rep
    for coid in duplicated:
        completed_by_id.pop(coid, None)
    report.ambiguous = sorted(duplicated & set(open_by_id))
    working_ids = {str(o.client_order_id) for o in (broker.get_open_orders() or [])}

    # ---- writes (under the lock)
    with apply_lock():
        _apply(store, report, open_by_id, fills or [], completed_by_id, working_ids, has_completed)

    logger.bind(component="broker_reconcile").info(
        f"reconciled: fills={report.fills_recorded} settled={len(report.settled)} "
        f"cancelled={len(report.cancelled)} unseen={len(report.unseen_open)} "
        f"filled_without_exec={len(report.filled_without_executions)} "
        f"ambiguous={len(report.ambiguous)} sources={report.sources}"
    )
    return report


def _apply(
    store: OrderStore,
    report: ReconcileReport,
    open_by_id: dict[str, Any],
    fills: list[Any],
    completed_by_id: dict[str, BrokerOrderReport],
    working_ids: set[str],
    has_completed: bool,
) -> None:
    touched: set[str] = set()
    for fill in fills:
        if not isinstance(fill, Fill):
            continue
        if fill.order_id not in open_by_id:
            report.unmatched_executions += 1
            continue
        before = len(store.load_fills(client_order_id=fill.order_id))
        store.save_fill(fill, client_order_id=fill.order_id)
        if len(store.load_fills(client_order_id=fill.order_id)) > before:
            report.fills_recorded += 1  # new executions only, not re-reads
            touched.add(fill.order_id)
    for coid in sorted(touched):
        status = store.settle_status(coid)
        if status is not None:
            report.settled[coid] = status.value

    still_open = {order.client_order_id for order, _s, _b in store.open_orders()}
    for coid in sorted(open_by_id):
        rep = completed_by_id.get(coid)
        if rep is not None and rep.broker_order_id:
            store.record_broker_order_id(coid, rep.broker_order_id)
            report.broker_ids_recorded += 1
        if coid not in still_open:
            continue  # settled by fills above
        if coid in report.ambiguous:
            continue
        if rep is not None and rep.is_cancelled:
            if rep.filled_quantity > _filled_locally(store, coid) + 1e-6:
                # Partly executed before the cancel, and those executions
                # are no longer visible: cancelling would hide real shares.
                report.filled_without_executions.append(coid)
                continue
            store.update_status(coid, OrderStatus.CANCELLED)
            report.cancelled.append(coid)
            report.settled.pop(coid, None)
            continue
        if rep is not None and rep.is_filled:
            report.filled_without_executions.append(coid)
            continue
        if coid not in working_ids and has_completed and rep is None:
            report.unseen_open.append(coid)
