"""Writes to the broker must never be auto-retried.

``_bounded`` reconnects and re-runs the call when it times out. That is
correct for reads — asking for the account twice is free — and wrong for
every write, because a timeout does not mean the order failed to reach
IBKR. It means we stopped waiting for the answer.

``submit_order`` has used the non-retrying ``_call_with_timeout`` since
the 2026-07-15 external review. ``convert_currency`` was still on
``_bounded`` and would re-submit an FX conversion after a timeout.

A doubled conversion is worse than a doubled equity order: it can drive
the source currency negative, and at ``MAX_MARGIN_BORROWING_PCT=0.0``
a negative balance makes the risk manager reject every later basket —
the shape of the June 2026 incident, three weeks of refused cycles.
"""

from __future__ import annotations

import inspect

from trading.execution import ibkr

#: Methods that MUTATE broker state. Retrying any of these can duplicate
#: a real-money action.
WRITE_METHODS = ("submit_order", "convert_currency")


def _source(name: str) -> str:
    return inspect.getsource(getattr(ibkr.IbkrBroker, name))


class TestNoWritePathAutoRetries:
    def test_submit_order_does_not_use_the_retrying_helper(self) -> None:
        assert "_bounded(" not in _source("submit_order")

    def test_convert_currency_does_not_use_the_retrying_helper(self) -> None:
        """The regression: an FX conversion re-sent after a timeout."""
        assert "_bounded(" not in _source("convert_currency")

    def test_every_write_uses_the_non_retrying_helper(self) -> None:
        for name in WRITE_METHODS:
            src = _source(name)
            assert "_call_with_timeout(" in src, f"{name} must bound its call"
            assert "_bounded(" not in src, (
                f"{name} uses _bounded, which reconnects and RETRIES — "
                "a timed-out write may already have reached IBKR"
            )

    def test_the_retrying_helper_still_exists_for_reads(self) -> None:
        """Guards against 'fixing' this by deleting the retry entirely —
        transient session drops are real and reads should ride through."""
        assert "_bounded" in _source("get_positions")
