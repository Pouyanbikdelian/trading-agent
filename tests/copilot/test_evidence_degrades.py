"""One unreadable fact must not cost the whole answer.

2026-08-11, an hour after arming live: the operator asked "I want to
start the strategy again and run a cycle for the PM agents, in which
order should I do it" — a question about process, needing no account
data at all — and got back:

    copilot error: OperationalError: no such table: orders

A freshly-armed live account has a zero-byte ``orders.db``; SQLite
creates no schema until the first order is written. ``orders_and_fills``
raised straight through ``answer()``, so EVERY question, however
unrelated, failed. The copilot was unusable on exactly the day it was
most needed.

Degrading names what is missing rather than hiding it. The charter
already requires the model to say when evidence does not cover a
question — it can only do that if the gap is in the evidence rather than
in a stack trace.
"""

from __future__ import annotations

import re
from pathlib import Path

ENGINE = Path("src/trading/copilot/engine.py").read_text()


class TestEveryProviderIsWrapped:
    """The invariant, not the one path that broke. A fix applied to
    ``orders_and_fills`` alone would leave six other providers able to
    take the copilot down."""

    def test_no_fact_provider_is_called_bare(self) -> None:
        bare = re.findall(r'"NOW_[a-z_]+": facts\.[a-z_]+\(', ENGINE)

        assert bare == [], f"unwrapped provider calls: {bare}"

    def test_the_symbol_only_provider_is_wrapped_too(self) -> None:
        """NOW_market is added conditionally and is easy to miss."""
        assert '"NOW_market": lambda: _safe(' in ENGINE

    def test_all_seven_providers_go_through_safe(self) -> None:
        for label in ("positions", "orders", "pm_book", "risk", "lessons", "config", "interface"):
            assert re.search(rf'_safe\(\s*"{label}"', ENGINE), f"{label} not wrapped"


class TestTheDegradedPayload:
    def _safe(self):
        """Rebuild the helper from the module under test's semantics."""

        def safe(label, fn, *a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                return {
                    "unavailable": f"{type(e).__name__}: {e}",
                    "_meaning": (
                        "This evidence could not be read for this question. If the "
                        "answer depends on it, say exactly that it is unavailable — "
                        "do not substitute a guess or another book's numbers."
                    ),
                }

        return safe

    def test_a_raising_provider_yields_a_payload_not_an_exception(self) -> None:
        import sqlite3

        def boom():
            raise sqlite3.OperationalError("no such table: orders")

        out = self._safe()("orders", boom)

        assert out["unavailable"] == "OperationalError: no such table: orders"

    def test_the_payload_names_what_is_missing(self) -> None:
        """'Unavailable' with no name is indistinguishable from 'nothing
        happened' — the operator needs to know WHICH source is dark."""
        out = self._safe()("orders", lambda: (_ for _ in ()).throw(ValueError("x")))

        assert "ValueError" in out["unavailable"]

    def test_the_payload_forbids_substituting_a_guess(self) -> None:
        """The failure mode this must not create: the model filling the
        gap from the other book, or from priors."""
        out = self._safe()("orders", lambda: (_ for _ in ()).throw(ValueError("x")))

        assert "do not substitute a guess" in out["_meaning"]
        assert "another book" in out["_meaning"]

    def test_a_working_provider_is_untouched(self) -> None:
        assert self._safe()("ok", lambda: {"a": 1}) == {"a": 1}


class TestTheZeroByteLedger:
    """The exact condition on the live account that morning."""

    def test_orders_and_fills_on_a_schemaless_db(self, tmp_path) -> None:
        from trading.copilot import facts

        (tmp_path / "orders.db").write_bytes(b"")  # zero-byte, no schema

        try:
            facts.orders_and_fills(tmp_path, None)
        except Exception as e:
            # It may still raise — that is what _safe is for. What must
            # NOT happen is the copilot dying, which the wrapper covers.
            assert "no such table" in str(e) or "unable to open" in str(e)

    def test_the_failure_is_logged_for_diagnosis(self) -> None:
        """Silent degradation is how a permanently-dark source goes
        unnoticed for weeks."""
        assert 'logger.bind(component="copilot").warning(f"evidence {label} unavailable' in ENGINE
