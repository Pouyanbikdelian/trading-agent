"""A cycle speaks in cards, not in a stream of side notes (2026-09-26).

A normal cycle used to send 8-12 Telegram messages: the PM bridge note,
caps, the pinned-holdings note, holds and exclusions respected, the cash
trim, the approval card, the fills, then the portfolio. The side notes now
ride inside the card they explain, and fills + portfolio are one message.
"""

from __future__ import annotations

from types import SimpleNamespace

from trading.runner import NullAlerts
from trading.runner.cycle import Cycle, _cycle_note, _take_cycle_notes


def _stub(**kw):
    return SimpleNamespace(alerts=NullAlerts(), **kw)


def test_notes_are_held_for_the_card_and_taken_once() -> None:
    c = _stub(_card_notes=[])
    _cycle_note(c, "📌 Holds respected — skipped 1 order")
    _cycle_note(c, "🤖 Agent PM → market: 9 names")
    assert c.alerts.sent == []  # nothing sent yet
    block = _take_cycle_notes(c)
    assert block.startswith("\n\n🗒 *Notes*") and "Holds respected" in block and "9 names" in block
    assert _take_cycle_notes(c) == ""


def test_outside_a_cycle_a_note_is_sent_directly() -> None:
    c = _stub()
    _cycle_note(c, "hello")
    assert c.alerts.sent == [("info", "hello")]


def test_what_no_card_absorbed_goes_out_once() -> None:
    c = _stub(_card_notes=["a", "b"], _fill_summary=None)
    Cycle._flush_cycle_messages(c)
    assert len(c.alerts.sent) == 1 and "Cycle notes" in c.alerts.sent[0][1]
    assert "a" in c.alerts.sent[0][1] and "b" in c.alerts.sent[0][1]


def test_fills_and_the_new_portfolio_are_one_message() -> None:
    c = _stub(_fill_summary="📊 *Summary* — 3 execution(s)", broker=SimpleNamespace())
    snap = SimpleNamespace(equity=1000.0, cash=1000.0, positions={}, base_currency="CHF")
    Cycle._announce_post_cycle_state(c, snap)
    assert len(c.alerts.sent) == 1
    text = c.alerts.sent[0][1]
    assert text.index("3 execution(s)") < text.index("Portfolio after this cycle")
    assert c._fill_summary is None
