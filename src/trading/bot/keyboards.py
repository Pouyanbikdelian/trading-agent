"""Inline keyboards, and the callback data behind them.

A tap is worth more than a typed command when the operator is on a phone:
it cannot be misspelled, it cannot be sent with the wrong argument, and it
carries the context of the message it is attached to. That last property
is also the danger.

The stale-button problem
------------------------
A Telegram inline keyboard lives forever in the chat history. Scroll back
three weeks, tap ``✅ Approve``, and without a binding token that tap
approves *whatever cycle happens to be pending right now* — a basket the
operator has never seen. So every action that touches state carries the id
of the thing it was created for, and the handler refuses when the id no
longer matches. A stale tap is answered with an explanation, never with an
order.

What deliberately has no button
-------------------------------
Nothing that submits or cancels an order: no ``/buy``, ``/sell``,
``/close``, ``/flatten``, and no ``/halt`` (which force-flattens on the
next cycle). Those stay typed. A mis-tap while the phone is in a pocket
should not be able to move money, and the friction of typing is the point.
Cycle approval is the exception that proves the rule: it is a decision the
operator is *already being asked to make*, it is bound to one specific
basket, and it still runs through the risk manager.

Wire format
-----------
Telegram caps ``callback_data`` at 64 bytes, so the encoding is terse:
``1|action|token``. Version first, so a bot restarted mid-deploy can
recognise and reject buttons minted by an older format rather than
misreading them.
"""

from __future__ import annotations

from typing import Any

CB_VERSION = "1"
_SEP = "|"
MAX_CALLBACK_BYTES = 64

# --- actions. Keep short: they share 64 bytes with the token.
ACT_APPROVE = "a"
ACT_APPROVE_SCALED = "as"
ACT_REJECT = "rj"
ACT_MODE_CONFIRM = "mc"
ACT_MODE_CANCEL = "mx"
ACT_RUN = "run"  # a read-only command, named in the token

# Read-only commands a button may run. An allowlist rather than "any
# command": it means a malformed or hostile callback cannot reach a
# handler that trades, even if the dispatch table changes later.
SAFE_COMMANDS = frozenset(
    {"/positions", "/status", "/health", "/orders", "/pending", "/regime", "/holds", "/detail"}
)


def encode(action: str, token: str = "") -> str:
    """Build callback data, raising if it will not fit Telegram's cap.

    Raising here (rather than truncating) is deliberate: a silently
    truncated token would produce a button that always looks stale.
    """
    data = f"{CB_VERSION}{_SEP}{action}{_SEP}{token}"
    if len(data.encode("utf-8")) > MAX_CALLBACK_BYTES:
        raise ValueError(f"callback data too long ({len(data)}B): {data!r}")
    return data


def decode(data: str) -> tuple[str, str] | None:
    """Parse callback data into ``(action, token)``.

    Returns None for anything unparseable or from another version — the
    caller then tells the operator to use the current message instead of
    guessing at intent.
    """
    if not data:
        return None
    parts = data.split(_SEP)
    if len(parts) != 3 or parts[0] != CB_VERSION:
        return None
    action, token = parts[1], parts[2]
    if not action:
        return None
    return action, token


def _button(text: str, action: str, token: str = "") -> dict[str, str]:
    return {"text": text, "callback_data": encode(action, token)}


def _markup(rows: list[list[dict[str, str]]]) -> dict[str, Any]:
    return {"inline_keyboard": rows}


def approval_keyboard(cycle_id: str) -> dict[str, Any]:
    """Buttons for the per-cycle approval prompt.

    ``cycle_id`` is truncated to the 8 chars the prompt already displays,
    which is what bounds the callback data. Collisions across 8 hex chars
    are not a practical concern for one operator's weekly cycles, and the
    handler re-checks against the currently-pending id anyway.
    """
    cid = str(cycle_id)[:8]
    return _markup(
        [
            [
                _button("✅ Approve", ACT_APPROVE, cid),
                _button("📉 80%", ACT_APPROVE_SCALED, f"{cid}:80"),
            ],
            [
                _button("❌ Reject", ACT_REJECT, cid),
                _button("📊 Positions", ACT_RUN, "/positions"),
            ],
        ]
    )


def sentinel_keyboard() -> dict[str, Any]:
    """Buttons for a sentinel alert.

    No trim or close button by design — see the module docstring. The
    useful actions here are "look closer" and "argue about it", both of
    which are free of the order path. ``/committee`` costs one LLM call.
    """
    return _markup(
        [
            [
                _button("🧠 Debate it", ACT_RUN, "/committee"),
                _button("📊 Positions", ACT_RUN, "/positions"),
            ]
        ]
    )


def mode_keyboard(mode: str) -> dict[str, Any]:
    """Confirm/cancel for a staged mode change, bound to that mode.

    The binding matters: staging ``defense``, changing your mind, staging
    ``bear``, then tapping the older message's ``Confirm`` must not apply
    ``defense``.
    """
    return _markup(
        [
            [
                _button("✅ Confirm", ACT_MODE_CONFIRM, str(mode)[:16]),
                _button("✖️ Cancel", ACT_MODE_CANCEL, str(mode)[:16]),
            ]
        ]
    )


def read_only_keyboard(*commands: str) -> dict[str, Any] | None:
    """A row of buttons for read-only commands, or None if none are valid."""
    row = [_button(c, ACT_RUN, c) for c in commands if c in SAFE_COMMANDS]
    return _markup([row]) if row else None
