"""A command the operator cannot find is a command that does not exist.

``/gateway`` shipped working and invisible: ``HELP_TEXT`` is a
hand-written string with no connection to the command registry, so
registering a Spec made the command dispatch and typo-match but never
listed it. The operator read /help, did not see it, and reasonably
concluded the deploy had failed.

Two hand-maintained lists of the same thing — the drift was inevitable.
Pinning the known gap rather than demanding it be zero: the six below are
aliases and sub-operations that are deliberately not advertised, and a
test that fails today teaches people to delete tests.
"""

from __future__ import annotations

import re
from pathlib import Path

from trading.bot.telegram import HELP_TEXT

#: Deliberately unlisted: /help itself, mandate sub-ops, and aliases.
KNOWN_UNLISTED = {"/cancel_order", "/forget", "/harden", "/help", "/mandates", "/soften"}


def _registry_commands() -> set[str]:
    src = Path("src/trading/bot/registry.py").read_text()
    return set(re.findall(r'Spec\(\s*"(/[a-z_-]+)"', src))


def test_gateway_is_listed() -> None:
    """The regression."""
    assert "/gateway" in HELP_TEXT


def test_the_help_says_what_gateway_is_for() -> None:
    """'stop|start|status' alone does not tell you it frees your session."""
    line = next(ln for ln in HELP_TEXT.splitlines() if ln.startswith("/gateway"))

    assert "IBKR session" in line
    assert "halts" in line


def test_no_new_command_goes_unlisted() -> None:
    missing = _registry_commands() - {c for c in _registry_commands() if c in HELP_TEXT}

    assert missing <= KNOWN_UNLISTED, f"new commands missing from /help: {missing - KNOWN_UNLISTED}"
