"""Noticing a problem must not become the problem.

``preflight_unheld`` raised when it found a position neither held nor
acknowledged. Inside a service with ``restart: unless-stopped`` that is
not a refusal, it is a loop: exit, restart, fail identically, and one
CRITICAL message per attempt. It happened three times on 2026-08-11 —
after the first live cycle bought the PM basket, then again the moment
the operator deliberately ``/unhold``ed WMT.

The second-order harm was worse than the noise. Refusing to start takes
down the position guards protecting every OTHER name, so one unprotected
position disarmed the entire book.

Halting gets what the raise was reaching for — nothing can trade, guard
exits included, since they route through the same halt-aware pipeline —
while the runner stays up, observable, and able to receive the ``/hold``
or ``/resume`` that clears it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path("src/trading/runner/runner.py").read_text()


class TestItNoLongerRaises:
    def test_the_refusal_is_gone(self) -> None:
        assert "refusing to start live with unprotected positions" not in RUNNER

    def test_it_halts_instead(self) -> None:
        body = RUNNER[RUNNER.index("def preflight_unheld") :][:4000]

        assert "set_halted(" in body
        assert "halted=True" in body

    def test_the_reason_names_the_symbols(self) -> None:
        """'/status' must explain itself without a log dig."""
        body = RUNNER[RUNNER.index("def preflight_unheld") :][:4000]

        assert 'reason=f"unprotected positions: {unheld}"' in body

    def test_the_alert_still_fires(self) -> None:
        body = RUNNER[RUNNER.index("def preflight_unheld") :][:4000]

        assert "alerts.critical(format_unheld_alert(result))" in body


class TestTheSquelch:
    """One message per NEW situation, not one per restart."""

    class _R:
        from trading.runner.runner import Runner as _Real

        _announce_unheld = _Real._announce_unheld
        UNHELD_ALERT_FILE = _Real.UNHELD_ALERT_FILE

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        monkeypatch.setattr("trading.runner.runner.settings", SimpleNamespace(state_dir=tmp_path))
        return self._R(), tmp_path

    def test_the_first_time_announces(self, runner) -> None:
        r, _ = runner

        assert r._announce_unheld({"WMT"}) is True

    def test_a_restart_with_the_same_set_is_silent(self, runner) -> None:
        """The regression: three identical messages were three restarts."""
        r, _ = runner
        r._announce_unheld({"WMT"})

        assert r._announce_unheld({"WMT"}) is False
        assert r._announce_unheld({"WMT"}) is False

    def test_a_new_unprotected_position_re_announces(self, runner) -> None:
        """A position that appeared is new information and must be heard."""
        r, _ = runner
        r._announce_unheld({"WMT"})

        assert r._announce_unheld({"WMT", "TSLA"}) is True

    def test_a_shrinking_set_re_announces(self, runner) -> None:
        """Going from two to one is also a change of state — and the
        operator may have expected it to reach zero."""
        r, _ = runner
        r._announce_unheld({"WMT", "TSLA"})

        assert r._announce_unheld({"WMT"}) is True

    def test_order_does_not_matter(self, runner) -> None:
        r, _ = runner
        r._announce_unheld({"WMT", "TSLA"})

        assert r._announce_unheld({"TSLA", "WMT"}) is False

    def test_the_record_is_readable(self, runner) -> None:
        r, tmp = runner
        r._announce_unheld({"WMT", "TSLA"})

        payload = json.loads((tmp / r.UNHELD_ALERT_FILE).read_text())
        assert payload["symbols"] == ["TSLA", "WMT"]

    def test_an_unwritable_state_dir_still_announces(self, tmp_path, monkeypatch) -> None:
        """Failing to remember must not mean failing to warn."""
        monkeypatch.setattr(
            "trading.runner.runner.settings",
            SimpleNamespace(state_dir=Path("/proc/nonexistent/nope")),
        )

        assert self._R()._announce_unheld({"WMT"}) is True


def test_a_clean_account_returns_early() -> None:
    """ok=True must not halt anything."""
    body = RUNNER[RUNNER.index("def preflight_unheld") :][:4000]
    i_ok = body.index('if result["ok"]:')
    i_halt = body.index("set_halted(")

    assert i_ok < i_halt
    assert "return" in body[i_ok : i_ok + 60]
