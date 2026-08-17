"""Pre-arm check for positions nothing is protecting.

2026-08-07, seven minutes into the first live session: the position
guards sold 63 shares of VST at ~139.12. The guards apply to every
position in the account regardless of who opened it — on paper they had
only ever seen system positions, so nothing had made them visible. The
runbook step that would have prevented it ("resolve WMT/VST with /hold")
was dropped when the scope narrowed to PM-only.
"""

from __future__ import annotations

from types import SimpleNamespace

from trading.runner.holds import save_holds
from trading.runtime.broker_ready import (
    check_unheld_positions,
    format_unheld_alert,
    save_unheld_ack,
)


def _pos(symbol: str, qty: float = 10.0):
    return SimpleNamespace(instrument=SimpleNamespace(symbol=symbol), quantity=qty)


class _Broker:
    def __init__(self, *positions, exc: Exception | None = None) -> None:
        self._positions = list(positions)
        self._exc = exc

    def get_positions(self):
        if self._exc:
            raise self._exc
        return self._positions


def test_a_personal_position_is_flagged(tmp_path) -> None:
    result = check_unheld_positions(_Broker(_pos("VST"), _pos("WMT")), state_dir=tmp_path)
    assert not result["ok"]
    assert result["unheld"] == ["VST", "WMT"]


def test_hold_clears_it(tmp_path) -> None:
    save_holds(tmp_path, {"VST", "WMT"})
    result = check_unheld_positions(_Broker(_pos("VST"), _pos("WMT")), state_dir=tmp_path)
    assert result["ok"]
    assert result["held"] == ["VST", "WMT"]


def test_partial_holds_still_flag_the_rest(tmp_path) -> None:
    save_holds(tmp_path, {"VST"})
    result = check_unheld_positions(_Broker(_pos("VST"), _pos("WMT")), state_dir=tmp_path)
    assert not result["ok"]
    assert result["unheld"] == ["WMT"]


def test_acknowledgement_clears_it(tmp_path) -> None:
    save_unheld_ack(tmp_path, {"VST"})
    result = check_unheld_positions(_Broker(_pos("VST")), state_dir=tmp_path)
    assert result["ok"]
    assert result["acked"] == ["VST"]


def test_a_new_position_is_not_covered_by_an_old_acknowledgement(tmp_path) -> None:
    """The acknowledgement records a symbol SET, not a blanket waiver —
    otherwise one 'yes' silently covers every position opened afterwards."""
    save_unheld_ack(tmp_path, {"VST"})
    result = check_unheld_positions(_Broker(_pos("VST"), _pos("NVDA")), state_dir=tmp_path)
    assert not result["ok"]
    assert result["unheld"] == ["NVDA"]


def test_closed_positions_are_ignored(tmp_path) -> None:
    result = check_unheld_positions(_Broker(_pos("VST", 0.0)), state_dir=tmp_path)
    assert result["ok"]


def test_short_positions_count(tmp_path) -> None:
    result = check_unheld_positions(_Broker(_pos("VST", -10.0)), state_dir=tmp_path)
    assert not result["ok"]


def test_symbols_are_matched_case_insensitively(tmp_path) -> None:
    save_holds(tmp_path, {"vst"})
    assert check_unheld_positions(_Broker(_pos("VST")), state_dir=tmp_path)["ok"]


def test_an_unreachable_broker_does_not_block_startup(tmp_path) -> None:
    """A gate that fires on a network blip is a gate that gets removed."""
    result = check_unheld_positions(_Broker(exc=ConnectionError("no gateway")), state_dir=tmp_path)
    assert result["ok"]
    assert result["reachable"] is False
    assert "skipped" in result["reason"]


def test_strict_preflight_requires_a_real_position_read(tmp_path) -> None:
    """The live bootstrap must not schedule execution on an unknown book."""
    result = check_unheld_positions(
        _Broker(exc=ConnectionError("no gateway")),
        state_dir=tmp_path,
        require_reachable=True,
    )

    assert result["ok"] is False
    assert result["reachable"] is False
    assert result["unheld"] == []


def test_strict_preflight_prefers_the_nonrecovering_position_reader(tmp_path) -> None:
    """IBKR bootstrap must not route its read through auto-recovery."""

    class _StrictBroker:
        def __init__(self) -> None:
            self.strict_calls = 0
            self.normal_calls = 0

        def get_positions(self):
            self.normal_calls += 1
            raise AssertionError("strict bootstrap must not use the normal reader")

        def get_positions_strict(self):
            self.strict_calls += 1
            return [_pos("VST")]

    broker = _StrictBroker()
    result = check_unheld_positions(broker, state_dir=tmp_path, require_reachable=True)

    assert result["reachable"] is True
    assert result["unheld"] == ["VST"]
    assert broker.strict_calls == 1
    assert broker.normal_calls == 0


def test_default_preflight_keeps_the_legacy_position_reader(tmp_path) -> None:
    """The nonrecovering reader is opt-in rather than a behavior change."""

    class _BrokerWithStrict(_Broker):
        def __init__(self) -> None:
            super().__init__(_pos("VST"))
            self.strict_calls = 0

        def get_positions_strict(self):
            self.strict_calls += 1
            raise AssertionError("legacy preflight should not use the strict reader")

    broker = _BrokerWithStrict()
    result = check_unheld_positions(broker, state_dir=tmp_path)

    assert result["unheld"] == ["VST"]
    assert broker.strict_calls == 0


def test_empty_account_is_fine(tmp_path) -> None:
    assert check_unheld_positions(_Broker(), state_dir=tmp_path)["ok"]


def test_the_alert_names_both_paths_and_the_fix() -> None:
    msg = format_unheld_alert({"unheld": ["VST"]})
    assert "VST" in msg
    # Disabling the guards is the tempting half-fix; the message must say
    # why it is not enough.
    assert "rebalance" in msg and "guards" in msg
    assert "GUARDS_ENABLED=false" in msg
    assert "/hold" in msg


def test_saving_an_acknowledgement_replaces_rather_than_merges(tmp_path) -> None:
    save_unheld_ack(tmp_path, {"VST"})
    save_unheld_ack(tmp_path, {"WMT"})
    result = check_unheld_positions(_Broker(_pos("VST"), _pos("WMT")), state_dir=tmp_path)
    assert result["unheld"] == ["VST"]
