"""Blending the Agent PM into the cycle's signal.

The dangerous property here is not "does the PM's view get through" — it
is that turning the bridge on must not change the SIZE of the book. One
knob (``AGENT_PM_SLEEVE_PCT``) splits the account between the mechanical
strategy and the PM; if the strategy side is not scaled down by the same
fraction, enabling the PM silently levers the account to ``1 + sleeve``.
The risk manager would catch the gross breach — but as a rejection at the
end of a cycle, not as the configuration error it actually is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from trading.core.types import Signal

NOW = datetime(2026, 8, 7, 21, 5, tzinfo=timezone.utc)


class _Alerts:
    def __init__(self) -> None:
        self.info_msgs: list[str] = []
        self.error_msgs: list[str] = []

    def info(self, m: str) -> None:
        self.info_msgs.append(m)

    def error(self, m: str) -> None:
        self.error_msgs.append(m)


class _Cycle:
    """Bare host for the method under test — the real Cycle needs a broker,
    a store and a risk manager, none of which this touches."""

    alerts: Any

    def __init__(self) -> None:
        self.alerts = _Alerts()

    from trading.runner.cycle import Cycle as _Real

    _merge_pm_signal = _Real._merge_pm_signal


def strategy_signal(**weights: float) -> Signal:
    return Signal(ts=NOW, strategy="top_k_momentum", target_weights=dict(weights))


@pytest.fixture
def cycle() -> _Cycle:
    return _Cycle()


def patch_bridge(monkeypatch, result) -> None:
    import trading.agents.pm_signal as mod

    monkeypatch.setattr(mod, "load_pm_signal", lambda *a, **k: result)


def make_result(
    weights: dict[str, float] | None,
    *,
    sleeve: float = 0.0,
    reason: str = "ok",
    dropped: list[str] | None = None,
):
    from trading.agents.pm_signal import PMSignalResult

    sig = (
        Signal(ts=NOW, strategy="agent_pm", target_weights=weights, metadata={"decided_at": "x"})
        if weights is not None
        else None
    )
    return PMSignalResult(
        sig,
        reason,
        dropped=dropped or [],
        gross=sum((weights or {}).values()),
        sleeve_pct=sleeve,
    )


class TestGrossIsPreserved:
    """The property that makes the knob safe."""

    def test_enabling_the_pm_does_not_lever_the_book(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.30}, sleeve=0.30))
        base = strategy_signal(**{"equity:AAPL": 0.5, "equity:MSFT": 0.5})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert sum(out.target_weights.values()) == pytest.approx(1.0)

    def test_the_strategy_is_scaled_by_one_minus_sleeve(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.25}, sleeve=0.25))
        base = strategy_signal(**{"equity:AAPL": 0.6})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out.target_weights["equity:AAPL"] == pytest.approx(0.45)
        assert out.target_weights["equity:NVDA"] == pytest.approx(0.25)

    def test_a_full_sleeve_hands_the_account_to_the_pm(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 1.0}, sleeve=1.0))
        base = strategy_signal(**{"equity:AAPL": 0.6, "equity:MSFT": 0.4})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out.target_weights["equity:AAPL"] == 0.0
        assert out.target_weights["equity:NVDA"] == pytest.approx(1.0)


class TestOverlappingNames:
    def test_shared_convictions_add_rather_than_overwrite(self, cycle, monkeypatch) -> None:
        """Both managers wanting AAPL is two independent votes, not one."""
        patch_bridge(monkeypatch, make_result({"equity:AAPL": 0.20}, sleeve=0.40))
        base = strategy_signal(**{"equity:AAPL": 1.0})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out.target_weights["equity:AAPL"] == pytest.approx(0.60 + 0.20)


class TestDisabledAndRefusals:
    def test_disabled_passes_the_strategy_through_untouched(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result(None, reason="PM bridge disabled (…)"))
        base = strategy_signal(**{"equity:AAPL": 1.0})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out is base
        assert cycle.alerts.error_msgs == []  # the off switch is not an event

    def test_a_refusal_is_announced(self, cycle, monkeypatch) -> None:
        """'PM held everything' and 'the bridge refused' are identical in a
        position report. The operator must be told which happened."""
        patch_bridge(monkeypatch, make_result(None, reason="PM decision 96.0h old, limit 6h"))
        base = strategy_signal(**{"equity:AAPL": 1.0})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out is base
        assert any("96.0h" in m for m in cycle.alerts.error_msgs)

    def test_the_bridge_raising_does_not_take_the_cycle_down(
        self, cycle, monkeypatch
    ) -> None:
        import trading.agents.pm_signal as mod

        def boom(*a, **k):
            raise RuntimeError("unreadable state")

        monkeypatch.setattr(mod, "load_pm_signal", boom)
        base = strategy_signal(**{"equity:AAPL": 1.0})

        assert cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW) is base


class TestAttribution:
    def test_the_merged_signal_names_both_producers(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.1}, sleeve=0.1))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.9}), instruments_by_key={}, ts=NOW
        )
        assert out.strategy == "top_k_momentum+agent_pm"

    def test_the_sleeve_and_drops_are_recorded_on_the_signal(
        self, cycle, monkeypatch
    ) -> None:
        patch_bridge(
            monkeypatch, make_result({"equity:NVDA": 0.1}, sleeve=0.1, dropped=["XLK"])
        )
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.9}), instruments_by_key={}, ts=NOW
        )
        assert out.metadata["pm_sleeve_pct"] == "0.1000"
        assert out.metadata["pm_dropped"] == "XLK"
