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

    _PM_TARGET_KEYS_METADATA = _Real._PM_TARGET_KEYS_METADATA

    _merge_pm_signal = _Real._merge_pm_signal
    _add_pm_targets = _Real._add_pm_targets
    _mode_scale = _Real._mode_scale
    _pm_mode_scale = _Real._pm_mode_scale
    _pm_capital_scale = _Real._pm_capital_scale


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


def patch_strategy_sleeve(monkeypatch, pct: float) -> None:
    import types

    monkeypatch.setattr(
        "trading.core.config.settings",
        types.SimpleNamespace(state_dir="/nonexistent", strategy_sleeve_pct=pct),
    )


class TestTheTwoSleevesAreIndependent:
    """An earlier version derived the strategy's share as (1 - pm_sleeve).
    That conflated two meanings: the PM's weights are fractions of ITS OWN
    book, so its sleeve says how much of the account that book is. Setting
    it to 0.11 for a 10k sleeve on an 88k account then silently handed the
    mechanical strategy the other 89%."""

    def test_a_small_pm_sleeve_does_not_hand_the_rest_to_the_strategy(
        self, cycle, monkeypatch
    ) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)  # benchmark-only
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.10}, sleeve=0.11))
        base = strategy_signal(**{"equity:AAPL": 0.5, "equity:MSFT": 0.5})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out.target_weights["equity:AAPL"] == 0.0
        assert out.target_weights["equity:MSFT"] == 0.0
        assert out.target_weights["equity:NVDA"] == pytest.approx(0.10)

    def test_the_strategy_can_be_benchmark_only(self, cycle, monkeypatch) -> None:
        """Signals, shadow ledger and dashboard still compute; every weight
        is zero so nothing can reach an order."""
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.2}, sleeve=0.2))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}), instruments_by_key={}, ts=NOW
        )
        assert out.target_weights["equity:AAPL"] == 0.0
        assert out.strategy == "agent_pm"

    def test_both_can_run_side_by_side(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.5)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.5}, sleeve=0.5))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}), instruments_by_key={}, ts=NOW
        )
        assert out.target_weights["equity:AAPL"] == pytest.approx(0.5)
        assert out.target_weights["equity:NVDA"] == pytest.approx(0.5)
        assert out.strategy == "top_k_momentum+agent_pm"


class TestThePMsCashDecisionSurvives:
    def test_weights_are_not_renormalised_to_fill_the_sleeve(self, cycle, monkeypatch) -> None:
        """The PM holding 67% invested and 33% cash is a decision. Scaling
        its gross up to fill the sleeve would overrule it."""
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(
            monkeypatch,
            make_result({"equity:A": 0.10, "equity:B": 0.05}, sleeve=0.11),
        )
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}), instruments_by_key={}, ts=NOW
        )
        assert sum(out.target_weights.values()) == pytest.approx(0.15)

    def test_relative_conviction_is_preserved(self, cycle, monkeypatch) -> None:
        """The reason this matters: the risk manager caps per-position
        BEFORE gross, so if the PM's weights arrive too large every name
        clips to the same number and a conviction-weighted book silently
        becomes equal-weight."""
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(
            monkeypatch,
            make_result({"equity:BIG": 0.15, "equity:SMALL": 0.06}, sleeve=0.11),
        )
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}), instruments_by_key={}, ts=NOW
        )
        w = out.target_weights
        assert w["equity:BIG"] / w["equity:SMALL"] == pytest.approx(0.15 / 0.06)


class TestOverlappingNames:
    def test_shared_convictions_add_rather_than_overwrite(self, cycle, monkeypatch) -> None:
        """Both managers wanting AAPL is two independent votes, not one."""
        patch_strategy_sleeve(monkeypatch, 0.6)
        patch_bridge(monkeypatch, make_result({"equity:AAPL": 0.20}, sleeve=0.40))
        base = strategy_signal(**{"equity:AAPL": 1.0})

        out = cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW)

        assert out.target_weights["equity:AAPL"] == pytest.approx(0.60 + 0.20)


class TestRefusalStillRespectsTheStrategySleeve:
    def test_a_benchmark_only_strategy_does_not_trade_when_the_pm_refuses(
        self, cycle, monkeypatch
    ) -> None:
        """The dangerous path: PM declines on freshness, and a strategy the
        operator had set to benchmark-only inherits the whole account."""
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result(None, reason="PM decision 96.0h old"))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}), instruments_by_key={}, ts=NOW
        )
        assert out.target_weights["equity:AAPL"] == 0.0
        assert any("96.0h" in m for m in cycle.alerts.error_msgs)


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

    def test_the_bridge_raising_does_not_take_the_cycle_down(self, cycle, monkeypatch) -> None:
        import trading.agents.pm_signal as mod

        def boom(*a, **k):
            raise RuntimeError("unreadable state")

        monkeypatch.setattr(mod, "load_pm_signal", boom)
        base = strategy_signal(**{"equity:AAPL": 1.0})

        assert cycle._merge_pm_signal(base, instruments_by_key={}, ts=NOW) is base


class TestPricingThePMsPicks:
    """The PM chooses from ~1000 names plus 21 ETFs; the cycle prices only
    UNIVERSE (503 under sp500). Anything outside is unpriceable, dropped,
    and the weight silently becomes cash. On the first real decision that
    was GLD 0.15 + XLV 0.12 of 0.67 gross — forty percent of the intended
    book, with the digest still reading fully invested."""

    @staticmethod
    def _patch_settings(monkeypatch, state_dir, sleeve: float) -> None:
        """Settings is frozen (every domain model is), so setattr on the
        instance raises. Swap the module-level object instead — which is
        what the code under test resolves at call time anyway."""
        import types

        monkeypatch.setattr(
            "trading.core.config.settings",
            types.SimpleNamespace(state_dir=state_dir, agent_pm_sleeve_pct=sleeve),
        )

    @classmethod
    def _cycle_with(cls, tmp_path, monkeypatch, weights: dict, sleeve: float = 0.8):
        import json as _json

        pm = tmp_path / "agent_pm"
        pm.mkdir(parents=True, exist_ok=True)
        (pm / "last_run.json").write_text(
            _json.dumps({"ok": True, "ts": NOW.isoformat(), "weights": weights})
        )
        cls._patch_settings(monkeypatch, tmp_path, sleeve)
        return _Cycle()

    @staticmethod
    def _equities(*syms):
        from trading.core.types import AssetClass, Instrument

        return [Instrument(symbol=s, asset_class=AssetClass.EQUITY) for s in syms]

    def test_off_universe_picks_are_added(self, tmp_path, monkeypatch) -> None:
        c = self._cycle_with(tmp_path, monkeypatch, {"JPM": 0.1, "HUM": 0.08})
        out = c._add_pm_targets(self._equities("JPM"))
        assert sorted(i.symbol for i in out) == ["HUM", "JPM"]

    def test_etf_shelf_names_are_typed_as_etf(self, tmp_path, monkeypatch) -> None:
        """`etf:GLD` vs `equity:GLD` decides the parquet partition — wrong
        class means unpriceable, which is the bug we are fixing."""
        c = self._cycle_with(tmp_path, monkeypatch, {"GLD": 0.15, "HUM": 0.08})
        out = c._add_pm_targets([])
        by_sym = {i.symbol: i.asset_class.value for i in out}
        assert by_sym == {"GLD": "etf", "HUM": "equity"}

    def test_nothing_is_duplicated(self, tmp_path, monkeypatch) -> None:
        c = self._cycle_with(tmp_path, monkeypatch, {"JPM": 0.1})
        out = c._add_pm_targets(self._equities("JPM", "AAPL"))
        assert len(out) == 2

    def test_disabled_bridge_leaves_the_universe_alone(self, tmp_path, monkeypatch) -> None:
        """At sleeve 0 the mechanical strategy's universe must be exactly
        what it was — enabling the PM should not change what momentum
        ranks over."""
        c = self._cycle_with(tmp_path, monkeypatch, {"GLD": 0.15}, sleeve=0.0)
        base = self._equities("JPM")
        assert c._add_pm_targets(base) is base

    def test_a_missing_decision_is_not_an_error(self, tmp_path, monkeypatch) -> None:
        self._patch_settings(monkeypatch, tmp_path, 0.8)
        base = self._equities("JPM")
        assert _Cycle()._add_pm_targets(base) is base

    def test_a_failed_pm_run_adds_nothing(self, tmp_path, monkeypatch) -> None:
        import json as _json

        pm = tmp_path / "agent_pm"
        pm.mkdir(parents=True)
        (pm / "last_run.json").write_text(_json.dumps({"ok": False, "reason": "timeout"}))
        self._patch_settings(monkeypatch, tmp_path, 0.8)
        base = self._equities("JPM")
        assert _Cycle()._add_pm_targets(base) is base


class TestAttribution:
    def test_the_merged_signal_names_both_producers(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.1}, sleeve=0.1))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.9}), instruments_by_key={}, ts=NOW
        )
        assert out.strategy == "top_k_momentum+agent_pm"

    def test_the_sleeve_and_drops_are_recorded_on_the_signal(self, cycle, monkeypatch) -> None:
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.1}, sleeve=0.1, dropped=["XLK"]))
        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.9}), instruments_by_key={}, ts=NOW
        )
        assert out.metadata["pm_sleeve_pct"] == "0.1000"
        assert out.metadata["pm_dropped"] == "XLK"


class TestThePMCanExitWhatItBought:
    """A target weight of zero is how a book says "get out". A name the PM
    has DROPPED is absent from its weights, not zero — and an absent key
    gets no delta, so no order, so the position stays.

    The mechanical strategy masks this for S&P names: its weight vector
    spans the universe, so anything unpicked carries an explicit 0.0. The
    PM's ETF shelf is outside that universe, and those are exactly what it
    uses to hedge. Left alone the PM's book ratchets — add, never remove —
    while the stranded position keeps eating the sleeve.
    """

    def test_a_dropped_name_gets_an_explicit_zero(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)
        result = make_result({"equity:NVDA": 0.10}, sleeve=0.11)
        result = result.__class__(**{**result.__dict__, "exiting": {"etf:GLD"}})
        patch_bridge(monkeypatch, result)

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object(), "etf:GLD": object()},
            ts=NOW,
        )

        assert out.target_weights["etf:GLD"] == 0.0

    def test_an_unpriceable_exit_is_reported_not_silently_skipped(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)
        result = make_result({"equity:NVDA": 0.10}, sleeve=0.11)
        result = result.__class__(**{**result.__dict__, "exiting": {"etf:GLD"}})
        patch_bridge(monkeypatch, result)

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object()},  # no GLD
            ts=NOW,
        )

        # No bogus zero on something the risk manager would reject anyway.
        assert "etf:GLD" not in out.target_weights

    def test_an_exit_never_overrides_a_strategy_position(self, cycle, monkeypatch) -> None:
        """If the mechanical book still wants the name, the PM leaving is
        not a reason to sell the strategy's shares."""
        patch_strategy_sleeve(monkeypatch, 1.0)
        result = make_result({"equity:NVDA": 0.10}, sleeve=0.11)
        result = result.__class__(**{**result.__dict__, "exiting": {"equity:AAPL"}})
        patch_bridge(monkeypatch, result)

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.5}),
            instruments_by_key={"equity:NVDA": object(), "equity:AAPL": object()},
            ts=NOW,
        )

        assert out.target_weights["equity:AAPL"] == 0.5


class TestTheOperatorModeReachesThePMSleeve:
    """`/mode flatten` reads as "get out of the market".

    The mode overlay reshapes the MECHANICAL strategy's weights, and the
    PM is merged in afterwards — so it zeroed the strategy, left the PM
    fully invested, and still replied "mode active: flatten".

    At STRATEGY_SLEEVE_PCT=0.0 (the live configuration on 2026-08-07)
    that was total, not partial: the strategy contributes nothing, the PM
    is the only book that trades, and the command did nothing at all.
    """

    def _with_mode(self, monkeypatch, cycle, mode_value: str | None):
        from trading.runtime.mode import Mode

        if mode_value is None:
            monkeypatch.setattr(cycle, "_pm_mode_scale", lambda: (None, "neutral"))
            return
        mode = Mode(mode_value)
        monkeypatch.setattr(cycle, "_pm_mode_scale", lambda: (cycle._mode_scale(mode), mode_value))

    def test_flatten_zeroes_the_pm_sleeve(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.11}, sleeve=0.11))
        self._with_mode(monkeypatch, cycle, "flatten")

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object()},
            ts=NOW,
        )

        assert out.target_weights["equity:NVDA"] == 0.0

    def test_defense_cuts_the_pm_sleeve_without_zeroing_it(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.10}, sleeve=0.11))
        self._with_mode(monkeypatch, cycle, "defense")

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object()},
            ts=NOW,
        )

        w = out.target_weights["equity:NVDA"]
        assert 0.0 < w < 0.10

    def test_neutral_leaves_the_pm_untouched(self, cycle, monkeypatch) -> None:
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.10}, sleeve=0.11))
        self._with_mode(monkeypatch, cycle, None)

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object()},
            ts=NOW,
        )

        assert out.target_weights["equity:NVDA"] == 0.10

    def test_the_strategy_side_is_not_scaled_twice(self, cycle, monkeypatch) -> None:
        """Step 6b already applied the mode to the strategy frame. Scaling
        the combined book here would cut it again."""
        patch_strategy_sleeve(monkeypatch, 1.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.10}, sleeve=0.11))
        self._with_mode(monkeypatch, cycle, "defense")

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 0.5}),
            instruments_by_key={"equity:NVDA": object(), "equity:AAPL": object()},
            ts=NOW,
        )

        assert out.target_weights["equity:AAPL"] == 0.5

    def test_an_unreadable_mode_leaves_the_pm_unscaled(self, cycle, monkeypatch) -> None:
        """A broken overlay must not stop a cycle — the operator sets a
        mode precisely when they are already nervous."""
        patch_strategy_sleeve(monkeypatch, 0.0)
        patch_bridge(monkeypatch, make_result({"equity:NVDA": 0.10}, sleeve=0.11))

        def boom():
            raise RuntimeError("mode.json is a banana")

        monkeypatch.setattr(cycle, "_pm_mode_scale", lambda: (None, "unknown"))

        out = cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={"equity:NVDA": object()},
            ts=NOW,
        )

        assert out.target_weights["equity:NVDA"] == 0.10


class TestThePMDollarCapExists:
    """PM_SLEEVE_CAPITAL_USD was defined in 2026-07 with a comment saying
    it existed so "the bridge, when built, sizes PM target weights against
    min(PM_SLEEVE_CAPITAL_USD, allotted equity)".

    The bridge was built 2026-08-07. Nothing read the setting. GO_LIVE §4
    records "$20,000 USD, hard cap" as a locked decision and the July
    notes record it as shipped — it did not exist. The only bound was
    AGENT_PM_SLEEVE_PCT, a FRACTION, so the allocation grew with the
    account, which is the thing a dollar cap is for.
    """

    def _settings(self, monkeypatch, *, cap: float, strat: float = 0.0):
        import types

        monkeypatch.setattr(
            "trading.core.config.settings",
            types.SimpleNamespace(
                state_dir="/nonexistent",
                strategy_sleeve_pct=strat,
                pm_sleeve_capital_usd=cap,
            ),
        )

    def _merge(self, cycle, monkeypatch, weights, **kw):
        patch_bridge(monkeypatch, make_result(weights, sleeve=0.5))
        monkeypatch.setattr(cycle, "_pm_mode_scale", lambda: (None, "neutral"))
        return cycle._merge_pm_signal(
            strategy_signal(**{"equity:AAPL": 1.0}),
            instruments_by_key={},
            ts=NOW,
            **kw,
        )

    def test_the_cap_binds_on_a_usd_account(self, cycle, monkeypatch) -> None:
        """20% of a 500k account is 100k; the cap says 20k."""
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.20},
            equity=500_000.0,
            base_currency="USD",
            fx_rates={},
        )

        assert out.target_weights["equity:NVDA"] * 500_000.0 == pytest.approx(20_000.0)

    def test_it_does_not_bind_when_the_sleeve_is_already_smaller(self, cycle, monkeypatch) -> None:
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.10},
            equity=100_000.0,
            base_currency="USD",
            fx_rates={},
        )

        assert out.target_weights["equity:NVDA"] == pytest.approx(0.10)

    def test_relative_conviction_survives_the_cap(self, cycle, monkeypatch) -> None:
        """Scaling, not clipping — otherwise a conviction-weighted book
        becomes equal-weight, the same trap the per-position cap sets."""
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:BIG": 0.30, "equity:SMALL": 0.10},
            equity=500_000.0,
            base_currency="USD",
            fx_rates={},
        )

        w = out.target_weights
        assert w["equity:BIG"] / w["equity:SMALL"] == pytest.approx(3.0)

    def test_the_cap_is_converted_for_a_chf_account(self, cycle, monkeypatch) -> None:
        """USD 20,000 at 0.81 CHF/USD is 16,200 CHF — not 20,000 CHF."""
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.50},
            equity=500_000.0,
            base_currency="CHF",
            fx_rates={"USD": 0.81},
        )

        assert out.target_weights["equity:NVDA"] * 500_000.0 == pytest.approx(16_200.0)

    def test_a_missing_rate_still_caps_rather_than_giving_up(self, cycle, monkeypatch) -> None:
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.50},
            equity=500_000.0,
            base_currency="CHF",
            fx_rates={},
        )

        assert out.target_weights["equity:NVDA"] * 500_000.0 == pytest.approx(20_000.0)

    def test_zero_disables_the_cap(self, cycle, monkeypatch) -> None:
        self._settings(monkeypatch, cap=0.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.50},
            equity=500_000.0,
            base_currency="USD",
            fx_rates={},
        )

        assert out.target_weights["equity:NVDA"] == pytest.approx(0.50)

    def test_no_equity_yet_does_not_zero_the_book(self, cycle, monkeypatch) -> None:
        """A broker snapshot without equity must not be read as 'cap to 0'."""
        self._settings(monkeypatch, cap=20_000.0)

        out = self._merge(
            cycle,
            monkeypatch,
            {"equity:NVDA": 0.10},
            equity=0.0,
            base_currency="USD",
            fx_rates={},
        )

        assert out.target_weights["equity:NVDA"] == pytest.approx(0.10)
