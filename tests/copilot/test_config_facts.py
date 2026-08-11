"""The copilot must be able to answer configuration questions.

It could cite positions, orders, fills, the PM book, lessons and halt
state — and had NO source at all for configuration. Asked "what's our
guard %" or "how big is the PM sleeve", it answered from the model's
priors, which is a polite way of saying it invented them.

Everything here is read from the LIVE objects (Settings, the guards' own
env helpers, mode.json). A hand-maintained summary would become exactly
the thing the 2026-08-07 audit kept finding — documentation describing
wiring that does not exist — except the copilot would state it with
confidence to an operator making a decision.
"""

from __future__ import annotations

from pathlib import Path

from trading.copilot import facts


def _cfg(monkeypatch, **env) -> dict:
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from trading.core.config import get_settings

    get_settings.cache_clear()
    try:
        return facts.config_now(Path("/nonexistent"))
    finally:
        get_settings.cache_clear()


class TestItReadsTheLiveSettings:
    def test_sleeves_come_from_settings_not_a_copy(self, monkeypatch) -> None:
        c = _cfg(monkeypatch, AGENT_PM_SLEEVE_PCT=0.11, STRATEGY_SLEEVE_PCT=0.0)

        assert c["who_trades"]["agent_pm_sleeve_pct"] == 0.11
        assert c["who_trades"]["strategy_sleeve_pct"] == 0.0

    def test_a_changed_setting_is_reflected_immediately(self, monkeypatch) -> None:
        """The whole point: no cached snapshot to go stale."""
        assert _cfg(monkeypatch, AGENT_PM_SLEEVE_PCT=0.25)["who_trades"][
            "agent_pm_sleeve_pct"
        ] == 0.25

    def test_risk_limits_are_present(self, monkeypatch) -> None:
        c = _cfg(monkeypatch, MAX_POSITION_PCT=0.03, MAX_DRAWDOWN_PCT=0.02)

        assert c["risk_limits"]["max_position_pct"] == 0.03
        assert c["risk_limits"]["max_drawdown_pct"] == 0.02

    def test_the_dollar_cap_is_reported(self, monkeypatch) -> None:
        assert _cfg(monkeypatch, PM_SLEEVE_CAPITAL_USD=12000)["who_trades"][
            "pm_sleeve_capital_usd"
        ] == 12000.0


class TestGuardSettings:
    def test_guard_numbers_come_from_the_guards_own_env_helpers(self, monkeypatch) -> None:
        c = _cfg(monkeypatch, GUARDS_ENABLED="true", GUARD_TRAIL_MIN_PCT=8, GUARD_ATR_MULT=3.0)

        g = c["position_guards"]
        assert g["enabled"] is True
        assert g["trail_min_pct"] == 8.0
        assert g["atr_mult"] == 3.0

    def test_the_ratchet_is_reported_as_on_when_both_knobs_are_set(self, monkeypatch) -> None:
        """Operators ask 'will this chop me' — the ratchet is the answer."""
        c = _cfg(monkeypatch, GUARD_TRAIL_FLOOR=0.4, GUARD_TRAIL_TIGHTEN=1.2)

        assert c["position_guards"]["profit_ratchet_on"] is True

    def test_the_ratchet_is_off_when_only_one_knob_is_set(self, monkeypatch) -> None:
        monkeypatch.delenv("GUARD_TRAIL_TIGHTEN", raising=False)
        c = _cfg(monkeypatch, GUARD_TRAIL_FLOOR=0.4)

        assert c["position_guards"]["profit_ratchet_on"] is False

    def test_the_stop_is_explained_as_from_the_high_water_mark(self, monkeypatch) -> None:
        """The operator's actual confusion: 'is 8% per day or from the peak?'"""
        c = _cfg(monkeypatch)

        assert "HIGH-WATER MARK" in c["position_guards"]["_stop_formula"]

    def test_it_says_guards_apply_to_personal_positions_too(self, monkeypatch) -> None:
        """The VST sale. Anyone asking about guards must be told this."""
        c = _cfg(monkeypatch)

        assert "regardless of who opened it" in c["position_guards"]["_what_they_are"]


class TestTheThingsThatMisledTheOperatorBefore:
    def test_it_states_that_kill_switches_do_not_sell(self, monkeypatch) -> None:
        c = _cfg(monkeypatch)

        assert "never liquidate" in c["risk_limits"]["_kill_switches_HALT_they_do_not_sell"]

    def test_it_states_that_risk_yaml_is_not_read(self, monkeypatch) -> None:
        c = _cfg(monkeypatch)

        assert "no loader exists" in c["risk_limits"]["_config_risk_yaml_is_NOT_read"]

    def test_it_warns_that_compose_overrides_env(self, monkeypatch) -> None:
        """The phantom disarm."""
        c = _cfg(monkeypatch)

        assert "overrides .env" in c["environment"]["_armed_means"]

    def test_it_reports_execution_as_plain_market_orders(self, monkeypatch) -> None:
        c = _cfg(monkeypatch)

        assert c["execution"]["order_type"] == "MARKET"
        assert "No VWAP" in c["execution"]["_no_algos"]

    def test_it_says_nothing_redeploys_after_a_guard_exit(self, monkeypatch) -> None:
        c = _cfg(monkeypatch)

        assert "NOTHING is re-deployed" in c["position_guards"]["_after_a_guard_exit"]

    def test_it_says_jobs_need_a_running_runner(self, monkeypatch) -> None:
        """Three days of a stale dashboard came from not knowing this."""
        c = _cfg(monkeypatch)

        assert "NOTHING updates" in c["schedule"]["_all_jobs_live_in_the_runner"]


class TestOperatingManual:
    def test_it_puts_pm_run_before_cycle(self, monkeypatch) -> None:
        steps = facts.operating_manual()["refresh_the_pm_and_trade"]

        assert steps[0].startswith("/pm run")
        assert any(s.startswith("/cycle") for s in steps)

    def test_it_explains_that_cycle_alone_does_not_reassess(self, monkeypatch) -> None:
        assert "does not ask" in facts.operating_manual()["_pm_run_first"]

    def test_it_states_the_arming_order(self, monkeypatch) -> None:
        arming = facts.operating_manual()["arming_order_matters"]

        assert "FIRST" in arming[0]
        assert "-91.82%" in arming[2]

    def test_it_says_halt_does_not_close_positions(self, monkeypatch) -> None:
        assert "does NOT close" in facts.operating_manual()["stopping_and_starting"]["halt"]

    def test_the_pipeline_is_in_order(self, monkeypatch) -> None:
        steps = facts.operating_manual()["what_a_cycle_does_in_order"]

        assert any("Merge the Agent PM" in s for s in steps)
        assert any("Risk manager" in s for s in steps)


def test_it_never_raises_on_a_missing_state_dir(monkeypatch) -> None:
    """It runs on every question. It must not be able to break the copilot."""
    c = facts.config_now(Path("/definitely/not/here"))

    assert c["holds"]["symbols"] == []
    assert "environment" in c
