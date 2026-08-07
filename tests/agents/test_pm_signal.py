"""The PM execution bridge.

The PM has been advisory since it was built. Giving it a path to the
broker changes what a bug costs: a hallucinated ticker or a fat-fingered
weight used to be a wrong line in a Telegram digest, and now reaches the
risk manager. So the properties pinned here are mostly about REFUSING to
trade, not about trading.

The three that matter most:

* **off by default.** ``AGENT_PM_SLEEVE_PCT=0`` means no bridge. Reaching
  the market must be an explicit act, never a side effect of deploying.
* **stale decisions do not execute.** The PM runs Monday, the cycle runs
  Friday. A four-day-old view executed against Friday's tape is not the
  view the PM formed.
* **dropped names are reported.** The PM may hold ETFs; the cycle's
  instrument map comes from the equity universe. Silently dropping an ETF
  target turns a deliberate allocation into cash with no log line.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.agents.pm_signal import (
    STRATEGY_NAME,
    format_bridge_note,
    load_pm_signal,
    pm_decision_path,
)

NOW = datetime(2026, 8, 7, 21, 5, tzinfo=timezone.utc)


def write_decision(
    state_dir: Path,
    *,
    weights: dict[str, float] | None = None,
    ts: datetime | None = None,
    ok: bool = True,
    **extra,
) -> Path:
    path = pm_decision_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": ok,
        "ts": (ts or NOW - timedelta(hours=1)).isoformat(),
        "equity": 1_012_345.67,
        "weights": {"AAPL": 0.10, "MSFT": 0.08} if weights is None else weights,
        **extra,
    }
    path.write_text(json.dumps(payload))
    return path


class TestDisabledByDefault:
    def test_zero_sleeve_produces_no_signal(self, tmp_path: Path) -> None:
        """Deploying the bridge must not, by itself, put the PM in the
        market."""
        write_decision(tmp_path)
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.0)
        assert not r.ok and "disabled" in r.reason

    def test_a_positive_sleeve_turns_it_on(self, tmp_path: Path) -> None:
        write_decision(tmp_path)
        assert load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05).ok


class TestDefaultsAreReadAtCallTime:
    """The sleeve and the freshness window used to be module-level
    ``os.getenv`` calls, so their values were frozen the first time
    anything imported the module. In a long-lived process — the runner,
    the bot — that means editing ``.env`` and restarting the *service*
    changed nothing if the module had been imported before settings were
    reloaded, and "I changed the sleeve and nothing happened" is a
    failure mode this codebase has paid for more than once.
    """

    def test_the_sleeve_default_follows_settings(self, tmp_path: Path, monkeypatch) -> None:
        from trading.agents import pm_signal

        write_decision(tmp_path)

        def sleeve(value: float):
            return lambda name, env, fallback: value if "sleeve" in name else fallback

        monkeypatch.setattr(pm_signal, "_setting", sleeve(0.0))
        assert not pm_signal.load_pm_signal(tmp_path, now=NOW).ok

        monkeypatch.setattr(pm_signal, "_setting", sleeve(0.11))
        assert pm_signal.load_pm_signal(tmp_path, now=NOW).ok

    def test_the_env_fallback_applies_when_settings_are_unavailable(self, monkeypatch) -> None:
        from trading.agents import pm_signal

        monkeypatch.setenv("AGENT_PM_SLEEVE_PCT", "0.07")
        monkeypatch.setattr(
            pm_signal,
            "_setting",
            lambda name, env, fallback: float(__import__("os").getenv(env, str(fallback))),
        )
        assert pm_signal.default_sleeve_pct() == pytest.approx(0.07)

    def test_defaults_are_functions_not_frozen_constants(self) -> None:
        """Pins the shape, not just the behaviour — reintroducing a
        module-level constant would silently restore the old bug."""
        from trading.agents import pm_signal

        assert callable(pm_signal.default_sleeve_pct)
        assert callable(pm_signal.default_max_age_h)
        assert not hasattr(pm_signal, "DEFAULT_SLEEVE_PCT")
        assert not hasattr(pm_signal, "DEFAULT_MAX_AGE_H")


class TestFreshness:
    def test_a_decision_within_the_window_trades(self, tmp_path: Path) -> None:
        write_decision(tmp_path, ts=NOW - timedelta(hours=2))
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05, max_age_h=6)
        assert r.ok and r.age_hours == pytest.approx(2.0)

    def test_the_current_four_day_schedule_gap_is_refused(self, tmp_path: Path) -> None:
        """PM Monday 14:30, cycle Friday 21:05. Until those are moved
        together, the bridge must decline rather than trade a stale view."""
        write_decision(tmp_path, ts=NOW - timedelta(days=4))
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05, max_age_h=6)
        assert not r.ok
        assert "96.0h old" in r.reason and "limit 6h" in r.reason

    def test_a_naive_timestamp_is_treated_as_utc(self, tmp_path: Path) -> None:
        """Rather than raising, or worse, comparing wall clocks."""
        path = pm_decision_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "ts": (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat(),
                    "weights": {"AAPL": 0.1},
                }
            )
        )
        assert load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05).ok


class TestSleeveScaling:
    def test_weights_are_scaled_onto_the_sleeve(self, tmp_path: Path) -> None:
        """The PM sizes against its own simulated book. On a real account
        its 10% must become 10% OF THE SLEEVE, not of the account."""
        write_decision(tmp_path, weights={"AAPL": 0.10, "MSFT": 0.20})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10)
        assert r.signal is not None
        assert r.signal.target_weights["equity:AAPL"] == pytest.approx(0.010)
        assert r.signal.target_weights["equity:MSFT"] == pytest.approx(0.020)

    def test_gross_is_bounded_by_the_sleeve(self, tmp_path: Path) -> None:
        """PM gross is capped at 1.0 of its own book, so account exposure
        can never exceed the sleeve — before the risk manager even looks."""
        write_decision(tmp_path, weights={f"S{i}": 0.10 for i in range(10)})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10)
        assert r.gross <= 0.10 + 1e-9

    def test_attribution_names_the_pm(self, tmp_path: Path) -> None:
        write_decision(tmp_path)
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert r.signal is not None and r.signal.strategy == STRATEGY_NAME


class TestInstrumentMapping:
    def test_etf_shelf_names_get_the_etf_asset_class(self, tmp_path: Path) -> None:
        """`etf:XLK` vs `equity:XLK` is the parquet partition and the key
        the cycle matches on — wrong class means unpriceable, not just
        mislabelled."""
        write_decision(tmp_path, weights={"XLK": 0.2, "AAPL": 0.1})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10)
        assert r.signal is not None
        assert "etf:XLK" in r.signal.target_weights
        assert "equity:AAPL" in r.signal.target_weights

    def test_untradeable_names_are_dropped_and_reported(self, tmp_path: Path) -> None:
        """The cycle builds its instrument map from the equity universe,
        so an ETF target would otherwise vanish into cash silently."""
        write_decision(tmp_path, weights={"XLK": 0.2, "AAPL": 0.1})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10, tradeable_keys={"equity:AAPL"})
        assert r.ok
        assert r.dropped == ["XLK"]
        assert list(r.signal.target_weights) == ["equity:AAPL"]  # type: ignore[union-attr]

    def test_everything_dropped_is_a_refusal_not_an_empty_signal(self, tmp_path: Path) -> None:
        """An empty target set means 'sell everything' to the risk
        manager. It must never be produced by an accident of mapping."""
        write_decision(tmp_path, weights={"XLK": 0.2})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10, tradeable_keys=set())
        assert not r.ok and "survived instrument mapping" in r.reason

    def test_non_numeric_and_non_positive_weights_are_discarded(self, tmp_path: Path) -> None:
        write_decision(tmp_path, weights={"AAPL": 0.1, "MSFT": "junk", "NVDA": -0.5})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10)
        assert r.signal is not None
        assert list(r.signal.target_weights) == ["equity:AAPL"]


class TestNeverRaises:
    """This runs inside a scheduled cycle. A signal source that can throw
    is a signal source that can stop the cycle."""

    def test_missing_file(self, tmp_path: Path) -> None:
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert not r.ok and "no PM decision yet" in r.reason

    def test_corrupt_json(self, tmp_path: Path) -> None:
        path = pm_decision_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert not r.ok and "unreadable" in r.reason

    def test_a_failed_pm_run_is_not_traded(self, tmp_path: Path) -> None:
        write_decision(tmp_path, ok=False, reason="PM call failed: timeout")
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert not r.ok and "last PM run failed" in r.reason

    def test_unparseable_timestamp(self, tmp_path: Path) -> None:
        write_decision(tmp_path)
        p = pm_decision_path(tmp_path)
        d = json.loads(p.read_text())
        d["ts"] = "last tuesday"
        p.write_text(json.dumps(d))
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert not r.ok and "unparseable ts" in r.reason

    def test_empty_weights(self, tmp_path: Path) -> None:
        write_decision(tmp_path, weights={})
        r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.05)
        assert not r.ok and "no weights" in r.reason


class TestOperatorNote:
    """'Held everything' and 'the bridge refused' look identical in a
    position report. The operator has to be told which happened."""

    def test_success_states_the_exposure(self, tmp_path: Path) -> None:
        write_decision(tmp_path, weights={"AAPL": 0.5, "MSFT": 0.5})
        note = format_bridge_note(load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10))
        assert "2 name(s)" in note and "10.0% of account" in note

    def test_refusal_states_the_reason(self, tmp_path: Path) -> None:
        write_decision(tmp_path, ts=NOW - timedelta(days=4))
        note = format_bridge_note(load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10, max_age_h=6))
        assert "not traded" in note and "96.0h" in note

    def test_drops_are_visible(self, tmp_path: Path) -> None:
        write_decision(tmp_path, weights={"XLK": 0.2, "AAPL": 0.1})
        note = format_bridge_note(
            load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.10, tradeable_keys={"equity:AAPL"})
        )
        assert "dropped XLK" in note
