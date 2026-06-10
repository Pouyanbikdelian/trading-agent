"""Options-structure monitor — hermetic tests over synthetic chains."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from trading.runtime import options_monitor as om


def _chain(strikes: list[float], ivs: list[float], oi: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"strike": strikes, "impliedVolatility": ivs, "openInterest": oi})


@pytest.fixture
def calm_metrics() -> om.OptionsMetrics:
    spot = 100.0
    near_calls = _chain([90, 95, 100, 105, 110], [0.20, 0.19, 0.18, 0.17, 0.16], [100] * 5)
    near_puts = _chain([90, 95, 100, 105, 110], [0.22, 0.20, 0.18, 0.17, 0.16], [120] * 5)
    far_calls = _chain([90, 95, 100, 105, 110], [0.21, 0.20, 0.19, 0.18, 0.17], [80] * 5)
    return om.compute_metrics(
        underlier="SPY",
        spot=spot,
        near_calls=near_calls,
        near_puts=near_puts,
        far_calls=far_calls,
        asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )


def test_compute_metrics_calm_surface(calm_metrics: om.OptionsMetrics) -> None:
    m = calm_metrics
    assert m.atm_iv == pytest.approx(0.18, abs=1e-9)  # mean of call/put ATM
    assert m.put_skew == pytest.approx(0.02, abs=1e-9)  # 0.95x-spot put @20% minus 18%
    assert m.term_slope == pytest.approx(0.01, abs=1e-9)  # contango
    assert m.pc_oi_ratio == pytest.approx(1.2, abs=1e-9)
    assert om.evaluate(m) == []  # nothing triggers on a calm surface


def test_evaluate_fires_on_stressed_surface() -> None:
    spot = 100.0
    # ATM IV 35%, steep put wing, inverted term structure, heavy put OI.
    near_calls = _chain([95, 100, 105], [0.36, 0.35, 0.34], [100, 100, 100])
    near_puts = _chain([95, 100, 105], [0.46, 0.35, 0.33], [600, 600, 600])
    far_calls = _chain([95, 100, 105], [0.31, 0.30, 0.29], [80, 80, 80])
    m = om.compute_metrics(
        underlier="SPY", spot=spot, near_calls=near_calls, near_puts=near_puts, far_calls=far_calls
    )
    names = {name for name, _ in om.evaluate(m)}
    assert names == {
        "ATM_IV_ELEVATED",
        "PUT_SKEW_STEEP",
        "TERM_STRUCTURE_INVERTED",
        "PUT_CALL_OI_HEAVY",
    }


def test_poll_debounces_and_recovers(tmp_path, monkeypatch, calm_metrics) -> None:
    sent: list[str] = []

    async def _fake_send(text: str) -> bool:
        sent.append(text)
        return True

    monkeypatch.setattr(om, "_send_telegram", _fake_send)
    state = tmp_path / "options_monitor.json"

    stressed = om.OptionsMetrics(
        underlier="SPY",
        asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        atm_iv=0.35,
        put_skew=0.11,
        term_slope=-0.05,
        pc_oi_ratio=1.9,
    )

    out1 = asyncio.run(om.poll_and_alert(metrics=stressed, state_path=state))
    assert out1["alert_sent"] is True and len(sent) == 1
    assert "Vol-surface signal" in sent[0]

    # Same stress again → debounced, no second alert.
    out2 = asyncio.run(om.poll_and_alert(metrics=stressed, state_path=state))
    assert out2["alert_sent"] is False and len(sent) == 1

    # Back to calm → recovery message.
    out3 = asyncio.run(om.poll_and_alert(metrics=calm_metrics, state_path=state))
    assert out3["cleared"] is True and out3["alert_sent"] is True
    assert "normalized" in sent[1]


def test_poll_survives_fetch_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(om, "fetch_chain_metrics", lambda underlier="SPY": None)
    out = asyncio.run(om.poll_and_alert(state_path=tmp_path / "s.json"))
    assert out == {"polled": False, "alert_sent": False}
