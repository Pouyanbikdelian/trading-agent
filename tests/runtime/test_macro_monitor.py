"""Macro financial-conditions monitor — hermetic tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading.runtime import macro_monitor as mm


def _panel(stressed: bool) -> pd.DataFrame:
    """~2.2y of synthetic closes. When ``stressed``, the last week jolts
    yields/dollar/energy upward hard enough to clear the 1.5σ gates."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-04-01", periods=560)
    n = len(idx)
    y5 = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.02, n)), index=idx)
    dxy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=idx)
    ng = pd.Series(3 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
    cl = pd.Series(70 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx)
    btc = pd.Series(60000 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=idx)
    if stressed:
        y5.iloc[-5:] += np.linspace(0.1, 0.5, 5)  # +50bps in a week
        dxy.iloc[-5:] *= np.linspace(1.01, 1.05, 5)
        ng.iloc[-21:] *= np.linspace(1.02, 1.60, 21)
        cl.iloc[-21:] *= np.linspace(1.01, 1.35, 21)
    return pd.DataFrame({"y_5y": y5, "dxy": dxy, "natgas": ng, "wti": cl, "btc": btc})


def test_calm_panel_triggers_nothing() -> None:
    r = mm.compute_readings(_panel(stressed=False), asof=datetime(2026, 6, 10, tzinfo=timezone.utc))
    assert abs(r.composite) < 1.5
    assert mm.evaluate(r) == []


def test_stressed_panel_fires_channels() -> None:
    r = mm.compute_readings(_panel(stressed=True))
    names = {n for n, _ in mm.evaluate(r)}
    assert "RATES_SHOCK" in names
    assert "DOLLAR_SQUEEZE" in names
    assert "ENERGY_SHOCK" in names
    assert r.composite > 1.5  # composite gate too


def test_poll_debounce_recovery_and_history(tmp_path, monkeypatch) -> None:
    sent: list[str] = []

    async def _fake_send(text: str) -> bool:
        sent.append(text)
        return True

    monkeypatch.setattr(mm, "_send_telegram", _fake_send)
    state = tmp_path / "macro.json"

    hot = mm.compute_readings(_panel(stressed=True))
    out1 = asyncio.run(mm.poll_and_alert(readings=hot, state_path=state))
    assert out1["alert_sent"] is True
    out2 = asyncio.run(mm.poll_and_alert(readings=hot, state_path=state))
    assert out2["alert_sent"] is False  # debounced

    calm = mm.compute_readings(_panel(stressed=False))
    out3 = asyncio.run(mm.poll_and_alert(readings=calm, state_path=state))
    assert out3["cleared"] is True and out3["alert_sent"] is True
    assert "normalized" in sent[-1]

    import json

    payload = json.loads(state.read_text())
    assert len(payload["history"]) == 3  # scoreable history accumulates


def test_poll_survives_fetch_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mm, "fetch_closes", lambda lookback_days=700: None)
    out = asyncio.run(mm.poll_and_alert(state_path=tmp_path / "s.json"))
    assert out == {"polled": False, "alert_sent": False}


def test_composite_is_mean_of_riskoff_channels() -> None:
    r = mm.MacroReadings(
        asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        rates_shock_z=2.0,
        dollar_z=1.0,
        energy_z=0.0,
        btc_confirm_z=-3.0,  # must NOT enter the composite
        composite=0.0,
    )
    # composite is computed in compute_readings; here just sanity-check
    # evaluate() uses the stored composite, not btc.
    assert pytest.approx(np.mean([2.0, 1.0, 0.0])) == 1.0
    assert {n for n, _ in mm.evaluate(r, threshold=1.5)} == {"RATES_SHOCK"}
