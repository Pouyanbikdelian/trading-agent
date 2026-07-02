"""Rotation analytics — pure-function tests, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.dashboard.rotation import classify_regimes, compute_rotation


def _synthetic_closes() -> pd.DataFrame:
    """~15 months of daily closes: SPY flat-ish, XLE outperforming
    (should land Leading), XLK underperforming (should land Lagging)."""
    idx = pd.bdate_range("2025-04-04", "2026-06-26")
    rng = np.random.default_rng(7)
    spy = 100 * np.cumprod(1 + rng.normal(0.0002, 0.002, len(idx)))
    t = np.linspace(0.0, 1.0, len(idx))
    # RRG quadrants react to the *rate* of relative-strength change, so a
    # "leader" must accelerate (rs growth rate rising), not just outperform.
    xle = spy * np.exp(0.25 * t**2)  # accelerating winner -> leading
    xlk = spy * np.exp(-0.25 * t**2)  # accelerating loser -> lagging
    return pd.DataFrame({"SPY": spy, "XLE": xle, "XLK": xlk}, index=idx)


def test_compute_rotation_quadrants() -> None:
    out = compute_rotation(_synthetic_closes(), {"XLE": 5e8, "XLK": 2e9})
    by = {s["sym"]: s for s in out["sectors"]}
    assert by["XLE"]["quadrant"] == "leading"
    assert by["XLE"]["trail"][-1]["x"] > 100
    assert by["XLK"]["quadrant"] == "lagging"
    assert by["XLK"]["rel_3m"] < 0 < by["XLE"]["rel_3m"]
    assert by["XLE"]["dollar_vol"] == 5e8
    assert by["XLE"]["days_in_quadrant"] >= 1
    # trails are daily, bounded, and chronological
    ts = [p["t"] for p in by["XLE"]["trail"]]
    assert len(ts) <= 504 and ts == sorted(ts)


def test_compute_rotation_degrades() -> None:
    assert compute_rotation(pd.DataFrame()) == {}
    short = _synthetic_closes().iloc[:30]
    assert compute_rotation(short) == {}


def test_classify_regimes() -> None:
    months = pd.period_range("2019-01", "2020-12", freq="M").astype(str)
    cpi = [2.0] * 12 + [4.0] * 12  # cool year, then hot year
    ff = list(np.linspace(2.5, 1.0, 12)) + list(np.linspace(1.0, 3.5, 12))
    econ = {
        "series": {
            "cpi_yoy": {"points": [{"t": t, "v": v} for t, v in zip(months, cpi, strict=True)]},
            "fed_funds": {"points": [{"t": t, "v": v} for t, v in zip(months, ff, strict=True)]},
        }
    }
    out = classify_regimes(econ)
    assert out["history"][0]["r"].startswith("cool")
    cur = out["current"]
    assert cur["r"] == "hot_hiking"
    assert cur["label"] and cur["since"].startswith("20")
    # ribbon history is monthly and ordered
    ts = [h["t"] for h in out["history"]]
    assert ts == sorted(ts) and len(ts) == 24


def test_classify_regimes_degrades() -> None:
    assert classify_regimes({}) == {}
    assert classify_regimes({"series": {"cpi_yoy": {"points": []}}}) == {}
