r"""Tests for the HMM-based advisory regime classifier."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.runtime import hmm_advisor


def _make_returns(seed: int, n: int = 1500) -> pd.Series:
    """Mix three regimes (bull / neutral / bear) so the 3-state HMM has
    enough signal to distinguish them. The synthetic bear regime is
    deliberately extreme (-0.4% daily drift, 4% daily vol) to be clearly
    separable from the bull (+0.08% drift, 1% vol) and neutral (0 drift,
    1.5% vol) regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC")
    # Segments: 50% bull, 25% neutral, 25% bear (most recent)
    n_bull = int(n * 0.50)
    n_neut = int(n * 0.25)
    n_bear = n - n_bull - n_neut
    bull = rng.normal(0.0010, 0.010, n_bull)
    neutral = rng.normal(0.0001, 0.015, n_neut)
    bear = rng.normal(-0.0040, 0.040, n_bear)
    returns = np.concatenate([bull, neutral, bear])
    return pd.Series(returns, index=idx, name="SPY_ret")


def test_fit_and_classify_returns_snapshot() -> None:
    r = _make_returns(seed=0, n=1500)
    _model, snap = hmm_advisor.fit_and_classify(r)
    assert snap.label in {"BEAR", "NEUTRAL", "BULL"}
    assert snap.state in {0, 1, 2}
    # Posterior probabilities should sum to 1.
    assert abs(snap.p_bear + snap.p_neutral + snap.p_bull - 1.0) < 1e-6
    # Last 500 bars were the bear regime — we expect P(bear) to be dominant.
    assert snap.p_bear > 0.4


def test_fit_too_little_data_raises() -> None:
    r = _make_returns(seed=0, n=100)
    with pytest.raises(ValueError, match="300"):
        hmm_advisor.fit_and_classify(r)


def test_state_zero_is_always_bear() -> None:
    """The HmmRegime base class sorts states by mean — verify our
    label mapping aligns with that invariant."""
    r = _make_returns(seed=1, n=1200)
    model, _snap = hmm_advisor.fit_and_classify(r)
    # The means of the underlying HMM, remapped, should be monotone.
    means = np.asarray(model._model.means_).ravel()
    order = model._state_order
    # After remap, sorted_state 0 (bear) should have the lowest mean
    sorted_means = np.empty_like(means)
    for raw_i, sorted_i in enumerate(order):
        sorted_means[sorted_i] = means[raw_i]
    assert sorted_means[0] < sorted_means[-1]


# --- async advisor wrapper -------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    msgs: list[str] = []

    async def fake_send(text: str) -> bool:
        msgs.append(text)
        return True

    monkeypatch.setattr(hmm_advisor, "_send_telegram", fake_send)
    return msgs


def test_poll_initial_regime_publishes(captured, tmp_path: Path) -> None:
    r = _make_returns(seed=0, n=1500)
    out = asyncio.run(hmm_advisor.poll_and_alert(spy_returns=r, state_path=tmp_path / "state.json"))
    assert out["regime_changed"] is True
    assert len(captured) == 1


def test_poll_unchanged_regime_silent(captured, tmp_path: Path) -> None:
    r = _make_returns(seed=0, n=1500)
    state_path = tmp_path / "state.json"
    asyncio.run(hmm_advisor.poll_and_alert(spy_returns=r, state_path=state_path))
    # Second poll on the SAME data — regime is unchanged, no alert
    asyncio.run(hmm_advisor.poll_and_alert(spy_returns=r, state_path=state_path))
    assert len(captured) == 1  # only the first


def test_poll_persists_state(captured, tmp_path: Path) -> None:
    r = _make_returns(seed=0, n=1500)
    state_path = tmp_path / "state.json"
    asyncio.run(hmm_advisor.poll_and_alert(spy_returns=r, state_path=state_path))
    payload = json.loads(state_path.read_text())
    assert "label" in payload
    assert payload["label"] in {"BEAR", "NEUTRAL", "BULL"}
    assert 0.0 <= payload["p_bear"] <= 1.0


def test_advisor_never_writes_mode_file(captured, tmp_path: Path) -> None:
    """Hard invariant — even the HMM advisor must NEVER write mode.json."""
    mode_path = tmp_path / "mode.json"
    mode_path.write_text('{"mode":"bull","set_at":"","set_by":"test","reason":""}')
    before = mode_path.read_text()

    r = _make_returns(seed=0, n=1500)
    asyncio.run(hmm_advisor.poll_and_alert(spy_returns=r, state_path=tmp_path / "advisor.json"))
    assert mode_path.read_text() == before
