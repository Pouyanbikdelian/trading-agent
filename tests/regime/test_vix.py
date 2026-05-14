"""VixRegime classifier tests + fetch helper smoke test with injected client."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.regime import DEFAULT_VIX_LABELS, VixRegime, fetch_vix_levels


def _vix_history(seed: int = 0) -> pd.Series:
    """Three equal-sized blocks (200 each) at well-separated VIX levels.
    Equal sizes mean the 33rd / 66th percentile edges align with the block
    boundaries, which is what makes the per-block accuracy tests strict."""
    rng = np.random.default_rng(seed)
    calm = 12 + rng.normal(0, 0.8, 200)
    normal = 22 + rng.normal(0, 0.8, 200)
    storm = 40 + rng.normal(0, 2.0, 200)
    idx = pd.date_range("2022-01-01", periods=600, freq="1D", tz="UTC")
    return pd.Series(np.concatenate([calm, normal, storm]), index=idx, name="vix")


def test_three_states_separated() -> None:
    levels = _vix_history()
    classifier = VixRegime(lookback_days=252, n_states=3).fit(levels)
    labels = classifier.predict(levels)
    # Each equally-sized block should land in its bucket with high accuracy.
    assert (labels.iloc[:200] == 0).mean() > 0.95
    assert (labels.iloc[200:400] == 1).mean() > 0.95
    assert (labels.iloc[400:] == 2).mean() > 0.95


def test_label_for_returns_human_names() -> None:
    assert VixRegime.label_for(0) == "low_vol"
    assert VixRegime.label_for(2) == "high_vol"
    assert VixRegime.label_for(99).startswith("state_")


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        VixRegime().predict(pd.Series([15.0]))


def test_fit_rejects_too_few() -> None:
    with pytest.raises(ValueError, match="at least"):
        VixRegime(n_states=3).fit(pd.Series([15.0] * 5))


# --- fetch_vix_levels with a fake yfinance ---------------------------------


class _FakeYf:
    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw = raw
        self.last_kwargs: dict[str, object] = {}

    def download(self, **kwargs: object) -> pd.DataFrame:
        self.last_kwargs = kwargs
        return self.raw


def test_fetch_vix_handles_naive_index() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D")  # naive
    raw = pd.DataFrame({"Close": [12.0, 13.0, 14.0, 15.0, 16.0]}, index=idx)
    fake = _FakeYf(raw)
    out = fetch_vix_levels(downloader=fake)
    assert str(out.index.tz) == "UTC"
    assert out.iloc[-1] == 16.0


def test_fetch_vix_handles_multiindex_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    raw = pd.DataFrame(
        {("Close", "^VIX"): [12.0, 13.0, 14.0]},
        index=idx,
    )
    fake = _FakeYf(raw)
    out = fetch_vix_levels(downloader=fake)
    assert out.iloc[-1] == 14.0


def test_default_labels_map_complete() -> None:
    # Three integer states map to the three named labels we use in playbooks.
    assert set(DEFAULT_VIX_LABELS.values()) == {"low_vol", "mid_vol", "high_vol"}
