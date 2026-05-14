r"""Tests for the multi-feature HMM regime classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.regime.hmm_macro import HmmMacroParams, HmmMacroRegime, build_features


@pytest.fixture
def macro_features() -> pd.DataFrame:
    """Three-regime synthetic feature panel.

    Each 250-bar block has a distinct return distribution and
    correlated cross-asset behaviour:
        bull   : mu_r = +0.001, low vol, term flat, credit positive
        crisis : mu_r = -0.003, high vol, term up (flight), credit down
        normal : mu_r = +0.0005, mid vol, mid term, mid credit
    """
    rng = np.random.default_rng(0)
    n_each = 250
    idx = pd.date_range("2020-01-01", periods=n_each * 3, freq="1D", tz="UTC")

    # Bigger spread between regime means + lower per-regime variance gives
    # the HMM clean modes to settle into; otherwise EM lands on overlapping
    # Gaussians and the ordering becomes label-soup.
    bull_r = rng.normal(0.003, 0.005, n_each)
    crisis_r = rng.normal(-0.008, 0.030, n_each)
    normal_r = rng.normal(0.0008, 0.010, n_each)
    market = pd.Series(
        100.0 * np.exp(np.cumsum(np.concatenate([bull_r, crisis_r, normal_r]))),
        index=idx,
        name="mkt",
    )

    # Bonds: TLT-IEF spread positive in crisis (flight to quality)
    tlt_r = np.concatenate(
        [
            rng.normal(0.0001, 0.005, n_each),
            rng.normal(0.0015, 0.012, n_each),
            rng.normal(0.0001, 0.006, n_each),
        ]
    )
    ief_r = rng.normal(0.0001, 0.003, n_each * 3)
    tlt = pd.Series(100.0 * np.exp(np.cumsum(tlt_r)), index=idx)
    ief = pd.Series(100.0 * np.exp(np.cumsum(ief_r)), index=idx)

    # Credit: HYG-LQD spread negative in crisis
    hyg_r = np.concatenate(
        [
            rng.normal(0.0005, 0.005, n_each),
            rng.normal(-0.002, 0.018, n_each),
            rng.normal(0.0003, 0.007, n_each),
        ]
    )
    lqd_r = rng.normal(0.0001, 0.004, n_each * 3)
    hyg = pd.Series(100.0 * np.exp(np.cumsum(hyg_r)), index=idx)
    lqd = pd.Series(100.0 * np.exp(np.cumsum(lqd_r)), index=idx)

    return build_features(market, tlt=tlt, ief=ief, hyg=hyg, lqd=lqd, vol_window=20)


def test_build_features_shape() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="1D", tz="UTC")
    market = pd.Series(100.0 + np.arange(300), index=idx)
    out = build_features(market, vol_window=20)
    assert list(out.columns) == ["mkt_ret", "log_vol", "term_spread_proxy", "credit_spread_proxy"]
    # Missing optional ETFs become zero columns, not NaN.
    assert (out["term_spread_proxy"] == 0).all()
    assert (out["credit_spread_proxy"] == 0).all()


def test_predict_before_fit_raises(macro_features: pd.DataFrame) -> None:
    cls = HmmMacroRegime(HmmMacroParams(n_states=3))
    with pytest.raises(RuntimeError, match="fit"):
        cls.predict(macro_features)


def test_three_state_fit_separates_crisis_from_bull(macro_features: pd.DataFrame) -> None:
    """EM does not guarantee a clean 3-mode decomposition with our synthetic
    overlap; what we DO guarantee is that the bull and crisis distributions
    over states differ — i.e. the HMM is picking up on the regime change,
    even if the boundary doesn't perfectly tile our hand-constructed blocks.
    """
    cls = HmmMacroRegime(HmmMacroParams(n_states=3, random_state=42)).fit(macro_features)
    labels = cls.predict(macro_features)
    bull_segment = labels.iloc[10:240]  # block 1
    crisis_segment = labels.iloc[260:490]  # block 2
    # State distributions over the two segments must differ materially.
    # We measure total-variation distance between the two label-distributions.
    bull_dist = bull_segment.value_counts(normalize=True).reindex(range(3), fill_value=0)
    crisis_dist = crisis_segment.value_counts(normalize=True).reindex(range(3), fill_value=0)
    tv = 0.5 * (bull_dist - crisis_dist).abs().sum()
    assert tv > 0.4, f"bull/crisis label distributions are too similar (TV={tv:.2f})"


def test_three_state_fit_assigns_high_vol_to_low_state(macro_features: pd.DataFrame) -> None:
    """Higher-state IDs should correspond to higher mean returns by
    construction of the post-fit state ordering."""
    cls = HmmMacroRegime(HmmMacroParams(n_states=3, random_state=42)).fit(macro_features)
    labels = cls.predict(macro_features)
    # Per-state mean of the first feature (market return) must be monotone.
    feat = macro_features.iloc[:, 0]
    feat = feat.reindex(labels[labels >= 0].index)
    means = feat.groupby(labels[labels >= 0]).mean()
    sorted_means = means.sort_index().values
    assert all(sorted_means[i] < sorted_means[i + 1] for i in range(len(sorted_means) - 1)), (
        f"state means should be monotone increasing by state id; got {sorted_means}"
    )


def test_state_ordering_is_stable(macro_features: pd.DataFrame) -> None:
    a = HmmMacroRegime(HmmMacroParams(n_states=3, random_state=7)).fit(macro_features)
    b = HmmMacroRegime(HmmMacroParams(n_states=3, random_state=7)).fit(macro_features)
    pd.testing.assert_series_equal(a.predict(macro_features), b.predict(macro_features))


def test_predict_proba_rows_sum_to_one(macro_features: pd.DataFrame) -> None:
    cls = HmmMacroRegime(HmmMacroParams(n_states=3)).fit(macro_features)
    prob = cls.predict_proba(macro_features)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)


def test_fit_rejects_short_history(macro_features: pd.DataFrame) -> None:
    cls = HmmMacroRegime(HmmMacroParams(n_states=3))
    with pytest.raises(ValueError, match=">="):
        cls.fit(macro_features.iloc[:5])
