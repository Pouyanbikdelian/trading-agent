"""The mechanical forecaster: learns a real effect, and never peeks."""

from __future__ import annotations

import numpy as np

from trading.forecasts.baseline import fit, forecast_row, prepare, walk_forward
from trading.forecasts.features import FEATURES, load_close_panel


def test_features_at_a_row_do_not_change_when_the_future_does(panel) -> None:
    close, bench = panel
    row = 600
    _, ranked, _ = prepare(close, bench)
    tampered = close.copy()
    tampered.iloc[row + 1 :] *= np.linspace(0.5, 2.0, close.shape[1])
    _, ranked_after, _ = prepare(tampered, bench)
    for name in FEATURES:
        np.testing.assert_allclose(
            ranked[name].iloc[: row + 1].to_numpy(),
            ranked_after[name].iloc[: row + 1].to_numpy(),
            equal_nan=True,
        )


def test_a_forecast_made_today_cannot_depend_on_tomorrow(panel) -> None:
    """The whole value of the baseline is that its backtest is honest. Fit
    and forecast at one row, then rewrite everything after it."""
    close, bench = panel
    row, horizon = 700, 21
    _, ranked, excess = prepare(close, bench)
    model = fit(ranked, excess[horizon], horizon=horizon, last_row=row)
    assert model is not None
    before = forecast_row(model, ranked, row)

    tampered, bench_after = close.copy(), bench.copy()
    tampered.iloc[row + 1 :] = tampered.iloc[row + 1 :].to_numpy()[::-1] * 3.0
    bench_after.iloc[row + 1 :] *= 0.5
    _, ranked2, excess2 = prepare(tampered, bench_after)
    after = forecast_row(
        fit(ranked2, excess2[horizon], horizon=horizon, last_row=row), ranked2, row
    )
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())


def test_it_learns_a_planted_momentum_effect_out_of_sample(panel) -> None:
    close, bench = panel
    report = walk_forward(close, bench)
    result = report["horizons"]["21"]
    assert result["skill_vs_coin_flip"] > 0
    assert result["rank_ic_mean"] > 0.05
    assert result["decile_spread_mean"] > 0
    assert 0.0 < result["prob_range"][0] < result["prob_range"][1] < 1.0
    assert "survivorship" in report["caveat"]


def test_too_little_history_gives_no_model_rather_than_a_guess(panel) -> None:
    close, bench = panel
    _, ranked, excess = prepare(close.iloc[:400], bench.iloc[:400])
    assert fit(ranked, excess[5], horizon=5, last_row=399) is None


def test_the_panel_follows_the_benchmark_calendar_and_never_fills(data_dir, panel) -> None:
    import pandas as pd

    close, _ = panel
    path = data_dir / "equity" / "S00" / "1D.parquet"
    holed = pd.read_parquet(path).drop(close.index[700])
    holed.to_parquet(path)
    loaded, bench = load_close_panel(data_dir, [*close.columns, "NOPE"])
    assert list(loaded.index) == list(bench.index)
    assert np.isnan(loaded["S00"].iloc[700])  # a missing bar stays missing
    assert "NOPE" not in loaded.columns
