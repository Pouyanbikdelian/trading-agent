"""Walk-forward harness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import ZERO_COSTS, expanding


def _always_long(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Trivial 'strategy': hold every symbol with weight 1 over the OOS window."""
    return pd.DataFrame(1.0, index=test.index, columns=test.columns)


def _last_train_mean_signal(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """A toy strategy that uses train data: long if the last train return was positive."""
    signal = (train.iloc[-1] > train.iloc[0]).astype(float)  # 1.0 or 0.0 per symbol
    return pd.DataFrame(
        np.tile(signal.values, (len(test), 1)),
        index=test.index,
        columns=test.columns,
    )


@pytest.fixture
def prices_50d() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=50, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"A": np.linspace(100, 149, 50), "B": np.linspace(50, 99, 50)},
        index=idx,
    )


def test_expanding_produces_expected_fold_count(prices_50d: pd.DataFrame) -> None:
    folds, _ = expanding(
        prices_50d,
        _always_long,
        train_size=20,
        test_size=5,
        step=5,
        costs=ZERO_COSTS,
    )
    # 50 bars total, 20 train, 5 test, step 5
    # Fold 0: train 0..19, test 20..24
    # Fold 1: train 0..24, test 25..29  (expanding!)
    # ... continuing until train_end + test_size > 50
    # train_end starts at 20, steps by 5; last valid is 45 (45+5=50).
    # That's [20,25,30,35,40,45] = 6 folds.
    assert len(folds) == 6
    assert folds[0].train_end == prices_50d.index[19]
    assert folds[0].test_start == prices_50d.index[20]
    assert folds[-1].test_end == prices_50d.index[49]


def test_expanding_concatenates_oos_results(prices_50d: pd.DataFrame) -> None:
    _, result = expanding(
        prices_50d,
        _always_long,
        train_size=20,
        test_size=5,
        step=5,
        costs=ZERO_COSTS,
    )
    # 30 OOS bars: indices 20..49.
    assert len(result.equity) == 30
    assert result.equity.index[0] == prices_50d.index[20]
    assert result.equity.index[-1] == prices_50d.index[49]


def test_expanding_default_step_equals_test_size(prices_50d: pd.DataFrame) -> None:
    folds_default, _ = expanding(
        prices_50d,
        _always_long,
        train_size=20,
        test_size=5,
        costs=ZERO_COSTS,
    )
    folds_explicit, _ = expanding(
        prices_50d,
        _always_long,
        train_size=20,
        test_size=5,
        step=5,
        costs=ZERO_COSTS,
    )
    assert len(folds_default) == len(folds_explicit)


def test_expanding_passes_growing_train_window(prices_50d: pd.DataFrame) -> None:
    seen_train_sizes: list[int] = []

    def spy(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        seen_train_sizes.append(len(train))
        return pd.DataFrame(0.0, index=test.index, columns=test.columns)

    expanding(prices_50d, spy, train_size=20, test_size=5, step=5, costs=ZERO_COSTS)
    # Train sizes should grow by 5 each fold: 20, 25, 30, 35, 40, 45.
    assert seen_train_sizes == [20, 25, 30, 35, 40, 45]


def test_signal_fn_using_train_data_works(prices_50d: pd.DataFrame) -> None:
    # Both symbols trend up, so the strategy stays long for every OOS window.
    _, result = expanding(
        prices_50d,
        _last_train_mean_signal,
        train_size=20,
        test_size=10,
        costs=ZERO_COSTS,
    )
    assert result.total_return > 0


def test_expanding_rejects_undersized_data() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": np.arange(10.0)}, index=idx)
    with pytest.raises(ValueError, match="at least"):
        expanding(prices, _always_long, train_size=20, test_size=5)


def test_expanding_rejects_bad_signal_fn(prices_50d: pd.DataFrame) -> None:
    def bad(train: pd.DataFrame, test: pd.DataFrame) -> dict:  # type: ignore[type-arg]
        return {"not": "a dataframe"}

    with pytest.raises(TypeError, match="DataFrame"):
        expanding(prices_50d, bad, train_size=20, test_size=5)  # type: ignore[arg-type]
