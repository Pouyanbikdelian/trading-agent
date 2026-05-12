"""Walk-forward (expanding-window) OOS evaluation.

The simplest defense against overfitting: never let your strategy see the
data it's evaluated on. We carve the price history into successive
``(train, test)`` folds, ask the user-supplied ``signal_fn`` to fit on the
train slice and emit weights for the test slice, and concatenate the
out-of-sample weights into a single series we then backtest.

Window types
------------
* *Expanding*: train slice grows; test slice slides forward by ``step`` bars.
  This is the default and matches the literature for "anchored walk-forward".
* *Rolling* (fixed-size train): not implemented here. Add when needed.

Contract for ``signal_fn``
--------------------------
``signal_fn(train_prices, test_prices) -> weights_for_test_period``

* ``train_prices``: prices up to and including the last train bar.
* ``test_prices``: prices for the OOS window (function may use these only
  to know the index; it MUST NOT look at price values *ahead* of each row
  when emitting weights).
* Return value: DataFrame aligned to ``test_prices.index`` with the same
  columns and the target weight per (ts, symbol).

Trust model: we can't actually prevent lookahead from inside this harness —
that's a discipline question for the strategy author. We document and rely
on review.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

from trading.backtest.costs import CostModel
from trading.backtest.engine import BacktestResult, run_vectorized

SignalFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _folds(
    index: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[Fold]:
    n = len(index)
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, step must all be positive")
    if train_size + test_size > n:
        raise ValueError(
            f"need at least train_size+test_size={train_size + test_size} bars; got {n}"
        )

    fold = 0
    train_end_idx = train_size  # exclusive
    while train_end_idx + test_size <= n:
        test_end_idx = train_end_idx + test_size
        yield Fold(
            index=fold,
            train_start=index[0],
            train_end=index[train_end_idx - 1],
            test_start=index[train_end_idx],
            test_end=index[test_end_idx - 1],
        )
        fold += 1
        train_end_idx += step


def expanding(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    costs: CostModel | None = None,
    initial_equity: float = 1.0,
) -> tuple[list[Fold], BacktestResult]:
    """Run an expanding-window walk-forward backtest.

    Returns the list of folds and a single ``BacktestResult`` over the
    concatenated OOS period. The concatenated equity curve is what you
    report; the per-fold list is what you use to diagnose regime breaks.
    """
    if step is None:
        step = test_size

    folds = list(_folds(prices.index, train_size, test_size, step))
    if not folds:
        raise ValueError("no folds produced — check window sizing vs data length")

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        train = prices.loc[fold.train_start : fold.train_end]
        test = prices.loc[fold.test_start : fold.test_end]
        w = signal_fn(train, test)
        if not isinstance(w, pd.DataFrame):
            raise TypeError("signal_fn must return a DataFrame of weights")
        # Align defensively — the harness owns the OOS index even if the
        # strategy gets sloppy.
        w = w.reindex(index=test.index, columns=test.columns).fillna(0.0)
        pieces.append(w)

    oos_weights = pd.concat(pieces, axis=0)
    # Folds with step < test_size can overlap; keep first occurrence so
    # earlier folds win for the overlap (a later fold has more train data
    # but we already trusted its earlier weight in the previous fold).
    oos_weights = oos_weights[~oos_weights.index.duplicated(keep="first")]

    oos_prices = prices.loc[oos_weights.index]
    result = run_vectorized(
        oos_prices,
        oos_weights,
        costs=costs,
        initial_equity=initial_equity,
    )
    return folds, result
