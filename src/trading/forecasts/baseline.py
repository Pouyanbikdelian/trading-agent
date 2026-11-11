"""The mechanical forecaster, and the walk-forward test of whether it works."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from trading.forecasts.features import (
    FEATURES,
    design_matrix,
    forward_excess,
    ranked_features,
    raw_features,
)

HORIZONS = (5, 21)
#: Training uses every Nth session: neighbouring days share almost all of
#: their forward window, so using them all adds rows, not information.
TRAIN_STEP = 5
MIN_TRAIN_SESSIONS = 504
REFIT_EVERY = 63


@dataclass(frozen=True)
class Model:
    """A fitted logistic model. Plain numbers so it can be stored and read."""

    horizon: int
    coef: tuple[float, ...]
    intercept: float
    n_train: int
    base_rate: float
    trained_through: str

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = x @ np.asarray(self.coef) + self.intercept
        probs: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-z))
        return probs


def fit(
    ranked: dict[str, pd.DataFrame],
    excess: pd.DataFrame,
    *,
    horizon: int,
    last_row: int,
) -> Model | None:
    """Fit on feature rows whose ``horizon``-session outcome is already known
    at calendar row ``last_row``. Nothing later is touched."""
    from sklearn.linear_model import LogisticRegression

    newest_labelled = last_row - horizon
    if newest_labelled < MIN_TRAIN_SESSIONS:
        return None
    rows = np.arange(newest_labelled % TRAIN_STEP, newest_labelled + 1, TRAIN_STEP)
    x, keys = design_matrix(ranked, rows)
    if not len(keys):
        return None
    ex = excess.to_numpy()
    col = {s: i for i, s in enumerate(excess.columns)}
    y_raw = np.array([ex[r, col[s]] for r, s in keys])
    ok = ~np.isnan(y_raw)
    x, y = x[ok], (y_raw[ok] > 0).astype(int)
    if len(y) < 2_000 or y.min() == y.max():
        return None
    model = LogisticRegression(C=1.0, max_iter=300).fit(x, y)
    return Model(
        horizon=horizon,
        coef=tuple(float(c) for c in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        n_train=len(y),
        base_rate=float(y.mean()),
        trained_through=str(excess.index[newest_labelled].date()),
    )


def forecast_row(model: Model, ranked: dict[str, pd.DataFrame], row: int) -> pd.Series:
    """Probabilities for every symbol with complete features on ``row``."""
    x, keys = design_matrix(ranked, np.array([row]))
    if not len(keys):
        return pd.Series(dtype="float64")
    return pd.Series(model.predict(x), index=[s for _, s in keys])


def prepare(
    close: pd.DataFrame, bench: pd.Series
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[int, pd.DataFrame]]:
    raw = raw_features(close)
    ranked = ranked_features(raw)
    excess = {h: forward_excess(close, bench, h) for h in HORIZONS}
    return raw, ranked, excess


def _tstat(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) < 8 or arr.std(ddof=1) == 0:
        return None
    return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))


def walk_forward(close: pd.DataFrame, bench: pd.Series) -> dict[str, Any]:
    """Refit every quarter on the past only; score the following quarter.

    Every number reported is out of sample. Significance uses one date per
    ``horizon`` sessions, because overlapping windows are one observation
    counted many times.
    """
    _, ranked, excess = prepare(close, bench)
    n = len(close)
    report: dict[str, Any] = {
        "sessions": n,
        "symbols": int(close.shape[1]),
        "from": str(close.index[0].date()),
        "to": str(close.index[-1].date()),
        "features": list(FEATURES),
        "caveat": (
            "Universe is today's index members with surviving price history, so "
            "results are flattered by survivorship. No costs. Out of sample, but one "
            "history: treat a small positive skill as a reason to keep measuring, "
            "not as an edge."
        ),
        "horizons": {},
    }
    for h in HORIZONS:
        ex = excess[h].to_numpy()
        col = {s: i for i, s in enumerate(close.columns)}
        probs: list[float] = []
        outs: list[int] = []
        climo: list[float] = []
        per_date: dict[int, tuple[float, float, float]] = {}
        coef_log: list[dict[str, Any]] = []
        start = MIN_TRAIN_SESSIONS + h
        for refit_at in range(start, n, REFIT_EVERY):
            model = fit(ranked, excess[h], horizon=h, last_row=refit_at)
            if model is None:
                continue
            coef_log.append(
                {"through": model.trained_through, **dict(zip(FEATURES, model.coef, strict=True))}
            )
            for row in range(refit_at, min(refit_at + REFIT_EVERY, n)):
                p = forecast_row(model, ranked, row)
                if p.empty:
                    continue
                realised = np.array([ex[row, col[s]] for s in p.index])
                ok = ~np.isnan(realised)
                if ok.sum() < 20:
                    continue
                pv, rv = p.to_numpy()[ok], realised[ok]
                probs.extend(pv.tolist())
                outs.extend((rv > 0).astype(int).tolist())
                climo.extend([model.base_rate] * int(ok.sum()))
                order = np.argsort(pv)
                k = max(1, len(pv) // 10)
                spread = float(rv[order[-k:]].mean() - rv[order[:k]].mean())
                ic = float(pd.Series(pv).rank().corr(pd.Series(rv).rank()))
                per_date[row] = (ic, spread, float(((pv - (rv > 0)) ** 2).mean()))
        if not probs:
            report["horizons"][str(h)] = {"status": "not enough history"}
            continue
        p_arr, y_arr, c_arr = np.asarray(probs), np.asarray(outs), np.asarray(climo)
        brier = float(((p_arr - y_arr) ** 2).mean())
        brier_climo = float(((c_arr - y_arr) ** 2).mean())
        independent = sorted(per_date)[::h]
        ics = [per_date[r][0] for r in independent]
        spreads = [per_date[r][1] for r in independent]
        report["horizons"][str(h)] = {
            "forecasts": len(p_arr),
            "dates": len(per_date),
            "independent_periods": len(independent),
            "base_rate": round(float(y_arr.mean()), 4),
            "hit_rate": round(float(((p_arr > 0.5) == (y_arr == 1)).mean()), 4),
            "brier": round(brier, 5),
            "skill_vs_coin_flip": round(1 - brier / 0.25, 5),
            "skill_vs_base_rate": round(1 - brier / brier_climo, 5),
            "rank_ic_mean": round(float(np.nanmean(ics)), 4),
            "rank_ic_t": _tstat(ics),
            "decile_spread_mean": round(float(np.nanmean(spreads)), 5),
            "decile_spread_t": _tstat(spreads),
            "prob_range": [round(float(p_arr.min()), 3), round(float(p_arr.max()), 3)],
            "latest_coefficients": coef_log[-1] if coef_log else {},
        }
    return report
