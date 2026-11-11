"""Price features and forward labels. Everything at row t uses rows <= t."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Feature names, in model-column order.
FEATURES = ("mom_12_1", "mom_6_1", "rev_1m", "rev_1w", "vol_3m", "near_high")
#: Rows of history a symbol needs before its first feature row exists.
MIN_HISTORY = 253
#: Fewer names than this on a date and a cross-sectional rank means little.
MIN_CROSS_SECTION = 20
BENCHMARK = "SPY"
_ASSET_DIRS = ("equity", "etf")


def _read_close(data_dir: Path, symbol: str) -> pd.Series | None:
    for sub in _ASSET_DIRS:
        for fname in ("1D.parquet", "1d.parquet"):  # Linux is case-sensitive
            path = Path(data_dir) / "parquet" / sub / symbol / fname
            if not path.exists():
                path = Path(data_dir) / sub / symbol / fname
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                return None
            column = "adj_close" if "adj_close" in frame.columns else "close"
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            index = pd.to_datetime(series.index, utc=True).normalize()
            series.index = index
            series = series[~series.index.duplicated(keep="last")].sort_index()
            return series[series > 0]
    return None


def load_close_panel(
    data_dir: Path, symbols: list[str], *, benchmark: str = BENCHMARK
) -> tuple[pd.DataFrame, pd.Series]:
    """Adjusted closes on the benchmark's calendar, and the benchmark itself.

    The benchmark's dates are the calendar: a stock with no bar on a session
    is NaN there, never forward-filled. A filled price would manufacture a
    0% return and a feature the market never showed.
    """
    bench = _read_close(data_dir, benchmark)
    if bench is None or len(bench) < MIN_HISTORY:
        raise ValueError(f"benchmark {benchmark} has no usable cached history")
    columns: dict[str, pd.Series] = {}
    for symbol in sorted({s.upper() for s in symbols} - {benchmark}):
        series = _read_close(data_dir, symbol)
        if series is not None and len(series) >= MIN_HISTORY:
            columns[symbol] = series
    if not columns:
        raise ValueError("no universe symbol has enough cached history")
    close = pd.DataFrame(columns).reindex(bench.index).astype("float64")
    return close, bench


def raw_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unranked features. Shifts are positional on the shared calendar."""
    daily = close / close.shift(1) - 1.0
    return {
        "mom_12_1": close.shift(21) / close.shift(252) - 1.0,
        "mom_6_1": close.shift(21) / close.shift(126) - 1.0,
        "rev_1m": close / close.shift(21) - 1.0,
        "rev_1w": close / close.shift(5) - 1.0,
        "vol_3m": daily.rolling(63, min_periods=63).std(),
        "near_high": close / close.rolling(252, min_periods=252).max(),
    }


def ranked_features(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Each feature as a same-day cross-sectional rank in [-0.5, 0.5].

    Ranks, not values: a 40% month and a 400% month are both "top of the
    list", so one meme stock cannot set the coefficients, and the number
    means the same thing in a calm year and a wild one.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in FEATURES:
        frame = raw[name]
        enough = frame.notna().sum(axis=1) >= MIN_CROSS_SECTION
        ranked = frame.rank(axis=1, pct=True) - 0.5
        out[name] = ranked.where(enough, np.nan)
    return out


def forward_excess(close: pd.DataFrame, bench: pd.Series, horizon: int) -> pd.DataFrame:
    """Return over the next ``horizon`` sessions minus the benchmark's.

    The only function here that looks forward. Row t is NaN until the
    calendar holds row t + horizon.
    """
    stock = close.shift(-horizon) / close - 1.0
    market = bench.shift(-horizon) / bench - 1.0
    return stock.sub(market, axis=0)


def design_matrix(
    ranked: dict[str, pd.DataFrame], rows: npt.NDArray[np.int_]
) -> tuple[npt.NDArray[np.float64], list[tuple[int, str]]]:
    """Stack the chosen calendar rows into (n, features), dropping NaN rows.

    Returns the matrix and, for each kept row, its (row position, symbol).
    """
    blocks = [ranked[name].to_numpy()[rows] for name in FEATURES]
    cube = np.stack(blocks, axis=-1)  # (dates, symbols, features)
    keep = ~np.isnan(cube).any(axis=-1)
    date_idx, sym_idx = np.nonzero(keep)
    symbols = list(ranked[FEATURES[0]].columns)
    keys = [(int(rows[d]), symbols[s]) for d, s in zip(date_idx, sym_idx, strict=True)]
    return cube[keep], keys
