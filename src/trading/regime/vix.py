"""VIX-based market regime classifier.

The CBOE VIX index measures *implied* 30-day volatility on S&P 500 options;
it's the market's forward-looking estimate of how shaky things are about
to get. Historically:

* VIX < 15  : complacent / risk-on environment.
* VIX 15-25 : normal.
* VIX 25-40 : elevated stress.
* VIX > 40  : crisis (GFC, March 2020).

We classify by **percentile against the rolling history** rather than
absolute thresholds, because regime boundaries drift over decades (a
"high" VIX in 2017 = a "low" VIX in 2022). The classifier returns integer
labels per the ``RegimeClassifier`` protocol; ``label_for`` resolves an
integer back into a human-readable name the playbook YAML can use.

Data source
-----------
The classifier expects a ``VIX_LEVELS`` Series — daily close of ``^VIX``,
tz-aware UTC. The ``fetch_vix_levels()`` helper pulls it via yfinance
(same dep we already use elsewhere). Use a cached Parquet (via the data
layer) in production to avoid a network call per cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    import yfinance as _yf  # noqa: F401


class VixParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lookback_days: int = Field(default=252, ge=20)
    """Bars used to compute percentile bands. 252 ≈ 1 year of daily data."""
    n_states: int = Field(default=3, ge=2, le=5)


# Human-readable labels for default n_states=3. Subclassing the protocol
# only requires integer labels, but the playbook layer uses these names.
DEFAULT_VIX_LABELS: dict[int, str] = {
    0: "low_vol",
    1: "mid_vol",
    2: "high_vol",
}


class VixRegime:
    """Percentile-bucketed VIX regime classifier.

    Identical fit/predict shape to ``RealizedVolRegime`` — but the input is
    a level (VIX), not a return. We treat VIX itself as a stationary-ish
    quantity over the lookback window.
    """

    def __init__(self, params: VixParams | None = None, **kwargs: object) -> None:
        if params is None:
            params = VixParams(**kwargs)  # type: ignore[arg-type]
        self.params = params
        self.n_states = params.n_states
        self._edges: np.ndarray | None = None

    def fit(self, vix_levels: pd.Series) -> VixRegime:
        """Learn percentile edges from a VIX history. Idempotent."""
        s = vix_levels.dropna().astype(float)
        if len(s) < self.params.n_states * 4:
            raise ValueError(f"need at least {self.params.n_states * 4} VIX observations to fit")
        qs = np.linspace(0.0, 1.0, self.params.n_states + 1)[1:-1]
        self._edges = np.quantile(s.values, qs)
        return self

    def predict(self, vix_levels: pd.Series) -> pd.Series:
        """Bucket each VIX observation into its percentile-derived regime."""
        if self._edges is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        s = vix_levels.astype(float)
        labels = np.full(len(s), -1, dtype=int)
        mask = s.notna().values
        labels[mask] = np.digitize(s.values[mask], self._edges)
        return pd.Series(labels, index=s.index, name="vix_regime")

    @staticmethod
    def label_for(state_id: int, mapping: dict[int, str] | None = None) -> str:
        """Resolve an integer state id to a human-readable name. The
        playbook YAML uses these names as rule keys."""
        m = mapping if mapping is not None else DEFAULT_VIX_LABELS
        return m.get(state_id, f"state_{state_id}")


def fetch_vix_levels(
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    downloader: Any | None = None,
) -> pd.Series:
    """Pull ``^VIX`` daily closes from yfinance. Returns tz-aware UTC Series.

    ``downloader`` is the optional yfinance-shaped object the tests inject;
    None means "import yfinance lazily and use it directly".
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf
    end = end or datetime.now(tz=timezone.utc)
    start = start or end.replace(year=end.year - 5)
    raw = downloader.download(  # type: ignore[attr-defined]
        tickers="^VIX",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        return pd.Series(dtype="float64", name="vix")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    close = raw["Close"].astype(float).dropna()
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.tz_convert("UTC")
    close.name = "vix"
    return close
