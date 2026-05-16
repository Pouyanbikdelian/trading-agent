r"""Apply the operator-set mode to the strategy's target weights.

This is the *defensive sleeve* logic. Every cycle the runner asks
``apply_mode()`` to reshape the model's raw weights for the current
mode. The function is pure: no I/O, no globals.

Mechanics
---------

* ``BULL``     — return the strategy weights unchanged.
* ``NEUTRAL``  — same as BULL today. Kept distinct so we can introduce
                 a mild de-risk here in a future build without
                 ambiguating the "bull" semantic.
* ``DEFENSE``  — scale strategy weights to ``defense_strategy_gross``
                 (default 0.70), fill the remainder with the
                 ``defensive_sleeve`` equal-weighted.
* ``BEAR``     — scale strategy weights to ``bear_strategy_gross``
                 (default 0.20), allocate ``bear_defensive_weight``
                 (default 0.30) to the defensive sleeve, the rest (~50%)
                 is held as cash (no weight on it).
* ``FLATTEN``  — return zero weights.

Defensive sleeve
----------------
By default the sleeve is ``[XLP, XLU, GLD, QUAL]``:

  * **XLP**  — Consumer Staples Select Sector SPDR. Defensive equities.
  * **XLU**  — Utilities Select Sector SPDR. Dividend-yielding defensives.
  * **GLD**  — SPDR Gold Trust. Negative-correlation tail hedge.
  * **QUAL** — iShares MSCI USA Quality Factor. Quality tilt — same long-
               term return as SPY with less drawdown.

Equal-weighted because the user wants simplicity and these are
already-low-correlation assets — fancy weighting buys little here.

Cost model
----------
Mode changes are not free — turnover at the rotation point pays
slippage. The runner's existing CostModel handles this. Mode-overlay
itself is *just* the target weights; the engine does the bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading.runtime.mode import Mode

# Default defensive ETFs — must be present in the price frame for the
# overlay to use them. If absent (e.g. test fixture has no ETFs) the
# overlay falls back to "cash" (just scales gross down, no rotation).
DEFAULT_DEFENSIVE_SLEEVE: tuple[str, ...] = ("XLP", "XLU", "GLD", "QUAL")


@dataclass(frozen=True)
class ModePolicy:
    """All the knobs for how each mode reshapes weights.

    Frozen so a tight test fixture is unambiguous, but we expose every
    parameter — the user might want a softer DEFENSE (e.g. 80/20) or a
    harder BEAR (e.g. 10/40/50 cash).
    """

    defensive_sleeve: tuple[str, ...] = DEFAULT_DEFENSIVE_SLEEVE

    # DEFENSE: scale strategy down, fill remainder with sleeve
    defense_strategy_gross: float = 0.70
    defense_defensive_gross: float = 0.30

    # BEAR: scale strategy way down, allocate sleeve, rest is cash
    bear_strategy_gross: float = 0.20
    bear_defensive_gross: float = 0.30
    # ↑ bear gross = 0.50; the rest is cash by virtue of weights summing < 1

    # Per-name caps inside the defensive sleeve. Each sleeve ETF gets
    # ``sleeve_alloc / len(sleeve)`` — but we cap to avoid concentration
    # if the sleeve is short.
    max_sleeve_per_name: float = 0.20

    extra: dict[str, float] = field(default_factory=dict)


def apply_mode(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    mode: Mode,
    *,
    policy: ModePolicy | None = None,
) -> pd.DataFrame:
    r"""Re-shape ``weights`` for the active ``mode``.

    Parameters
    ----------
    weights
        The strategy's target weights — wide DataFrame, index = bars,
        columns = symbols, values = target weight per name. Output keeps
        the same index but may add ETF columns from the defensive
        sleeve.
    prices
        Price frame. Used to know which defensive ETFs are actually
        available (any missing from ``prices.columns`` is skipped).
    mode
        The active mode.
    policy
        Tunable parameters. Defaults are conservative.
    """
    policy = policy or ModePolicy()

    if mode in (Mode.BULL, Mode.NEUTRAL):
        return weights.copy()

    if mode == Mode.FLATTEN:
        return weights * 0.0

    # DEFENSE / BEAR — both rebuild the same way, just with different scales.
    if mode == Mode.DEFENSE:
        strat_scale = policy.defense_strategy_gross
        def_scale = policy.defense_defensive_gross
    elif mode == Mode.BEAR:
        strat_scale = policy.bear_strategy_gross
        def_scale = policy.bear_defensive_gross
    else:
        raise ValueError(f"unhandled mode: {mode}")

    # 1) Scale strategy weights to their target gross.
    #    The strategy already normalised to its own target_gross; we
    #    just multiply through.
    out = weights.astype(float).copy() * strat_scale

    # 2) Add the defensive sleeve.
    #    Skip ETFs not present in ``prices``; reweight what's left so
    #    the sleeve still sums to ``def_scale``.
    available = [t for t in policy.defensive_sleeve if t in prices.columns]
    if available:
        per_name = min(def_scale / len(available), policy.max_sleeve_per_name)
        for tkr in available:
            if tkr not in out.columns:
                out[tkr] = 0.0
            out[tkr] = out[tkr] + per_name
    # If no defensive ETFs available the gross is just strat_scale (rest cash).

    # Reorder columns to keep the deterministic output shape: original
    # strategy columns first, then any sleeve ETFs appended.
    original_cols = list(weights.columns)
    extra_cols = [c for c in out.columns if c not in weights.columns]
    return out.reindex(columns=original_cols + extra_cols, fill_value=0.0)


def estimate_mode_impact(
    current_weights: pd.Series,
    target_weights: pd.Series,
    *,
    equity: float,
    cost_bps: float = 10.0,
) -> dict[str, object]:
    r"""Bot-side preview: what trades does this mode change imply?

    Returns a dict with:
      - ``buys`` / ``sells`` — list of {symbol, weight_delta, dollar_delta}
      - ``turnover_pct`` — sum of |delta| / 1.0
      - ``trading_cost_dollar`` — estimated cost at ``cost_bps`` per side
      - ``net_gross_delta`` — change in sum(|weights|)
    """
    union = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(union, fill_value=0.0).astype(float)
    new = target_weights.reindex(union, fill_value=0.0).astype(float)
    delta = new - cur

    buys = []
    sells = []
    for sym, d in delta.items():
        if abs(d) < 1e-6:
            continue
        rec = {
            "symbol": sym,
            "weight_delta": float(d),
            "dollar_delta": float(d) * equity,
        }
        (buys if d > 0 else sells).append(rec)

    buys.sort(key=lambda r: -r["weight_delta"])
    sells.sort(key=lambda r: r["weight_delta"])  # most negative first

    turnover = float(delta.abs().sum())  # fraction of equity traded
    cost = turnover * equity * cost_bps / 10_000.0

    return {
        "buys": buys,
        "sells": sells,
        "turnover_pct": turnover,
        "turnover_dollar": turnover * equity,
        "trading_cost_dollar": cost,
        "net_gross_delta": float(new.abs().sum() - cur.abs().sum()),
        "n_changes": len(buys) + len(sells),
    }
