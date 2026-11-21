"""Step 9 — try to break the regime-switch ensemble before trusting it.

Attacks: (1) switching cost on every regime change; (2) execute every
component ONE DAY LATE (no same-close trading, e.g. IBS via MOC); (3) 3x
trading costs; (4) regime label lagged one extra day; (5) year by year vs
SPY; (6) deflated Sharpe charged for all 83 zoo strategies + 4 ensembles.
"""

# ruff: noqa: N803, N812
from __future__ import annotations

import sys

import harness as H
import pandas as pd
from harness import OUT, TRAIN_END, sharpe
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(OUT.parents[1] / "src"))
from trading.selection.scores import deflated_sharpe

PICKS = {
    "bull/calm": ["QQQ EMA50/200", "QQQ IBS<0.2", "Risk parity (ERC) core 6"],
    "bull/stressed": ["GTAA-5 (Faber)", "Donchian 100/50 ETFs, 10% vol", "Min variance core 6"],
    "bear": ["SPY IBS<0.2", "IWM IBS<0.2", "QQQ IBS<0.2"],
}


def rebuild(delay: int = 0, cost_mult: float = 1.0) -> pd.DataFrame:
    """Re-run the 9 components with an execution delay / cost multiplier."""
    import s02_trend
    import s03_meanrev
    import s06_allocation

    H.COST_BP_ETF *= cost_mult
    H.COST_BP_STOCK *= cost_mult
    orig = H.evaluate
    keep: dict[str, pd.Series] = {}
    wanted = {n for v in PICKS.values() for n in v}

    def ev(name, family, w, cost_bp=None):
        if name in wanted:
            w2 = w.reindex(H.prices().index).ffill().shift(delay) if delay else w
            keep[name] = orig(name, family, w2, cost_bp).ret
        return orig.__wrapped__ if False else H.Result(name, family, pd.Series(dtype=float))

    for mod in (s02_trend, s03_meanrev, s06_allocation):
        mod.evaluate = ev
        mod.save = lambda res, tag: pd.DataFrame()
        mod.show = lambda tab: None
        mod.main()
    H.COST_BP_ETF /= cost_mult
    H.COST_BP_STOCK /= cost_mult
    return pd.DataFrame(keep).loc["2019-03-01":]


def ensemble(R: pd.DataFrame, lab: pd.Series, switch_bp: float = 0.0) -> pd.Series:
    tr = R[R.index <= TRAIN_END]
    out = pd.Series(0.0, index=R.index)
    for g, names in PICKS.items():
        iv = 1 / tr[names].std()
        w = iv / iv.sum()
        m = lab.reindex(R.index) == g
        out[m] = (R.loc[m, names] * w).sum(axis=1)
    if switch_bp:
        changes = lab.reindex(R.index) != lab.reindex(R.index).shift(1)
        out[changes] -= switch_bp / 1e4 * 2  # sell one book, buy the other
    return out


def summary(name: str, r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {
        "test": name,
        "sharpe": sharpe(r),
        "train": sharpe(r[r.index <= TRAIN_END]),
        "oos": sharpe(r[r.index > TRAIN_END]),
        "cagr": eq.iloc[-1] ** (252 / len(r)) - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


def main() -> None:
    lab = H.regimes()
    base = rebuild()
    rows = [summary("baseline (as reported)", ensemble(base, lab))]
    n_sw = int((lab.loc["2019-03":] != lab.loc["2019-03":].shift(1)).sum())
    rows.append(summary(f"+ switching cost 10bp ({n_sw} switches)", ensemble(base, lab, 10)))
    rows.append(summary("regime label lagged 1 extra day", ensemble(base, lab.shift(1), 10)))
    late = rebuild(delay=1)
    rows.append(summary("all components executed 1 day LATE", ensemble(late, lab, 10)))
    pricey = rebuild(cost_mult=3.0)
    rows.append(summary("3x trading costs", ensemble(pricey, lab, 30)))
    both = rebuild(delay=1, cost_mult=3.0)
    rows.append(summary("1 day late AND 3x costs", ensemble(both, lab.shift(1), 30)))
    spy = H.returns()["SPY"].loc[base.index]
    rows.append(summary("SPY buy&hold (same window)", spy))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    e = ensemble(base, lab, 10)
    yr = (
        pd.DataFrame({"ensemble": e, "SPY": spy})
        .groupby(e.index.year)
        .apply(lambda x: (1 + x).prod() - 1)
    )
    yr["ens_sharpe"] = e.groupby(e.index.year).apply(sharpe)
    print("\nYear by year:\n", yr.round(3).to_string())
    days = lab.reindex(e.index).value_counts()
    print("\nDays per regime:", days.to_dict())
    sr = e.mean() / e.std()
    for n_trials in (87, 300):
        d = deflated_sharpe(sr, len(e), float(skew(e)), float(kurtosis(e, fisher=False)), n_trials)
        print(f"Deflated Sharpe probability, charged for {n_trials} trials: {d:.3f}")
    # what share of each component's days were in market?
    print("\nComponent Sharpe by regime it was USED in (all years):")
    for g, names in PICKS.items():
        m = lab.reindex(base.index) == g
        print(f"  {g}: " + ", ".join(f"{n} {sharpe(base.loc[m, n]):.2f}" for n in names))
    e.to_frame("regime_switch_ensemble").to_parquet(OUT / "ensemble_regime_switch.parquet")


if __name__ == "__main__":
    main()
