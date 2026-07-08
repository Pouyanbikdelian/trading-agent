# Rotation study — defensive vs risk-on cluster seesaw

**Question (Yan):** the RRG tab *looks* like Health/REIT + Biotech rotate against Tech/Semis and especially Uranium. Is that real, is it tradable, and is there a lesson worth saving?

**Data:** IBKR monthly closes, 2021-07 → 2026-07 (61 monthly obs), pulled via the market-data MCP for the same 15-ETF universe the dashboard RRG uses. Analysis reproduces the dashboard's RS-Ratio / RS-Momentum math. Script: `scripts/rotation_cluster_analysis.py` (re-runs in seconds).

Clusters tested:
- **DEF (defensive/rate-sensitive):** XLV healthcare, XLRE real-estate/REITs, IBB biotech
- **RISK (risk-on/cyclical):** XLK tech, SMH semis, URA uranium

## Verdict: the seesaw is real — as a *relative* rotation, not a naive alpha

**1. The anti-correlation is genuine and robust.** Monthly returns *relative to SPY*:

- `corr(DEF, RISK) = −0.52`. On **absolute** returns it's **+0.46** — both are equities and share market beta. The negative sign only appears once you strip the market out, which is exactly what an RRG is built to show: money **rotating between** the two groups.
- Rolling 12-month correlation is **negative in 100% of windows** (mean −0.52, never above −0.03). It holds in every sub-period: −0.49 (2021–22 rate shock), −0.34 (2023–24 AI bull), **−0.67 (2025–26, strongest)**. This is structural, not a one-regime fluke.
- Clusters are internally coherent (within-DEF avg +0.41, within-RISK +0.28).

**2. The cleanest pair is Healthcare ↔ Tech, not Uranium.** Pairwise relative-return correlations:

| DEF \ RISK | XLK | SMH | URA |
|---|---|---|---|
| XLV | **−0.59** | −0.50 | −0.34 |
| XLRE | −0.42 | −0.28 | −0.14 |
| IBB | −0.34 | −0.19 | −0.18 |

Your instinct on the *direction* is right — URA sits on the risk-on/leading side — but URA is the **loosest** member of the cluster (uranium trades on its own supply/demand cycle). The tightest seesaw is **XLV/IBB ↔ XLK/SMH**.

**3. It's coincident, not a lead you can front-run.** Cross-correlation peaks at lag 0 (−0.52); no meaningful 1–2 month lead in either direction. There's a mild mean-reversion bump at +2 months (+0.35). Practically: the two clusters are a live seesaw you read *now*, not a predictor of each other's future.

**4. RRG confirms it visually.** Using the dashboard's quadrant logic, the two clusters sit on **opposite sides** (one leading-tilt, the other lagging-tilt) in **55% of months** — that's the rotation you were seeing by eye.

## Backtest: honest result — don't trade it naively

Monthly rebalance, 2021-07 → 2026-07 (58 months). Signal = 3-month relative-momentum spread (RISK − DEF).

| Strategy | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| SPY (benchmark) | 12.3% | 16.0% | 0.81 | −24.8% |
| Always DEF | 3.4% | 15.4% | 0.29 | −24.6% |
| Always RISK | **25.5%** | 27.5% | **0.97** | −29.5% |
| Rotation, long leader (momentum) | 15.3% | 22.3% | 0.75 | −28.2% |
| Long leader / short laggard | −0.9% | 25.4% | 0.09 | −36.2% |
| Mean-reversion, buy laggard | 12.5% | 22.8% | 0.63 | −24.0% |
| Mean-reversion long / short | −5.4% | 25.4% | −0.09 | −46.5% |

**Read:** over a historic semis/tech bull, the winning move was simply *holding the risk cluster*. Every attempt to **time** the seesaw with 3-month momentum **underperformed** (long-only rotation lost to Always-RISK; both market-neutral long/short versions lost money and whipsawed). The anti-correlation is a real **structure**, but a naive momentum or mean-reversion timing rule does **not** monetize it on this sample. Treat this as a **risk / context signal**, not a standalone alpha.

## What it's actually good for

1. **Crowding / hedge check.** The Top-8 momentum book naturally piles into whatever's leading (lately semis/tech). Because DEF is −0.5 anti-correlated to RISK *relative to SPY*, a small XLV/IBB sleeve is a genuine rotation hedge when the book is crowded risk-on — not just cash drag. This is the most defensible use.
2. **Committee / regime context.** Feed the live cluster spread + RRG tilt to the committee as a structured "rotation state" fact, so the debate starts from *which way money is actually rotating* instead of eyeballing the chart.
3. **Sentinel tie-in.** The rotation state corroborated the July-2 Sentinel alarm (semis selling). A "rotation turning defensive" flag is a natural companion to the late-day de-risk gate.

## Current state (as of last obs, early Jul 2026)

The seesaw has **just tilted defensive**: 3-month momentum spread flipped to favor DEF (−0.046); last month DEF **+1.2%** vs SPY while RISK **−4.3%**; RRG shows XLV + IBB **leading**, SMH **weakening**, URA **lagging**. Money is rotating *out of* semis/uranium *into* healthcare/biotech right now — consistent with the recent Sentinel chip-selloff alarm.

## Proposed lesson (for the lesson memory)

> **Statement:** US sector ETFs split into a persistent risk-on/risk-off seesaw — Tech/Semis/Uranium (XLK/SMH/URA) vs Healthcare/REIT/Biotech (XLV/XLRE/IBB). Their returns *relative to SPY* are ~−0.5 correlated and negative in 100% of rolling 12-month windows (2021–2026); the tightest pair is XLV↔XLK. It is a **coincident** rotation (no exploitable lead) and does **not** survive as a naive momentum/mean-reversion long-short (whipsaws). Use it as a **crowding/hedge and regime-context signal**: when the book is crowded in one cluster, the opposite cluster is a real hedge; a flip in the 3-month relative-momentum spread marks the rotation turning.
>
> **Tags:** rotation, sector-etf, risk-on-risk-off, hedging, rrg, regime
>
> **Status:** candidate (needs forward paper-confirmation before it drives sizing)

## Suggested next steps (none applied yet — your call)

- **Add the lesson** above to the memory store (`add_lesson`), status *candidate*.
- **Wire a `rotation_state` into the committee context** (`agents/context.py`): cluster spread + RRG tilt, computed from the same `rotation.py` the dashboard uses. Low-risk, advisory-only.
- **Prototype a hedge-overlay** in the backtester only: when book RISK-exposure > X, add a small DEF sleeve; measure drawdown reduction. Backtest → walk-forward → paper before anything live (per CLAUDE.md).
