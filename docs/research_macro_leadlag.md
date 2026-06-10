# Cross-asset lead/lag vs Nasdaq 100, 2018–2026

*Research note, 2026-06-10. Data: daily closes 2018-01 → 2026-06 (yfinance:
NDX, gold, WTI, Henry Hub natgas (LNG proxy), BTC, 13w/5y/10y/30y UST
yields, DXY, VIX). Code: `experiments/fetch_macro_data.py` +
`experiments/analyze_macro_leadlag.py` (rerunnable).*

## Q1 — Highest negative correlation with NDX since 2018

Correlation of weekly/monthly moves with NDX returns (yields as changes):

| | daily | weekly | monthly |
|---|---|---|---|
| VIX | **-0.74** | **-0.70** | **-0.65** |
| DXY (dollar) | -0.14 | **-0.34** | **-0.35** |
| 10y yield | +0.06 | -0.02 | **-0.27** |
| 5y / 30y yield | +0.05/+0.07 | -0.01 | -0.23/-0.27 |
| 3m yield | +0.03 | +0.01 | +0.01 |
| natgas (LNG) | +0.06 | +0.09 | +0.11 |
| WTI | +0.15 | +0.15 | +0.28 |
| gold | +0.09 | +0.19 | +0.12 |
| BTC | **+0.30** | +0.26 | **+0.37** |

**Answers.**
- Of the assets you listed, **long-end bond yields (10y/30y) carry the
  highest negative correlation — but only at the monthly horizon, and
  only since 2022.** The sub-period table is the real story: yields
  correlated *positively* with NDX in 2018–21 (growth-shock regime:
  yields up = good news) and *negatively* in 2022–23 (inflation-shock
  regime: yields up = discount-rate pain for long-duration tech,
  weekly corr -0.27). Since 2024 it's back near zero. The stock–bond
  correlation sign is a *regime variable*, not a constant — treat it
  as something to monitor, not assume.
- The cleanest inverse asset overall is actually the **dollar** (DXY,
  -0.34 weekly, negative in *every* sub-period since 2020) — the
  global-liquidity / financial-conditions channel. It wasn't on your
  list but it beats everything on it.
- **VIX is the most negative of all (-0.74 daily) but it is
  contemporaneous** — a thermometer, not a forecast (lead-lag peak at
  k=0, nothing at k≥1).
- **BTC is the opposite of a hedge**: the most positively correlated
  asset in the panel (+0.37 monthly), and the correlation has *grown*
  (0.09 → 0.26 → 0.38 → 0.34 across sub-periods). It's a high-beta
  NDX twin.
- **Gold ≈ uncorrelated** (diversifier, not an inverse signal); WTI
  and natgas mildly positive on average.

## Q2 — Which reacts fastest / picks up first?

Three independent tests:

**(a) Daily lead-lag:** no asset's day-t-1 move meaningfully predicts
NDX at day t (all |corr| ≤ 0.07 at k≥1). At daily frequency markets are
near-simultaneous; "fastest" cannot mean literal daily lead.

**(b) Predictive regressions** (trailing move → NDX forward return,
overlap-deflated t):
- **Rates shock is the fastest actionable channel**: a 1σ rise in
  5y/10y yields over 5 days → ≈ -21bps NDX over the next 5 days
  (t ≈ -2.0). Economically: duration repricing transmits to mega-cap
  tech within a week.
- **Energy is the slow-burn channel**: 1σ 21d natgas rally → -82bps
  NDX next 21d (t ≈ -2.0); WTI same sign, weaker. Supply-shock →
  input-cost / real-income squeeze takes weeks, not days.
- BTC 21d momentum is a *positive* confirm (+64bps, t 1.5): top-quintile
  BTC momentum preceded +275bps avg NDX next-21d vs +26bps otherwise.
  Risk-ON gauge, not a warning light.

**(c) Event study at the four >15% drawdown peaks** (first |z|≥1.5 of a
20d move within ±40d of peak; negative = before the peak):
- **2020-02 (COVID)**: gold ≤-40d, DXY -30d, WTI -27d, VIX -23d,
  natgas -20d — yields and BTC only confirmed *after* (+6 to +19d).
- **2025-02**: WTI ≤-40d, natgas -20d, gold -16d led; yields/VIX/BTC
  lagged (+12 to +18d).
- **2021-11**: nearly everything was already stressed ≥1 month out
  (inflation was visible in rates and energy well before equities broke).
- **2018-08**: *nothing* led — an endogenous valuation/positioning break
  gives no cross-asset warning.

**Economist's conclusion.** There is no single fast *and* reliable
indicator: the fast ones (VIX, daily yield moves) are contemporaneous
or noisy; the reliable ones (energy complex, dollar) lead by weeks but
only for *exogenous* (macro-driven) bear markets, and give nothing for
endogenous breaks like 2018. The best operational compromise is a
**financial-conditions dial**, in priority order:
1. **5d change in 5y/10y yields** (fast, weakly predictive, watch the
   *sign regime* — it only bites when inflation drives the bus),
2. **DXY 5d strength** (most consistent inverse since 2020),
3. **energy complex 21d momentum** (slow but led 3 of 4 majors),
4. **BTC 21d momentum** as the risk-ON confirm,
with VIX kept as the coincident thermometer. Linear combos don't
improve t-stats much, but the quintile asymmetry is economically real:
calm-quintile periods averaged +119 to +201bps per 21d vs +18 to +44bps
in stress-quintile periods — i.e. this is a **risk dial** (when to be
brave), not an alpha signal (what to buy).

## Integration

`runtime/macro_monitor.py` (advisory, same pattern as the SPY/VIX,
options and style advisors): daily poll computes the four channel
z-scores + composite, alerts on threshold breaches with debounce, and
persists history to `state/macro_monitor.json` so we can score it
against realized drawdowns after a quarter. It never touches the order
path; if a quarter of live readings looks good, the natural next step
is feeding the composite into `Strategy.modulate` as a *continuous*
exposure scalar (the one good idea from the Instagram reel).

**Honesty box:** t-stats of ~2 over 8 years with overlapping windows
are *suggestive, not proven*; the event study has n=4; correlations are
regime-dependent and the 2022 inflation regime dominates the negative
readings. That is exactly why this ships as an advisor and not as a
position-sizing input on day one.
