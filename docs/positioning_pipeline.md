# Positioning pipeline — November 6, 2026

The free-source pipeline is implemented. It adds evidence to committee/PM
reasoning and prospective shadow experiments. It does not add an executable
positioning strategy or bypass the existing risk, halt and approval gates.

## Sources and limitations

- **COT:** seven explicitly mapped futures markets (S&P, Nasdaq, EUR, CHF,
  10-year Treasury, WTI, gold), 12 separate category observations. Net/OI,
  exact one/four-week changes, long versus short changes, trailing percentile
  requiring 26 prior reports. Futures-only datasets remain separate. Current
  historical series can contain revisions; these are not historical release
  vintages. CHF and Treasury data remain contextual, without direct targets.
- **Gamma:** free yfinance SPY/QQQ partial chains, up to three expirations within
  45 days and ±20% strikes. Requires a recent underlying bar and two-sided
  coverage; excludes 0DTE and nonstandard multipliers. Black-Scholes sensitivity
  uses r=q=0 and assumes European exercise. Last trade is not a quote timestamp;
  OI date and dealer inventory are unknown. Unsigned concentration and opposite
  dealer-side scenarios are research estimates, never observed dealer positions.
  Closed-market/stale data is unavailable. No subscriptions purchased.
- **Goldman:** checked public-report imports, not a continuous Prime Services
  feed. Preserve publication date, covered period, units, source URL and original
  report ID. Reprints deduplicate; conflicting revisions need a new report ID.
  Reports expire at the earlier of publication+14 days or covered-period+21 days.
  No applicable report is an explicit gap. No report has been invented or seeded.

## Operation

`uv run trading positioning refresh` collects read-only data into the configured
state directory. `uv run trading positioning status` reports coverage, errors,
actual observation dates and collector-watchdog status. Neither command calls
IBKR or creates orders.

The runner polls at 15:31/15:46 New York on weekdays. It also **awaits collection
before committee and PM context assembly**, with a ten-minute gamma cache and
hourly COT cache only when the normally expected report is already present.
The selected Friday committee/PM/cycle schedule remains 15:35/16:15/17:00.
Actual publication completion is checked from returned report dates, not assumed
from cron success. A normal Friday 15:30 expectation is conservative: holiday or
other delays are shown as unavailable; a full exception-calendar service remains
future work. Optional feed gaps do not cancel the desk or risk-reducing exits.
A data update does not auto-submit a trade or launch another decision.

Immutable observations retain first-seen timestamps in `positioning/archive/`;
`positioning.json` is the atomic current collection/status file. Context filters
future/expired observations each time it reads them. Missing/overdue attempts
appear in the watchdog; fetch failures appear in status and runner logs.
Actual observation inputs are also included in the existing immutable PM audit.

For public Goldman evidence, fill a copy of
`config/goldman_public_report.example.json`, check its dates/metrics and short
excerpt against the public source, and only then set `verified_against_source`.
Run `uv run trading positioning import-goldman /path/to/report.json`, then
`uv run trading positioning refresh` to include it in current context. This is
an explicit operator assertion, not automatic independent fact verification.
Private client reports and paid feeds are not collected.

## Decision checks and measured influence

Every reviewed/targeted candidate requires all three source assessments, linked
IDs, scope, effect and explanation, or explicit unavailable/not-applicable.
SPY macro context must be considered for stocks but can only justify a timing
annotation, not stock-specific ranking/size claims. Incorrect scope, ignored
available evidence and missing assessments fail the existing reasoning review;
one repair attempt is allowed, then exact-decision operator acknowledgement is
required. Model prose cannot be made semantically true by schema validation.
The independent basket approval remains required.

Two predeclared prospective shadow arms compare PM targets with a maximum 10%
relative reduction for a directly mapped instrument when COT percentile ≥90%
or unsigned gamma's largest-strike share ≥25%. No stacking, leverage increase,
forced replacement buy or live target mutation. These are **unvalidated research
hypotheses**, not trading recommendations. Individual-stock portfolios can have
no direct mapped adjustments; futures/ETF evidence does not become stock flows.
Goldman prose has no numeric sizing rule. Arms use the existing matched
5/21/63-observation cost/FX/risk grader and disclose coverage alongside effects.
The PM digest and `/pm review` expose coverage, assessments and adjustments.

Live deterministic overlays still require backtest/walk-forward and ≥30 days
paper evidence, plus operator authorization. Reliable options quotes/greeks may
require an approved broker data subscription; they still won't reveal actual
dealer inventory. A continuous Goldman PB feed requires legitimate client access.
