# Positioning sources and integration research — November 5, 2026

This document records the November 5 research and selected Friday schedule.
The free-source implementation followed on November 6; current behavior and
remaining limitations are documented in [the pipeline runbook](positioning_pipeline.md).
No subscriptions were purchased and no executable positioning overlay was added.
The owner has no Goldman Prime Services or dealer-gamma subscription; budget
remains approximately $20/month.

## Source selection

| Source | Access/cost found | Recommendation |
| --- | --- | --- |
| CFTC TFF and Disaggregated COT | Public API, no token needed for reasonable use | First integration: weekly macro positioning |
| Goldman public Prime Services insights | Public articles; client reporting through Marquee | Optional dated evidence; no continuous free PB feed found |
| IBKR underlying/options data | Published non-professional bundles $10 + $4.50/month | Candidate for internally estimated gamma, subject to account entitlements and owner approval |
| SqueezeMetrics aggregate DIX/GEX | Public chart and CSV accessible | Research reference; permission needed before systematic collection |
| SpotGamma | Standard listed at $89/month; page contains other plan blocks | Above budget; introductory discounts are temporary |
| gexbot | Documented REST API; current checkout price not verified | Optional future provider; Classic explicitly describes a naive GEX model |

Sources: [CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm),
[Goldman](https://www.goldmansachs.com/what-we-do/ficc-and-equities/prime-services),
[IBKR pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php?menu=B),
[IBKR coverage](https://www.interactivebrokers.com/docs/general/market-data-subscriptions/popular-market-data-subscriptions/introduction),
[SqueezeMetrics](https://squeezemetrics.com/monitor/dix),
[SpotGamma](https://spotgamma.com/subscribe-to-spotgamma/),
[gexbot API](https://docs.gexbot.com/apidocs/classic/).

The IBKR combination is $14.50/month before applicable taxes, other subscriptions
or chargeable snapshots. Eligibility, API entitlements and the actual total need
account-specific confirmation. SPY/QQQ underlying and options are the initial
scope; SPX index data needs separate attention. Quotes and greeks do not reveal
dealer inventory. No purchase is authorized merely by this research.

## COT access and use

Direct unauthenticated probes succeeded for both datasets:

- TFF futures-only: `https://publicreporting.cftc.gov/resource/gpe5-46if.json`
- Disaggregated futures-only: `https://publicreporting.cftc.gov/resource/72hh-3qpy.json`

Query: `$limit=1&$order=report_date_as_yyyy_mm_dd DESC`. Both returned HTTP 200
and report date October 27, 2026. Samples contained 82 and 186 fields respectively.
These probes verify reachability/sample schema, not a production availability SLA.
References: [TFF](https://publicreporting.cftc.gov/w/gpe5-46if/default),
[Disaggregated](https://publicreporting.cftc.gov/w/72hh-3qpy/default),
[API guide](https://publicreporting.cftc.gov/stories/s/COT-Help/p2fg-u73y/).

Proposed features: `(long-short)/open_interest`, weekly/four-week changes,
long additions versus short covering, and trailing historical percentiles with
minimum coverage. Calculate separately by contract market and trader category.
Keep futures-only and combined reports separate; never add them together.
Use stable contract-market codes and explicit instrument exposure mappings.

Treat crowding as a hypothesis alongside price trend, not an automatic reversal
signal. Crude positioning does not measure buying of MPC. Index positioning
does not establish a TMO thesis. CHF futures need consistent quote direction
when mapped to USD/CHF. Treasury shorts may reflect hedged relative-value trades;
directional interpretation remains an assumption.

Normal COT publication is Friday 15:30 New York for Tuesday's positions. The
previous cycle was Friday 15:00 every two weeks, with committee at 13:00. The
owner selected the post-release Friday schedule below. Use the latest actually
published report and respect delayed releases and revisions in tests.
[CFTC timing](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm).

## Goldman evidence

Public articles can provide sector flows, leverage and crowding observations
about Goldman's Prime Services client sample. This
[dated semiconductor article](https://www.goldmansachs.com/insights/articles/how-hedge-funds-are-trading-semiconductor-stocks)
demonstrates availability, not a current trade signal.

Proposed import: public URL or permitted public-content source, retaining the
original date, covered period, sector/region, metric, units, source and extraction
confidence. Keep a reported observation distinct from an independently measured
whole-market flow. Check numerical extraction against the original text.
Deduplicate articles repeating one Goldman note; do not invent dollar flows
from phrases such as 'most sold', or forward-fill old commentary indefinitely.
No fresh applicable report means unavailable. A 13F filing is not a replacement
for client flows or a net-leverage series.

## Selected Friday schedule

The owner chose Friday and may execute manually after hours. The earlier Monday
recommendation is superseded. Keep the existing two-week cadence and October 9
anchor. All times below use America/New_York and follow its daylight saving time:

| Job | Selected time on cycle Fridays |
| --- | --- |
| Committee | 15:35 |
| Broker readiness | 16:00 (derived from cycle) |
| PM | 16:15 (45-minute lead) |
| Cycle/proposal | 17:00 |

The settings are in `config/friday_positioning_schedule.env.example`. On
November 6 and November 20, 2026, the cycle is 23:00 Europe/Warsaw. The runner's
cadence/trigger functions were checked for both dates, and 19 schedule/cadence
tests passed. This change does not enable automatic outside-hours order routing,
clear the existing halt, or remove basket approval. No cycle was manually fired.

Freshness still needs completed data dependencies, not just cron offsets. Moving
only the cycle to 15:45 would leave a 45-minute-lead PM deciding before COT.
The explicit committee override must also move, as it does in this profile.

The remaining data-aware scheduling work is:

- On each actual COT publication, collect and validate the new report, preserve
  the publication/ingestion timestamps and calculate positioning features.
  Friday 15:35 New York is an initial polling slot, not a guarantee of release.
- Refresh cheap market/positioning observations daily and ingest applicable
  Goldman observations only when new material appears. Refresh options-derived
  inputs immediately ahead of a decision when the feed is available.
- On rebalance Fridays, feed validated post-release observations through the
  committee/PM sequence and refresh relevant prices/options ahead of planning.
- Treat that time as a target, not a deadline to bypass incomplete inputs. Use
  exchange and release calendars for holidays/early closes. Revalidate quotes,
  account state and relevant short-lived evidence before approved execution.
- Material changes between scheduled reviews can produce an alert or refreshed
  proposal. A data update alone does not mandate a trade or a full LLM debate.

The normal COT release leaves 30 minutes before the core-session close, but
after-hours equity trading already exists. A 17:00 proposal gives time for
research and review without a deadline to execute before 16:00.
[CFTC calendar](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm),
[NYSE session hours](https://www.nyse.com/trade/trading-information).

The committee override, PM, broker readiness and cycle move together; the anchor
and frequency stay unchanged. Source collectors and verified-release dependencies
are not implemented merely by shifting the cron.

NYSE Arca's filing states an intended December 6, 2026 launch for 23/5 trading;
Nasdaq also announces that date. Core-session hours remain distinct. Venue hours
alone do not change our broker routing, data entitlements, baseline handling or
order policy. References:
[NYSE Arca SEC filing](https://www.sec.gov/files/rules/sro/nysearca/2026/34-106378.pdf),
[Nasdaq announcement](https://www.nasdaq.com/newsroom/nasdaq-explains-what-is-23-5-trading).

## Gamma research

Current `runtime/options_monitor.py` measures SPY IV/skew/term structure and
put/call open interest. It does not calculate dealer gamma. Its missing-value
fallbacks must not be copied into the new collector: absence must remain unknown.

Begin with SPY/QQQ, bounded strikes/expirations and a read-only collector that
respects IBKR pacing. Preserve underlying, option quote and open-interest dates
separately. Current greeks do not turn previous-day OI into intraday inventory.

First calculate unsigned gamma concentration by strike and expiry. For signed
GEX, store assumed dealer-side signs and compare sensitivity scenarios. A common
dollar sensitivity per 1% move is `gamma * OI * multiplier * spot^2 * 0.01`;
the dealer sign is a separate assumption. If results flip under plausible sign
assumptions, they cannot support confident directional instructions. Separate
zero-day expiries from longer-dated exposure and reject stale/incomplete chains.

Initially test timing annotations and bounded reductions in initial size. Do not
use estimated gamma to pick long-term winners or automatically increase leverage.
[Method limitations](https://squeezemetrics.com/monitor/docs),
[IBKR API requirements](https://www.interactivebrokers.com/docs/tws-api/doc/md/introduction).

SqueezeMetrics' public `/monitor/static/DIX.csv` returned HTTP 200 with fields
`date,price,dix,gex` and a final row dated November 4, 2026. It is an aggregate
daily estimate. Its terms require written permission for systematic collection;
its separate full web/API plan lists $720/month. Public download availability
does not establish permission for an automated database.
[Terms](https://squeezemetrics.com/monitor/terms),
[plans](https://squeezemetrics.com/monitor/plans).

## Proposed implementation

1. Frozen observation types in `core/types.py`; adapters under `data/` for COT,
   approved reported-observation imports and option snapshots.
2. `runtime/positioning_monitor.py`: immutable snapshots plus freshness/coverage
   status and a watchdog for failed collection or missing expected releases.
3. `agents/context.py`: source-specific expiry. Weekly COT must not inherit a
   generic 36-hour options-monitor expiry.
4. `agents/pm_contract.py`: mandatory positioning assessment for every candidate:
   evidence IDs, market scope, support/conflict, ranking/size/timing effect, or
   explicit unavailable/not-applicable reason. Distinguish reported, estimated
   and directly observed evidence; a verified extraction is not a verified thesis.
5. Deterministic overlay on combined targets before the existing risk manager.
   Effects are bounded in configuration, not selected by model prose. Preserve
   exclusions, risk caps, halt and human approval. Source absence never blocks
   risk-reducing exits. Strategies specifically dependent on a missing feed
   cannot open new entries; optional-feed gaps must not freeze the whole desk.
6. `agents/pm_audit.py`: persist inputs, assumptions and overlay version. Telegram
   shows original target, adjustment, final target and supporting observations.
7. `reporting/pm_shadow.py`: baseline/COT/gamma/reported-evidence comparisons using
   matched dates, FX, costs and risk. Include return, drawdown, turnover, coverage
   and missed opportunities. Never fabricate historical reports or observations.

Deliver collectors, mandatory assessments and shadow effects first. Executable
rules require backtest/walk-forward and paper evaluation. Test publication-time
look-ahead, revisions, category separation, FX direction, narrative deduplication,
stale chains, sign uncertainty, overlay bounds and reduction-only behavior.

Recommendation: start with zero-cost COT plus optional primary-source observations.
Confirm IBKR eligibility/total price before requesting subscription approval.
Avoid purchasing a dashboard until exportability and benefit for this system's
holding horizon are demonstrated. No paid vendor has been selected or contacted.
