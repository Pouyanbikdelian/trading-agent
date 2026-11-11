# Shadow forecasts

Added 2026-11-11. By default this is research only: nothing here reaches the
PM, the risk manager or an order. `FORECAST_PM_EVIDENCE=true` changes that for
the PM — see "Letting the forecasts influence trades" below. Code in `src/trading/forecasts/`, ledger in
`<state>/forecasts.db`, last-run status in `<state>/forecast_shadow_status.json`.

## The question

Every session, for every stock: *what is the probability it beats SPY over the
next 5 and the next 21 sessions?* Each forecast is written once, never edited,
and graded automatically when its window has closed.

Two forecasters answer it:

| Forecaster | Sees | Covers | Cost |
|---|---|---|---|
| Mechanical baseline (`baseline`) | Six price features, ranked across the universe each day: 12-1 and 6-1 momentum, 1-month and 1-week return, 3-month volatility, distance from the 52-week high. Logistic regression, refit daily on outcomes already known. | Whole universe | None |
| LLM, numbers only (`llm:<model>:numbers`) | Exactly the baseline's numbers, as plain percentages. | Top `FORECAST_SHADOW_TOP_N` names by 6-1 momentum | Tokens |
| LLM, numbers + headlines (`llm:<model>:news`) | The same, plus up to four recent headlines per stock. | Same names | Tokens |

Neither LLM forecaster is shown the baseline's probability. The two are an
experiment, not personas: "numbers" asks whether a model reads the same
figures any better than a regression, and the gap between "numbers" and
"news" is what the headlines were worth.

An LLM is worth paying for only if it beats the baseline on the same stocks
and days. That paired comparison is what the dashboard's Agents tab reports.

## What the baseline can do: walk-forward result

`trading forecasts backtest`, run 2026-11-11 on the local cache: 500 current
S&P 500 members, 2018-01-02 to 2026-05-14, refit every 63 sessions on the past
only, every figure out of sample.

| | 5 sessions | 21 sessions |
|---|---|---|
| Forecasts scored | 780,828 | 765,212 |
| Independent periods | 318 | 75 |
| Right (p > 0.5 and it beat SPY, or the reverse) | 50.7% | 50.7% |
| Skill vs coin flip (1 − Brier ÷ 0.25) | −0.04% | −0.22% |
| Rank correlation, forecast vs outcome | 0.020 (t = 1.6) | 0.024 (t = 1.1) |
| Top decile minus bottom decile | +0.31% (t = 1.6) | +0.98% (t = 1.2) |

**Reading.** The baseline is a coin flip. Its ordering of stocks leans the
right way, but not by enough to separate from luck, and that is on a universe
flattered by survivorship (today's index members) with no costs. This is the
bar: an LLM that cannot beat ~0.25 Brier on the same forecasts is adding
nothing.

## Reading the dashboard panel

- **Skill** is 1 − Brier ÷ reference. 0% is "no better", negative is worse.
- **Independent periods**, not forecasts, are the sample size. Fifty stocks on
  one day mostly move together, and tomorrow's 21-session window is today's
  shifted by one. One period is counted per `horizon` sessions.
- Under 8 independent periods no verdict is given. For 21-session forecasts
  that is about eight months; the 5-session ones answer in about two.

## Limits worth remembering

- An LLM cannot be backtested: it already knows what happened. Only forecasts
  made after its training cutoff count. Sonnet 5.5's is June 2026.
- The LLM half is conditional on the top-momentum names, so its base rate can
  differ from the universe's. The paired comparison handles that; the raw
  "vs coin flip" column does not.
- LLM requests go through the Message Batches API (half price). The run
  submits one batch, waits up to eight minutes for it, and otherwise leaves
  it for the next run to collect; a forecast always keeps the session it was
  asked about. A batch unfinished after 30 hours is written off. The request
  shapes follow the published docs and are covered by tests against a
  stand-in, but had not been exercised against the live API when this was
  written: check the first real run. `FORECAST_SHADOW_BATCH=false` falls
  back to direct calls at full price.

## Operating it

```bash
FORECAST_SHADOW_ENABLED=true     # baseline, free, 18:15 New York Mon-Fri
FORECAST_SHADOW_LLM=true         # also ask the model: TOP_N/10 calls a day
FORECAST_SHADOW_MODEL=claude-sonnet-5-5
FORECAST_SHADOW_TOP_N=50
FORECAST_SHADOW_BATCH=true       # half-price batch route (default)

trading forecasts run            # once, now (add --llm to spend tokens,
                                 #   --wait 480 to linger for the batch)
trading forecasts status         # last run and the skill table
trading forecasts backtest       # the table above, on the current cache
```

The scheduled run is a separate process with a 15-minute deadline, so it
cannot hold memory in, or hang, the trading runner. Token spend shows on its
own `shadow-batch` tier line in `scripts/llm_cost.py`, priced at half.

## Related changes made the same day

- The PM charter's "allocate at least 5% when creative or scout is 70%+
  sure" rule was removed. The scorecard showed those confidence numbers carry
  no information; a take is now an idea to examine, never a sizing instruction.
- The Cockpit's PM panel also reports the PM against the mechanical momentum
  picks on the same dates (the matched cohorts from `reporting/pm_shadow`),
  which is a fairer yardstick than SPY.

## Letting the forecasts influence trades

`FORECAST_PM_EVIDENCE=true` (operator's decision, 2026-11-11, taken before any
forecaster had a track record). The PM is then shown, for every candidate and
holding, each forecaster's latest probabilities as `forecast:SYMBOL` entries in
its evidence catalog, with the forecasters' live track record beside them.

- It may use them to choose between candidates it would otherwise rate alike.
- A forecast never opens a position without other dated evidence and never
  sets or raises a size. A claim citing one must be marked uncertain.
- The PM's proposal card lists what the forecasters said about each chosen
  name ("Forecasts, unproven"), so the operator judges with the same numbers.
- Forecasts older than five days are not shown.

The ledger keeps grading every forecast either way. What is lost is a clean
split of credit between the PM and the forecasts in LIVE results; the shadow
table still answers whether the forecasts themselves have skill.
