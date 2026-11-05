# PM decision audit and matched shadow cohorts — October 30, 2026

Implemented on top of the October 29 candidate review. Deployed on November 5
with the owner's authorization to review every proposed basket personally;
see [deployment and rollback record](pm_deployment_2026-11-05.md). No broker
operation is required for validation or the offline grader.

## Decision behavior

- Excluded and pinned names remain hard-filtered. When an excluded name was
  a mechanical pick, the PM must review eligible alternatives from the ladder.
  Passing on all alternatives is allowed; buying a replacement is not mandatory.
- Candidate reasons link to evidence IDs. Fresh numeric predicates are checked
  against the supplied market snapshot. Headlines and committee opinions do not
  independently establish their claims; uncertainty needs a verification step.
  Repeated opinions cannot count as independent measurements. Free-text claims
  remain explicitly unverified, even when a related numeric predicate passes.
- Every target needs its final weight, a downside scenario, the corresponding
  portfolio loss fraction, invalidation, overlap and reasons for rejecting both
  smaller and larger sizes. The scenario is an estimate, not a stop guarantee.
- Cash needs a reconciled safety floor and discretionary amount, an alternative
  comparison, a trigger and a timezone-aware review deadline within seven days.
  A majority-cash decision expires at that deadline; ordinary decision freshness
  can expire it sooner.
- An incomplete response gets one semantic repair. Only reasoning fields are
  accepted from the repair: final allocations and exception requests cannot
  change. Persistent issues block the PM execution bridge until the operator
  acknowledges the exact decision. Simulation bookkeeping remains research.
- A failed initial model call replaces the previous eligible decision with an
  explicit failed record, so the old proposal cannot remain executable.

## Review and audit

`/pm review` displays blocked names, requested alternatives, reasons, omissions
and the decision ID. `/pm accept <decision-ID> <reason>` records a reasoning
acknowledgement for that decision's content hash only. It does not approve orders,
verify an uncertain event, grant an exception or bypass risk/freshness checks.
New decisions require their own acknowledgement when incomplete. Legacy records
retain the previous bridge semantics.

Each new decision has an exclusive-write directory under
`state/agent_pm/decisions/pm-<ID>/`. It stores the exact system/prompt, model
configuration, relevant source-file hashes, prior book, observed prices, model
attempts (including the repair prompt), final decision and any review receipt.
Records are content-hashed and linked; a mismatch blocks the new decision.
This is application-level append-only, tamper-evident storage, not protection
against an administrator rewriting the entire directory. Source hashes identify
code; external provider determinism and byte-identical replay are not promised.

## Forward comparisons

Candidate preparation snapshots the actual mechanical target weights, resolved
risk configuration, sector tags and quote currencies without touching execution
members. Missing metadata yields an explicit unavailable comparison.

Every cohort starts with the same 100,000 units of base-currency cash on the first
completed daily observation strictly after the decision. It compares PM,
mechanical, mechanical at PM's requested gross, equal-weight PM picks, mechanical
with PM vetoes, cash and an experimental scoped-Sentinel mechanical arm. All
arms respect exclusions. The experimental arm lifts the caution cap only for
pinned-only triggers; the actual Sentinel policy is unchanged.

Five-, 21- and 63-observation outcomes use the same dates, dated FX, per-side
costs, integer sizing, concentration limits and daily risk-manager checks. Daily
loss halts flatten the simulated cohort. Grading requires complete price and FX
coverage; it never fills missing FX with 1:1 or reports unavailable data as zero.
Today's potentially unfinished daily candle cannot complete a grade. Saved grades
retain the exact price/FX observations and are bound to the decision and horizon.

Attribution reports PM minus mechanical, selection/sizing at matched requested
gross, cash allocation, sizing and the experimental Sentinel difference. Risk
caps and rounding can change achieved gross. Cohorts overlap: their averages
are descriptive, not independent samples or a compounded strategy track record.
They assume prefunded FX, zero cash interest and daily-close fills; intraday
guards, dividends, taxes and market impact are not reproduced. This is the
prospective cohort foundation, not a continuous production-equivalent backtest.

Grading runs after a successful PM decision; `/pm shadow` reads its saved status,
including coverage gaps and overdue/failed grading. An offline catch-up is:

```bash
uv run python scripts/grade_pm_shadow.py --state-dir <paper-state> --data-dir <parquet-root>
```

## Validation and remaining evidence

Hermetic regressions cover repair isolation, exact-decision receipts, tampering,
model failure, cash expiry, contradictory measurements, repeated opinions,
broker-free candidate preparation, Telegram routing, risk limits, daily loss
exits, data gaps, no look-ahead and immutable forward grades.

Final validation on November 4, 2026:

- Offline suite: `uv run pytest -q -m 'not slow and not live'` — 2,384 passed,
  one deselected; two existing multiprocessing/fork deprecation warnings.
- `make lint` — passed, including formatting.
- Focused mypy on `pm_audit.py`, `pm_contract.py`, `pm_reasoning.py` and
  `reporting/pm_shadow.py` with `--follow-imports=silent` — passed. This is
  not a claim that repository-wide type checking is clean.
- Offline grader CLI help and `git diff --check` — passed.

Fresh paper decisions and matured forward prices are still needed to judge
whether reasoning or returns improve. No historical PM answers are regenerated
and presented as prospective evidence. The full continuous strategy/PM race
remains open in TODO.md; paper-first requirements and live approval gates remain.
