# PM candidate and reasoning review — October 29, 2026

Follow-up: [October 30 decision audit and shadow implementation](pm_decision_audit_2026-10-30.md)
adds a bounded reasoning repair and an exact-decision execution gate. It
supersedes the warning-only behavior described below. Both changes remain local.

## Observed failure

The 10:41 UTC PM decision chose MPC and TMO at 3% each, with 94% cash.
The mechanical sleeve was zero. DD was operator-excluded, but the PM had no
explicit replacement-review obligation. Its October 28 candidate data was
mixed with October 23 committee opinions and an unverified MRNA crash rumor.
It described keeping existing sizes although the previous targets were 5%
MPC and 4% TMO. Both were simulated holdings, absent from the live snapshot.
A QNT-specific Sentinel caution contributed to the cash decision; the hard
70% deployment ceiling did not require 94% cash.

## Change

- Build a separate, deterministic review list from the existing ladder and
  universe, removing excluded and pinned names. Identify blocked mechanical
  picks and request review of the next eligible alternatives (at least three
  when available). The source ranking and mechanical strategy are unchanged.
- Request a structured allocate/pass reason and evidence for each required
  candidate and each final target. Require a cash explanation that compares
  alternatives and separates safety cash from discretionary cash.
- Preserve old committee views with explicit ages. After 72 hours, or when
  the timestamp is unknown/future, mark them as requiring revalidation.
  Only the newest fresh take per agent is eligible for current prompt voting.
  A dated thesis can survive, but historical rumor is not current evidence.
- Filter PM news by article publication time, not collection time. Headlines
  older than 72 hours, undated, or future-dated are omitted with a count.
  This is a PM-context filter; stored news and other agents are unchanged.
- Supply marked simulated weights to the model and calculate before/after
  weight changes independently. Report live ownership only as an account
  snapshot fact; missing account evidence is unknown, not zero holdings.
- Persist the review plan, completeness issues, allocation facts and evidence
  freshness in the decision file and append-only journal. Show omissions on
  the digest and `/pm`; `/pm review` exposes exclusions, requested alternatives,
  candidate reasons and cash reasoning without running a PM or cycle.

## Boundaries and validation

Review completeness checks presence and consistency, **not the truth of model
claims or investment merit**. Missing reasoning is visibly flagged, not silently
filled in and not a new execution gate. Replacements are evaluated, never
automatically bought or funded by redistribution. The Sentinel ceiling and
rotation restriction, risk manager, blocklists and approval gates remain.

Hermetic regression cases reproduce DD exclusion, MPC/TMO reductions, stale
rumor evidence, missing account state, malformed/duplicate/inconsistent review
records, explicit cash decisions, persistence and read-only Telegram review.
No live cycle, approval, or broker operation is part of this validation.

This changes model inputs and decision instructions. It is prepared locally
for paper/shadow evaluation; it has not been deployed to the live server.
Passing unit tests establishes these mechanics, not improved returns or that
the model will always use sound evidence. Evaluate fresh paper decisions and
their review completeness before adopting the changed reasoning in live use.
