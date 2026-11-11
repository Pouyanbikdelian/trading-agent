"""Shadow forecasts: resolvable probabilities, graded against a free baseline.

Why this package exists (2026-11-11). The committee scorecard showed the
agents' confidence is worse than a coin flip, and the PM's lead over SPY
needs about thirty more months of marks to mean anything. Both problems are
sample size. This package asks one precise question of every stock, every
day — "what is the probability it beats SPY over the next H sessions?" —
of two forecasters:

* a mechanical model on price features (``baseline``): free, and testable
  on years of history because it cannot know the future;
* an LLM reading the same numbers plus headlines (``llm``): costs tokens,
  and can only be judged going forward.

The LLM is worth paying for only if it beats the baseline, so that — not a
coin flip — is the comparison everything here reports.

Nothing in this package reaches the PM, the risk manager or an order. It
reads the price cache and writes its own SQLite file.
"""
