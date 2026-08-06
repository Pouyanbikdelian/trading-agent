# How this system works — start here

Written 2026-08-06 by reading the code, not the older docs. Where this
disagrees with `docs/system_map.md` or the brochure, trust this.

Four documents, in reading order:

| | |
|---|---|
| **[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** | The map. What each package does, what runs when, where the two books live. |
| **[02_CONFIGURATION.md](02_CONFIGURATION.md)** | Every knob that matters and where it lives. The gates that stop live trading. |
| **[03_AGENT_INPUTS.md](03_AGENT_INPUTS.md)** | Exactly what each agent is fed, what it returns, and who reads the answer. |
| **[04_PRODUCTION_READINESS.md](04_PRODUCTION_READINESS.md)** | The audit: what is solid, what is not, and what would have to be true before live. |

---

## The one-paragraph version

A scheduler (`runner/`) wakes up on cron. Twice a week it convenes a
**committee** of eight LLM personas who each read a slice of a shared
context blob and return a stance plus a falsifiable prediction; a
challenger attacks the two loudest, a manager synthesises a posture.
Every prediction is stored and graded against real prices weeks later,
which produces a per-agent skill score. Separately, once a week, a **PM
agent** reads those debates and allocates a *simulated* $1M book. A
**historian** distils lessons weekly. None of this touches the broker.

In parallel and completely independently, a mechanical **momentum
strategy** trades a paper IBKR account through a hard-blocking risk
manager. The agents do not place its orders and cannot.

## The thing most likely to confuse you

**There are two portfolios and they have nothing to do with each other.**

| | Traded book | Agent PM book |
|---|---|---|
| Who decides | `strategies/top_k_momentum` — mechanical | An LLM, weekly |
| Money | Paper IBKR account, CHF | Simulated, USD, `state/agent_pm/` |
| Path to orders | risk manager → broker | **none — it cannot trade** |
| Currently holds | AMD, CIEN, DELL, INTC, LITE, SNDK, STX, WDC | JPM, V, GLD, PM, HUM, LMT, XLV, NVDA |

The committee debates the *traded* book. The PM allocates its *own*
book. On 2026-08-05 the PM read its neighbour's positions as its own and
announced it had exited a semiconductor cluster it never held — see
`OPERATOR_ACCOUNT_KEYS` in `agents/pm.py`. If you are ever unsure which
book a number refers to, that is the correct instinct.
