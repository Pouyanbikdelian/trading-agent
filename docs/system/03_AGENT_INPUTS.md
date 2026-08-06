# What we feed the agents, and what they do with it

## One context blob, sliced per agent

`agents/context.py:build_context(state_dir, data_dir)` runs once per
cycle and does no network I/O — it reads local state files and the
parquet cache only. Everything every agent sees comes from here.

| Key | Source | Meaning |
|---|---|---|
| `account`, `positions` | `runner.db` snapshot | **the traded book** — equity, per-name qty, 52w entry/now percentile, unrealized, sector |
| `book_concentration` | computed | effective number of bets (ENB) from correlation eigenvalues |
| `candidate_ladder` | `agents/candidates.py` | **the ranked scoreboard — the only source of names not already held** |
| `holds`, `k_override` | `state/holds.json` | operator pins |
| `operator_objections` | copilot thread | what Yan argued recently, in his words |
| `operator_mandates` | `copilot/mandates.py` | standing instructions, graded strong/medium/soft |
| `macro_dial` | `macro_monitor.json` | macro readings |
| `vol_surface` | `options_monitor.json` | implied/realized vol |
| `spy_vix_triggers`, `style_leader` | `advisor.json`, `style_advisor.json` | |
| `established_lessons` | memory | up to 6 — what the desk has learned |
| `operator_lessons_under_consideration` | memory | Yan's un-hardened lessons |
| `dossiers`, `source_trust` | memory | world-state notes, per-source Beta trust |
| `economy` | FRED via `econ_watch` | CPI, claims, HY spreads |
| `sector_momentum_vs_spy_pct`, `headlines` | `news_watch` | last, so it is first to be trimmed |

`_VIEW_KEYS` in `committee.py` then hands each persona only its slice.
The **creative** agent is deliberately denied `positions` — position
blindness is the whole point of that role.

## The eight personas

| Agent | Sees | Job |
|---|---|---|
| `quant` | ladder, positions, macro, vol, economy | Reason from numbers. Two hard rules: a high 52w percentile says *where* price is, not that reward-to-risk is good; correlated holdings are ONE bet. |
| `narrator` | macro, dossiers, trust, lessons, headlines | Geopolitics, CB psychology, what is priced in |
| `street` | ladder, positions, style, sector momentum, headlines | Sell-side ratings, revisions, crowded love/hate |
| `position_coach` | positions, holds, lessons, objections, mandates | Only our book — does each thesis still hold |
| `risk_officer` | positions, vol, macro, triggers, economy | What can hurt us this week. May say "nothing actionable". |
| `trader` | ladder, positions, vol, triggers, headlines | Tactical, this-week horizon |
| `scout` | ladder, sector momentum, headlines, trust, lessons | The NEXT theme, weeks to months. Carries a standing operator directive on quantum computing. |
| `creative` | ladder, sector momentum, headlines, economy, macro, trust — **not the book** | The idea the committee is missing because they are staring at their holdings. Weighs named social sources by trust score. |

Each returns the same schema:

```json
{"stance": "bullish|neutral|bearish",
 "take": "<4-6 sentences, specific>",
 "prediction": {"subject": "NVDA", "direction": "up",
                "horizon_days": 14, "confidence": 0.72},
 "sources": ["..."], "cited_lessons": ["ls-..."]}
```

The `prediction` is the point. It is stored, and graded weeks later
against real closes, producing the per-agent hit rate and Brier score
that everything downstream uses to weight the voice. **Track record, not
eloquence.**

## The debate

1. **Specialists** (mid-tier model) → 8 takes, each journaled as
   `kind="take"`, each prediction stored.
2. **Guards** — deterministic, non-LLM checks the personas miss.
   Advisory; never an order gate.
3. **Challenger** (frontier) — reads all takes plus market context,
   attacks the two highest-conviction ones.
4. **Manager** (frontier) — synthesises posture from takes, objections,
   calibration, guard flags, and the lesson book. Journaled as
   `kind="committee"` — **note this row stores only the ruling, not the
   takes.**

## The PM (weekly, frontier model)

Reads, in prompt order:

```
operator_mandates, operator_held_do_not_trade,
sim_portfolio { equity, cash, holdings, performance, cycles_since_name_change },
allowed_universe_etfs (21 ETFs), allowed_universe_stocks (~1600, as a note),
agent_takes (16 most recent),
agent_calibration,
sentinel_alert,
today_context  ← with live-account keys renamed *_NOT_YOUR_BOOK
recent_committee_rulings (6)
```

Assembled by `_budgeted_prompt`, which fits 24k characters by **dropping
whole items** — headlines first, then oldest history — never by slicing
the string. The book, the caps, the mandates and today's context survive
every reduction, so the result is always valid JSON.

Returns target weights, a rationale, and a watch item. Weights then pass
through `_clamp_weights`: whitelist → per-name cap → ETF-count cap →
cluster cap → gross cap. Anything off-whitelist or operator-pinned goes
to cash, never to a creative substitute.

**The PM cannot construct an `Order`.** It writes `state/agent_pm/` and
sends Telegram. That isolation is structural, not conventional.

## The historian (Fridays)

Reads a week of journal plus the current lesson book, proposes at most
two new **candidate** lessons, and votes on existing ones. Promotion to
`established` needs +3 net supporting evidence.

## The copilot (on demand, Telegram)

Answers questions over `NOW_trading_account_paper`,
`NOW_trading_account_orders`, `NOW_agent_pm_simulated_book`,
`NOW_risk_state`, `NOW_lesson_book`, plus recent chat turns. Read-only by
construction — `tests/copilot/` asserts the package never imports
`execution/`. It can capture mandates and, since 2026-08-06, discuss the
lesson book.

## Feedback loops — which are closed

| Loop | Closed? |
|---|---|
| prediction → graded vs price → agent calibration → weighting | **yes**, nightly |
| source cited → prediction graded → source trust | **yes** |
| PM decision → daily mark → equity curve vs SPY | **yes** |
| candidate considered but passed → shadow book → forward return | **yes** |
| lesson → cited by an agent → credited | **no** — `cited_lessons` is collected and never read |
| closed trade → episode → lesson evidence | **no** — `add_episode` is never called |

The last two are the subject of finding #1 in
[04_PRODUCTION_READINESS.md](04_PRODUCTION_READINESS.md).
