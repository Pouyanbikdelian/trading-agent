# Go-live plan — momentum sleeve first, PM stays sim

Status: DRAFT · created 2026-07-02 · owner: Yan · prepared with Claude.
This is the working checklist for taking real money live. Any AI session
(Claude, Opus, whatever) working this repo: read this first, work items
top-to-bottom, check things off with a commit, never skip a gate.

**Ground rules (from CLAUDE.md, restated because money):**
- Live = `TRADING_ENV=live` AND `ALLOW_LIVE_TRADING=true` AND
  `IBKR_TRADING_MODE=live`. Nobody flips these except Yan, by hand.
- The agent PM does NOT go live. It has no path to the order pipeline
  (rule #4 isolation). Its bridge (PM → Signal → risk manager) must be
  built, tested, and paper-traded ≥30 days before that conversation.
- First live config is sized DOWN (see §3). Scale up only after 2 clean
  weeks of live/paper parity.

---

## 1. Dashboard: real PnL & per-sleeve attribution  [build before live]

Goal: see at a glance what each sleeve actually made, in money, correctly.

Built 2026-07-09 as a new FIRST tab **"Live"** (`dashboard/live.py` +
`app.py`). All curves USD — CHF book converted via daily USDCHF
(Yan's call 2026-07-09: USD everywhere, CHF only on the account card).

- [x] **Per-sleeve PnL cards** — realized + unrealized + fees per strategy
      sleeve (momentum live, PM sim, later sleeves), from orders.db fills
      joined to positions; not just account equity deltas.
- [x] **FX-correct performance line** — account is CHF-based; convert the
      equity curve to USD (or plot both) so the race vs SPY/PM isn't
      polluted by USDCHF. (Bug found 2026-07-02: race chart mixes CHF book
      vs USD benchmarks.)
- [x] **Live vs paper split** — once live, the dashboard must show live
      book and paper book as separate series, never merged. (Set
      `DASHBOARD_LIVE_STATE_DIR` to the live state dir; it renders as its
      own sleeve, never merged with paper.)
- [x] **Daily PnL bars + attribution table** — per symbol contribution,
      so "which position hurt today" is one glance.
- [x] **Exclude intraday last point** from daily curves (today's point is
      the latest snapshot, not a close — makes drops look fake). Applied
      to the Portfolio tab's curve too; "Today" view keeps intraday.

## 2. Sanity checks before arming  [Claude runs, Yan reviews]

Code / config audit:
- [x] `.env` on VPS: full lint done 2026-07-09 via `cat -A` — no glued
      lines, no duplicates, no stray characters; the 2026-07-02
      `MAX_DRAWDOWN_PCT` suspicion was a false alarm (inline comments
      parse fine, `trading status` matches intent). GUARD_* confirmed
      present + enabled. Remaining: verify the IBKR_USERNAME line ends
      at the username (`grep -n "for live" .env` must be empty).
- [x] Risk limits sanity — PRECEDENCE RESOLVED 2026-07-09: the risk
      manager reads limits from .env ONLY; config/risk.yaml's limit
      numbers are never loaded (its header was wrong; fixed). Values
      confirmed for paper: 0.10 / 1.0 / 0.02 / 0.15. Live-day values
      are set in §3 (sized down). Yan still owns the final "yes these
      are my live numbers" on live day.
- [x] Kill-switch drill (2026-07-09) — REAL BUG, FIXED: the daily-loss
      baseline never rolled forward past the first-ever cycle, so that
      kill switch could never fire (7 weeks stale). Fixed + regression
      test. Drawdown halt drilled live: fired at a tightened limit,
      alerted, persisted across restart, blocked the next cycle,
      /resume cleared it. PASS.
- [x] Startup reconciliation drill (2026-07-14): clean restart → ✅ match
      alert; forced drift (1 SPY bought behind the trader's back) →
      `startup drift: 1 symbol(s)` + 🚨 alert. PASS. Side-finding: the
      IBKR web-portal login steals the gateway session (single-session
      accounts) — expect a gateway wedge + auto-restart after any manual
      portal visit.
- [x] Order idempotency (2026-07-14): `docker kill` mid-cycle + a
      double-triggered cycle two minutes apart — zero duplicate
      client_order_ids, second cycle correctly saw the first's fills.
      PASS.
- [x] IB Gateway death (2026-07-14): fired organically during the
      session-steal wedge — trader detected the dead session, restarted
      the gateway itself, reconnected. PASS (alert delivery confirmed by
      operator in Telegram).
- [x] Currency check — REAL BUG, FIXED 2026-07-14: sizing divided CHF
      equity by USD prices, undersizing every position by the USDCHF
      factor (~19%). Now: `broker.get_fx_rates()` (IBKR ExchangeRate
      rows) → `signal_to_orders(fx_rates=...)` sizes against the
      base-currency price. Missing rate degrades to old behavior with a
      logged decision. NOTE: first cycle after deploy will top positions
      up ~20% to true target weights.
- [x] Pinned holds review (2026-07-14): all 3 released by Yan — 'A' was
      a ghost pin (no position) that had silently eaten a basket slot;
      MU/SNDK released and promptly trailing-stopped at a profit. Code
      fix: pins without positions no longer reserve slots.
- [x] Clock/timezone audit (2026-07-14): CRON=5 21 * * FRI (UTC) =
      17:05 ET in summer (fine, 65 min after close) but 16:05 ET in
      winter — only 5 min after close, when the daily bar may not be
      final. RECOMMENDATION before November: set `CRON=5 22 * * FRI` in
      the VPS .env (year-round ≥65 min after close; fills still land at
      Monday's open either way). Sentinel/PM-mark UTC crons drift 1h in
      winter — cosmetic, accepted. Committee crons are already
      NYSE-anchored.
- [x] Sector cap — REAL BUG, FIXED 2026-07-14 (found via "why only
      semis?"): the 30% sector cap NEVER bound in production because no
      fundamentals cache existed and the failure was silent. Now: cycle
      falls back to `<data_dir>/fundamentals.parquet`, warns loudly
      (log + Telegram) when the cap is disabled, and a new CLI populates
      it: `trading data fundamentals sp500` (weekly refresh is plenty —
      add an ofelia label later if manual gets old). NOTE: once enabled,
      the first cycle will cut the ~90% tech book down toward 30% tech —
      expect a big rebalance.
- [x] External code review (OpenAI, 2026-07-15) — verified 7 claims:
      5 real, fixed same day: (1) SMART-routed open-order contracts broke
      key matching (pending netting silently no-op'd on real IBKR) +
      etf:/equity: key mismatch (would have bitten the PM bridge) — both
      normalized; (2) freeze not atomic — added last-instant halt
      re-check + zombie-cycle guard at the submit gate; (4) placeOrder
      was auto-retried after timeout (duplicate risk) — writes are never
      retried now, reads still are; (6) unfilled BUYs counted as
      sellable — sellable base is settled + pending sells only.
      Claim 3 (halt blocks manual /buy /sell /flatten) is INTENDED
      design — open policy question for Yan: should /flatten work while
      halted (panic button vs frozen-means-frozen)?
      Claims 5 & 7 (Telegram offset replay, no global execution lock)
      real but bounded (15-min command TTL; per-job locks + cooldown) —
      queued in Phase 11.
- [ ] Full test suite green on the exact commit being deployed; tag it
      (`git tag live-candidate-1`).

## 3. Config & gating changes for live day  [Yan only]

- [ ] Open a SEPARATE IBKR live sub-account (or confirm main) — never
      share the paper account's state dir. Fresh `state/` for live.
- [ ] `.env` for the live container:
      - `TRADING_ENV=live`, `ALLOW_LIVE_TRADING=true`, `IBKR_TRADING_MODE=live`
      - sized-down first month: `MAX_POSITION_PCT=0.05`,
        `MAX_GROSS_EXPOSURE=0.50`, keep loss/DD kills as-is
      - guards stay on; ratchet (FLOOR=0.4/TIGHTEN=1.2) only if its paper
        month looked clean
- [ ] Decide capital split. Proposal on the table: 50% momentum / 50% PM —
      REJECTED for now (PM has no live path). Counter-proposal: momentum
      live at reduced size, PM stays sim, rest in cash/T-bills until PM
      earns its bridge.
- [ ] IB Gateway login for live (noted 2026-07-09): the LIVE account
      username is the paper username WITHOUT the trailing "2"; password
      unchanged. Update `IBKR_USERNAME` in the VPS `.env` (feeds
      `TWS_USERID` in docker-compose) on live day, switch the gateway to
      live mode, and point the trader at port 4001 (live) instead of
      4002 (paper). Then `docker compose up -d --force-recreate` —
      plain restart does NOT re-read `.env`.
- [ ] Telegram alerting: confirm alerts arrive on a channel Yan actually
      watches with sound ON for the first live week.
- [ ] Write down the abort procedure ON PAPER (literally): how to halt
      (`/halt`), how to flatten manually in TWS, who does what if the VPS
      dies during market hours.

## 4. Agent PM path to live (later, separate track)

Decisions locked 2026-07-09 (Yan):
- **Sleeve capital: $20,000 USD, hard cap.** `PM_SLEEVE_CAPITAL_USD` in
  `.env` (Settings.pm_sleeve_capital_usd, default 20000). The $1M sim
  book is untouched so the observation window stays comparable; the
  bridge MUST size PM weights against this cap, never account equity.
- **Operator holds are off-limits to the PM.** One list for the whole
  system: `/hold` (state/holds.json). The PM prompt discloses pinned
  symbols and `_clamp_weights` hard-drops any allocation to them (done,
  live in sim now); at bridge time `filter_held_orders` blocks the order
  side too. Yan's long-term positions: `/hold` them and every automated
  path leaves them alone.

- [x] Finish sim observation window (ended ~2026-07-12). Next: pull the
      PM's final record vs SPY + the paper sleeve and decide the bridge.
- [ ] Build PM → `Signal` bridge through the REAL risk manager (never a
      new order path); code-reviewed + tested.
- [ ] Run as ~20% PAPER sleeve ≥30 days.
- [ ] Only then discuss live, sized down.

## 5. Nice-to-have before live (not blocking)

- [ ] Wire Rotation-radar crossings into committee context as a standing
      input (analysis 2026-07-02: reaction speed belongs in the fast
      layers, not the slow sleeve).
- [x] Race chart: rebase both sleeves to a common start AND common
      currency; label axes. (Done 2026-07-09 — the Live tab's "Sleeve
      race" is USD-converted, common-window, rebased to 100.)
