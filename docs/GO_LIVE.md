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
- [ ] Kill-switch drill on paper: force a >2% daily loss scenario (or
      inject via test) and watch the halt actually fire + alert + block
      subsequent orders. Same for MAX_DRAWDOWN_PCT.
- [ ] Startup reconciliation drill: restart trader mid-day with open
      positions; verify broker-vs-state drift detection works and alerts.
- [ ] Order idempotency: kill trader mid-cycle (paper), restart, confirm
      no duplicated orders (orders.db + broker both).
- [ ] IB Gateway death drill: `docker stop ibkr-gateway` during market
      hours on paper; verify self-heal restart + trader reconnect + alert.
- [ ] Currency check: CHF base account buying USD stocks — verify sizing
      uses the right FX rate and MAX_POSITION_PCT is computed on the
      right equity number (found the fallback comment in execution/ibkr.py).
- [ ] Pinned holds review: 3 pins currently active — decide each one
      deliberately before live (pins block BOTH strategy sells AND guard
      exits... guard exits are blocked by design; confirm Yan wants pins live).
- [ ] Clock/timezone audit of the live cron (21:05 UTC Friday = 4:05pm ET
      standard, 5:05pm during DST — confirm intended behavior in July).
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

- [ ] Finish sim observation window (~2026-07-12 per plan).
- [ ] Build PM → `Signal` bridge through the REAL risk manager (never a
      new order path); code-reviewed + tested.
- [ ] Run as ~20% PAPER sleeve ≥30 days.
- [ ] Only then discuss live, sized down.

## 5. Nice-to-have before live (not blocking)

- [ ] Wire Rotation-radar crossings into committee context as a standing
      input (analysis 2026-07-02: reaction speed belongs in the fast
      layers, not the slow sleeve).
- [ ] Race chart: rebase both sleeves to a common start AND common
      currency; label axes.
