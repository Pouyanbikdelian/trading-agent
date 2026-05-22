r"""Long-polling Telegram command bot.

Runs as a separate process from the trading runner. Receives commands
from the operator's phone, mutates the on-disk state (halt.json), and
responds with status snapshots pulled from the same SQLite stores the
runner writes to.

Why long polling and not webhooks
---------------------------------
Webhooks require a publicly reachable HTTPS endpoint with a valid TLS
cert. That's needless complexity for a single-operator bot — long
polling works through any NAT, no inbound ports, no TLS. The trade-off
is a single open HTTP connection 24/7, which is trivial.

Authorization
-------------
Only the chat ID configured in ``settings.telegram_chat_id`` is
accepted. Every other message is silently ignored. Bots can be added
to groups; the authorization check prevents anyone else in such a
group (or a typo'd chat) from issuing commands.

Commands
--------
``/start``       — greeting + help
``/help``        — command list
``/status``      — env, halted, heartbeat age, last cycle outcome
``/positions``   — current positions and weights
``/report``      — generate a fresh weekly report and send the summary
``/halt``        — set halt.json. Optional reason after the slash.
``/resume``      — clear halt.json
``/heartbeat``   — runner heartbeat age in seconds
"""

from __future__ import annotations

import asyncio
import json
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger

BOT_API_BASE = "https://api.telegram.org"
POLL_TIMEOUT = 25  # seconds — long-poll
HELP_TEXT = (
    "*Trading bot — commands*\n\n"
    "*Read-only*\n"
    "/status — env, halted, heartbeat, mode\n"
    "/positions — current positions and weights\n"
    "/balances — cash per currency\n"
    "/orders — recent orders (last 7d, grouped by status)\n"
    "/pending — only orders currently in flight\n"
    "/heartbeat — last cycle age\n"
    "/report — generate the weekly report\n"
    "/fx-rate FROM TO — current reference rate (e.g. `/fx-rate USD CHF`)\n\n"
    "*Mode (rebalance posture)*\n"
    "/mode [bull|neutral|defense|bear|flatten] — preview a mode change\n"
    "/confirm — apply the previewed mode + run an off-cycle rebalance\n"
    "/cancel — discard a pending mode preview\n\n"
    "*Manual orders* (queued; runner executes within ~5s)\n"
    "/buy SYM QTY [LIMIT] — e.g. `/buy AAPL 10` or `/buy 10 AAPL`\n"
    "/sell SYM [QTY|all] [LIMIT] — e.g. `/sell AAPL all`\n"
    "/close SYM — close a single position\n"
    "/flatten — close every open position\n"
    "/cancel_order CLIENT_ID — cancel a specific pending order\n\n"
    "*FX*\n"
    "/fx 5000 CHF to USD — convert at market (also: `/convert`)\n"
    "/fx 5000 CHF — to USD by default\n\n"
    "*Reliability*\n"
    "/health — broker / heartbeat / queue at a glance\n"
    "/cycle — force one off-cycle rebalance now\n"
    "/refresh — queue a data refresh\n"
    "/reconnect — bounce the broker connection\n\n"
    "*Safety*\n"
    "/halt [reason] — kill switch: refuse to trade + force flatten\n"
    "/resume — clear halt\n"
)


# ---------------------------------------------------------------------------
# Low-level Bot API helpers
# ---------------------------------------------------------------------------


async def _send(client: httpx.AsyncClient, token: str, chat_id: str, text: str) -> None:
    """POST sendMessage. Never raises — bot loop must keep running.

    Markdown parse failures (400 with "can't parse entities") have caused us
    to drop critical alerts (e.g. broker rejection messages with unbalanced
    backticks). On any 400 we retry once as plain text so the operator
    *always* sees the message; aesthetics lose to deliverability.
    """
    url = f"{BOT_API_BASE}/bot{token}/sendMessage"
    base = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = await client.post(url, json={**base, "parse_mode": "Markdown"}, timeout=10.0)
        if r.status_code == 400:
            logger.warning(
                f"telegram markdown parse failed, retrying as plain text: {r.text[:200]}"
            )
            r = await client.post(url, json=base, timeout=10.0)
        if r.status_code >= 400:
            logger.warning(f"telegram send failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.warning(f"telegram send error: {e}")


async def _get_updates(client: httpx.AsyncClient, token: str, offset: int) -> list[dict[str, Any]]:
    """Long-poll for the next batch of updates."""
    try:
        r = await client.get(
            f"{BOT_API_BASE}/bot{token}/getUpdates",
            params={"offset": offset, "timeout": POLL_TIMEOUT},
            timeout=POLL_TIMEOUT + 5,
        )
        if r.status_code >= 400:
            logger.warning(f"telegram getUpdates failed: {r.status_code}")
            return []
        data = r.json()
        return data.get("result", []) if data.get("ok") else []
    except httpx.TimeoutException:
        return []  # normal — no new messages
    except Exception as e:
        logger.warning(f"telegram getUpdates error: {e}")
        await asyncio.sleep(2)
        return []


# ---------------------------------------------------------------------------
# Command handlers — each returns the reply text
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _cmd_halt(args: list[str]) -> str:
    from trading.core.file_lock import file_lock

    reason = " ".join(args) if args else "telegram"
    halt_path = settings.state_dir / "halt.json"
    # Lock the halt file across processes — audit fix #9 prevents racing
    # writes from /halt and the risk manager's auto-halt or /resume.
    with file_lock(halt_path):
        _atomic_write_json(
            halt_path,
            {
                "halted": True,
                "reason": reason,
                "halted_at": datetime.now(tz=timezone.utc).isoformat(),
                "flatten_on_next_cycle": True,
            },
        )
    logger.bind(reason=reason).warning("telegram halt")
    return f"🛑 *HALTED* — reason: `{reason}`\nNext cycle will force-flatten positions."


def _cmd_resume() -> str:
    from trading.core.file_lock import file_lock

    halt_path = settings.state_dir / "halt.json"
    with file_lock(halt_path):
        _atomic_write_json(
            halt_path,
            {
                "halted": False,
                "reason": "",
                "halted_at": None,
                "flatten_on_next_cycle": False,
            },
        )
    logger.info("telegram resume")
    return "✅ *RESUMED* — halt cleared."


def _cmd_status() -> str:
    halt_path = settings.state_dir / "halt.json"
    halted = False
    halt_reason = ""
    if halt_path.exists():
        try:
            payload = json.loads(halt_path.read_text())
            halted = bool(payload.get("halted", False))
            halt_reason = str(payload.get("reason", ""))
        except Exception:
            pass

    hb_age = _heartbeat_age()
    hb_line = "unknown" if hb_age is None else f"{hb_age:.0f}s ago"
    halt_line = f"🛑 *HALTED* — `{halt_reason}`" if halted else "🟢 running"
    return (
        "*Trading status*\n"
        f"env: `{settings.trading_env}`\n"
        f"live armed: `{settings.is_live_armed()}`\n"
        f"state: {halt_line}\n"
        f"heartbeat: {hb_line}\n"
    )


def _heartbeat_age() -> float | None:
    hb_path = settings.state_dir / "heartbeat.json"
    if not hb_path.exists():
        return None
    age = datetime.now(tz=timezone.utc).timestamp() - hb_path.stat().st_mtime
    return float(age)


def _cmd_heartbeat() -> str:
    age = _heartbeat_age()
    if age is None:
        return "_no heartbeat file yet — runner hasn't completed a cycle._"
    return f"heartbeat updated `{age:.0f}s` ago"


# How old a snapshot can be before we prepend a "snapshot is stale" warning
# to /positions and /balances. The runner writes a fresh snapshot at the end
# of every successful cycle; on a normal weekly-rebalance cadence the
# snapshot will be at most a week old between scheduled cycles. We use a
# tighter threshold here (30 min) because the operator usually only checks
# /positions to verify a recent action just took effect — anything older
# than that probably means a cycle failed and the file is fossilised.
_SNAPSHOT_STALE_AFTER_MINUTES = 30


def _snapshot_age_warning(snap_ts: datetime) -> str | None:
    """Return a warning prefix when the snapshot is older than the threshold.

    Saved as a separate helper so /positions and /balances stay in sync —
    they were both lying about state in the May 2026 incident (broker was
    flat post-/flatten, snapshot still showed the over-bought basket).
    """
    now = datetime.now(tz=timezone.utc)
    ts = snap_ts if snap_ts.tzinfo else snap_ts.replace(tzinfo=timezone.utc)
    age_min = (now - ts).total_seconds() / 60.0
    if age_min < _SNAPSHOT_STALE_AFTER_MINUTES:
        return None
    if age_min < 60:
        age_s = f"{age_min:.0f} min"
    elif age_min < 60 * 48:
        age_s = f"{age_min / 60:.1f} h"
    else:
        age_s = f"{age_min / 60 / 24:.1f} d"
    return (
        f"⚠️ snapshot is {age_s} old — broker state may have changed.\n"
        "   The runner refreshes the snapshot every 60s; if this warning "
        "is firing, the trader's snapshot job is failing (broker down, "
        "halt, etc.). Check `/health` or trader logs.\n\n"
    )


def _cmd_positions() -> str:
    """``/positions`` — monospace table of holdings as of the last cycle.

    Note: this reads the *snapshot* the runner writes at end-of-cycle,
    not a live broker query (the bot has no broker connection). If the
    last cycle crashed mid-flight the data here is from the prior good
    cycle; the snapshot timestamp at the top tells you how old it is.
    A stale-snapshot warning is prepended when the snapshot is older than
    the threshold so the operator isn't misled.
    """
    try:
        from trading.runner.state import RunnerStore

        store = RunnerStore(settings.state_dir / "runner.db")
        snap = store.latest_snapshot()
    except Exception as e:
        return f"could not read positions: {e}"
    if snap is None:
        return "no snapshot yet — runner hasn't completed a cycle."

    prefix = _snapshot_age_warning(snap.ts) or ""

    ccy = getattr(snap, "base_currency", None) or "USD"
    if not snap.positions:
        return (
            prefix
            + f"📊 Portfolio (snapshot {snap.ts:%Y-%m-%d %H:%M UTC})\n"
            + f"  Equity: {ccy} {snap.equity:,.2f}    Cash: {ccy} {snap.cash:,.2f}\n"
            + "  No open positions."
        )

    rows: list[tuple[str, float, float, float, float, float]] = []
    for _key, pos in sorted(snap.positions.items()):
        mv = pos.quantity * pos.avg_price + pos.unrealized_pnl
        weight = mv / snap.equity if snap.equity > 0 else 0.0
        rows.append(
            (pos.instrument.symbol, pos.quantity, pos.avg_price, mv, weight, pos.unrealized_pnl)
        )

    header = (
        f"{'Symbol':<7} {'Qty':>10} {'Avg cost':>11} "
        f"{'Mkt value':>12} {'Weight':>7} {'P&L':>11}"
    )
    sep = "-" * len(header)
    body = [
        f"{sym:<7} {qty:>10.2f} {avg:>11,.2f} {mv:>12,.0f} {w:>6.1%} {pnl:>+11,.0f}"
        for sym, qty, avg, mv, w, pnl in rows
    ]

    n = len(rows)
    long_count = sum(1 for _, qty, *_ in rows if qty > 0)
    short_count = n - long_count

    summary = (
        f"📊 Portfolio (snapshot {snap.ts:%Y-%m-%d %H:%M UTC})\n"
        f"  Equity: {ccy} {snap.equity:,.2f}    Cash: {ccy} {snap.cash:,.2f}\n"
        f"  Positions: {n} ({long_count} long, {short_count} short)"
    )
    table = "```\n" + "\n".join([header, sep, *body]) + "\n```"
    return prefix + summary + "\n" + table


# ---------------------------------------------------------------------------
# Mode change — preview / confirm / cancel
# ---------------------------------------------------------------------------


def _mode_paths() -> tuple[Path, Path, Path]:
    """(mode.json, pending_mode.json, trigger_now flag) — atomic single source."""
    sd = settings.state_dir
    return sd / "mode.json", sd / "pending_mode.json", sd / "trigger_now.flag"


def _cmd_mode(args: list[str]) -> str:
    """Preview a mode change. Operator must /confirm to apply."""
    from trading.runtime.mode import (
        Mode,
        PendingModeChange,
        read_mode,
        write_pending,
    )

    if not args:
        # No arg → just show current mode
        cur = read_mode(_mode_paths()[0])
        return (
            f"*Current mode:* `{cur.mode.value}`\n"
            f"set by `{cur.set_by}` at `{cur.set_at or 'default'}`\n"
            f"reason: _{cur.reason or 'n/a'}_\n\n"
            "Send `/mode bull|neutral|defense|bear|flatten` to preview a change."
        )

    try:
        target = Mode.parse(args[0])
    except ValueError as e:
        return f"❌ {e}"

    cur = read_mode(_mode_paths()[0])
    if cur.mode == target:
        return f"already in `{target.value}` mode — nothing to do."

    # Best-effort impact preview from the latest snapshot.
    preview_lines = _build_mode_preview(cur.mode, target)

    # Stage the change. /confirm reads it back.
    pending = PendingModeChange(
        new_mode=target,
        requested_at=datetime.now(tz=timezone.utc).isoformat(),
        requested_by="telegram",
        reason=" ".join(args[1:]) if len(args) > 1 else "",
    )
    _, pending_path, _ = _mode_paths()
    write_pending(pending_path, pending)

    return (
        f"📋 *Mode change preview — `{target.value.upper()}`*\n"
        f"current: `{cur.mode.value}` → proposed: `{target.value}`\n\n"
        f"{preview_lines}\n\n"
        "Reply *CONFIRM* (or /confirm) within 10 min, or /cancel."
    )


def _cmd_confirm() -> str:
    """Apply the staged mode change + fire an off-cycle rebalance."""
    from trading.runtime.mode import (
        clear_pending,
        read_pending,
        write_mode,
    )

    mode_path, pending_path, trigger_path = _mode_paths()
    pending = read_pending(pending_path)
    if pending is None:
        return "no pending mode change. Send `/mode <name>` first."
    if pending.is_expired():
        clear_pending(pending_path)
        return "⏱️ pending mode change expired. Re-send `/mode <name>`."

    state = write_mode(mode_path, pending.new_mode, set_by="telegram", reason=pending.reason)
    clear_pending(pending_path)

    # Drop the trigger-now flag — the runner picks this up and runs a
    # cycle immediately instead of waiting for the cron.
    trigger_path.parent.mkdir(parents=True, exist_ok=True)
    trigger_path.write_text(
        json.dumps({"reason": f"mode change to {state.mode.value}", "ts": state.set_at})
    )
    logger.bind(mode=state.mode.value).info("telegram confirmed mode change")
    return (
        f"✅ Mode set to *{state.mode.value.upper()}*.\n"
        f"🔄 Off-cycle rebalance queued — runner will pick it up within ~30s."
    )


def _cmd_cancel() -> str:
    from trading.runtime.mode import clear_pending, read_pending

    _, pending_path, _ = _mode_paths()
    pending = read_pending(pending_path)
    if pending is None:
        return "nothing to cancel."
    clear_pending(pending_path)
    return f"❌ pending `{pending.new_mode.value}` change cancelled."


def _build_mode_preview(current_mode: object, target_mode: object) -> str:
    """Estimate trades + cost for the staged mode change.

    Falls back to a generic preview if we can't read a current snapshot
    (e.g. the runner hasn't completed its first cycle yet).
    """
    try:
        from trading.runner.state import RunnerStore
        from trading.selection.mode_overlay import ModePolicy, apply_mode

        store = RunnerStore(settings.state_dir / "runner.db")
        snap = store.latest_snapshot()
    except Exception:
        snap = None

    if snap is None or not snap.positions or snap.equity <= 0:
        return (
            "_No snapshot yet — can't compute trade list. Trades will be "
            "computed by the runner on the next cycle._"
        )

    # Current weights from the snapshot
    cur_weights = {}
    for _key, pos in snap.positions.items():
        mv = pos.quantity * pos.avg_price + pos.unrealized_pnl
        cur_weights[pos.instrument.symbol] = mv / snap.equity if snap.equity > 0 else 0.0
    cur_w = pd.Series(cur_weights, dtype=float)  # type: ignore[name-defined]

    # Estimate target by applying the mode to the *current* weights as a proxy
    # (the real recompute happens in the runner — this preview is just illustrative)
    cur_frame = cur_w.to_frame().T
    prices_proxy = cur_frame.copy() * 0 + 1.0  # placeholder; mode overlay reads columns only
    # Inject defensive ETF columns so the overlay can place them
    for tkr in ModePolicy().defensive_sleeve:
        prices_proxy[tkr] = 1.0
    target_frame = apply_mode(cur_frame, prices_proxy, target_mode, policy=ModePolicy())  # type: ignore[arg-type]
    target_w = target_frame.iloc[-1]

    from trading.selection.mode_overlay import estimate_mode_impact

    impact = estimate_mode_impact(cur_w, target_w, equity=snap.equity)

    sells = impact["sells"][:5]
    buys = impact["buys"][:5]
    lines = []
    if sells:
        lines.append("*Sells (top 5):*")
        for r in sells:
            lines.append(
                f"  `{r['symbol']:<6}` Δw `{r['weight_delta']:+.2%}` ≈ `${r['dollar_delta']:+,.0f}`"
            )
    if buys:
        lines.append("*Buys (top 5):*")
        for r in buys:
            lines.append(
                f"  `{r['symbol']:<6}` Δw `{r['weight_delta']:+.2%}` ≈ `${r['dollar_delta']:+,.0f}`"
            )
    lines.append("")
    lines.append(
        f"_Turnover:_ `{impact['turnover_pct']:.1%}` (`${impact['turnover_dollar']:,.0f}`)"
    )
    lines.append(f"_Est. cost:_ `${impact['trading_cost_dollar']:,.2f}` (~10 bps)")
    lines.append(f"_Net gross change:_ `{impact['net_gross_delta']:+.2%}`")
    return "\n".join(lines)


def _cmd_report() -> str:
    """Generate a fresh weekly report and return its executive summary."""
    try:
        import contextlib

        from trading.reporting import (
            fetch_news_for_symbols,
            gather_weekly_report,
            summarise,
        )

        report = gather_weekly_report(fetch_vix=True)
        if report.positions:
            with contextlib.suppress(Exception):
                # News fetch is best-effort; never let a network blip
                # kill the report.
                report.news_by_symbol = fetch_news_for_symbols(
                    list(report.positions.keys()), max_per_symbol=3
                )
        return summarise(report)
    except Exception as e:
        return f"report generation failed: `{e}`"


# ---------------------------------------------------------------------------
# Manual trading commands — queue work, return acknowledgement, runner
# executes asynchronously and pushes a result alert.
# ---------------------------------------------------------------------------


def _queue_command(cmd_type: str, args: dict[str, Any]) -> str | None:
    r"""Submit a command to the runner via the file queue.

    Returns ``None`` on success — the bot stays silent until the runner
    posts the actual outcome (✅ submitted / ❌ rejected). Operators
    flagged the previous "📋 Queued ID — runner will execute within ~5s"
    message as spam because the real result lands ~5s later and the
    queue ack added no useful information.

    Returns an error string only when the request itself is malformed
    (unknown command type) so the bot can tell the user immediately.
    """
    from trading.runtime.commands import Command, CommandType, submit

    try:
        cmd = Command.new(CommandType(cmd_type), args=args, requested_by="telegram")
    except ValueError:
        return f"❌ unknown command `{cmd_type}`"
    submit(cmd, settings.state_dir)
    return None  # silent — wait for runner's result message


def _looks_like_number(s: str) -> bool:
    """True if ``s`` parses as a positive number (or 'all')."""
    if s.lower() == "all":
        return True
    try:
        return float(s) > 0
    except ValueError:
        return False


def _looks_like_ticker(s: str) -> bool:
    r"""True if ``s`` looks like a stock ticker: 1-5 alphanumeric chars,
    optionally with `.`, `-` (BRK.A, RDS-A). Not bulletproof — just to
    catch typos like ``/buy 10 appl`` early with a clear message."""
    if not s or len(s) > 8:
        return False
    cleaned = s.replace(".", "").replace("-", "")
    return cleaned.isalnum() and cleaned[0].isalpha()


def _parse_order_args(args: list[str], action: str) -> tuple[str, str, str | None] | str:
    r"""Parse buy/sell args supporting either order:

      /buy AAPL 10           — SYMBOL first
      /buy 10 AAPL           — QTY first (more natural English)
      /buy AAPL 10 195.50    — with limit price
      /buy 10 AAPL 195.50    — with limit price (qty-first)
      /sell AAPL all
      /sell all AAPL

    Returns (symbol, qty, limit_or_None) on success, or an error string.
    """
    if not args:
        return (
            f"usage: `/{action} SYMBOL QTY [LIMIT_PRICE]` — "
            f"e.g. `/{action} AAPL 10` or `/{action} 10 AAPL`"
        )

    # Strategy: identify which arg is the ticker (alpha-leading) and
    # which is the quantity (numeric or "all"). Then any third arg is
    # the limit price.
    if len(args) == 1:
        # Only one arg — must be a symbol (for sell-all)
        if not _looks_like_ticker(args[0]):
            return f"❌ `{args[0]}` doesn't look like a ticker (1-5 letters)"
        if action == "sell":
            return (args[0].upper(), "all", None)
        return f"usage: `/{action} SYMBOL QTY` — e.g. `/{action} AAPL 10`"

    a, b = args[0], args[1]
    if _looks_like_ticker(a) and _looks_like_number(b):
        symbol, qty = a, b
    elif _looks_like_number(a) and _looks_like_ticker(b):
        symbol, qty = b, a
    else:
        return (
            f"❌ couldn't parse `{a}` `{b}` as (symbol, qty). "
            f"Expected one to be a ticker (e.g. AAPL) and the other a "
            f"number (e.g. 10) — got `{a}`, `{b}`."
        )

    limit = args[2] if len(args) >= 3 else None
    if limit is not None and not _looks_like_number(limit):
        return f"❌ limit price `{limit}` is not a number"
    return symbol.upper(), qty, limit


def _cmd_buy(args: list[str]) -> str:
    r"""``/buy AAPL 10`` or ``/buy 10 AAPL`` — accepts either arg order."""
    parsed = _parse_order_args(args, "buy")
    if isinstance(parsed, str):
        return parsed
    symbol, qty, limit = parsed
    payload: dict[str, Any] = {"symbol": symbol, "qty": qty}
    if limit is not None:
        payload["limit"] = limit
    return _queue_command("buy", payload)


def _cmd_sell(args: list[str]) -> str:
    r"""``/sell AAPL 5`` or ``/sell 5 AAPL`` or ``/sell AAPL all`` — defaults to closing the full position."""
    parsed = _parse_order_args(args, "sell")
    if isinstance(parsed, str):
        return parsed
    symbol, qty, limit = parsed
    payload: dict[str, Any] = {"symbol": symbol, "qty": qty}
    if limit is not None:
        payload["limit"] = limit
    return _queue_command("sell", payload)


def _cmd_close(args: list[str]) -> str:
    r"""``/close SYM`` — flatten a specific position (alias for ``/sell SYM all``)."""
    if len(args) < 1:
        return "usage: `/close SYMBOL` — e.g. `/close AAPL`"
    return _queue_command("close", {"symbol": args[0].upper()})


def _cmd_flatten() -> str:
    r"""``/flatten`` — close every open position at market."""
    return _queue_command("flatten", {})


def _cmd_cancel_order(args: list[str]) -> str:
    r"""``/cancel_order CLIENT_ORDER_ID`` — cancel a specific pending order."""
    if not args:
        return "usage: `/cancel_order CLIENT_ORDER_ID`"
    return _queue_command("cancel_order", {"client_order_id": args[0]})


def _cmd_orders() -> str:
    r"""``/orders`` — recent orders from the local store, grouped by status.

    Previously this iterated ``load_orders`` results as if each row were
    a flat ``Order``, but the store actually returns
    ``(Order, OrderStatus, broker_id)`` tuples — every call was throwing
    AttributeError. Now: properly unpacked, status shown, and pending
    orders surface at the top so the operator sees in-flight work first.
    """
    try:
        from trading.core.types import OrderStatus
        from trading.execution.store import OrderStore

        store = OrderStore(settings.state_dir / "orders.db")
        from datetime import timedelta

        recent = store.load_orders(since=datetime.now(tz=timezone.utc) - timedelta(days=7))
    except Exception as e:
        return f"could not read orders: {e}"

    if not recent:
        return "no orders in the last 7 days."

    pending_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED}
    pending: list[tuple[Any, OrderStatus, str | None]] = []
    other: list[tuple[Any, OrderStatus, str | None]] = []
    for o, st, bid in recent:
        if st in pending_statuses:
            pending.append((o, st, bid))
        else:
            other.append((o, st, bid))

    lines: list[str] = []
    if pending:
        lines.append(f"⏳ *In flight: {len(pending)} order(s)*")
        for o, st, _bid in pending[-15:]:
            lines.append(
                f"  `{o.client_order_id[:14]:<14}` "
                f"{o.side.value} {o.quantity:g} {o.instrument.symbol} "
                f"({o.order_type.value}) — {st.value}"
            )
        lines.append("")
    lines.append(f"*Recent (last 7d, {len(other)} order(s)):*")
    for o, st, _bid in other[-15:]:
        lines.append(
            f"  `{o.client_order_id[:14]:<14}` "
            f"{o.side.value} {o.quantity:g} {o.instrument.symbol} — {st.value}"
        )
    return "\n".join(lines)


def _cmd_pending_orders() -> str:
    r"""``/pending`` — only orders currently in flight.

    Audit fix #7. The operator submits a basket or manual /buy, then
    needs visibility into "did it actually fill yet?" Live broker queries
    happen via the cycle (slow); this is the bot-side view of what's
    SUBMITTED but not yet terminal.
    """
    try:
        from trading.core.types import OrderStatus
        from trading.execution.store import OrderStore

        store = OrderStore(settings.state_dir / "orders.db")
        from datetime import timedelta

        recent = store.load_orders(since=datetime.now(tz=timezone.utc) - timedelta(days=2))
    except Exception as e:
        return f"could not read orders: {e}"

    pending_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED}
    in_flight = [(o, st, bid) for (o, st, bid) in recent if st in pending_statuses]
    if not in_flight:
        return "✅ no orders currently in flight (per local store)."

    lines = [f"⏳ *{len(in_flight)} order(s) in flight* (per local store):"]
    now = datetime.now(tz=timezone.utc)
    for o, st, bid in in_flight:
        age_s = (now - o.created_at).total_seconds()
        age_str = f"{age_s / 60:.0f}m" if age_s >= 60 else f"{age_s:.0f}s"
        bid_str = f" broker={bid}" if bid else ""
        lines.append(
            f"  `{o.client_order_id[:14]:<14}` {st.value} {age_str} ago — "
            f"{o.side.value} {o.quantity:g} {o.instrument.symbol}"
            f" ({o.order_type.value}){bid_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FX commands
# ---------------------------------------------------------------------------


def _cmd_balances() -> str:
    r"""``/balances`` — show per-currency cash from the last account snapshot.

    Reads the snapshot the runner writes at end-of-cycle, so the data
    can be hours old if cycles have been failing. A stale-snapshot
    warning is prepended when the snapshot is past the freshness
    threshold — without that, the operator can't tell a fresh balance
    from a fossilised one.
    """
    try:
        from trading.runner.state import RunnerStore

        store = RunnerStore(settings.state_dir / "runner.db")
        snap = store.latest_snapshot()
    except Exception as e:
        return f"could not read snapshot: `{e}`"
    if snap is None:
        return (
            "_no account snapshot yet — runner hasn't completed a cycle._\n"
            "Try `/cycle` to force one, or wait for Friday."
        )
    prefix = _snapshot_age_warning(snap.ts) or ""
    ccy = getattr(snap, "base_currency", None) or "USD"
    lines = [f"*Account balances* (as of {snap.ts.strftime('%Y-%m-%d %H:%M UTC')}):"]
    lines.append(f"  total cash: `{ccy} {snap.cash:,.2f}`")
    lines.append(f"  total equity: `{ccy} {snap.equity:,.2f}`")

    per_ccy = getattr(snap, "cash_by_currency", None) or {}
    if per_ccy:
        lines.append("")
        lines.append("*Cash by currency:*")
        for code, amt in sorted(per_ccy.items()):
            if abs(amt) < 1.0:
                continue
            lines.append(f"  `{code:<5} {amt:>14,.2f}`")
    return prefix + "\n".join(lines)


def _cmd_fx_rate(args: list[str]) -> str:
    r"""``/fx-rate USD CHF`` — show reference rate from yfinance (delayed).

    Argument parsing: either two args (``USD CHF``) or one pair (``USDCHF``).
    """
    if not args:
        return "usage: `/fx-rate USD CHF` — show reference rate"
    if len(args) == 1 and len(args[0]) >= 6:
        base, quote = args[0][:3].upper(), args[0][3:6].upper()
    else:
        base, quote = args[0].upper(), args[1].upper() if len(args) >= 2 else "USD"
    try:
        import yfinance as yf

        symbol = f"{base}{quote}=X"
        data = yf.download(symbol, period="2d", auto_adjust=True, progress=False)
        if data.empty:
            return f"❌ no FX quote for `{symbol}`"
        close = data["Close"]
        # yfinance can return either a Series or 1-col DataFrame here.
        last = close.iloc[-1]
        if hasattr(last, "iloc"):
            last = last.iloc[0]
        rate = float(last)
        return (
            f"*FX reference rate* `{base}/{quote}`: `{rate:.4f}`\n"
            f"_source: yfinance (delayed up to ~15 min). The trade fill "
            f"will use IBKR's live IDEALPRO rate at the time of execution._"
        )
    except Exception as e:
        return f"❌ FX rate lookup failed: `{e}`"


_KNOWN_CCYS = {"USD", "CHF", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD"}


def _cmd_fx(args: list[str]) -> str:
    r"""Convert currency at market — flexible parsing.

    All of these work:
      /fx 5000 CHF                 — 5000 CHF → USD  (default destination)
      /fx 5000 CHF to USD          — explicit destination (English)
      /fx 5000 CHF USD             — explicit destination (terse)
      /fx CHF 5000                 — old form, still supported
      /fx CHF 5000 USD             — old form with destination
      /fx 5000 USD to CHF          — works either direction

    ``/convert`` is registered as an alias for the same handler.
    """
    if not args:
        return (
            "*Usage:*\n"
            "  `/fx 5000 CHF` — convert 5000 CHF to USD\n"
            "  `/fx 5000 CHF to USD` — explicit destination\n"
            "  `/fx 5000 USD to CHF` — either direction\n\n"
            "_Same command also responds to `/convert`._"
        )

    # Drop English filler words so natural phrasing works.
    tokens = [t for t in args if t.lower() not in ("to", "→", "->", "into", "for")]

    amount: float | None = None
    ccys: list[str] = []
    for t in tokens:
        if _looks_like_number(t) and t.lower() != "all":
            if amount is None:
                amount = float(t)
            else:
                return f"❌ unexpected second number: `{t}`"
        else:
            ccys.append(t.upper())

    if amount is None or amount <= 0:
        return "❌ missing or invalid amount. Try `/fx 5000 CHF to USD`"
    if not ccys:
        return "❌ missing currency. Try `/fx 5000 CHF` or `/fx 5000 CHF to USD`"

    bad = [c for c in ccys if c not in _KNOWN_CCYS]
    if bad:
        return (
            f"❌ unsupported currency: `{', '.join(bad)}`. "
            f"Supported: {', '.join(sorted(_KNOWN_CCYS))}"
        )

    from_ccy = ccys[0]
    # Default destination: the OTHER major between USD and CHF (matches user's CHF base account)
    to_ccy = ccys[1] if len(ccys) >= 2 else ("USD" if from_ccy != "USD" else "CHF")
    if from_ccy == to_ccy:
        return f"❌ source and destination are the same (`{from_ccy}`)"

    return _queue_command(
        "fx_convert",
        {"from_ccy": from_ccy, "to_ccy": to_ccy, "amount": amount},
    )


# ---------------------------------------------------------------------------
# Reliability / health commands
# ---------------------------------------------------------------------------


def _cmd_health() -> str:
    r"""``/health`` — broker, scheduler, and heartbeat state at a glance."""
    sd = settings.state_dir
    hb_age = _heartbeat_age()
    hb_line = "unknown (no cycle yet)" if hb_age is None else f"{hb_age:.0f}s ago"
    halt_path = sd / "halt.json"
    halt_state = "🟢 not halted"
    if halt_path.exists():
        try:
            payload = json.loads(halt_path.read_text())
            if payload.get("halted"):
                halt_state = f"🛑 HALTED — `{payload.get('reason', '')}`"
        except Exception:
            halt_state = "⚠️ halt.json unparseable"
    pending_path = sd / "commands" / "pending"
    n_pending = len(list(pending_path.glob("*.json"))) if pending_path.exists() else 0
    running_path = sd / "commands" / "running"
    n_running = len(list(running_path.glob("*.json"))) if running_path.exists() else 0
    return (
        "*Health*\n"
        f"env: `{settings.trading_env}`  live armed: `{settings.is_live_armed()}`\n"
        f"halt: {halt_state}\n"
        f"heartbeat: {hb_line}\n"
        f"command queue: `{n_pending}` pending, `{n_running}` running\n"
    )


def _cmd_cycle_now() -> str:
    r"""``/cycle`` — force one off-cycle execution immediately."""
    sd = settings.state_dir
    trigger_path = sd / "trigger_now.flag"
    sd.mkdir(parents=True, exist_ok=True)
    trigger_path.write_text(
        json.dumps(
            {
                "reason": "telegram /cycle",
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    )
    return "🔄 cycle triggered — basket preview + orders in ~4 minutes."


def _cmd_refresh() -> str:
    r"""``/refresh`` — queue a data-refresh command for the runner."""
    return _queue_command("refresh_data", {})


def _cmd_reconnect() -> str:
    r"""``/reconnect`` — bounce the broker connection."""
    return _queue_command("reconnect_broker", {})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _dispatch(text: str) -> str | None:
    """Parse a command and return a reply, or None if not a command."""
    if not text or not text.startswith("/"):
        return None
    parts = shlex.split(text)
    cmd = parts[0].lower().split("@")[0]  # strip "@botname" suffix
    args = parts[1:]

    if cmd in ("/start", "/help"):
        return HELP_TEXT
    if cmd == "/status":
        return _cmd_status()
    if cmd == "/heartbeat":
        return _cmd_heartbeat()
    if cmd == "/positions":
        return _cmd_positions()
    if cmd == "/halt":
        return _cmd_halt(args)
    if cmd == "/resume":
        return _cmd_resume()
    if cmd == "/report":
        return _cmd_report()
    if cmd == "/mode":
        return _cmd_mode(args)
    if cmd == "/confirm":
        return _cmd_confirm()
    if cmd == "/cancel":
        return _cmd_cancel()
    # --- manual orders ---
    if cmd == "/buy":
        return _cmd_buy(args)
    if cmd == "/sell":
        return _cmd_sell(args)
    if cmd == "/close":
        return _cmd_close(args)
    if cmd == "/flatten":
        return _cmd_flatten()
    if cmd == "/orders":
        return _cmd_orders()
    if cmd in ("/pending", "/pending_orders", "/pending-orders"):
        return _cmd_pending_orders()
    if cmd == "/cancel_order":
        return _cmd_cancel_order(args)
    # --- FX ---
    if cmd == "/balances":
        return _cmd_balances()
    if cmd in ("/fx-rate", "/fx_rate", "/fxrate"):
        return _cmd_fx_rate(args)
    if cmd in ("/fx", "/convert"):
        return _cmd_fx(args)
    # --- Reliability ---
    if cmd == "/health":
        return _cmd_health()
    if cmd in ("/cycle", "/cycle_now"):
        return _cmd_cycle_now()
    if cmd == "/refresh":
        return _cmd_refresh()
    if cmd == "/reconnect":
        return _cmd_reconnect()
    return f"unknown command `{cmd}` — try /help"


async def run_bot() -> None:
    """Run the long-poll loop. Blocks until cancelled.

    Authorization: only messages from ``settings.telegram_chat_id`` are
    processed. Everything else is silently ignored.
    """
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set in .env")

    logger.info("telegram bot starting (long-poll)")
    offset = 0
    async with httpx.AsyncClient() as client:
        # Greet on startup so the operator knows the bot is up.
        await _send(
            client,
            token,
            chat_id,
            "🤖 *Bot online.* Use /help for commands.",
        )
        while True:
            updates = await _get_updates(client, token, offset)
            for upd in updates:
                offset = max(offset, upd["update_id"] + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                # Authorization: ignore anything not from the configured chat.
                msg_chat = str(msg.get("chat", {}).get("id"))
                if msg_chat != str(chat_id):
                    logger.warning(f"telegram unauthorized chat {msg_chat}")
                    continue
                text = msg.get("text", "")
                reply = await _dispatch(text)
                if reply is not None:
                    await _send(client, token, chat_id, reply)
