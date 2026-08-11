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
import contextlib
import json
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from trading.bot import registry
from trading.core.config import settings
from trading.core.logging import logger

BOT_API_BASE = "https://api.telegram.org"
POLL_TIMEOUT = 25  # seconds — long-poll
HELP_TEXT = (
    "*Trading bot — commands*\n\n"
    "*Status & data*\n"
    "/status — env, halted, heartbeat\n"
    "/health — broker, heartbeat, queue at a glance\n"
    "/positions — open positions + weights\n"
    "/balances — cash by currency, equity\n"
    "/orders — last 7d of orders (grouped)\n"
    "/pending — orders currently working\n"
    "/heartbeat — age of last cycle\n"
    "/report — weekly report\n\n"
    "*Cycle approval* (when REQUIRE\\_CYCLE\\_APPROVAL=true)\n"
    "/approve — submit pending basket as-is\n"
    "/approve N — scale to N% (e.g. `/approve 80`)\n"
    "/approve flat — flatten instead of basket\n"
    "/pick 1 3 5 8 — override basket with these ranks\n"
    "/reject — skip this cycle, no orders\n\n"
    "*Regime & signal*\n"
    "/regime — HMM bull/bear state + SPY/VIX triggers\n"
    "/signal \\[N] — top-N candidates the strategy would pick NOW (no order)\n\n"
    "*Trigger work*\n"
    "/cycle — force a rebalance now (~2 min)\n"
    "/refresh — queue a data refresh\n\n"
    "*Copilot* (read-only — explains, never trades)\n"
    "/ask QUESTION — anything about past decisions or current state\n"
    "/why SYM — why did we buy/sell/hold it, and what happened\n"
    "/thesis SYM — latest thesis + is it still valid\n"
    "/committee SYM — decision history for a symbol (bare /committee still convenes)\n\n"
    "*Manual orders* (queued, run within ~5s)\n"
    "/buy SYM QTY \\[LIMIT] — e.g. `/buy AAPL 10 180`\n"
    "/sell SYM \\[QTY|all] \\[LIMIT] — e.g. `/sell AAPL all`\n"
    "/close SYM — close one position\n"
    "/flatten — close every open position\n"
    "/hold SYM | /unhold SYM | /holds — pin/release positions the cycle must not touch\n"
    "/k N | /k clear — override the strategy top-K at runtime\n"
    "/correlation — 12m correlation matrix of current holdings\n"
    "/memory — permanent-memory vitals: calibration, trust, lessons\n"
    "/lesson <text> — teach the desk something; a firm tone makes it binding\n"
    "/lessons [harden|soften <id>] — review or re-weight what we've learned\n"
    "/edge [5|21|63] — did our picks beat the names we passed on?\n"
    "/edge why — the breakdown: rank, market conditions, entry level\n"
    "/detail — full transcript of the latest committee debate\n"
    "/committee — convene the agents for a fresh debate right now\n"
    "/pm — simulated agent-PM book · /pm run — rebalance it now\n"
    "/cancel\\_order CLIENT\\_ID — cancel a pending order\n\n"
    "*Mode (rebalance posture)*\n"
    "/mode bull|neutral|defense|bear|flatten — preview\n"
    "/confirm — apply previewed mode + run now\n"
    "/cancel — discard preview\n\n"
    "*FX*\n"
    "/fx 5000 CHF to USD — convert at market\n"
    "/fx-rate USD CHF — reference rate\n\n"
    "*Safety*\n"
    "/halt \\[reason] — refuse new trades; force flatten\n"
    "/resume — clear halt, reset failure counter\n"
    "/reconnect — bounce the broker connection\n"
    "/gateway stop|start|status — release your IBKR session so you can trade by hand in TWS or mobile; stop also halts\n"
)


# ---------------------------------------------------------------------------
# Low-level Bot API helpers
# ---------------------------------------------------------------------------


def _split_for_telegram(text: str, limit: int = 3800, max_chunks: int = 4) -> list[str]:
    """Split a long reply into <=limit chunks on line boundaries.

    Telegram hard-caps messages at 4096 chars; long committee transcripts
    were being cut mid-sentence. Multiple messages beat truncation."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit and current:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        chunks[-1] += "\n…(truncated)"
    return chunks


class ButtonReply(str):
    """A reply that carries an inline keyboard.

    str subclass for the same reason as PlainReply: every existing caller
    and test that treats a reply as text keeps working, and only the send
    path needs to know about the extra attribute.
    """

    markup: dict[str, Any]

    def __new__(cls, text: str, markup: dict[str, Any]) -> ButtonReply:
        obj = super().__new__(cls, text)
        obj.markup = markup
        return obj


class PlainReply(str):
    """A reply that must be sent WITHOUT Telegram markdown parsing.

    LLM-generated copilot answers are not markdown-safe: legacy Telegram
    markdown silently eats underscore pairs, so an answer quoting an
    evidence key like ``NOW_agent_pm_simulated_book`` rendered as
    ``NOWagentpmsimulatedbook`` (operator report, 2026-07-16). str
    subclass so every existing dispatch/test path is unaffected.
    """


async def _send(
    client: httpx.AsyncClient,
    token: str,
    chat_id: str,
    text: str,
    *,
    plain: bool = False,
    reply_markup: dict[str, Any] | None = None,
) -> None:
    """POST sendMessage. Never raises — bot loop must keep running.

    Markdown parse failures (400 with "can't parse entities") have caused us
    to drop critical alerts (e.g. broker rejection messages with unbalanced
    backticks). On any 400 we retry once as plain text so the operator
    *always* sees the message; aesthetics lose to deliverability.
    ``plain=True`` skips markdown entirely (see PlainReply).

    ``reply_markup`` attaches an inline keyboard. It survives the plain-text
    retry: losing the buttons would leave the operator staring at an
    approval prompt with no way to answer it except typing.
    """
    url = f"{BOT_API_BASE}/bot{token}/sendMessage"
    base: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        base["reply_markup"] = reply_markup
    try:
        if plain:
            r = await client.post(url, json=base, timeout=10.0)
            if r.status_code >= 400:
                logger.warning(f"telegram send failed: {r.status_code} {r.text[:200]}")
            return
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


async def _answer_callback(
    client: httpx.AsyncClient, token: str, callback_id: str, text: str = ""
) -> None:
    """Acknowledge a button tap so Telegram stops showing the spinner.

    Unanswered callbacks leave the button spinning for ~30s, which reads
    as "the bot is dead". Answered first, acted on second — the ack is
    cosmetic and must never be blocked by the work."""
    try:
        await client.post(
            f"{BOT_API_BASE}/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text[:200]},
            timeout=10.0,
        )
    except Exception as e:
        logger.warning(f"telegram answerCallbackQuery error: {e}")


async def _strip_keyboard(
    client: httpx.AsyncClient, token: str, chat_id: str, message_id: int
) -> None:
    """Remove the inline keyboard from a message that has been acted on.

    Idempotency by removal: once the buttons are gone the same prompt
    cannot be tapped twice, which matters when the two taps would be
    ``Approve`` and then ``Approve`` again on a re-run cycle."""
    try:
        await client.post(
            f"{BOT_API_BASE}/bot{token}/editMessageReplyMarkup",
            json={"chat_id": chat_id, "message_id": message_id, "reply_markup": {}},
            timeout=10.0,
        )
    except Exception as e:
        logger.warning(f"telegram editMessageReplyMarkup error: {e}")


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
    """Stop trading. Does NOT liquidate — see risk/halt_file.py.

    This used to reply "Next cycle will force-flatten positions" while
    writing a flag that nothing read. An operator who halted in an
    emergency would have walked away believing the book was being
    closed. The reply now states what is actually true, including
    whether anything is still watching the positions.
    """
    from trading.risk.halt_file import describe_exposure, set_halted

    reason = " ".join(args) if args else "telegram"
    # set_halted read-modify-writes under the cross-process file lock, so
    # the kill-switch baselines survive (audit fix #9 kept, plus the
    # baseline-erasure bug found 2026-08-07).
    set_halted(settings.state_dir, halted=True, reason=reason)
    logger.bind(reason=reason).warning("telegram halt")
    return f"🛑 *HALTED* — reason: `{reason}`\nNo new orders will be submitted.\n{describe_exposure(settings.state_dir)}"


def _cmd_hold(args: list[str]) -> str:
    """``/hold NVDA`` — pin a symbol: the cycle will neither sell nor add
    to it until ``/unhold``. Manual /buy /sell /close /flatten still work."""
    from trading.runner.holds import load_holds, save_holds

    if not args:
        return registry.usage_for("/hold") + " — freeze a position out of the rebalance cycle"
    sym = args[0].upper()
    holds = load_holds(settings.state_dir)
    if sym in holds:
        return f"`{sym}` is already held. `/holds` to list."
    holds.add(sym)
    save_holds(settings.state_dir, holds)
    logger.bind(symbol=sym).info("telegram hold added")
    msg = (
        f"\U0001f4cc `{sym}` pinned — the cycle will not touch it (no sells, no adds).\n"
        f"_Manual `/sell {sym} ...` and `/flatten` still work. "
        f"`/unhold {sym}` to release._"
    )
    # Ghost pins were one of the six defects found in the July 2026 audit.
    # The SLOT half was fixed then — apply_runtime_overrides now ignores a
    # pin with no position, so the basket is no longer one name short.
    # This message still said it "reserves a basket slot" until 2026-08-10,
    # describing behaviour that had been gone for a month.
    #
    # What remains true is the part operators actually trip over: a hold
    # freezes a symbol in BOTH directions. Pinning something you do not
    # own does not protect anything — it stops the cycle ever buying it.
    if not _holds_position(sym):
        msg += (
            f"\n\n⚠️ `{sym}` isn't in the current book, so there's nothing to "
            "protect. A hold blocks BUYS as well as sells, so this stops the "
            f"cycle ever opening `{sym}` until you `/unhold {sym}`."
        )
    return msg


def _holds_position(symbol: str) -> bool:
    """True if the latest snapshot shows a non-zero position in ``symbol``.

    Best-effort: an unreadable snapshot returns True so we stay quiet
    rather than warn wrongly."""
    try:
        from trading.runner.state import RunnerStore

        snap = RunnerStore(settings.state_dir / "runner.db").latest_snapshot()
        if snap is None or not snap.positions:
            return False
        return any(
            p.instrument.symbol.upper() == symbol.upper() and p.quantity
            for p in snap.positions.values()
        )
    except Exception:
        return True


def _cmd_unhold(args: list[str]) -> str:
    from trading.runner.holds import load_holds, save_holds

    if not args:
        return registry.usage_for("/unhold")
    sym = args[0].upper()
    holds = load_holds(settings.state_dir)
    if sym not in holds:
        return f"`{sym}` is not held. `/holds` to list."
    holds.discard(sym)
    save_holds(settings.state_dir, holds)
    logger.bind(symbol=sym).info("telegram hold removed")
    return f"\U0001f513 `{sym}` released — the next cycle manages it normally again."


def _cmd_holds() -> str:
    from trading.runner.holds import load_holds

    holds = load_holds(settings.state_dir)
    if not holds:
        return "_no pinned positions — `/hold <sym>` to pin one._"
    return "\U0001f4cc *Pinned positions* (cycle won't touch):\n" + "\n".join(
        f"  • `{s}`" for s in sorted(holds)
    )


def _cmd_k(args: list[str]) -> str:
    """``/k 12`` — override the strategy's top-K at runtime; ``/k`` shows;
    ``/k clear`` reverts to the configured default."""
    from trading.runner.holds import load_holds, load_k_override, save_k_override

    current = load_k_override(settings.state_dir)
    held = len(load_holds(settings.state_dir))
    if not args:
        base = f"`{current}`" if current else "_not set (strategy default applies)_"
        tail = f"\n_{held} pinned position(s) each reserve one slot on top._" if held else ""
        return f"Top-K override: {base}{tail}"
    if args[0].lower() in ("clear", "off", "none", "reset"):
        save_k_override(settings.state_dir, None)
        return "Top-K override cleared — next cycle uses the configured default."
    try:
        k = int(args[0])
    except ValueError:
        return "usage: `/k 12` to set, `/k` to show, `/k clear` to reset"
    if not 1 <= k <= 50:
        return "k must be between 1 and 50"
    save_k_override(settings.state_dir, k)
    note = f" ({held} held position(s) will reserve slots on top)" if held else ""
    return f"Top-K set to `{k}` from the next cycle{note}."


def _cmd_correlation() -> str:
    """``/correlation`` — pairwise 12m correlation of current holdings."""
    from trading.runner.state import RunnerStore
    from trading.runtime.portfolio_stats import format_correlation, holdings_correlation

    try:
        store = RunnerStore(settings.state_dir / "runner.db")
        snap = store.latest_snapshot()
    except Exception as e:
        return f"could not read snapshot: `{e}`"
    if snap is None or not snap.positions:
        return "_portfolio is flat — nothing to correlate._"
    syms = sorted({p.instrument.symbol for p in snap.positions.values()})
    if len(syms) < 2:
        return f"_only one holding (`{syms[0]}`) — correlation needs at least two._"
    corr = holdings_correlation(syms, settings.data_dir)
    if corr is None:
        return "_not enough cached price history for these names — run a data fetch first._"
    return format_correlation(corr)


def _cmd_lesson(args: list[str]) -> str:
    """``/lesson <statement>`` — write a lesson into permanent memory.

    Status follows the operator's TONE, the same grading the mandate path
    uses: an instruction phrased as an instruction ("I want you to add
    this lesson…") lands as ``established`` and reaches every agent from
    the next cycle; anything softer lands as a ``candidate`` and is shown
    to the desk as under consideration.

    Why tone rather than a flag: the operator already writes to this
    system in natural language and the mandate path already reads
    firmness from phrasing. A second, different convention for lessons
    would be a thing to remember rather than a thing to use.

    Before this existed there was no way to write a lesson at all — the
    copilot filed "we should buy quality after 4-5 down days" as a soft
    MANDATE, which influences one run and then expires, when what was
    wanted was a durable, gradeable belief.
    """
    from trading.copilot.mandates import STRONG, grade_strength
    from trading.core.text import clip
    from trading.memory.store import default_store

    statement = " ".join(args).strip()
    if not statement:
        return (
            "usage: `/lesson <what the desk should remember>`\n"
            "Tone sets the weight — a firm instruction is added outright, "
            "anything softer arrives as a candidate for the desk to weigh.\n"
            "`/lessons` to review, `/lessons harden <id>` to promote."
        )
    if len(statement) < 15:
        return "_that is too short to be a lesson — say what to do and when it applies._"

    strength = grade_strength(statement)
    status = "established" if strength == STRONG else "candidate"
    try:
        mem = default_store()
        lid = mem.add_lesson(statement, tags=f"operator {strength}", status=status)
    except Exception as e:
        return f"could not write the lesson: `{e}`"

    if status == "established":
        head = "🧠 *Lesson added* — established, in front of every agent from the next cycle."
        tail = f"Too strong? `/lessons soften {lid}`."
    else:
        head = "🧠 *Lesson recorded* — candidate, the desk will weigh it and may disagree."
        tail = f"Want it binding? `/lessons harden {lid}`."
    return f"{head}\n`{lid}` · {clip(statement, 300)}\n_{tail}_"


def _cmd_lessons(args: list[str]) -> str:
    """``/lessons [harden|soften <id>]`` — review or re-weight the book."""
    from trading.core.text import clip
    from trading.memory.store import default_store

    try:
        mem = default_store()
    except Exception as e:
        return f"could not open memory: `{e}`"

    if args and args[0].lower() in ("harden", "soften"):
        if len(args) < 2:
            return f"usage: `/lessons {args[0].lower()} <lesson-id>`"
        want = "established" if args[0].lower() == "harden" else "candidate"
        lid = args[1].strip()
        try:
            ok = mem.set_lesson_status(lid, want)
        except Exception as e:
            return f"could not change `{lid}`: `{e}`"
        if not ok:
            return f"_no live lesson `{lid}` — `/lessons` to list them._"
        return f"`{lid}` is now *{want}*."

    est = mem.lessons(status="established")
    cand = mem.lessons(status="candidate")
    if not est and not cand:
        return "_no lessons yet. `/lesson <statement>` writes the first one._"

    def _rows(rows: list[Any], limit: int) -> list[str]:
        out = []
        for r in rows[:limit]:
            who = "👤" if "operator" in (r["tags"] or "") else "📚"
            score = (
                f" ({r['support']}/{r['contradict']})" if r["support"] or r["contradict"] else ""
            )
            out.append(f"  {who} `{r['id']}`{score} {clip(r['statement'], 220)}")
        return out

    lines = ["🧠 *Lessons* — 👤 yours · 📚 the historian's"]
    if est:
        lines.append(f"\n*Established* ({len(est)}) — every agent sees these:")
        lines.extend(_rows(est, 8))
    if cand:
        lines.append(f"\n*Candidates* ({len(cand)}) — proposed, not yet earned:")
        lines.extend(_rows(cand, 8))
    lines.append("\n_`/lessons harden <id>` promotes · `/lesson <text>` adds_")
    return "\n".join(lines)


def _cmd_memory() -> str:
    """``/memory`` — permanent-memory vitals: counts, calibration, trust."""
    from trading.memory.store import default_store

    try:
        mem = default_store()
        s = mem.stats()
    except Exception as e:
        return f"could not open memory: `{e}`"
    lines = [
        "\U0001f9e0 *Permanent memory*",
        f"  journal `{s['journal']}` | episodes `{s['episodes']}` | "
        f"lessons `{s['lessons']}` | dossiers `{s['dossiers']}`",
        f"  predictions `{s['predictions']}` | sources tracked `{s['sources']}`",
    ]
    cal = mem.calibration()
    if cal:
        lines.append("")
        lines.append("*Agent calibration* (graded, hit rate, Brier):")
        for c in cal[:6]:
            lines.append(
                f"  `{c['agent']:<10}` n={c['n']}  hit {c['hit_rate']:.0%}  brier {c['brier']:.2f}"
            )
    trust = mem.trust_table(min_graded=2)
    if trust:
        lines.append("")
        lines.append("*Source trust* (top by evidence):")
        for r in trust[:6]:
            lines.append(f"  `{r['source']:<16}` {r['trust']:.2f} ({r['graded']} graded)")
    est = mem.lessons(status="established")
    if est:
        lines.append("")
        lines.append(f"*Top lesson:* {est[0]['statement'][:140]}")
    return "\n".join(lines)


# Below this many graded names on a side, a spread is an anecdote and
# gets shown with a warning rather than as a result.
_EDGE_MIN_N = 20


def _cmd_edge(args: list[str]) -> str:
    """``/edge [5|21|63]`` — did our picks beat the names we passed on?
    ``/edge why [5|21|63]`` — where that answer comes from.

    The one report the desk had no way to produce before the shadow
    ledger existed. Everything is quoted net of SPY over the identical
    window, because absolute forward returns on a list of passed names
    look wonderful in any rising market.
    """
    from trading.memory.store import default_store

    args = list(args)
    want_why = bool(args) and args[0].lower() in ("why", "detail", "breakdown")
    if want_why:
        args = args[1:]

    leg = 21
    if args:
        try:
            leg = int(args[0])
        except ValueError:
            return "usage: `/edge [why] [5|21|63]` — the forward window in days"
    if leg not in (5, 21, 63):
        return "usage: `/edge [why] [5|21|63]` — the forward window in days"

    if want_why:
        return _cmd_edge_why(leg)

    try:
        mem = default_store()
        report = mem.edge_report(leg_days=leg)
    except Exception as e:
        return f"could not read the shadow ledger: `{e}`"

    if not report:
        return (
            f"📊 *Selection edge, {leg}d* — nothing graded yet.\n"
            f"_The ledger records every ranked candidate each cycle and needs "
            f"{leg} days to mature. Check back._"
        )

    lines = [f"📊 *Selection edge, {leg}d forward, net of SPY*", ""]
    thin = False
    for r in report:
        taken, passed, spread = r.get("taken_excess"), r.get("passed_excess"), r.get("spread")
        # Distinct names, not rows. The ladder re-ranks the same universe
        # daily, so rows pile up while the real sample barely grows — and
        # each row's forward window overlaps its neighbour's almost
        # entirely. Quoting rows would make forty observations read as
        # twelve hundred.
        s_t, s_p = r.get("n_taken_symbols", 0), r.get("n_passed_symbols", 0)
        head = f"*{r['origin']}*  picks {s_t} names · passes {s_p} names"
        if spread is None:
            lines.append(f"{head}\n  _no counterfactual on this origin yet_")
            continue
        mark = "✅" if spread > 0 else "❌"
        if min(s_t, s_p) < _EDGE_MIN_N:
            mark, thin = "⚠️", True
        lines.append(
            f"{head}\n  picks {taken:+.2%} · passes {passed:+.2%} · spread {spread:+.2%} {mark}"
        )
    lines.append("")
    if thin:
        lines.append("⚠️ _= fewer than 20 distinct names on a side. Not yet a result._")
    obs = sum(r.get("n_taken", 0) + r.get("n_passed", 0) for r in report)
    names = sum(r.get("n_taken_symbols", 0) + r.get("n_passed_symbols", 0) for r in report)
    if obs > names:
        lines.append(
            f"_Counts are distinct names. {obs} observations underlie them — "
            "the same names are re-ranked daily, so rows overstate the sample._"
        )
    lines.append(
        "_Positive spread means the ranking step added something. "
        "Negative means the names we rejected did better._"
    )
    lines.append("_`/edge why` for the breakdown._")
    return "\n".join(lines)


def _cmd_edge_why(leg: int) -> str:
    """``/edge why`` — the three slices behind the headline number.

    All three are SQL over columns recorded at decision time. Nothing
    here asks a model to explain a result: a fluent invented cause is
    worse than an admitted gap, because it gets remembered and repeated.
    """
    from trading.memory.store import default_store

    try:
        mem = default_store()
        by_rank = mem.edge_by_rank(leg_days=leg)
        by_vol = mem.edge_by_condition("vol_bucket", leg_days=leg)
        by_entry = mem.edge_by_entry(leg_days=leg)
    except Exception as e:
        return f"could not read the shadow ledger: `{e}`"

    if not (by_rank or by_vol or by_entry):
        return (
            f"📊 *Why, {leg}d* — nothing graded yet.\n"
            f"_Needs {leg} days of matured rows. `/edge` for the headline._"
        )

    # Distinct names throughout — see the note in _cmd_edge on why row
    # counts flatter the sample here.
    def thin(n_symbols: int) -> str:
        return " ⚠️" if n_symbols < _EDGE_MIN_N else ""

    lines = [f"🔍 *Why — {leg}d forward, net of SPY*", ""]

    if by_rank:
        lines.append("*1. Does the score rank anything?*")
        for r in by_rank:
            s = r.get("n_symbols", 0)
            lines.append(f"  rank `{r['rank_bucket']:<6}` {r['excess']:+.2%}  {s} names{thin(s)}")
        spread = _rank_discrimination(by_rank)
        if spread is not None:
            verdict = (
                "top of the ladder beats the bottom — the score works, the cut may be misplaced"
                if spread > 0.005
                else "flat across the ladder — the score is not discriminating"
            )
            lines.append(f"  _{verdict}_")
        lines.append("")

    if by_vol:
        lines.append("*2. When does the edge exist?*")
        for r in by_vol:
            sp = r.get("spread")
            s_t, s_p = r.get("n_taken_symbols", 0), r.get("n_passed_symbols", 0)
            if sp is None:
                lines.append(f"  `{r['condition']:<9}` no counterfactual yet")
                continue
            lines.append(
                f"  `{r['condition']:<9}` spread {sp:+.2%}  {s_t}/{s_p} names{thin(min(s_t, s_p))}"
            )
        lines.append("")

    if by_entry:
        lines.append("*3. Does buying near the highs work?*")
        for r in by_entry:
            s = r.get("n_symbols", 0)
            lines.append(f"  `{r['entry_bucket']:<9}` {r['excess']:+.2%}  {s} names{thin(s)}")
        lines.append("")

    lines.append("⚠️ _= fewer than 20 distinct names. Read it as a hint, not a result._")
    return "\n".join(lines)


def _rank_discrimination(by_rank: list[dict[str, Any]]) -> float | None:
    """Best bucket minus worst, or None when there isn't enough to compare.

    The single number that says whether the ranking is doing any work.
    """
    vals = [r["excess"] for r in by_rank if r.get("excess") is not None and r["n"] > 0]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def _cmd_detail() -> str:
    """``/detail`` — full transcript of the latest committee debate."""
    import json as _json

    from trading.agents.committee import format_digest

    path = settings.state_dir / "last_committee.json"
    if not path.exists():
        return (
            "_no committee run recorded yet — it convenes Mon & Fri ~1pm ET "
            "plus a late-day de-risk check; or run /committee now._"
        )
    try:
        digest = _json.loads(path.read_text())
    except Exception as e:
        return f"could not read last committee run: `{e}`"
    # Long transcripts are split into multiple messages by the send loop.
    return format_digest(digest)


def _cmd_committee() -> str:
    """``/committee`` — convene the agent committee now (fresh debate)."""
    flag = settings.state_dir / "committee_now.flag"
    try:
        flag.write_text(datetime.now(tz=timezone.utc).isoformat())
    except Exception as e:
        return f"could not request committee: `{e}`"
    return (
        "\U0001f3db Committee convening — fresh debate underway.\n"
        "_Summary lands here in ~2 minutes; `/detail` for the full transcript._"
    )


def _cmd_pm(args: list[str]) -> str:
    """``/pm`` — simulated agent-PM book; ``/pm run`` — rebalance now."""
    import json as _json

    if args and args[0].lower() == "run":
        flag = settings.state_dir / "agent_pm_now.flag"
        try:
            flag.write_text(datetime.now(tz=timezone.utc).isoformat())
        except Exception as e:
            return f"could not request PM run: `{e}`"
        return "🧪 Agent PM convening — rebalance lands here in ~1 minute."

    from trading.agents.pm import _marks_age_note, _money, format_holdings, performance
    from trading.core.text import clip as _clip

    pm_dir = settings.state_dir / "agent_pm"
    try:
        book = _json.loads((pm_dir / "portfolio.json").read_text())
    except Exception:
        return "_no agent-PM book yet — it trades Mondays 14:30 UTC, or `/pm run` to convene now._"
    perf = performance(settings.state_dir)
    # Name the book and the currency on every figure: this sim is USD,
    # the real account reports CHF, and they land in the same chat.
    lines = [
        "🧪 *Agent PM — simulated book* (USD, not the trading account)",
        f"equity {_money(float(perf.get('equity', 0.0)))} "
        f"({perf.get('return_pct', 0.0):+.2f}% since inception)",
    ]
    if "spy_return_pct" in perf:
        alpha = perf["return_pct"] - perf["spy_return_pct"]
        lines.append(
            f"vs SPY {perf['spy_return_pct']:+.2f}% same window (alpha {alpha:+.2f}pp) "
            f"· max DD {perf['max_drawdown_pct']:.1f}% · {perf['points']} marks"
        )
    lines.append("")
    lines.append(f"*Holdings*{_marks_age_note(book)}")
    lines.extend(format_holdings(book))
    try:
        last = _json.loads((pm_dir / "last_run.json").read_text())
        lines.append("")
        lines.append(f"_Last rationale: {_clip(last.get('rationale', ''), 400)}_")
    except Exception:
        pass
    return "\n".join(lines)


def _halt_state() -> tuple[bool, str]:
    """(halted, reason) from halt.json; (False, "") when absent/unreadable."""
    halt_path = settings.state_dir / "halt.json"
    if not halt_path.exists():
        return False, ""
    try:
        payload = json.loads(halt_path.read_text())
        return bool(payload.get("halted", False)), str(payload.get("reason", ""))
    except Exception:
        return False, ""


def _cmd_resume() -> str:
    # Report the state rather than acting on a no-op. "✅ RESUMED" when
    # nothing was halted reads as confirmation that a halt was lifted,
    # which is exactly the wrong thing to believe on a phone.
    halted, _reason = _halt_state()
    counter_path = settings.state_dir / "consecutive_errors.json"
    if not halted:
        stale = 0
        try:
            if counter_path.exists():
                stale = int(json.loads(counter_path.read_text()).get("count", 0))
        except Exception:
            pass
        if not stale:
            return "not halted — nothing to resume. `/status` for the current state."
        # A live failure counter is still worth clearing even unhalted:
        # it is what would trip the next auto-halt.

    from trading.risk.halt_file import set_halted

    # Preserves daily_equity_open / equity_high_watermark / last_day. The
    # old four-key overwrite reset both kill switches on every resume,
    # so a drawdown spanning a halt could never trip the drawdown cap.
    set_halted(settings.state_dir, halted=False)

    # Reset the consecutive-error counter. Without this, an operator who
    # auto-halted at N failures and /resume'd would re-halt on the first
    # subsequent failure (counter still at N, next failure → N+1 ≥
    # threshold → auto-halt). The runner reloads this file at cycle
    # start so the zero takes effect on the next cycle.
    counter_was = None
    try:
        if counter_path.exists():
            existing = json.loads(counter_path.read_text())
            counter_was = int(existing.get("count", 0))
        _atomic_write_json(counter_path, {"count": 0})
    except Exception:
        logger.bind(component="bot").exception("could not reset consecutive_errors")

    logger.info("telegram resume")
    msg = "✅ *RESUMED* — halt cleared." if halted else "✅ Was not halted; state re-armed."
    if counter_was and counter_was > 0:
        msg += f"\n   Failure counter reset (was {counter_was})."
    return msg


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
    lines = [
        "*Trading status*",
        f"env: `{settings.trading_env}`",
        f"live armed: `{settings.is_live_armed()}`",
        f"state: {halt_line}",
        f"heartbeat: {hb_line}",
    ]

    pending = _read_cycle_pending()
    if pending is not None:
        short_id = str(pending.get("id", ""))[:8]
        n = int(pending.get("n_orders", 0))
        pct = float(pending.get("deploy_pct", 0.0))
        lines.append(
            f"\n⏸ *Cycle `{short_id}` awaiting approval* — {n} orders, {pct:.0f}% of equity"
        )
        lines.append("Reply: `/approve` · `/approve 80` · `/approve flat` · `/reject`")
    elif getattr(settings, "require_cycle_approval", False):
        lines.append(
            "\n_cycle approval required for every cycle (REQUIRE\\_CYCLE\\_APPROVAL=true)._"
        )

    # The three things an operator almost always checks straight after
    # /status. All read-only.
    from trading.bot.keyboards import read_only_keyboard

    markup = read_only_keyboard("/positions", "/orders", "/health")
    text = "\n".join(lines)
    return ButtonReply(text, markup) if markup else text


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
        f"{'Symbol':<7} {'Qty':>10} {'Avg cost':>11} {'Mkt value':>12} {'Weight':>7} {'P&L':>11}"
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

    from trading.bot.keyboards import mode_keyboard

    return ButtonReply(
        f"📋 *Mode change preview — `{target.value.upper()}`*\n"
        f"current: `{cur.mode.value}` → proposed: `{target.value}`\n\n"
        f"{preview_lines}\n\n"
        "Tap to decide, or reply /confirm · /cancel within 10 min.",
        mode_keyboard(target.value),
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
        from trading.runtime.mode import read_mode

        cur = read_mode(mode_path)
        return (
            f"nothing staged to confirm — mode is still `{cur.mode.value}`.\n"
            "Preview a change first: `/mode bull|neutral|defense|bear|flatten`."
        )
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
        return (
            "nothing to cancel — no mode change is staged.\n"
            "_(`/cancel` discards a `/mode` preview. To stop a working order "
            "use `/cancel_order`; to stop a pending cycle use `/reject`.)_"
        )
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
        return registry.usage_for("/close")
    return _queue_command("close", {"symbol": args[0].upper()})


def _cmd_flatten() -> str:
    r"""``/flatten`` — close every open position at market."""
    return _queue_command("flatten", {})


def _in_flight_order_ids() -> list[str] | None:
    """Client order ids currently PENDING/SUBMITTED, or None if unreadable."""
    try:
        from datetime import timedelta

        from trading.core.types import OrderStatus
        from trading.execution.store import OrderStore

        store = OrderStore(settings.state_dir / "orders.db")
        recent = store.load_orders(since=datetime.now(tz=timezone.utc) - timedelta(days=2))
    except Exception:
        return None
    live = {OrderStatus.PENDING, OrderStatus.SUBMITTED}
    return [o.client_order_id for (o, st, _bid) in recent if st in live]


def _cmd_cancel_order(args: list[str]) -> str:
    r"""``/cancel_order CLIENT_ORDER_ID`` — cancel a specific pending order.

    Checks the id against the local store before queueing. A typo'd id
    used to be accepted and silently do nothing; the operator, watching
    from a phone, would believe an order had been pulled when it was
    still working.
    """
    if not args:
        return f"{registry.usage_for('/cancel_order')}\n`/pending` lists what's in flight."

    wanted = args[0]
    ids = _in_flight_order_ids()
    if ids is None:
        # Store unreadable — queue it anyway rather than block a cancel,
        # but say the check did not happen. _queue_command returns None on
        # success (the runner posts the real outcome), so an error string
        # from it wins; otherwise we speak up about the skipped check.
        err = _queue_command("cancel_order", {"client_order_id": wanted})
        return err or "_couldn't verify against the local order store — queued anyway._"
    if not ids:
        return "no orders in flight — nothing to cancel. `/orders` for recent history."

    # Accept a unique prefix: /pending prints ids truncated to 14 chars.
    matches = [i for i in ids if i == wanted] or [i for i in ids if i.startswith(wanted)]
    if not matches:
        listed = "\n".join(f"  `{i[:14]}`" for i in ids[:8])
        return f"no in-flight order matching `{wanted}`. Currently working:\n{listed}"
    if len(matches) > 1:
        listed = "\n".join(f"  `{i[:14]}`" for i in matches[:8])
        return f"`{wanted}` matches {len(matches)} orders — be more specific:\n{listed}"
    return _queue_command("cancel_order", {"client_order_id": matches[0]})


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
        return registry.usage_for("/fx-rate")
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


def _cmd_gateway(args: list[str]) -> str:
    r"""``/gateway stop|start|status`` — release or retake the IBKR session.

    IBC runs with ``ExistingSessionDetectedAction=primary``, so the
    container holds the account's ONE session and the operator cannot log
    into TWS or the mobile app while it is up. Stopping it is the only way
    to trade by hand — and doing that previously required an SSH client on
    the phone.

    ``status`` answers locally (the bot has no docker socket, only the
    runner does) by asking the runner's last known container state via the
    command pipeline like everything else.
    """
    action = (args[0].lower() if args else "").strip()
    if action not in ("stop", "start", "status", "state"):
        return (
            "usage: `/gateway stop` · `/gateway start` · `/gateway status`\n"
            "_`stop` halts trading and frees your IBKR session so you can "
            "log into TWS or mobile. `start` brings it back (~90s + 2FA) "
            "and leaves trading HALTED until you `/resume`._"
        )
    if action == "status":
        return _cmd_health()
    # Acted on HERE, not queued. The runner executes queued commands, and
    # the runner cannot start without the gateway (cli.py connects the
    # broker before the scheduler). Queueing made /gateway start
    # unreachable in the one situation it exists for: 2026-08-11, gateway
    # down, runner crash-looping, two /gateway stop commands stuck in the
    # queue waiting for a process that could never come back.
    from trading.runtime.docker_ctl import GATEWAY_CONTAINER, container_state, docker_action

    if action == "state":
        return f"gateway: {container_state()}"
    try:
        if action == "stop":
            # Halt FIRST: the desk must not spend the manual session
            # firing orders at a socket that is about to disappear.
            from trading.risk.halt_file import set_halted

            set_halted(settings.state_dir, halted=True, reason="gateway stopped for manual trading")
            result = docker_action(GATEWAY_CONTAINER, "stop")
            return (
                f"🔌 {result}\n"
                "Trading is HALTED and your IBKR session is free — log into TWS or "
                "mobile now.\n_`/gateway start` when you are done._"
            )
        result = docker_action(GATEWAY_CONTAINER, "start")
        return (
            f"🔌 {result}\n"
            "Logging in (~90s, approve the 2FA push). Still HALTED — "
            "`/health` to watch, then `/resume`."
        )
    except Exception as e:
        return (
            f"⚠️ could not {action} the gateway: `{type(e).__name__}: {e}`\n"
            "_On the VPS: `docker compose up -d ib-gateway`_"
        )


def _cmd_signal(args: list[str]) -> str:
    r"""``/signal [N]`` — top-N candidates the strategy would consider RIGHT NOW.

    Out-of-cycle peek. Loads prices from the cache, runs the
    strategy's ``top_candidates`` on the latest bar, and returns the
    ranked list. Doesn't trigger a cycle, doesn't submit anything —
    pure read-only "what does the strategy see today".
    """
    import os

    n = 15
    if args:
        try:
            n = max(1, min(50, int(args[0])))
        except ValueError:
            return "usage: `/signal [N]` — top N candidates (default 15, max 50)"

    universe = os.getenv("UNIVERSE", "sp500")
    strategy_name = os.getenv("STRATEGY", "top_k_momentum")

    # Load universe symbols (try generated.yaml first for sp500 / nasdaq100,
    # then the curated universes.yaml).
    syms: list[str] = []
    import yaml

    for path in (
        settings.data_dir.parent / "config" / "universes.generated.yaml",
        settings.data_dir.parent / "config" / "universes.yaml",
        Path("/app/config/universes.generated.yaml"),
        Path("/app/config/universes.yaml"),
    ):
        try:
            doc = yaml.safe_load(path.read_text())
            unis = doc.get("universes", doc)
            if universe in unis:
                syms = list(unis[universe].get("symbols", []))
                break
        except Exception:
            continue
    if not syms:
        return (
            f"_universe `{universe}` not found in config/universes\\*.yaml._\n"
            "Set `UNIVERSE=` in .env to one that exists."
        )

    # Load cached prices.
    try:
        from trading.core.types import AssetClass, Instrument
        from trading.data.cache import ParquetCache

        cache = ParquetCache(settings.data_dir)
        series: dict[str, pd.Series] = {}
        for sym in syms:
            ins = Instrument(symbol=sym, asset_class=AssetClass.EQUITY)
            try:
                df = cache.read(ins, "1D")
                if not df.empty and "close" in df.columns:
                    series[sym] = df["close"].dropna()
            except Exception:
                continue
        if not series:
            return (
                f"_no cached prices for `{universe}`. Backfill first:_\n"
                f"`docker compose exec trader trading data fetch {universe} "
                "--from 2022-01-01 --freq 1d`"
            )
        prices = pd.concat(series, axis=1).ffill().dropna(how="all")
    except Exception as e:
        return f"_failed to load prices: {e}_"

    # Construct strategy with defaults + rebalance override if env is set.
    try:
        from trading.strategies.base import get_strategy

        cls = get_strategy(strategy_name)
        kwargs: dict[str, Any] = {}
        # Honor REBALANCE if set, matching how the runner does it.
        rebal = os.getenv("REBALANCE")
        if rebal:
            with contextlib.suppress(ValueError):
                kwargs["rebalance"] = int(rebal)
        strat = cls(cls.Params(**kwargs))
        cands = strat.top_candidates(prices, top_n=n)
    except Exception as e:
        return f"_strategy `{strategy_name}` couldn't rank: {e}_"

    if not cands:
        return (
            f"_strategy `{strategy_name}` returned no candidates._\n"
            f"Universe: `{universe}` ({len(syms)} names, "
            f"{prices.shape[0]} bars)\n"
            "Likely cause: insufficient history. Backfill with the command above."
        )

    last_bar = prices.index[-1]
    lines = [
        f"📊 *Top {len(cands)} candidates* — `{strategy_name}` on `{universe}`",
        f"_(latest bar: {last_bar.strftime('%Y-%m-%d')}, {prices.shape[0]} bars loaded)_",
        "",
    ]
    for i, (sym, score) in enumerate(cands, start=1):
        lines.append(f"  `{i:>2}` `{sym:<6}` `{score:+.2%}`")
    lines.append("")
    lines.append(
        "_This is informational only — to actually trade, run `/cycle`. "
        "These ranks match the candidate list you'd see in the approval prompt._"
    )
    return "\n".join(lines)


def _cmd_regime() -> str:
    r"""``/regime`` — current market regime snapshot.

    Reads two state files written by the runner's background advisors:
    ``state/hmm_advisor.json`` (daily HMM classification of SPY) and
    ``state/advisor.json`` (hourly SPY+VIX risk triggers). These are
    only refreshed at their schedule cadence (daily / hourly), so the
    snapshot can be up to ~24h old for the HMM. The age is shown.
    """
    lines = ["*Market regime snapshot*"]

    hmm_path = settings.state_dir / "hmm_advisor.json"
    if hmm_path.exists():
        try:
            hmm = json.loads(hmm_path.read_text())
            label = str(hmm.get("label", "?"))
            emoji = {"BEAR": "🛑", "NEUTRAL": "🟡", "BULL": "🟢"}.get(label, "ℹ️")
            as_of_iso = hmm.get("as_of", "")
            age = ""
            if as_of_iso:
                try:
                    as_of = datetime.fromisoformat(as_of_iso)
                    if as_of.tzinfo is None:
                        as_of = as_of.replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(tz=timezone.utc) - as_of).total_seconds() / 3600.0
                    age = f" _({age_h:.1f}h ago)_"
                except Exception:
                    pass
            lines.append(
                f"{emoji} HMM: `{label}`{age}\n"
                f"  P(bear)=`{hmm.get('p_bear', 0):.0%}` · "
                f"P(neutral)=`{hmm.get('p_neutral', 0):.0%}` · "
                f"P(bull)=`{hmm.get('p_bull', 0):.0%}`"
            )
        except Exception as e:
            lines.append(f"_HMM state file unreadable: {e}_")
    else:
        lines.append(
            "_HMM regime not classified yet — runs daily at 22:15 UTC. "
            "Returns blank if the runner hasn't completed a daily fit._"
        )

    adv_path = settings.state_dir / "advisor.json"
    if adv_path.exists():
        try:
            adv = json.loads(adv_path.read_text())
            active = list(adv.get("active", []))
            severities = dict(adv.get("severities", {}))
            if active:
                lines.append("\n*Active risk triggers* (SPY/VIX advisor):")
                for name in sorted(active, key=lambda n: -severities.get(n, 0)):
                    lines.append(f"  • `{name}` (severity `{severities.get(name, '?')}`)")
            else:
                lines.append("\n_No active SPY/VIX risk triggers._")
        except Exception:
            pass

    lines.append(
        "\n_Both advisors are informational — they don't auto-change "
        "your mode. Use `/mode bull|neutral|defense|bear|flatten` to act._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cycle approval — operator gates each cycle's basket before submission
# ---------------------------------------------------------------------------


_APPROVAL_PENDING_FILE = "cycle_approval_pending.json"
_APPROVAL_DECISION_FILE = "cycle_approval_decision.json"


def _read_cycle_pending() -> dict[str, Any] | None:
    path = settings.state_dir / _APPROVAL_PENDING_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cycle_decision(action: str, **extra: Any) -> bool:
    """Write the operator's decision for the currently-pending cycle.

    Returns True if a pending cycle was found, False otherwise. The
    cycle on the trader side is polling for this file every ~2s.
    """
    pending = _read_cycle_pending()
    if pending is None:
        return False
    payload = {
        "id": pending.get("id"),
        "action": action,
        "decided_at": datetime.now(tz=timezone.utc).isoformat(),
        **extra,
    }
    _atomic_write_json(settings.state_dir / _APPROVAL_DECISION_FILE, payload)
    return True


def _no_pending_cycle_reply(cmd: str) -> str:
    """Why there is nothing to approve, and what to do instead.

    An approval command arriving out of order is the commonest 'wrong
    order' case: the operator saw a basket, got distracted, and replied
    after the window closed. Saying only "no cycle awaiting approval"
    leaves them guessing whether they missed it, whether it executed, or
    whether the runner is even alive — so report the state we can see.
    """
    lines = [f"nothing awaiting approval, so `{cmd}` has no cycle to act on."]

    age = _heartbeat_age()
    if age is None:
        lines.append("The runner hasn't completed a cycle yet — check `/health`.")
    elif age > 3600:
        lines.append(f"Last cycle finished {age / 3600:.0f}h ago. `/health` for the full picture.")
    else:
        lines.append(f"Last cycle finished {age / 60:.0f} min ago — `/orders` shows what it did.")

    if not getattr(settings, "require_cycle_approval", False):
        lines.append(
            "_Approval isn't enabled (REQUIRE\\_CYCLE\\_APPROVAL=false), "
            "so cycles submit without asking._"
        )
    lines.append("Run `/cycle` to start one now.")
    return "\n".join(lines)


def _cmd_approve(args: list[str]) -> str:
    r"""``/approve [N|flat]`` — approve the currently-pending cycle.

    No arg → submit basket as-is.
    Integer 0-100 → scale order quantities to N% of original.
    ``flat`` → close every position instead of submitting the basket.
    """
    pending = _read_cycle_pending()
    if pending is None:
        return _no_pending_cycle_reply("/approve")

    short_id = str(pending.get("id", ""))[:8]
    if not args:
        if not _write_cycle_decision("approve"):
            return "_pending cycle expired before approval could land._"
        return f"✅ Approved cycle `{short_id}` as-is. Orders submitting…"

    first = args[0].lower()
    if first in ("flat", "flatten"):
        if not _write_cycle_decision("flatten"):
            return "_pending cycle expired._"
        return f"⏸ Cycle `{short_id}` → flatten instead of new basket."

    try:
        pct = float(first)
    except ValueError:
        return (
            f"_unrecognized argument `{args[0]}`._\n"
            "Use: `/approve` · `/approve 80` · `/approve flat`"
        )
    if not 0.0 <= pct <= 100.0:
        return "scale must be between 0 and 100 (% of basket size)."
    factor = pct / 100.0
    if not _write_cycle_decision("scale", scale_factor=factor):
        return "_pending cycle expired._"
    return f"✅ Approved cycle `{short_id}` at {pct:.0f}% — rescaling orders."


def _cmd_reject() -> str:
    r"""``/reject`` — refuse the currently-pending cycle; no orders go out."""
    pending = _read_cycle_pending()
    if pending is None:
        return _no_pending_cycle_reply("/reject")
    short_id = str(pending.get("id", ""))[:8]
    if not _write_cycle_decision("reject"):
        return "_pending cycle expired._"
    return f"❌ Rejected cycle `{short_id}` — no orders will be submitted."


def _cmd_pick(args: list[str]) -> str:
    r"""``/pick 1 3 5 8 11`` — replace the pending basket with a chosen
    subset of the top-N candidate list.

    Picks are 1-based ranks into the ``candidates`` array surfaced by the
    approval prompt. Selected names are equal-weighted at the per-position
    cap, then run back through the risk manager — so all the normal
    limits (gross, sector, no-margin) still apply.
    """
    pending = _read_cycle_pending()
    if pending is None:
        return _no_pending_cycle_reply("/pick")

    if not pending.get("can_pick", False):
        return (
            "_this cycle doesn't expose a candidate scoreboard._\n"
            "Use `/approve`, `/approve N`, `/approve flat`, or `/reject` instead."
        )

    if not args:
        return "usage: `/pick 1 3 5 8 11` — ranks from the candidate list in the approval prompt."

    candidates = pending.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return "_no candidates attached to this cycle — try `/approve` instead._"

    # Accept space- or comma-separated digits; deduplicate while
    # preserving order.
    raw_tokens: list[str] = []
    for a in args:
        raw_tokens.extend(t for t in a.replace(",", " ").split() if t)
    seen: set[int] = set()
    picked_ranks: list[int] = []
    for t in raw_tokens:
        try:
            r = int(t)
        except ValueError:
            return f"_unrecognized rank `{t}`._ Pass integers like `/pick 1 3 5`."
        if r < 1 or r > len(candidates):
            return f"_rank `{r}` out of range._ Pick from 1..{len(candidates)}."
        if r not in seen:
            seen.add(r)
            picked_ranks.append(r)

    picked_symbols = [str(candidates[r - 1].get("symbol", "")) for r in picked_ranks]
    picked_symbols = [s for s in picked_symbols if s]
    if not picked_symbols:
        return "_no valid symbols selected._"

    short_id = str(pending.get("id", ""))[:8]
    if not _write_cycle_decision("pick", picked_symbols=picked_symbols):
        return "_pending cycle expired._"
    return (
        f"✅ /pick on `{short_id}` → {len(picked_symbols)} names: "
        f"`{', '.join(picked_symbols)}`\nRebuilding basket…"
    )


async def _cmd_copilot(
    question: str, symbol: str | None = None, *, replied_to: str | None = None
) -> str:
    """Read-only Investment Committee Copilot (docs/COPILOT.md).

    Runs in a worker thread: the engine does SQLite + one HTTP call and
    must not block the poll loop. It has no order path by construction —
    it explains decisions, it cannot make them."""
    import asyncio as _asyncio

    from trading.copilot import answer as _copilot_answer

    try:
        text = await _asyncio.to_thread(
            _copilot_answer,
            question,
            state_dir=settings.state_dir,
            data_dir=settings.data_dir,
            symbol=symbol,
            replied_to=replied_to,
        )
        # LLM output is not markdown-safe — send verbatim (PlainReply).
        return PlainReply(text)
    except Exception as e:  # never let the copilot take the bot down
        logger.exception("copilot failed")
        return f"copilot error: {type(e).__name__}: {e}"


async def _dispatch(text: str, *, replied_to: str | None = None) -> str | None:
    """Parse a command and return a reply, or None to stay silent.

    ``replied_to`` is the text of the message the operator replied to —
    an alert, usually. It lets "why this alert?" mean something.
    """
    if not text:
        return None
    if not text.startswith("/"):
        # Plain chat goes to the read-only copilot — the bot IS the
        # assistant now, no /ask prefix needed. Guard rails: very short
        # fragments ("ok", "👍") get a hint instead of an LLM call, and
        # the copilot's own 15s rate limit bounds a chatty evening.
        stripped = text.strip()
        if len(stripped) < 8 or not any(c.isalpha() for c in stripped):
            return "Ask me a full question — e.g. `why did we buy MU?` — or /help for commands."
        # A forward-looking instruction ("high conviction on GS, look at
        # it next round") is stored for the next run rather than answered.
        # Captured BEFORE the copilot so it costs no LLM call.
        mandate_reply = _maybe_capture_mandate(stripped)
        if mandate_reply is not None:
            return mandate_reply
        return await _cmd_copilot(stripped, replied_to=replied_to)
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
    if cmd == "/hold":
        return _cmd_hold(args)
    if cmd == "/unhold":
        return _cmd_unhold(args)
    if cmd == "/holds":
        return _cmd_holds()
    if cmd == "/k":
        return _cmd_k(args)
    if cmd in ("/correlation", "/corr"):
        return _cmd_correlation()
    if cmd == "/memory":
        return _cmd_memory()
    if cmd == "/lesson":
        return _cmd_lesson(args)
    if cmd == "/lessons":
        return _cmd_lessons(args)
    if cmd == "/edge":
        return _cmd_edge(args)
    if cmd == "/detail":
        return _cmd_detail()
    if cmd == "/committee":
        # With a symbol argument: copilot history view (read-only).
        # Bare: original behavior — convene the committee.
        if args:
            return await _cmd_copilot(
                f"Show the committee's decision history and votes for {args[0].upper()}: "
                "what was decided, when, with what dissent, and did it execute?",
                symbol=args[0].upper(),
            )
        return _cmd_committee()
    if cmd == "/ask":
        if not args:
            return registry.usage_for("/ask")
        return await _cmd_copilot(" ".join(args))
    if cmd == "/why":
        if not args:
            return registry.usage_for("/why")
        return await _cmd_copilot(
            f"Why did we buy, sell, or hold {args[0].upper()}? What was the thesis, "
            "the votes and dissent, did the trade execute, and what happened after?",
            symbol=args[0].upper(),
        )
    if cmd == "/thesis":
        if not args:
            return registry.usage_for("/thesis")
        return await _cmd_copilot(
            f"What was the committee's most recent thesis for {args[0].upper()}, what were "
            "its invalidation conditions, and is it still valid given current data?",
            symbol=args[0].upper(),
        )
    if cmd == "/pm":
        return _cmd_pm(args)
    if cmd == "/resume":
        return _cmd_resume()
    # --- cycle approval (only meaningful when REQUIRE_CYCLE_APPROVAL=true) ---
    if cmd == "/approve":
        return _cmd_approve(args)
    if cmd == "/reject":
        return _cmd_reject()
    if cmd == "/pick":
        return _cmd_pick(args)
    if cmd in ("/regime", "/state"):
        return _cmd_regime()
    if cmd in ("/signal", "/candidates", "/signals"):
        return _cmd_signal(args)
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
    if cmd == "/gateway":
        return _cmd_gateway(args)
    if cmd in ("/forget", "/newtopic"):
        return _cmd_forget()
    if cmd in ("/mandates", "/notes"):
        return _cmd_mandates(args)
    if cmd == "/soften":
        return _cmd_restrength(args, "soft")
    if cmd == "/harden":
        return _cmd_restrength(args, "strong")
    return await _cmd_unknown(cmd, text)


_STRENGTH_ICON = {"strong": "🔴", "medium": "🟠", "soft": "🟡"}


def _maybe_capture_mandate(text: str) -> str | None:
    """Store a forward-looking instruction, or return None to pass through.

    The reply always states the strength that was read and how to change
    it. Tone parsing is deterministic but not infallible, and the moment
    to catch a misreading is now — while the operator is still looking at
    the screen — not after a run has acted on it.
    """
    from trading.copilot.mandates import MandateStore, mandate_span

    # The span is the instruction clause, not the whole bubble. A message
    # can be one thought wrapped in three others; storing the wrapper made
    # the echo unreadable and the stored mandate vague.
    span = mandate_span(text)
    if span is None:
        return None
    try:
        from trading.copilot.engine import _known_symbols
        from trading.copilot.thread import extract_symbols

        known = _known_symbols(settings.state_dir)
    except Exception:
        known = set()
    try:
        # Universe membership, not just held names: the point of a mandate
        # is often a name we do NOT hold yet.
        from trading.agents.pm import UNIVERSE, _stock_universe

        known = set(known) | set(UNIVERSE) | set(_stock_universe())
    except Exception:
        logger.bind(component="bot").warning("mandate: universe unavailable for symbol match")

    try:
        symbols = extract_symbols(span, known)
        m = MandateStore(settings.state_dir).add(span, symbols=symbols)
    except Exception as e:
        logger.exception("mandate capture failed")
        return f"couldn't save that instruction: `{type(e).__name__}: {e}`"

    syms = f"\nnames: {', '.join(f'`{s}`' for s in m.symbols)}" if m.symbols else ""
    nudge = {
        "strong": "The desk will act on it unless there's a concrete reason not to.",
        "medium": "The desk will weigh it seriously and tell you if it declines.",
        "soft": "The desk will consider it and may drop it.",
    }[m.strength]
    return (
        f"📌 Noted for the next run — read as {_STRENGTH_ICON[m.strength]} *{m.strength}*.\n"
        f"_{m.text}_{syms}\n\n"
        f"{nudge}\n"
        f"_Wrong reading? `/harden {m.id}` or `/soften {m.id}`. "
        f"`/mandates` to review, `/mandates drop {m.id}` to remove._\n"
        "_Risk caps still apply — a mandate can't lift a position or cluster limit._"
    )


def _format_mandate(m: Any) -> str:
    icon = _STRENGTH_ICON.get(m.strength, "•")
    syms = f" [{', '.join(m.symbols)}]" if m.symbols else ""
    return f"{icon} `{m.id}` *{m.strength}*{syms}\n   _{m.text}_"


def _cmd_mandates(args: list[str]) -> str:
    """``/mandates`` — list standing instructions; ``/mandates clear`` wipes;
    ``/mandates drop M123`` cancels one."""
    from trading.copilot.mandates import MandateStore

    store = MandateStore(settings.state_dir)
    if args and args[0].lower() in ("clear", "reset"):
        n = store.clear()
        return f"🧹 cleared {n} standing instruction(s)."
    if args and args[0].lower() in ("drop", "cancel", "remove"):
        if len(args) < 2:
            return "usage: `/mandates drop M123` — id from `/mandates`"
        return (
            f"❌ dropped `{args[1]}`."
            if store.cancel(args[1])
            else f"no active mandate `{args[1]}`."
        )
    active = store.active()
    if not active:
        return (
            "_no standing instructions._\n"
            "Just tell me one — e.g. `high conviction on GS, look at it next round`."
        )
    lines = ["📌 *Standing instructions for the next run*"]
    lines.extend(_format_mandate(m) for m in active)
    lines.append("\n_`/soften ID` · `/harden ID` · `/mandates drop ID`_")
    return "\n".join(lines)


def _cmd_restrength(args: list[str], strength: str) -> str:
    """Re-grade a mandate whose tone the parser read wrongly.

    The echo-back on capture exists so a misread is visible; this is how
    it gets corrected, without retyping the instruction."""
    from trading.copilot.mandates import MandateStore

    if not args:
        return f"usage: `/{'soften' if strength == 'soft' else 'harden'} M123`"
    m = MandateStore(settings.state_dir).set_strength(args[0], strength)
    if m is None:
        return f"no active mandate `{args[0]}`. `/mandates` to list."
    return f"✅ `{m.id}` is now *{strength}*.\n   _{m.text}_"


def _cmd_forget() -> str:
    """``/forget`` — drop the conversation thread and start clean.

    The thread expires on its own after 45 minutes, but an operator who
    has just changed subject shouldn't have to wait or work around a
    carried-over symbol. Journaled exchanges are untouched — this clears
    what the copilot *carries forward*, not what it recorded."""
    from trading.copilot.thread import Thread

    Thread(settings.state_dir).clear()
    return "🧹 conversation reset — next question starts fresh.\n_(journaled exchanges are kept)_"


async def _cmd_unknown(cmd: str, text: str) -> str:
    """Handle a leading-slash token that matched no command.

    Two outcomes, both better than the old "unknown command — try /help":

    * a near miss (``/postions``) gets the command it almost spelled,
      matched locally with no LLM call;
    * anything else is treated as a question. ``/whats up with INTC`` is
      an operator talking, not a malformed command, and the copilot can
      answer it. Being wrong here is cheap — the copilot is read-only and
      has no order path.
    """
    from trading.bot.registry import suggest

    # Trailing words point at a sentence ("/hows the book") rather than a
    # mistyped command ("/postions"), so demand a stronger match before
    # offering one.
    has_trailing_words = len(text.strip().split()) > 1
    hits = suggest(cmd, strict=has_trailing_words)
    if hits:
        options = " or ".join(f"`{h}`" for h in hits)
        return f"no command `{cmd}` — did you mean {options}?"

    # Not close to anything: read it as a sentence. Strip the slash so the
    # copilot sees plain prose.
    question = text.strip().lstrip("/").strip()
    if len(question) < 8 or not any(c.isalpha() for c in question):
        return f"no command `{cmd}` — `/help` for the list, or just ask me a question."
    return await _cmd_copilot(question)


async def _handle_callback(data: str) -> str:
    """Turn a button tap into the same work the typed command would do.

    Every state-changing action re-validates its binding token against
    what is pending *now*. The button in the chat history is a request to
    act on one specific cycle or mode; if that thing is gone, the tap is
    refused with an explanation rather than silently redirected onto
    whatever is current. This is the whole reason callback data carries a
    token at all — see ``bot.keyboards``.
    """
    from trading.bot import keyboards

    parsed = keyboards.decode(data)
    if parsed is None:
        return "that button is from an older version of the bot — use `/help` for current commands."
    action, tokentail = parsed

    if action == keyboards.ACT_RUN:
        # Read-only commands only, checked against the allowlist rather
        # than trusting the callback payload.
        if tokentail not in keyboards.SAFE_COMMANDS:
            logger.warning(f"callback requested non-allowlisted command: {tokentail!r}")
            return f"`{tokentail}` can't be run from a button."
        reply = await _dispatch(tokentail)
        return reply if reply is not None else "done."

    if action in (keyboards.ACT_APPROVE, keyboards.ACT_APPROVE_SCALED, keyboards.ACT_REJECT):
        cid, _, scale = tokentail.partition(":")
        pending = _read_cycle_pending()
        if pending is None:
            return _no_pending_cycle_reply("that button")
        current = str(pending.get("id", ""))[:8]
        if cid and current and cid != current:
            # The dangerous case: an old approval prompt still sitting in
            # the chat history. Approving it would submit a basket the
            # operator never looked at.
            return (
                f"that button belongs to cycle `{cid}`, but the pending cycle is "
                f"`{current}`. Scroll to the newest prompt — it shows the basket "
                "you'd actually be approving."
            )
        if action == keyboards.ACT_REJECT:
            return _cmd_reject()
        return _cmd_approve([scale] if scale else [])

    if action in (keyboards.ACT_MODE_CONFIRM, keyboards.ACT_MODE_CANCEL):
        from trading.runtime.mode import read_pending

        _, pending_path, _ = _mode_paths()
        staged = read_pending(pending_path)
        if staged is None:
            return "nothing staged any more — that preview was already applied or cancelled."
        staged_mode = getattr(staged.new_mode, "value", str(staged.new_mode))
        if tokentail and tokentail != staged_mode:
            return (
                f"that button was for `{tokentail}`, but `{staged_mode}` is staged now. "
                "Use the newest preview."
            )
        return _cmd_confirm() if action == keyboards.ACT_MODE_CONFIRM else _cmd_cancel()

    return "unrecognised button."


_OFFSET_FILE = "telegram_offset.json"


def _load_offset() -> int:
    """Last confirmed Telegram update offset, persisted across restarts.

    Starting from 0 after a crash re-delivers every update Telegram
    hasn't had confirmed — including an order command executed seconds
    before the crash (external review, 2026-07-15). The command TTL
    bounds the damage; persisting the offset removes it. Missing or
    corrupt file degrades to 0 — the old behavior."""
    import json as _json

    path = settings.state_dir / _OFFSET_FILE
    try:
        return int(_json.loads(path.read_text()).get("offset", 0))
    except Exception:
        return 0


def _save_offset(offset: int) -> None:
    """Atomic write; failure is logged, never fatal to the poll loop."""
    import json as _json
    import os as _os
    import tempfile as _tempfile

    path = settings.state_dir / _OFFSET_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
        with _os.fdopen(fd, "w") as f:
            _json.dump({"offset": offset}, f)
        _os.replace(tmp, path)
    except Exception:
        logger.exception("failed to persist telegram offset")


async def _send_reply(client: httpx.AsyncClient, token: str, chat_id: str, reply: str) -> None:
    """Send a dispatch result, honouring PlainReply and ButtonReply.

    A long reply is chunked; the keyboard rides on the LAST chunk so the
    buttons sit directly above the compose box where the operator's thumb
    already is, rather than scrolled off above three screens of text.
    """
    plain = isinstance(reply, PlainReply)
    markup = getattr(reply, "markup", None) if isinstance(reply, ButtonReply) else None
    chunks = _split_for_telegram(reply)
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        await _send(
            client,
            token,
            chat_id,
            chunk,
            plain=plain,
            reply_markup=markup if last else None,
        )


async def _process_callback(
    client: httpx.AsyncClient, token: str, chat_id: str, cb: dict[str, Any]
) -> None:
    """Authorize, acknowledge, disarm, then act — in that order.

    Disarming (stripping the keyboard) happens *before* the action runs.
    The alternative — act, then strip — leaves a live Approve button for
    the duration of the work, and the runner takes seconds to pick up a
    decision. Two impatient taps in that window is exactly the double
    submission this ordering prevents.
    """
    cb_id = str(cb.get("id", ""))
    cb_chat = str((cb.get("message") or {}).get("chat", {}).get("id"))
    if cb_chat != str(chat_id):
        logger.warning(f"telegram unauthorized callback from chat {cb_chat}")
        await _answer_callback(client, token, cb_id, "not authorized")
        return

    await _answer_callback(client, token, cb_id)
    message_id = (cb.get("message") or {}).get("message_id")
    if message_id is not None:
        await _strip_keyboard(client, token, chat_id, int(message_id))

    try:
        reply = await _handle_callback(str(cb.get("data", "")))
    except Exception as e:
        # One bad button must not take the poll loop down with it.
        logger.exception("callback handler failed")
        reply = f"button failed: {type(e).__name__}: {e}"

    await _send_reply(client, token, chat_id, reply)


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
    offset = _load_offset()
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
                # Persist BEFORE dispatching: crash mid-command must not
                # replay it on restart. Losing one un-dispatched command
                # (operator re-sends) beats re-executing one.
                offset = max(offset, upd["update_id"] + 1)
                _save_offset(offset)

                cb = upd.get("callback_query")
                if cb:
                    await _process_callback(client, token, chat_id, cb)
                    continue

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                # Authorization: ignore anything not from the configured chat.
                msg_chat = str(msg.get("chat", {}).get("id"))
                if msg_chat != str(chat_id):
                    logger.warning(f"telegram unauthorized chat {msg_chat}")
                    continue
                text = msg.get("text", "")
                # Telegram hands us the full original when the operator
                # replies to a message. That is how "why this alert?"
                # becomes answerable — otherwise "this" has no referent.
                replied_to = (msg.get("reply_to_message") or {}).get("text")
                reply = await _dispatch(text, replied_to=replied_to)
                if reply is not None:
                    await _send_reply(client, token, chat_id, reply)
