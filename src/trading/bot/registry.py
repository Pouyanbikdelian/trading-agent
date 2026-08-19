"""What every bot command is called, and how to use it.

One table, three consumers: the typo matcher (``suggest``), the usage
errors raised by handlers, and the test that pins every dispatched
command to an entry here. Keeping them together means a new command
cannot ship with a misspelt suggestion list or an absent usage line —
the coverage test fails first.

Why a registry rather than more strings inside ``_dispatch``: the
operator reads these messages on a phone, away from the machine, usually
right after something did not work. A near-miss like ``/postions`` used
to return "unknown command — try /help", which costs a round trip and
tells them nothing. Matching it to ``/positions`` costs no tokens and no
LLM call.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

# How close a typo must be to earn a suggestion.
#
# Similarity alone is not enough. At 0.6, "/whats" (as in "/whats up with
# INTC") matches BOTH "/halt" and "/status" — offering a halt to someone
# asking a question is the worst failure this matcher can have. So a
# candidate must also share the first letter: real typos transpose,
# double or drop letters in the middle ("/postions", "/aprove",
# "/statuss") and almost never change the first one. A near miss we
# reject merely falls through to the copilot, which answers it.
#
# 0.7 rather than difflib's usual 0.6: at 0.6 "/whats" still reaches
# "/why" (ratio exactly 0.6). Genuine typos score far higher — /postions
# 0.95, /aprove 0.93, /halr 0.80 — so the tighter bar costs nothing real.
_CUTOFF = 0.7

# Applied when the message carries trailing words, which usually means a
# sentence rather than a command. See ``suggest(strict=True)``.
_STRICT_CUTOFF = 0.85


@dataclass(frozen=True)
class Spec:
    """One command: what to type, and one example of typing it."""

    name: str
    summary: str
    usage: str = ""
    example: str = ""
    aliases: tuple[str, ...] = field(default=())

    def usage_line(self) -> str:
        """A one- or two-line 'how to type this' reply."""
        parts = [f"usage: `{self.usage or self.name}`"]
        if self.example:
            parts.append(f"e.g. `{self.example}`")
        return " — ".join(parts)


REGISTRY: tuple[Spec, ...] = (
    # --- status & data
    Spec("/help", "command list", aliases=("/start",)),
    Spec("/status", "env, halted, heartbeat"),
    Spec("/health", "broker, heartbeat, queue at a glance"),
    Spec("/positions", "open positions + weights"),
    Spec("/balances", "cash by currency, equity"),
    Spec("/orders", "last 7d of orders, grouped"),
    Spec("/pending", "orders currently working", aliases=("/pending_orders", "/pending-orders")),
    Spec("/heartbeat", "age of the last cycle"),
    Spec("/report", "weekly report"),
    Spec("/correlation", "12m correlation matrix of holdings", aliases=("/corr",)),
    Spec("/memory", "calibration, trust, lessons"),
    Spec(
        "/watchlist",
        "show the dashboard watchlist or propose an add/remove",
        usage="/watchlist [add|remove SYMBOL|undo]",
        example="/watchlist add NVDA",
        aliases=("/watch",),
    ),
    Spec(
        "/lesson",
        "propose a durable desk lesson; approval applies it",
        usage="/lesson <what the desk should remember, and when it applies>",
        example="/lesson I want you to add this lesson: never average into a falling knife",
    ),
    Spec(
        "/lessons",
        "review what we've learned; propose hardening or softening one",
        usage="/lessons [harden|soften <lesson-id>]",
        example="/lessons harden ls-3f2a91c4",
    ),
    Spec(
        "/edge",
        "did our picks beat the names we passed on?",
        usage="/edge [why] [5|21|63]",
        example="/edge why 21",
    ),
    # --- cycle approval
    Spec(
        "/approve",
        "submit the pending basket",
        usage="/approve [N|only SYM ...|all except SYM ...|flat]",
        example="/approve 80",
    ),
    Spec("/proposal", "repeat the exact pending buy/sell plan"),
    Spec(
        "/review",
        "show a non-executable live-account review",
        aliases=("/plan",),
    ),
    Spec("/candidates", "alternate ranks for the pending cycle"),
    Spec("/reject", "skip this cycle, no orders"),
    Spec(
        "/pick",
        "override the basket with these ranks",
        usage="/pick RANK [RANK ...]",
        example="/pick 1 3 5 8",
    ),
    # --- regime & signal
    Spec("/regime", "HMM bull/bear state + triggers", aliases=("/state",)),
    Spec(
        "/signal",
        "top-N candidates the strategy would pick now",
        usage="/signal [N]",
        example="/signal 10",
        aliases=("/signals",),
    ),
    # --- trigger work
    Spec(
        "/cycle",
        "force a rebalance; while halted, a reduce-only basket or a review",
        aliases=("/cycle_now",),
    ),
    Spec("/refresh", "queue a data refresh"),
    Spec("/reconnect", "bounce the broker connection"),
    Spec(
        "/gateway",
        "release or retake the IBKR session for manual trading",
        usage="/gateway stop|start|status",
        example="/gateway stop",
    ),
    # --- copilot
    Spec(
        "/ask", "ask about past decisions or state", usage="/ask QUESTION", example="/ask why MU?"
    ),
    Spec("/why", "why did we buy/sell/hold it", usage="/why SYMBOL", example="/why MU"),
    Spec(
        "/thesis", "latest thesis + is it still valid", usage="/thesis SYMBOL", example="/thesis MU"
    ),
    Spec(
        "/committee",
        "convene a debate, or show a symbol's history",
        usage="/committee [SYMBOL]",
        example="/committee MU",
    ),
    Spec("/detail", "full transcript of the latest debate"),
    Spec("/forget", "reset the conversation thread", aliases=("/newtopic",)),
    Spec(
        "/mandates",
        "standing instructions for the next run",
        usage="/mandates [clear|drop ID]",
        example="/mandates drop M123",
        aliases=("/notes",),
    ),
    Spec("/harden", "raise a mandate's strength", usage="/harden ID", example="/harden M123"),
    Spec("/soften", "lower a mandate's strength", usage="/soften ID", example="/soften M123"),
    # --- manual orders
    Spec("/buy", "buy a symbol", usage="/buy SYM QTY [LIMIT]", example="/buy AAPL 10 180"),
    Spec("/sell", "sell a symbol", usage="/sell SYM [QTY|all] [LIMIT]", example="/sell AAPL all"),
    Spec("/close", "close one position", usage="/close SYMBOL", example="/close AAPL"),
    Spec("/flatten", "close every open position"),
    Spec(
        "/cancel_order",
        "cancel a pending order",
        usage="/cancel_order CLIENT_ID",
        example="/cancel_order a1b2c3",
    ),
    Spec(
        "/hold",
        "pin a position the cycle must not touch",
        usage="/hold SYMBOL",
        example="/hold NVDA",
    ),
    Spec("/unhold", "release a pinned position", usage="/unhold SYMBOL", example="/unhold NVDA"),
    Spec("/holds", "list pinned positions"),
    Spec(
        "/exclude",
        "never buy this symbol again (selling still works)",
        usage="/exclude SYMBOL [reason]",
        example="/exclude PM tobacco, not for this book",
    ),
    Spec("/unexclude", "lift a standing ban", usage="/unexclude SYMBOL", example="/unexclude PM"),
    Spec("/exclusions", "list the standing never-buy names"),
    Spec("/k", "override the strategy top-K", usage="/k [N|clear]", example="/k 12"),
    Spec("/pm", "PM research simulation", usage="/pm [run]", example="/pm run"),
    # --- mode
    Spec(
        "/mode",
        "preview a rebalance posture change",
        usage="/mode bull|neutral|defense|bear|flatten",
        example="/mode defense",
    ),
    Spec("/confirm", "apply the previewed mode + run now"),
    Spec("/cancel", "discard the previewed mode"),
    # --- FX
    Spec(
        "/fx",
        "convert at market",
        usage="/fx AMT CCY to CCY",
        example="/fx 5000 CHF to USD",
        aliases=("/convert",),
    ),
    Spec(
        "/fx-rate",
        "reference rate",
        usage="/fx-rate CCY CCY",
        example="/fx-rate USD CHF",
        aliases=("/fx_rate", "/fxrate"),
    ),
    # --- safety
    Spec(
        "/baseline",
        "show or re-stamp the kill-switch high-water mark",
        usage="/baseline [reset [reason]]",
        example="/baseline reset accepted the August drawdown",
    ),
    Spec(
        "/halt",
        "refuse new exposure; explicit reduce-only exits remain available",
        usage="/halt [reason]",
        example="/halt bad data",
    ),
    Spec("/resume", "clear the halt, reset the failure counter"),
)


_BY_NAME: dict[str, Spec] = {}
for _spec in REGISTRY:
    _BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec


def all_names() -> tuple[str, ...]:
    """Every accepted command token, canonical names and aliases alike."""
    return tuple(sorted(_BY_NAME))


def find(cmd: str) -> Spec | None:
    """Look up a command by name or alias. Case-insensitive."""
    return _BY_NAME.get(cmd.lower())


def suggest(cmd: str, limit: int = 2, *, strict: bool = False) -> list[str]:
    """Canonical commands close enough to ``cmd`` to be worth offering.

    Deliberately conservative: a wrong guess sends the operator to a
    command they did not want, which on this system can mean orders. An
    empty list is the honest answer for anything that is not a near miss,
    and the caller then treats the text as a question instead.

    ``strict`` raises the bar for messages that carry trailing words.
    "/hows the book" matches /holds at 0.73 — good enough for a bare
    token, not good enough when the shape of the message says the
    operator was writing a sentence. Real typos still clear the higher
    bar (/aprove 80 → /approve scores 0.93).
    """
    token = cmd.lower()
    if not token.startswith("/"):
        token = "/" + token
    # Three-letter fragments are usually the beginning of prose ("/wat…")
    # rather than a command typo.  Suggesting a long command from one is
    # noisier than helpful and can hide the /help fallback.
    if len(token) < 5:
        return []
    pool = [n for n in all_names() if n[1:2] == token[1:2]]
    cutoff = _STRICT_CUTOFF if strict else _CUTOFF
    hits = difflib.get_close_matches(token, pool, n=limit * 2, cutoff=cutoff)
    out: list[str] = []
    for h in hits:
        spec = _BY_NAME[h]
        if spec.name not in out:
            out.append(spec.name)
        if len(out) >= limit:
            break
    return out


def usage_for(cmd: str) -> str:
    """The usage line for ``cmd``, or a bare pointer if it is unknown."""
    spec = find(cmd)
    return spec.usage_line() if spec else "try `/help` for the command list"
