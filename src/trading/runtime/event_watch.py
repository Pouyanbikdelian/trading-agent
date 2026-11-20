"""Event watch — macro & geopolitical news triage, pushed to Telegram.

The gap it fills: news_watch collects headlines for the committee's
*context* (sector/theme scent, read twice a week), and the Sentinel only
wakes on PRICE tripwires. Nothing told the operator "the Fed just moved
off-cycle" or "a strike hit Gulf oil infrastructure" when it happened.

Pipeline, cheapest first (Yan's rule: model it in code when code can):

1. **Fetch** — Google News RSS, ~13 macro/geopolitical queries limited to
   the last day (``when:1d``). Queries name events and institutions, never
   tickers (the news_watch lesson: a ticker in a query becomes the answer).
2. **Dedup** — normalized-title hashes kept 72h, so a headline is judged
   once no matter how many runs see it.
3. **Mechanical score** — keyword weights: hard-event words (emergency,
   invasion, sanctions, default, surprise...) score high; previews,
   explainers and "stocks to buy" listicles score negative. Near-duplicate
   titles are clustered and the number of distinct outlets is a boost —
   one blog is gossip, five wires is an event.
4. **LLM triage** — ONE call per run, and only when something cleared the
   mechanical bar (most runs: zero spend). The model grades severity 1-5
   against a strict rubric, folds duplicates, and is shown what was
   already alerted so it only re-alerts on a genuinely new development.
5. **Notify** — everything at/above ``EVENT_WATCH_MIN_SEVERITY`` (default
   4) from one run goes out as ONE Telegram message. Quiet hours hold
   severity-4 items for a morning digest; severity 5 always goes now.
   A daily cap stops a news storm from becoming a phone storm.

Severity-5 events are recorded with ``review_suggested: true``. That is the
hook for the next step (an approval-gated review cycle); today the alert
only *suggests* ``/committee``. NOTHING here touches the order path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading.core.logging import logger

STATE_FILENAME = "event_watch.json"
TIMEOUT_S = 15.0
MAX_PER_QUERY = 8
SEEN_TTL_HOURS = 72.0
ALERT_MEMORY_HOURS = 48.0
MAX_CANDIDATES_TO_LLM = 40

LlmFn = Callable[[str, str], dict[str, Any]]
FetchFn = Callable[[], list[dict[str, str]]]

# Event-shaped queries. No tickers. ``when:1d`` keeps Google News to the
# last 24h so stale explainers don't resurface.
_QUERIES: dict[str, str] = {
    "fed": "Federal Reserve OR FOMC OR Powell rate decision when:1d",
    "us_data": 'CPI OR inflation OR "jobs report" OR payrolls OR GDP surprise when:1d',
    "global_cb": 'ECB OR "Bank of Japan" OR "Bank of England" OR SNB OR PBOC rates when:1d',
    "war": "military strike OR invasion OR missile attack OR escalation when:1d",
    "middle_east": 'Iran OR Israel OR "Strait of Hormuz" OR Gulf attack when:1d',
    "china_taiwan": 'China Taiwan military OR "export controls" OR "South China Sea" when:1d',
    "russia_nato": "Russia Ukraine NATO attack OR ceasefire when:1d",
    "trade": 'tariffs OR sanctions OR "trade war" announced when:1d',
    "energy_supply": 'OPEC OR "oil supply" OR pipeline disruption OR refinery attack when:1d',
    "fiscal": '"government shutdown" OR "debt ceiling" OR "Treasury auction" OR downgrade when:1d',
    "fin_stress": '"bank failure" OR "credit crisis" OR liquidity crisis OR sovereign default when:1d',
    "market_shock": 'stocks plunge OR selloff OR "circuit breaker" OR "flash crash" when:1d',
    "cyber_infra": 'cyberattack OR "grid outage" OR "critical infrastructure" attack when:1d',
}

# (pattern, weight). Matched on the lowercased title, word-boundary-ish.
_HARD = 3
_SOFT = 1
_KEYWORDS: tuple[tuple[str, int], ...] = (
    # hard events
    (r"emergency", _HARD),
    (r"surpris", _HARD),
    (r"unexpected", _HARD),
    (r"invasion|invades|invade", _HARD),
    (r"declares? war|war with", _HARD),
    (r"missile|airstrike|air strike|drone strike|strikes? on", _HARD),
    (r"nuclear", _HARD),
    (r"sanction", _HARD),
    (r"tariff", _HARD),
    (r"default", _HARD),
    (r"bank (failure|collapse|run)|collapse", _HARD),
    (r"circuit breaker|plunge|crash|rout|tumble|soar", _HARD),
    (r"shutdown", _HARD),
    (r"rate (cut|hike)|cuts rates|hikes rates|raises rates", _HARD),
    (r"blockade|closes? (the )?strait", _HARD),
    (r"assassinat|coup|martial law", _HARD),
    (r"ceasefire|truce", _HARD),
    (r"intervene|intervention", _HARD),
    (r"downgrade", _HARD),
    (r"export (ban|controls?)", _HARD),
    (r"embargo", _HARD),
    (r"hotter|cooler|beats? expectations|misses? expectations", _HARD),
    # context words
    (r"\bfed\b|fomc|powell", _SOFT),
    (r"\becb\b|lagarde|bank of japan|\bboj\b|\bsnb\b", _SOFT),
    (r"\bcpi\b|inflation|payrolls|jobs report|unemployment|gdp", _SOFT),
    (r"opec|\boil\b|brent|crude|natural gas", _SOFT),
    (r"treasur|yields?|bond", _SOFT),
    (r"dollar|yen|yuan", _SOFT),
    (r"china|taiwan|iran|israel|russia|ukraine|nato", _SOFT),
    (r"recession", _SOFT),
    (r"attack", _SOFT),
)
_NOISE: tuple[str, ...] = (
    "what to know",
    "explained",
    "explainer",
    "opinion",
    "preview",
    "week ahead",
    "how to",
    "stocks to buy",
    "to watch",
    "could ",
    "might ",
    "analyst says",
    "here's why",
    "heres why",
    "live updates",
    "podcast",
    "newsletter",
    "quiz",
    "best ",
    "top 5",
    "top 10",
    "should you",
    "1 stock",
    "2 stocks",
    "3 stocks",
)
_NOISE_PENALTY = 2

SEVERITY_LABEL = {5: "🔴 *MAJOR*", 4: "🟠 *IMPORTANT*", 3: "🟡 Notable"}

CHARTER = (
    "You are the desk's macro & geopolitical news officer. You receive fresh "
    "headlines (already de-duplicated and keyword-screened) plus the book's "
    "holdings, and you decide what the operator must hear about NOW. You are "
    "stingy: most hours contain nothing worth a phone buzz.\n\n"
    "SEVERITY RUBRIC (be strict):\n"
    "5 MAJOR — regime-changing, markets will reprice broadly: war or direct "
    "military clash involving a major power or a key energy chokepoint; "
    "emergency/off-cycle central-bank action; systemic bank or sovereign "
    "failure; sweeping new tariffs/sanctions on a top-5 economy; oil supply "
    "shock; market-wide circuit breaker.\n"
    "4 IMPORTANT — likely to move indices, rates, FX or a sector held: a "
    "central-bank decision or guidance that surprised; a big surprise in "
    "CPI/payrolls/GDP; meaningful escalation or de-escalation (ceasefire) in "
    "an active conflict; new trade measures; a sovereign downgrade; a large "
    "event hitting a held name or its sector.\n"
    "3 NOTABLE — real but expected or already priced.\n"
    "1-2 NOISE — previews, opinion, explainers, rumours, recycled stories, "
    "single-source claims, listicles.\n\n"
    "RULES: Merge headlines about the same event into one event. Compare "
    "against `already_alerted`: if an event is the same story, set "
    "is_new_development=false unless something materially changed (a new "
    "strike, an actual decision vs a threat, a number released). Single-"
    "source sensational claims cap at 3. Do not invent facts beyond the "
    "headlines; say 'unconfirmed' when it is. `exposed` lists ONLY symbols "
    "from `holdings` (or broad sectors) plausibly hit. Respond ONLY with JSON: "
    '{"events": [{"ids": [<candidate ids>], "headline": "<one clean line>", '
    '"category": "central_bank|macro_data|geopolitical|trade|energy|fiscal|'
    'financial_stress|market|other", "severity": <1-5>, '
    '"is_new_development": <bool>, "why_it_matters": "<1-2 sentences, '
    'plain, no hype>", "market_read": "risk_off|risk_on|mixed|unclear", '
    '"exposed": ["..."], "watch": "<what to watch next, short>"}]}. '
    'Return {"events": []} when nothing reaches 3.'
)


# ----------------------------------------------------------------- utils
def _envf(key: str, default: float) -> float:
    raw = os.getenv(key)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "as",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "be",
        "after",
        "amid",
        "over",
        "says",
        "say",
        "said",
        "new",
        "its",
        "it",
        "that",
        "this",
        "than",
        "vs",
        "us",
    ]
)


def _tokens(title: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(title.lower()) if t not in _STOP and len(t) > 1}


def title_key(title: str) -> str:
    """Stable dedup key: sorted significant tokens, hashed."""
    toks = sorted(_tokens(title))
    return hashlib.sha1(" ".join(toks).encode()).hexdigest()[:16]


def score_title(title: str) -> int:
    t = title.lower()
    s = 0
    for pat, w in _KEYWORDS:
        if re.search(pat, t):
            s += w
    if any(n in t for n in _NOISE):
        s -= _NOISE_PENALTY
    return s


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster(items: list[dict[str, Any]], threshold: float = 0.45) -> list[dict[str, Any]]:
    """Greedy near-duplicate clustering. Each cluster keeps its best-scored
    title and the set of distinct outlets carrying it."""
    clusters: list[dict[str, Any]] = []
    for it in sorted(items, key=lambda x: -x["score"]):
        toks = _tokens(it["title"])
        for c in clusters:
            if _jaccard(toks, c["_toks"]) >= threshold:
                c["sources"].add(it.get("source") or "?")
                c["titles"].append(it["title"])
                c["_toks"] |= toks
                break
        else:
            clusters.append(
                {
                    "title": it["title"],
                    "topic": it.get("topic", ""),
                    "score": it["score"],
                    "published": it.get("published", ""),
                    "sources": {it.get("source") or "?"},
                    "titles": [it["title"]],
                    "_toks": set(toks),
                }
            )
    for c in clusters:
        # corroboration boost: +1 per extra outlet, capped
        c["score"] += min(len(c["sources"]) - 1, 3)
    return clusters


# ----------------------------------------------------------------- fetch
def _fetch_query(topic: str, query: str) -> list[dict[str, str]]:
    import httpx

    resp = httpx.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=TIMEOUT_S,
        follow_redirects=True,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    out: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source = (item.findtext("source") or "").strip()
        if " - " in title:
            head, _, tail = title.rpartition(" - ")
            title, source = head, source or tail
        out.append(
            {
                "topic": topic,
                "title": title[:220],
                "source": source[:60],
                "published": (item.findtext("pubDate") or "")[:40],
            }
        )
        if len(out) >= MAX_PER_QUERY:
            break
    return out


def fetch_headlines() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for topic, q in _QUERIES.items():
        try:
            items.extend(_fetch_query(topic, q))
        except Exception as e:
            logger.bind(component="event_watch").info(f"feed '{topic}' failed: {e}")
    return items


# ----------------------------------------------------------------- state
def _load_state(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(state_dir) / STATE_FILENAME).read_text())
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, default=str)
    os.replace(tmp, path)


def _prune(state: dict[str, Any], now: datetime) -> None:
    seen = state.get("seen") or {}
    cut = now - timedelta(hours=SEEN_TTL_HOURS)
    kept: dict[str, str] = {}
    for k, v in seen.items():
        ts = _parse(v)
        if ts is not None and ts >= cut:
            kept[k] = v
    state["seen"] = kept
    acut = now - timedelta(hours=ALERT_MEMORY_HOURS)
    state["alerted"] = [
        a for a in (state.get("alerted") or []) if (_parse(a.get("t")) or now) >= acut
    ]
    day = now.date().isoformat()
    if (state.get("day") or {}).get("date") != day:
        state["day"] = {"date": day, "alerts": 0, "llm_calls": 0}


def _parse(ts: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def _in_quiet_hours(now: datetime) -> bool:
    spec = os.getenv("EVENT_WATCH_QUIET_HOURS", "23-7").strip()
    if not spec or spec.lower() in ("off", "none", "0"):
        return False
    try:
        start, end = (int(x) for x in spec.split("-"))
        local = now.astimezone(ZoneInfo(os.getenv("EVENT_WATCH_TZ", "Europe/Zurich")))
    except Exception:
        return False
    h = local.hour
    return (start <= h or h < end) if start > end else (start <= h < end)


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


def _holdings(state_dir: Path) -> list[str]:
    try:
        from trading.runtime.sentinel import _held_symbols

        return _held_symbols(Path(state_dir))
    except Exception:
        return []


# ----------------------------------------------------------------- core
def run_event_watch(
    state_dir: Path,
    *,
    fetch: FetchFn | None = None,
    llm: LlmFn | None = None,
    holdings: list[str] | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """One pass. Returns ``{"quiet": True, ...}`` or
    ``{"send": [events...], "held_for_digest": n, ...}``. Never raises on
    feed/LLM failure — a blind watcher stays silent."""
    log = logger.bind(component="event_watch")
    state_dir = Path(state_dir)
    now = now or _now()
    state = _load_state(state_dir)
    _prune(state, now)
    seen: dict[str, str] = state["seen"]
    min_score = int(_envf("EVENT_WATCH_MIN_SCORE", 3))
    min_sev = int(_envf("EVENT_WATCH_MIN_SEVERITY", 4))
    max_alerts = int(_envf("EVENT_WATCH_MAX_ALERTS_PER_DAY", 8))
    max_llm = int(_envf("EVENT_WATCH_MAX_LLM_CALLS_PER_DAY", 40))
    quiet = _in_quiet_hours(now)

    result: dict[str, Any] = {"quiet": True, "t": now.isoformat()}

    # Release a held quiet-hours digest once the night is over.
    pending = list(state.get("pending") or [])
    send: list[dict[str, Any]] = []
    released_digest = bool(pending) and not quiet
    if released_digest:
        send.extend(pending)
        pending = []

    raw = (fetch or fetch_headlines)()
    # Judge against what was seen BEFORE this run, so the same story from
    # several outlets in one pass all reach clustering (that count is the
    # corroboration signal); mark seen afterwards.
    fresh = []
    new_keys: set[str] = set()
    for it in raw:
        k = title_key(it["title"])
        if k in seen:
            continue
        new_keys.add(k)
        fresh.append({**it, "score": score_title(it["title"])})
    for k in new_keys:
        seen[k] = now.isoformat()

    clusters = [c for c in cluster(fresh) if c["score"] >= min_score]
    clusters.sort(key=lambda c: -c["score"])
    clusters = clusters[:MAX_CANDIDATES_TO_LLM]
    result.update({"fetched": len(raw), "fresh": len(fresh), "candidates": len(clusters)})

    events: list[dict[str, Any]] = []
    if clusters and (state["day"]["llm_calls"] < max_llm or force):
        payload = {
            "now_utc": now.isoformat(timespec="minutes"),
            "holdings": holdings if holdings is not None else _holdings(state_dir),
            "already_alerted": [
                {"headline": a.get("headline"), "severity": a.get("severity"), "t": a.get("t")}
                for a in state["alerted"]
            ],
            "candidates": [
                {
                    "id": i,
                    "title": c["title"],
                    "also": c["titles"][1:4],
                    "outlets": sorted(c["sources"])[:6],
                    "n_outlets": len(c["sources"]),
                    "published": c["published"],
                    "screen_score": c["score"],
                }
                for i, c in enumerate(clusters)
            ],
        }
        state["day"]["llm_calls"] += 1
        try:
            resp = (llm or _default_llm)(CHARTER, json.dumps(payload, default=str))
            events = list(resp.get("events") or [])
        except Exception as e:
            log.warning(f"event triage LLM failed: {e}")
            events = []
    elif clusters:
        log.warning("event watch LLM daily cap reached; candidates not triaged")

    for ev in events:
        try:
            sev = int(ev.get("severity") or 0)
        except (TypeError, ValueError):
            continue
        if sev < min_sev or ev.get("is_new_development") is False:
            continue
        ids = [i for i in (ev.get("ids") or []) if isinstance(i, int) and 0 <= i < len(clusters)]
        outlets: set[str] = set()
        for i in ids:
            outlets |= clusters[i]["sources"]
        rec = {
            "t": now.isoformat(),
            "severity": sev,
            "headline": str(ev.get("headline") or (clusters[ids[0]]["title"] if ids else ""))[:200],
            "category": str(ev.get("category") or "other"),
            "why_it_matters": str(ev.get("why_it_matters") or "")[:400],
            "market_read": str(ev.get("market_read") or "unclear"),
            "exposed": [str(x) for x in (ev.get("exposed") or [])][:8],
            "watch": str(ev.get("watch") or "")[:200],
            "outlets": sorted(o for o in outlets if o and o != "?")[:5],
            "n_outlets": len(outlets),
            "review_suggested": sev >= 5,
        }
        if quiet and sev < 5:
            pending.append(rec)
        else:
            send.append(rec)

    # Daily cap — severity 5 always passes.
    capped: list[dict[str, Any]] = []
    for rec in sorted(send, key=lambda r: -r["severity"]):
        if rec["severity"] >= 5 or state["day"]["alerts"] < max_alerts:
            capped.append(rec)
            state["day"]["alerts"] += 1
    dropped = len(send) - len(capped)

    state["pending"] = pending[-20:]
    state["alerted"].extend(capped)
    state["last_run"] = {
        **{k: v for k, v in result.items() if k != "quiet"},
        "sent": len(capped),
        "held": len(pending),
        "dropped_by_cap": dropped,
    }
    if any(r["review_suggested"] for r in capped):
        state["last_major"] = next(r for r in capped if r["review_suggested"])
    _save_state(state_dir, state)

    log.info(
        f"event watch: fetched={result['fetched']} fresh={result['fresh']} "
        f"candidates={result['candidates']} sent={len(capped)} held={len(pending)}"
    )
    if not capped:
        return {**result, "held_for_digest": len(pending)}
    return {
        **result,
        "quiet": False,
        "send": capped,
        "digest": released_digest,
        "held_for_digest": len(pending),
    }


# ----------------------------------------------------------------- format
_CAT = {
    "central_bank": "Central bank",
    "macro_data": "Macro data",
    "geopolitical": "Geopolitics",
    "trade": "Trade / sanctions",
    "energy": "Energy",
    "fiscal": "Fiscal",
    "financial_stress": "Financial stress",
    "market": "Markets",
}
_READ = {"risk_off": "📉 risk-off", "risk_on": "📈 risk-on", "mixed": "↔️ mixed"}


def _md(s: str) -> str:
    """Neutralize Markdown v1 specials inside free text."""
    return re.sub(r"([_*`\[])", r"\\\1", s or "")


def format_event_alert(events: list[dict[str, Any]], *, digest: bool = False) -> str:
    head = "🌍 *Macro & geopolitics*"
    if digest:
        head += " — overnight digest"
    lines = [head, ""]
    for ev in sorted(events, key=lambda e: -e["severity"]):
        label = SEVERITY_LABEL.get(ev["severity"], f"sev {ev['severity']}")
        cat = _CAT.get(ev.get("category", ""), "Other")
        lines.append(f"{label} · {cat}")
        lines.append(f"*{_md(ev['headline'])}*")
        if ev.get("why_it_matters"):
            lines.append(_md(ev["why_it_matters"]))
        meta = []
        if ev.get("market_read") in _READ:
            meta.append(_READ[ev["market_read"]])
        if ev.get("exposed"):
            meta.append("exposed: " + _md(", ".join(ev["exposed"])))
        if meta:
            lines.append(" · ".join(meta))
        if ev.get("watch"):
            lines.append(f"👀 {_md(ev['watch'])}")
        if ev.get("outlets"):
            extra = ev.get("n_outlets", 0) - len(ev["outlets"])
            src = ", ".join(ev["outlets"]) + (f" +{extra}" if extra > 0 else "")
            lines.append(f"_{_md(src)}_")
        lines.append("")
    if any(e.get("review_suggested") for e in events):
        lines.append(
            "_Major event. Tap Debate it for a committee read; nothing trades "
            "without your approval._"
        )
    return "\n".join(lines).rstrip()


def format_recent(state_dir: Path, hours: float = 24.0) -> str:
    """``/events`` — what was alerted recently, plus anything held overnight."""
    state = _load_state(Path(state_dir))
    cut = _now() - timedelta(hours=hours)
    recent = [a for a in (state.get("alerted") or []) if (_parse(a.get("t")) or cut) > cut]
    pending = state.get("pending") or []
    last = state.get("last_run") or {}
    if not recent and not pending:
        lr = f"\n_last scan: {last.get('t', 'never')[:16]} UTC_" if last else ""
        return f"🌍 Nothing macro/geopolitical worth flagging in the last {hours:.0f}h.{lr}"
    parts: list[str] = []
    if recent:
        parts.append(
            format_event_alert(recent).replace(
                "🌍 *Macro & geopolitics*", f"🌍 *Last {hours:.0f}h*", 1
            )
        )
    if pending:
        parts.append(f"_{len(pending)} item(s) held for the morning digest (quiet hours)._")
    return "\n\n".join(parts)
