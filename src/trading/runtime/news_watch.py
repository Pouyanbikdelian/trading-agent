"""News watch — the scout's eyes: RSS headlines, Reddit signals, sector momentum.

The committee context builder is deliberately network-free, so anything
the agents "see" from the outside world must be collected here on a
schedule and written to state — same pattern as market_watch. Three feeds:

* **Headlines** — Google News RSS queries (free, no key) across broad
  market + sector/theme + financial influencer searches. Gossip-grade by
  design: the scout charter tells the LLM to weigh it as such.
* **Reddit signals** — top daily posts from key investment subreddits via
  Reddit's free public JSON API. Authors are tagged as ``reddit:u/<name>``
  so the source-trust ledger accumulates accuracy scores over time: an
  author who called $AMD three weeks before the move gets a higher trust
  weight on the next call; noise merchants get faded.
* **Sector momentum** — 1m/3m return of the SPDR sectors + theme ETFs
  *relative to SPY*, via yfinance.

Everything is None-tolerant and bounded; a dead feed degrades the scout,
never the runner.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

STATE_FILENAME = "news.json"
MAX_PER_QUERY = 5  # fewer per query so more queries fit in the context budget
MAX_HEADLINES = 100
TIMEOUT_S = 15.0

# Broad + thematic queries. Tuned for "what is the crowd excited about",
# not for completeness — the scout needs scent, not an archive.
# Grouped: core market, sector-specific, macro/rates, capital flows, themes.
_QUERIES: dict[str, str] = {
    # ---- core market pulse
    "market": "stock market this week",
    "earnings": "earnings beat miss revenue guidance quarterly results",
    "macro_rates": "Federal Reserve interest rates inflation treasury yields bond",
    # ---- sector specialists (each feeds the creative + sector scouts)
    "ai_semis": "AI chips semiconductor stocks Nvidia AMD TSMC",
    "financials": "banks financial stocks JPMorgan Goldman Sachs earnings rates",
    "healthcare": "healthcare biotech pharma FDA approval UnitedHealth Eli Lilly",
    "energy": "energy oil uranium nuclear stocks XOM Chevron",
    "defense": "defense aerospace government contract Lockheed Raytheon Boeing",
    "industrials": "industrials manufacturing supply chain infrastructure spending",
    "consumer": "consumer retail discretionary earnings spending Amazon Walmart",
    "utilities_real_estate": "utilities real estate REIT dividend interest rates",
    "materials": "copper gold silver mining commodities materials sector",
    # ---- global & macro backdrop
    "global_macro": "China emerging markets global economy GDP recession outlook",
    "dollar_credit": "US dollar DXY credit spreads high yield bonds default",
    # ---- capital flows: committed money is a different signal class than opinion
    "capital_flows": "billion investment fund launch acquisition stake pension",
    "institutional": "hedge fund positioning short interest institutional buying selling",
    "ai_capex": "AI infrastructure data center hyperscaler investment billion",
    # ---- standing operator directives
    "quantum": "quantum computing breakthrough stocks investment",
    "quantum_gov": "quantum computing government contract defense national lab partnership",
    # ---- late-cycle / rotation signals
    "sector_rotation": "sector rotation defensive growth cyclical reallocation",
    "valuation": "overvalued bubble peak earnings multiple compression expensive stocks",
    # ---- high-conviction public voices: news coverage of X posts / interviews
    # surfaces when a name makes a call big enough to be picked up by media.
    # Authors are not tagged here (no username in RSS); Reddit handles that.
    "macro_voices": "Druckenmiller Ackman Burry Tepper Einhorn portfolio position 2026",
    "tech_voices": "Chamath Palihapitiya ARK Invest Cathie Wood bought sold stake 2026",
    "activist_events": "activist investor stake board seat proxy fight demands 2026",
}

# Relative-momentum universe: 11 SPDRs + liquid theme ETFs.
SECTOR_ETFS: dict[str, str] = {
    "XLK": "tech",
    "XLE": "energy",
    "XLF": "financials",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "cons_discretionary",
    "XLP": "cons_staples",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "real_estate",
    "XLC": "communications",
    "SMH": "semiconductors",
    "IBB": "biotech",
    "ITA": "defense_aero",
    "URA": "uranium",
}


# Reddit: subreddits to monitor via their public RSS feeds.
# Reddit deprecated unauthenticated JSON in 2023; the .rss endpoints
# remain public and need no key. Author is embedded in <author> tags on
# some subs — when present it is tagged as "reddit:u/<name>" so the
# source-trust ledger can accumulate accuracy scores per user over time.
_REDDIT_SUBS: tuple[str, ...] = (
    "wallstreetbets",  # high-vol retail; useful as contrarian sentiment gauge
    "investing",  # longer-horizon fundamental analysis
    "stocks",  # broad retail sentiment
    "securityanalysis",  # institutional-style deep dives
    "options",  # derivatives positioning and flow ideas
)
_REDDIT_MAX_PER_SUB = 5  # posts per subreddit per collection pass


def fetch_reddit_signals() -> list[dict[str, str]]:
    """Top posts from key investment subreddits via Reddit's public RSS.
    Author tagged as 'reddit:u/<name>' feeds the source-trust ledger;
    the committee memory system accumulates accuracy scores per user.
    A 0.6s inter-request delay respects Reddit's public rate limit."""
    import time

    import httpx

    out: list[dict[str, str]] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for i, sub in enumerate(_REDDIT_SUBS):
        if i > 0:
            time.sleep(0.6)  # stay under Reddit's public rate limit
        try:
            resp = httpx.get(
                f"https://www.reddit.com/r/{sub}/top.rss",
                params={"t": "day"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; trading-research/1.0)"},
                timeout=TIMEOUT_S,
                follow_redirects=True,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            count = 0
            for entry in root.findall("atom:entry", ns)[: _REDDIT_MAX_PER_SUB * 2]:
                title_el = entry.find("atom:title", ns)
                author_el = entry.find("atom:author/atom:name", ns)
                title = (title_el.text or "").strip() if title_el is not None else ""
                raw_author = (author_el.text or "").strip() if author_el is not None else ""
                # Reddit RSS includes "/u/" prefix in the name field — strip it.
                author = raw_author.lstrip("/u").lstrip("/").strip() or ""
                if not title or title.lower() in ("", "[deleted]", "[removed]"):
                    continue
                source_key = (
                    f"reddit:u/{author}"
                    if author and author not in ("[deleted]", "AutoModerator")
                    else f"reddit:r/{sub}"
                )
                out.append(
                    {
                        "topic": f"reddit_{sub}",
                        "title": f"[r/{sub}] {title[:185]}",
                        "source": source_key,
                        "published": "",
                    }
                )
                count += 1
                if count >= _REDDIT_MAX_PER_SUB:
                    break
        except Exception as e:
            logger.bind(component="news_watch").info(f"reddit r/{sub} rss failed: {e}")
    return out


def _fetch_query(topic: str, query: str) -> list[dict[str, str]]:
    """One Google News RSS query -> [{title, source, published, topic}]."""
    import httpx

    url = "https://news.google.com/rss/search"
    resp = httpx.get(
        url,
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=TIMEOUT_S,
        follow_redirects=True,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        # Google News titles end " - Publisher"; split it out as source.
        source = item.findtext("source") or ""
        if not source and " - " in title:
            title, _, source = title.rpartition(" - ")
        items.append(
            {
                "topic": topic,
                "title": title[:200],
                "source": str(source).strip()[:60],
                "published": (item.findtext("pubDate") or "")[:40],
            }
        )
        if len(items) >= MAX_PER_QUERY:
            break
    return items


def fetch_headlines() -> list[dict[str, str]]:
    headlines: list[dict[str, str]] = []
    for topic, query in _QUERIES.items():
        try:
            headlines.extend(_fetch_query(topic, query))
        except Exception as e:
            logger.bind(component="news_watch").info(f"feed '{topic}' failed: {e}")
    return headlines[:MAX_HEADLINES]


def fetch_sector_momentum() -> dict[str, dict[str, float | None]]:
    """{sector: {ret_1m_vs_spy, ret_3m_vs_spy, off_52w_high_pct}}.

    off_52w_high_pct is the rotation lens: a sector at −1% off its high is
    extended (no valuation cushion); one at −20% with improving 1m momentum
    is a recovery candidate. Momentum says what's strong; this says what's
    stretched versus washed out — the agents debate the difference."""
    out: dict[str, dict[str, float | None]] = {}
    try:
        import yfinance as yf

        tickers = [*SECTOR_ETFS, "SPY"]
        raw = yf.download(
            " ".join(tickers),
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )

        def _stats(tkr: str) -> tuple[float, float, float] | None:
            try:
                s = raw[tkr]["Close"].dropna()
                if len(s) < 63:
                    return None
                last = float(s.iloc[-1])
                return (
                    last / float(s.iloc[-21]) - 1.0,
                    last / float(s.iloc[-63]) - 1.0,
                    last / float(s.max()) - 1.0,  # distance from 52w high (<= 0)
                )
            except Exception:
                return None

        spy = _stats("SPY")
        if spy is None:
            return out
        for tkr, name in SECTOR_ETFS.items():
            r = _stats(tkr)
            if r is None:
                out[name] = {
                    "etf": tkr,
                    "ret_1m_vs_spy": None,
                    "ret_3m_vs_spy": None,
                    "off_52w_high_pct": None,
                }
                continue
            out[name] = {
                "etf": tkr,
                "ret_1m_vs_spy": round((r[0] - spy[0]) * 100, 2),
                "ret_3m_vs_spy": round((r[1] - spy[1]) * 100, 2),
                "off_52w_high_pct": round(r[2] * 100, 2),
            }
    except Exception as e:
        logger.bind(component="news_watch").info(f"sector momentum fetch failed: {e}")
    return out


def collect(state_dir: Path) -> dict[str, Any]:
    """One collection pass; atomic write of state/news.json."""
    rss = fetch_headlines()
    reddit = fetch_reddit_signals()
    # Reddit entries go after RSS so context truncation loses them last
    # (they carry the most attribution-trackable signal for trust scoring).
    all_headlines = (rss + reddit)[:MAX_HEADLINES]
    reading: dict[str, Any] = {
        "t": datetime.now(tz=timezone.utc).isoformat(),
        "headlines": all_headlines,
        "sector_momentum": fetch_sector_momentum(),
    }
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(reading, f)
    os.replace(tmp, path)
    n_reddit = len(reddit)
    logger.bind(component="news_watch").info(
        f"news watch updated ({len(all_headlines)} headlines "
        f"[{len(rss)} rss + {n_reddit} reddit], "
        f"{len(reading['sector_momentum'])} sectors)"
    )
    return reading


def load(state_dir: Path, *, max_age_hours: float = 36.0) -> dict[str, Any]:
    """Read the latest collection if fresh enough; {} otherwise."""
    path = Path(state_dir) / STATE_FILENAME
    try:
        reading = json.loads(path.read_text())
        ts = datetime.fromisoformat(reading["t"])
        age_h = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600
        return reading if age_h <= max_age_hours else {}
    except Exception:
        return {}
