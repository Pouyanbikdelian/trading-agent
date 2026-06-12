"""News watch — the scout's eyes: free RSS headlines + sector momentum.

The committee context builder is deliberately network-free, so anything
the agents "see" from the outside world must be collected here on a
schedule and written to state — same pattern as market_watch. Two feeds:

* **Headlines** — Google News RSS queries (free, no key) across broad
  market + sector/theme searches. Gossip-grade by design: the scout
  charter tells the LLM to weigh it as such.
* **Sector momentum** — 1m/3m return of the SPDR sectors + a few theme
  ETFs *relative to SPY*, via yfinance. "Obvious industry direction" in
  numbers, so the scout's trendy-sector call is anchored to tape, not
  just chatter.

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
MAX_PER_QUERY = 7
MAX_HEADLINES = 70
TIMEOUT_S = 15.0

# Broad + thematic queries. Tuned for "what is the crowd excited about",
# not for completeness — the scout needs scent, not an archive.
_QUERIES: dict[str, str] = {
    "market": "stock market this week",
    "ai_semis": "AI chips semiconductor stocks",
    "energy": "energy oil uranium nuclear stocks",
    "biotech": "biotech pharma FDA stocks",
    "defense": "defense aerospace stocks",
    "crypto": "bitcoin crypto market",
    "consumer": "consumer retail stocks earnings",
    # Capital flows: announced money is a different signal class than
    # opinion — a $10bn committed fund is skin in the game, an analyst
    # note is words. The scout's charter weights these accordingly.
    "capital_flows": "billion investment fund launch acquisition stake",
    "ai_capex": "AI infrastructure data center investment billion",
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
    """{sector: {ret_1m_vs_spy, ret_3m_vs_spy}} — relative, in percent."""
    out: dict[str, dict[str, float | None]] = {}
    try:
        import yfinance as yf

        tickers = [*SECTOR_ETFS, "SPY"]
        raw = yf.download(
            " ".join(tickers),
            period="4mo",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )

        def _rets(tkr: str) -> tuple[float, float] | None:
            try:
                s = raw[tkr]["Close"].dropna()
                if len(s) < 63:
                    return None
                last = float(s.iloc[-1])
                return last / float(s.iloc[-21]) - 1.0, last / float(s.iloc[-63]) - 1.0
            except Exception:
                return None

        spy = _rets("SPY")
        if spy is None:
            return out
        for tkr, name in SECTOR_ETFS.items():
            r = _rets(tkr)
            if r is None:
                out[name] = {"etf": tkr, "ret_1m_vs_spy": None, "ret_3m_vs_spy": None}
                continue
            out[name] = {
                "etf": tkr,
                "ret_1m_vs_spy": round((r[0] - spy[0]) * 100, 2),
                "ret_3m_vs_spy": round((r[1] - spy[1]) * 100, 2),
            }
    except Exception as e:
        logger.bind(component="news_watch").info(f"sector momentum fetch failed: {e}")
    return out


def collect(state_dir: Path) -> dict[str, Any]:
    """One collection pass; atomic write of state/news.json."""
    reading: dict[str, Any] = {
        "t": datetime.now(tz=timezone.utc).isoformat(),
        "headlines": fetch_headlines(),
        "sector_momentum": fetch_sector_momentum(),
    }
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(reading, f)
    os.replace(tmp, path)
    logger.bind(component="news_watch").info(
        f"news watch updated ({len(reading['headlines'])} headlines, "
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
