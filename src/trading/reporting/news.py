r"""News fetcher for the daily report.

Two free sources, both optional:

* ``yfinance.Ticker(sym).news`` — gives us a few recent headlines per
  ticker with a published timestamp. Coverage is patchy and US-equity-
  biased but it costs nothing and integrates with the data layer we
  already use.
* A configurable list of RSS feeds (Reuters business, FT Markets,
  Bloomberg if subscribed) that the operator can extend. Headlines are
  matched against the held symbols by simple regex.

Both sources return :class:`Headline` records — a uniform shape that the
executive-summary layer (or a downstream LLM) can consume.

To activate the LLM-driven scoring + summary, set ``ANTHROPIC_API_KEY``
in the environment and call
:func:`trading.reporting.executive_summary.summarise`. This module just
provides the raw structured input.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trading.core.logging import logger


@dataclass
class Headline:
    """One news item. Provenance and timing are explicit so callers can
    de-duplicate and filter by recency."""

    title: str
    source: str
    url: str
    published: datetime
    matched_symbols: list[str] = field(default_factory=list)


def fetch_news_for_symbols(
    symbols: list[str],
    *,
    max_per_symbol: int = 5,
    max_age_hours: float = 48.0,
    yf_downloader: Any | None = None,
    pause_seconds: float = 0.2,
) -> dict[str, list[Headline]]:
    """Pull recent headlines per symbol from yfinance.

    yfinance's ``Ticker(sym).news`` returns a list of dicts; field names
    drift between versions. We defensively pick whichever timestamp key
    is present and fall back gracefully when the response is malformed.
    """
    if yf_downloader is None:
        try:
            import yfinance as yf

            yf_downloader = yf
        except ImportError:
            logger.bind(component="news").warning("yfinance not installed; skipping news")
            return {}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=max_age_hours)
    out: dict[str, list[Headline]] = {}

    for sym in symbols:
        try:
            ticker = yf_downloader.Ticker(sym)
            raw_items = ticker.news or []
        except Exception as e:
            logger.bind(symbol=sym).warning(f"news fetch failed: {e!r}")
            out[sym] = []
            continue

        items: list[Headline] = []
        for raw in raw_items:
            h = _parse_yf_news_item(raw, sym=sym)
            if h is None:
                continue
            if h.published < cutoff:
                continue
            items.append(h)
        items.sort(key=lambda h: h.published, reverse=True)
        out[sym] = items[:max_per_symbol]
        time.sleep(pause_seconds)

    return out


def _parse_yf_news_item(raw: dict[str, Any], *, sym: str) -> Headline | None:
    """Defensive parser for yfinance news payload — the schema has
    changed several times. We try the modern format first, then fall
    back to legacy field names."""
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") or raw
    title = content.get("title") or raw.get("title")
    if not title:
        return None

    # publishing time: try both legacy and modern paths
    ts: datetime | None = None
    pub_raw = content.get("pubDate") or content.get("displayTime")
    if pub_raw:
        try:
            ts = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            ts = None
    if ts is None:
        epoch = raw.get("providerPublishTime") or content.get("providerPublishTime")
        if epoch:
            try:
                ts = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                ts = None
    if ts is None:
        ts = datetime.now(tz=timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    url = (
        content.get("clickThroughUrl", {}).get("url")
        or content.get("canonicalUrl", {}).get("url")
        or raw.get("link")
        or ""
    )
    provider = (
        (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else None
        )
        or raw.get("publisher")
        or "unknown"
    )

    return Headline(
        title=str(title).strip(),
        source=str(provider),
        url=str(url),
        published=ts,
        matched_symbols=[sym],
    )


def fetch_rss(
    feeds: list[str],
    held_symbols: list[str],
    *,
    max_age_hours: float = 48.0,
    timeout: float = 10.0,
    user_agent: str = "trading-agent/0.1",
) -> list[Headline]:
    """Pull RSS feeds, parse headlines, and tag them with any held
    symbols whose ticker appears in the title.

    Match policy: case-insensitive whole-word match of the bare ticker
    (``\\bAAPL\\b``). Cheap and good enough for ``XLK``-class names;
    ambiguous on common short symbols (``A``, ``T``). Symbols shorter
    than 3 characters are skipped to keep false positives low.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=max_age_hours)
    sym_patterns = {
        s: re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE) for s in held_symbols if len(s) >= 3
    }
    headlines: list[Headline] = []
    for feed_url in feeds:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                xml = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.bind(feed=feed_url).warning(f"RSS fetch failed: {e!r}")
            continue
        for item in _parse_rss_items(xml):
            if item["published"] < cutoff:
                continue
            matched = [s for s, pat in sym_patterns.items() if pat.search(item["title"])]
            if not matched:
                continue
            headlines.append(
                Headline(
                    title=item["title"],
                    source=_domain_of(feed_url),
                    url=item["url"],
                    published=item["published"],
                    matched_symbols=matched,
                )
            )
    headlines.sort(key=lambda h: h.published, reverse=True)
    return headlines


def _parse_rss_items(xml: str) -> list[dict]:
    """Minimal RSS 2.0 / Atom parser. We don't pull a dependency for
    this; the standard library xml.etree handles both formats with a
    little dispatching."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    out: list[dict] = []
    # RSS 2.0: channel/item
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        ts = _parse_rfc822(pub)
        if title and ts:
            out.append({"title": title, "url": link, "published": ts})

    # Atom: feed/entry
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href") if link_el is not None else ""
        updated = entry.findtext("{http://www.w3.org/2005/Atom}updated") or ""
        ts = _parse_iso(updated)
        if title and ts:
            out.append({"title": title, "url": link, "published": ts})

    return out


def _parse_rfc822(s: str) -> datetime | None:
    try:
        from email.utils import parsedate_to_datetime

        ts = parsedate_to_datetime(s)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def _parse_iso(s: str) -> datetime | None:
    try:
        ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except (TypeError, ValueError):
        return None


def _domain_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc or url
    except Exception:
        return url
