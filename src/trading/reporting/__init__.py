"""Daily-report machinery.

Three stages, each independently testable:

  data        gather_daily_report   reads runner.db, orders.db, halt.json
  news        fetch_news_for_symbols, fetch_rss
  prose       summarise  (Claude API if available; deterministic fallback)
  output      render_markdown  (one Markdown doc)
"""

from __future__ import annotations

from trading.reporting.daily import (
    DailyReport,
    gather_daily_report,
    gather_weekly_report,
)
from trading.reporting.executive_summary import summarise
from trading.reporting.news import Headline, fetch_news_for_symbols, fetch_rss
from trading.reporting.render import render_markdown

__all__ = [
    "DailyReport",
    "Headline",
    "fetch_news_for_symbols",
    "fetch_rss",
    "gather_daily_report",
    "gather_weekly_report",
    "render_markdown",
    "summarise",
]
