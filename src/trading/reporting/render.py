r"""Markdown renderer for the daily report.

Renders a :class:`DailyReport` plus its optional executive summary into
a single Markdown document. The same payload feeds the JSON output for
machine consumers; this module owns the human-readable surface.
"""

from __future__ import annotations

from trading.reporting.daily import DailyReport
from trading.reporting.news import Headline


def render_markdown(report: DailyReport, *, executive_summary: str | None = None) -> str:
    lines: list[str] = []

    lines.append(f"# Trading Daily Report - {report.as_of.date().isoformat()}")
    lines.append("")
    lines.append(f"_Generated at {report.as_of.isoformat()}_")
    lines.append("")

    # --- Executive summary ----------------------------------------------
    if executive_summary:
        lines.append("## Executive summary")
        lines.append("")
        lines.append(executive_summary)
        lines.append("")

    # --- Headline numbers -----------------------------------------------
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Portfolio equity | ${report.equity:,.2f} |")
    lines.append(f"| Cash | ${report.cash:,.2f} |")
    lines.append(f"| Daily PnL ($) | ${report.daily_pnl:+,.2f} |")
    lines.append(f"| Daily PnL (%) | {report.daily_pnl_pct:+.2%} |")
    lines.append(f"| Week-to-date | {report.week_pnl_pct:+.2%} |")
    lines.append(f"| Month-to-date | {report.month_pnl_pct:+.2%} |")
    lines.append(f"| YTD | {report.ytd_pnl_pct:+.2%} |")
    if report.vix_level is not None:
        lines.append(f"| VIX | {report.vix_level:.2f} ({report.vix_regime}) |")
    lines.append(f"| Halted? | {'YES — ' + report.halt_reason if report.halted else 'no'} |")
    if report.heartbeat_age_seconds is not None:
        lines.append(
            f"| Heartbeat age | {report.heartbeat_age_seconds:.0f}s "
            f"({'stale' if report.heartbeat_age_seconds > 86400 else 'fresh'}) |"
        )
    lines.append("")

    # --- Positions ------------------------------------------------------
    if report.positions:
        lines.append("## Current positions")
        lines.append("")
        lines.append("| Symbol | Qty | Avg price | Market value | Weight | Unrealized PnL |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for sym, d in sorted(
            report.positions.items(),
            key=lambda kv: -kv[1]["market_value"],
        ):
            lines.append(
                f"| {sym} | {d['quantity']:,.2f} | ${d['avg_price']:,.2f} "
                f"| ${d['market_value']:,.2f} | {d['weight']:+.2%} "
                f"| ${d['unrealized_pnl']:+,.2f} |"
            )
        lines.append("")

    # --- Last cycle trades ----------------------------------------------
    if report.last_cycle_trades:
        lines.append("## Trades since last report (24h)")
        lines.append("")
        lines.append("| Time | Symbol | Qty | Price | Commission |")
        lines.append("|---|---|---:|---:|---:|")
        for t in report.last_cycle_trades[:25]:
            lines.append(
                f"| {t['ts'].strftime('%Y-%m-%d %H:%M')} | "
                f"{t.get('order_id', '?')[-12:]} | "
                f"{t['quantity']:+,.2f} | ${t['price']:,.2f} | "
                f"${t['commission']:,.4f} |"
            )
        lines.append("")

    # --- Recent cycle outcomes ------------------------------------------
    if report.recent_cycles:
        lines.append("## Recent cycles")
        lines.append("")
        lines.append("| Time | Status | Orders | Fills | Duration ms |")
        lines.append("|---|---|---:|---:|---:|")
        for c in report.recent_cycles[:10]:
            lines.append(
                f"| {c['ts'].strftime('%Y-%m-%d %H:%M')} | {c['status']} | "
                f"{c['orders_submitted']} | {c['fills_received']} | "
                f"{c['duration_ms']:.0f} |"
            )
        lines.append("")

    # --- News by held symbol --------------------------------------------
    if report.news_by_symbol:
        lines.append("## News on held names")
        lines.append("")
        for sym, items in report.news_by_symbol.items():
            if not items:
                continue
            lines.append(f"### {sym}")
            for h in items[:5]:
                if isinstance(h, Headline):
                    when = h.published.strftime("%Y-%m-%d %H:%M")
                    lines.append(f"- _{when}_ — {h.source} — [{h.title}]({h.url})")
                else:
                    title = h.get("title", "")
                    lines.append(f"- {title}")
            lines.append("")

    return "\n".join(lines)
