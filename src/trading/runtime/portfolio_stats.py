r"""Portfolio analytics for Telegram — beta and holdings correlation.

Everything reads the local Parquet price cache only (no network, no
broker calls), so both the daily beta line and ``/correlation`` cost a
handful of file reads. Symbols missing from the cache are skipped
rather than failing the whole report.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ASSET_DIRS = ("equity", "etf")

# Scorecard subjects are read from the daily US equity/ETF cache.  Those
# bars are date-labelled at midnight UTC even though their value is the US
# session close, so timestamp comparison alone cannot tell whether a daily
# close was available without looking ahead.
_NYSE_TZ = ZoneInfo("America/New_York")
# The cache refresh starts at 17:40 New York time. Do not call a missing
# 16:00 close stale before its 18:00 completion window has passed; the
# evening grader runs later at 18:45.
_DAILY_CACHE_SETTLE_GRACE = pd.Timedelta(hours=2)
_CACHE_READY_LOCAL_TIME = time(18, 0)
# Exact awaiting IDs are recorded by the daily grader. They stay valid
# through that scheduled 18:45 pass so the watchdog does not alarm in the
# gap between the cache window closing and the grader replacing its journal.
# If the grader/cache actually fails, ops raises the overdue alarm shortly
# after 19:00 instead of masking it until the next day.
_AWAITING_JOURNAL_EXPIRY_GRACE = pd.Timedelta(hours=3)
_AWAITING_JOURNAL_EXPIRY_LOCAL_TIME = time(19, 0)

# Predictions are sometimes expressed in index language while the free
# daily cache holds tradeable proxies. Keep this mapping narrow and
# explicit: silently substituting an arbitrary symbol would make a
# scorecard look complete while grading the wrong claim.
_CLOSE_SYMBOL_ALIASES = {
    "NDX": "QQQ",
    "NASDAQ100": "QQQ",
    "NASDAQ-100": "QQQ",
    "SPX": "SPY",
    "S&P500": "SPY",
    "S&P 500": "SPY",
}


def cache_symbol_for_subject(symbol: str) -> str:
    """Tradeable cached proxy for a scorecard subject, otherwise itself."""
    normalized = str(symbol).strip().upper()
    return _CLOSE_SYMBOL_ALIASES.get(normalized, normalized)


def close_at(series: pd.Series | None, when: Any) -> float | None:
    """Last close at or before ``when``, tolerant of tz-aware/naive mixing.

    Exists because that mixing silently disabled the whole scorecard. The
    nightly grader did::

        hist = s[s.index <= ts0.replace(tzinfo=None)]

    and the cache index is ``datetime64[ms, UTC]``, so pandas raised
    ``TypeError: Invalid comparison between dtype=datetime64[ms, UTC] and
    datetime`` on the FIRST prediction. The loop sat inside a broad
    ``except Exception``, so every night it logged once and graded
    nothing — for as long as the grader has existed. No prediction was
    ever scored, ``calibration()`` always returned empty, and every
    charter line about weighting agents by track record was reasoning
    over a table with no rows in it.

    Returns None when there is no bar at or before ``when`` — callers
    must treat that as "cannot grade yet", never as zero.
    """
    if series is None or len(series) == 0:
        return None
    ts = pd.Timestamp(when)
    tz = getattr(series.index, "tz", None)
    if tz is not None:
        ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    hist = series[series.index <= ts]
    if len(hist) == 0:
        return None
    return float(hist.iloc[-1])


def _utc_timestamp(value: Any) -> pd.Timestamp:
    """Interpret a naive timestamp as UTC and return an aware UTC value."""
    ts = pd.Timestamp(value)
    normalized = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return cast(pd.Timestamp, normalized)


@lru_cache(maxsize=256)
def _nyse_schedule(start: date, end: date) -> pd.DataFrame:
    """NYSE sessions from the installed local calendar — never a network call."""
    import pandas_market_calendars as mcal

    return cast(pd.DataFrame, mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end))


def _last_nyse_session_on_or_before(day: date) -> date:
    """The scorecard's final eligible session for a calendar-day horizon."""
    # Fourteen days comfortably spans every ordinary US market closure.  It
    # also keeps each per-prediction calendar lookup small and cacheable.
    schedule = _nyse_schedule(day - timedelta(days=14), day)
    if schedule.empty:  # Defensive only; NYSE has regular sessions.
        raise ValueError(f"no NYSE session found on or before {day.isoformat()}")
    return pd.Timestamp(schedule.index[-1]).date()


def last_completed_nyse_session(when: Any) -> date:
    """Last NYSE session whose actual close was known at ``when``.

    Daily cache labels are midnight timestamps, not the instant a close was
    printed.  A committee prediction made at 13:00 ET must therefore start
    from the preceding completed close, rather than acquire that afternoon's
    eventual close when the cache is read days later.
    """
    moment = _utc_timestamp(when)
    local_day = moment.tz_convert(_NYSE_TZ).date()
    schedule = _nyse_schedule(local_day - timedelta(days=14), local_day)
    completed = schedule.loc[schedule["market_close"] <= moment]
    if completed.empty:
        raise ValueError(f"no completed NYSE session found before {moment.isoformat()}")
    return pd.Timestamp(completed.index[-1]).date()


def completed_session_close(series: pd.Series | None, when: Any) -> float | None:
    """Daily close from the last NYSE session completed at ``when``.

    This is the scorecard-safe counterpart to :func:`close_at`.  It is used
    for prediction endpoints so the midnight label on a daily bar cannot
    smuggle a future same-session close into a mid-session forecast.
    """
    if series is None or len(series) == 0:
        return None
    target_session = last_completed_nyse_session(when)
    matches = [
        position
        for position, label in enumerate(series.index)
        if pd.Timestamp(label).date() == target_session
    ]
    if not matches:
        return None
    return float(series.iloc[matches[-1]])


def _session_deadline(
    market_close: Any,
    *,
    grace: pd.Timedelta,
    no_earlier_than_local: time,
) -> pd.Timestamp:
    """Session deadline respecting both the close and nightly job slots.

    NYSE early-close days finish at 13:00 ET, but the configured cache and
    grading jobs intentionally retain their 17:40/18:45 wall-clock slots.
    A close-plus-grace calculation alone would declare them failed hours
    before either job was scheduled to run.
    """
    close = _utc_timestamp(market_close)
    floor = _utc_timestamp(
        datetime.combine(
            close.tz_convert(_NYSE_TZ).date(),
            no_earlier_than_local,
            tzinfo=_NYSE_TZ,
        )
    )
    return cast(pd.Timestamp, max(close + grace, floor))


def _latest_settled_nyse_session(asof: Any) -> date | None:
    """Latest session whose daily cache bar should be available by ``asof``.

    The cache refresh starts after the US close and can take time across a
    large universe.  A short grace prevents the grader from calling a normal
    in-flight refresh stale, while still surfacing a Friday bar missing by
    Monday night.
    """
    now = _utc_timestamp(asof)
    local_day = now.tz_convert(_NYSE_TZ).date()
    schedule = _nyse_schedule(local_day - timedelta(days=14), local_day)
    cache_ready_at = schedule["market_close"].map(
        lambda close: _session_deadline(
            close,
            grace=_DAILY_CACHE_SETTLE_GRACE,
            no_earlier_than_local=_CACHE_READY_LOCAL_TIME,
        )
    )
    settled = schedule.loc[cache_ready_at <= now]
    if settled.empty:
        return None
    return pd.Timestamp(settled.index[-1]).date()


def nyse_session_settled_since(since: Any, asof: Any) -> bool:
    """Whether a session cleared the scorecard journal grace after ``since``.

    A daily grader can legitimately journal an ``awaiting_next_daily_bar``
    result over a weekend. That result stops being current only after the
    next NYSE session, its cache completion window, and the scheduled evening
    grading pass have had time to run. This avoids a false overdue alert in
    the short interval before that grader rewrites its journal. If the pass
    fails, the same-night watchdog still raises the real issue shortly after.
    """
    checked_at = _utc_timestamp(since)
    now = _utc_timestamp(asof)
    if now <= checked_at:
        return False
    start_day = checked_at.tz_convert(_NYSE_TZ).date()
    end_day = now.tz_convert(_NYSE_TZ).date()
    schedule = _nyse_schedule(start_day, end_day)
    journal_expiry_at = schedule["market_close"].map(
        lambda close: _session_deadline(
            close,
            grace=_AWAITING_JOURNAL_EXPIRY_GRACE,
            no_earlier_than_local=_AWAITING_JOURNAL_EXPIRY_LOCAL_TIME,
        )
    )
    settled_after_check = schedule.loc[
        (schedule["market_close"] > checked_at) & (journal_expiry_at <= now)
    ]
    return not settled_after_check.empty


def coverage_status(
    series: pd.Series | None,
    when: Any,
    *,
    asof: Any | None = None,
) -> str:
    """Return whether a close series can safely grade ``when``.

    Grading a 14-day horizon against the newest bar available is not
    grading it over 14 days. If the cache stops short of the due date the
    prediction has not matured *in our data* and must wait.

    Daily cache bars are labelled at midnight, while their values represent
    a completed US market session.  Therefore a due Friday, Saturday or
    Sunday must wait for Monday's daily bar before it can use Friday's close
    without relying on a same-session label.  That normal wait is reported
    as ``awaiting_next_daily_bar``, not ``cache_behind``.

    Pass ``asof`` from a scheduled grader.  It lets this helper promote an
    expected wait to ``cache_behind`` once a later NYSE session has settled
    and its bar is still absent (for example, a Friday cache still missing
    on Monday night).  Omitting ``asof`` preserves a data-only diagnosis,
    which is useful for deterministic callers and tests.
    """
    if series is None or len(series) == 0:
        return "unavailable"

    due = _utc_timestamp(when)
    # The cache's 1D labels use the US session date, even though their
    # timestamps are normalized to UTC.  Compare labels by date, not by
    # midnight timestamp, so a weekend is not misdiagnosed as missing bars.
    due_session = _last_nyse_session_on_or_before(due.tz_convert(_NYSE_TZ).date())
    session_labels = {pd.Timestamp(label).date() for label in series.index}
    last_session_label = max(session_labels)

    # A later date in the file only proves the file continued to update; it
    # does not prove it contains the eligible close.  Without this check a
    # cache with Thursday + Monday but no Friday would be marked covered for
    # a weekend expiry and silently grade the return through Thursday.
    if due_session not in session_labels:
        if last_session_label > due_session:
            return "cache_behind"
        if asof is not None:
            latest_settled = _latest_settled_nyse_session(asof)
            if latest_settled is not None and latest_settled >= due_session:
                return "cache_behind"
        elif last_session_label < due_session:
            return "cache_behind"
        return "awaiting_next_daily_bar"

    # A later session label is the conservative proof that the eligible
    # session's close is complete, after the eligible bar itself has been
    # verified above. Equality deliberately waits — using a date-labelled
    # bar on its own date can otherwise look ahead.
    if last_session_label > due_session:
        return "covered"

    if asof is not None:
        latest_settled = _latest_settled_nyse_session(asof)
        if latest_settled is not None and latest_settled > last_session_label:
            return "cache_behind"

    # If no observation time is available, a series that predates the final
    # eligible session is already demonstrably short.  With an ``asof``
    # before that session has settled, it remains a normal wait instead.
    if asof is None and last_session_label < due_session:
        return "cache_behind"
    return "awaiting_next_daily_bar"


def covers(series: pd.Series | None, when: Any, *, asof: Any | None = None) -> bool:
    """True when the cache has a post-due daily session for ``when``.

    Kept as the simple public predicate for existing callers; use
    :func:`coverage_status` when the caller needs an operator-facing reason.
    """
    return coverage_status(series, when, asof=asof) == "covered"


def _read_close(data_dir: Path, symbol: str) -> pd.Series | None:
    # The cache names files after the Frequency literal "1D", but older
    # CLI fetches wrote "1d". macOS hides the difference (case-insensitive
    # filesystem); Linux does not — so try both spellings explicitly.
    cached_symbol = cache_symbol_for_subject(symbol)
    for sub in _ASSET_DIRS:
        for fname in ("1D.parquet", "1d.parquet"):
            p = Path(data_dir) / sub / cached_symbol / fname
            if p.exists():
                try:
                    s = pd.read_parquet(p)["close"].dropna()
                    s.index = pd.to_datetime(s.index)
                    return s.sort_index()
                except Exception:
                    return None
    return None


def portfolio_beta(
    position_values: dict[str, float],
    data_dir: Path,
    *,
    market: str = "SPY",
    lookback: int = 252,
) -> tuple[float, int] | None:
    """Value-weighted portfolio beta vs ``market`` over ``lookback`` days.

    ``position_values``: symbol -> current market value (sign carries
    direction for shorts). Returns (beta, names_used) or None when the
    market series or every holding is missing from the cache.
    """
    mkt = _read_close(data_dir, market)
    if mkt is None or len(mkt) < 60:
        return None
    mkt_ret = mkt.pct_change().iloc[-lookback:].dropna()
    if mkt_ret.std() == 0:
        return None

    total = sum(abs(v) for v in position_values.values())
    if total <= 0:
        return None

    beta_acc = 0.0
    weight_acc = 0.0
    used = 0
    for sym, value in position_values.items():
        s = _read_close(data_dir, sym)
        if s is None:
            continue
        ret = s.pct_change().iloc[-lookback:].dropna()
        joined = pd.concat([ret, mkt_ret], axis=1, keys=["a", "m"]).dropna()
        if len(joined) < 60:
            continue
        beta_i = float(np.cov(joined["a"], joined["m"])[0, 1] / joined["m"].var())
        w = value / total  # signed weight, normalized by gross
        beta_acc += w * beta_i
        weight_acc += abs(w)
        used += 1
    if used == 0 or weight_acc == 0:
        return None
    return beta_acc, used


def holdings_correlation(
    symbols: list[str], data_dir: Path, *, lookback: int = 252
) -> pd.DataFrame | None:
    """Pairwise return correlation of ``symbols`` over ``lookback`` days."""
    series = {}
    for sym in symbols:
        s = _read_close(data_dir, sym)
        if s is not None and len(s) > 60:
            series[sym.upper()] = s.pct_change()
    if len(series) < 2:
        return None
    rets = pd.DataFrame(series).iloc[-lookback:].dropna(how="all")
    return rets.corr()


def format_correlation(corr: pd.DataFrame, *, max_matrix: int = 10) -> str:
    """Telegram-friendly rendering: compact monospace matrix for small
    books, ranked pair list for big ones. Values as integer percent."""
    syms = list(corr.columns)
    # Average pairwise correlation (off-diagonal).
    n = len(syms)
    off = corr.values[np.triu_indices(n, k=1)]
    avg = float(np.nanmean(off)) if len(off) else 0.0

    lines = [f"🔗 *Holdings correlation* — trailing 12m, {n} names", ""]
    if n <= max_matrix:
        tag = {s: s[:4] for s in syms}
        lines.append("```")
        lines.append("     " + " ".join(f"{tag[s]:>4}" for s in syms))
        for s in syms:
            cells = []
            for t in syms:
                v = corr.loc[s, t]
                cells.append("   ." if s == t else f"{v * 100:>4.0f}")
            lines.append(f"{tag[s]:<5}" + " ".join(cells))
        lines.append("```")
    pairs = [
        (syms[i], syms[j], float(corr.iloc[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if not np.isnan(corr.iloc[i, j])
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    if pairs:
        hi = pairs[0]
        lo = pairs[-1]
        lines.append(f"Average pairwise: `{avg * 100:+.0f}%`")
        lines.append(f"Most correlated:  `{hi[0]}–{hi[1]}  {hi[2] * 100:+.0f}%`")
        lines.append(f"Least correlated: `{lo[0]}–{lo[1]}  {lo[2] * 100:+.0f}%`")
    if avg > 0.7:
        lines.append("_⚠️ High average correlation — this book moves as one trade._")
    return "\n".join(lines)
