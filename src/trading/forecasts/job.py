"""The daily run: forecast today, grade what has come due. No order path."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.core.clock import UtcClock
from trading.core.logging import logger
from trading.forecasts import batch as batch_api
from trading.forecasts import llm as llm_forecaster
from trading.forecasts import store
from trading.forecasts.baseline import HORIZONS, fit, forecast_row, prepare
from trading.forecasts.features import load_close_panel

#: A newest bar older than this is a stale cache, not "today".
MAX_BAR_AGE_DAYS = 4
#: Share of the universe that must have a bar for a session to be forecast.
MIN_COVERAGE = 0.8
#: Below this share of names with at least one headline, the feed is treated
#: as down and the "news" forecaster is not asked. Without news it is the
#: "numbers" forecaster under another name, and its forecasts would dilute
#: the one comparison it exists for.
MIN_HEADLINE_COVERAGE = 0.2
#: A batch still unfinished after this is written off. The API itself
#: expires a batch at 24 hours.
BATCH_GIVE_UP = timedelta(hours=30)


def _pct(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(100 * float(value), 2)


def _cards(
    symbols: list[str],
    raw: dict[str, pd.DataFrame],
    ranked: dict[str, pd.DataFrame],
    row: int,
    headlines: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """What the LLM sees per stock. Deliberately NOT the baseline's
    probability: shown that, the model would anchor on it, and "does it add
    anything to the baseline" could no longer be measured."""
    cards = []
    for symbol in symbols:
        value = {name: frame[symbol].iloc[row] for name, frame in raw.items()}
        cards.append(
            {
                "symbol": symbol,
                "return_1w_pct": _pct(value["rev_1w"]),
                "return_1m_pct": _pct(value["rev_1m"]),
                "return_6m_ex_last_month_pct": _pct(value["mom_6_1"]),
                "return_12m_ex_last_month_pct": _pct(value["mom_12_1"]),
                "daily_volatility_3m_pct": _pct(value["vol_3m"]),
                "pct_of_52w_high": _pct(value["near_high"]),
                "momentum_percentile_in_universe": _pct(ranked["mom_6_1"][symbol].iloc[row] + 0.5),
                "recent_headlines": headlines.get(symbol, []),
            }
        )
    return cards


def _yahoo_headlines(symbols: list[str]) -> dict[str, list[str]]:
    from trading.reporting.news import fetch_news_for_symbols

    found = fetch_news_for_symbols(symbols, max_per_symbol=4, max_age_hours=96.0)
    return {
        s: [f"{h.published.date()} {h.title[:160]}" for h in items]
        for s, items in found.items()
        if items
    }


def _google_headlines(symbols: list[str], *, budget_s: float = 120.0) -> dict[str, list[str]]:
    """Google News RSS, one query per symbol, inside a total time budget."""
    from email.utils import parsedate_to_datetime

    from trading.runtime.news_watch import _fetch_query

    cutoff = UtcClock().now() - timedelta(hours=96)
    started = time.monotonic()
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        if time.monotonic() - started > budget_s:
            break
        try:
            titles = []
            for item in _fetch_query(symbol, f"{symbol} stock when:4d"):
                try:
                    published = parsedate_to_datetime(item["published"])
                except (TypeError, ValueError):
                    continue
                if published.tzinfo is not None and published >= cutoff:
                    titles.append(f"{published.date()} {item['title'][:160]}")
                if len(titles) >= 4:
                    break
            if titles:
                out[symbol] = titles
        except Exception:
            continue
    return out


def _headlines(symbols: list[str]) -> dict[str, list[str]]:
    """Headlines per symbol: Yahoo first, Google News for whatever it missed.

    The first live run (2026-11-11) got "Yahoo Finance is currently down"
    for all fifty names. That fetcher logs and returns nothing, so the
    "news" forecaster was asked with no news and nothing said so.
    """
    try:
        found = _yahoo_headlines(symbols)
    except Exception:
        found = {}
    missing = [s for s in symbols if s not in found]
    if missing:
        found.update(_google_headlines(missing))
    return found


def _record_usage(model: str, usage: dict[str, Any], tier: str) -> None:
    """Batch calls never pass through agents.llm, so log their tokens here
    or the cost report would show the shadow forecasts as free."""
    try:
        from trading.agents import llm

        llm._record_telemetry(
            provider="anthropic",
            model=model,
            tier=tier,
            effort=llm._anthropic_effort("shadow"),
            max_tokens=llm_forecaster.MAX_TOKENS,
            latency_ms=0.0,
            usage=usage,
            stop_reason=None,
            timeout_s=0.0,
        )
    except Exception:
        logger.bind(component="forecasts").warning("could not record batch token usage")


def collect_batches(
    state_dir: Path,
    *,
    now: datetime,
    client: Any = batch_api,
) -> dict[str, Any]:
    """Turn finished batches into forecasts. Safe to call any number of times.

    The forecast keeps the session it was ASKED about: the model saw only
    that day's numbers, whenever its answer happened to come back.
    """
    out: dict[str, Any] = {"collected": 0, "written": 0, "still_pending": 0, "errors": []}
    for pending in store.batches(state_dir, state="pending"):
        batch_id = str(pending["batch_id"])
        try:
            info = client.poll(batch_id)
            if info.get("processing_status") != "ended" or not info.get("results_url"):
                age = now - datetime.fromisoformat(str(pending["submitted_at"]))
                if age > BATCH_GIVE_UP:
                    store.close_batch(state_dir, batch_id, "failed", "never finished", now=now)
                    out["errors"].append(f"batch {batch_id[-8:]} never finished; written off")
                else:
                    out["still_pending"] += 1
                continue
            rows: list[dict[str, Any]] = []
            failed = 0
            for line in client.results(str(info["results_url"])):
                meta = pending["requests"].get(str(line.get("custom_id")))
                text, usage, _kind = batch_api.message_text(line)
                if meta is None or text is None:
                    failed += 1
                    continue
                _record_usage(str(pending["model"]), usage, "shadow-batch")
                try:
                    from trading.agents import llm

                    answers = llm_forecaster.parse(llm._extract_json(text), meta["symbols"])
                except Exception:
                    failed += 1
                    continue
                source = llm_forecaster.source_name(str(pending["model"]), str(meta["lens"]))
                rows.extend(
                    {
                        "asof": pending["asof"],
                        "symbol": symbol,
                        "horizon": horizon,
                        "source": source,
                        "prob": prob,
                        "note": answer["why"],
                    }
                    for symbol, answer in answers.items()
                    for horizon, prob in answer["probs"].items()
                )
            written = store.save(state_dir, rows, now=now)
            store.close_batch(
                state_dir, batch_id, "collected", f"{written} forecasts, {failed} failed", now=now
            )
            out["collected"] += 1
            out["written"] += written
            if failed:
                out["errors"].append(f"batch {batch_id[-8:]}: {failed} request(s) unusable")
        except Exception as exc:
            out["errors"].append(f"batch {batch_id[-8:]}: {type(exc).__name__}: {str(exc)[:120]}")
    return out


def _ask_llm(
    state_dir: Path,
    asof: str,
    cards: list[dict[str, Any]],
    status: dict[str, Any],
    *,
    model: str,
    use_batch: bool,
    now: datetime,
    llm_call: Callable[[str, str], dict[str, Any]] | None,
    client: Any,
    lenses: tuple[str, ...] = llm_forecaster.LENSES,
) -> None:
    """Either queue one batch for every lens, or ask synchronously."""
    if use_batch:
        if store.batches(state_dir, asof=asof):
            return  # already queued for this session; never pay twice
        requests: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        for lens in lenses:
            for n, chunk in enumerate(
                llm_forecaster.batches_for(llm_forecaster.lens_cards(cards, lens))
            ):
                custom_id = f"{asof.replace('-', '')}-{lens}-{n:02d}"
                meta[custom_id] = {"lens": lens, "symbols": [c["symbol"] for c in chunk]}
                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": batch_api.message_params(
                            llm_forecaster.SYSTEM,
                            llm_forecaster.build_prompt(asof, chunk),
                            model=model,
                            max_tokens=llm_forecaster.MAX_TOKENS,
                        ),
                    }
                )
        batch_id = client.submit(requests)
        store.add_batch(state_dir, batch_id, asof=asof, model=model, requests=meta, now=now)
        status["llm_batch_submitted"] = len(requests)
        return
    for lens in lenses:
        answers, errors = llm_forecaster.forecast(
            asof, llm_forecaster.lens_cards(cards, lens), model=model, call=llm_call
        )
        status["errors"].extend(errors)
        source = llm_forecaster.source_name(model, lens)
        status["llm_written"] += store.save(
            state_dir,
            [
                {
                    "asof": asof,
                    "symbol": symbol,
                    "horizon": horizon,
                    "source": source,
                    "prob": prob,
                    "note": answer["why"],
                }
                for symbol, answer in answers.items()
                for horizon, prob in answer["probs"].items()
            ],
            now=now,
        )


def run_daily(
    state_dir: Path,
    data_dir: Path,
    symbols: list[str],
    *,
    top_n: int = 50,
    use_llm: bool = False,
    use_batch: bool = True,
    model: str = llm_forecaster.DEFAULT_MODEL,
    now: datetime | None = None,
    wait_s: float = 0.0,
    llm_call: Callable[[str, str], dict[str, Any]] | None = None,
    news_fn: Callable[[list[str]], dict[str, list[str]]] | None = None,
    batch_client: Any = batch_api,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect finished batches, grade what is due, forecast the newest session.

    Writes a status file whatever happens, saying what was ACHIEVED — rows
    written and graded — so a job that runs and produces nothing is visible.
    ``wait_s`` lets a run linger for its own batch, so on a normal evening
    the answers land the same night instead of the next.
    """
    now = now or UtcClock().now()
    status: dict[str, Any] = {
        "ran_at": now.isoformat(),
        "asof": None,
        "baseline_written": 0,
        "llm_written": 0,
        "llm_asked": 0,
        "llm_batch_submitted": 0,
        "llm_pending_batches": 0,
        "llm_enabled": bool(use_llm),
        "llm_route": ("batch" if use_batch else "direct") if use_llm else None,
        "model": model if use_llm else None,
        "errors": [],
    }

    def collect() -> None:
        got = collect_batches(
            state_dir, now=UtcClock().now() if wait_s else now, client=batch_client
        )
        status["llm_written"] += got["written"]
        status["llm_pending_batches"] = got["still_pending"]
        status["errors"].extend(got["errors"])

    try:
        collect()
        close, bench = load_close_panel(data_dir, symbols)
        raw, ranked, excess = prepare(close, bench)
        # The session to forecast is the newest one MOST of the universe has
        # a bar for, not the last row of the calendar. A refresh that reached
        # the benchmark and a handful of names would otherwise produce a
        # "universe" of twenty stocks ranked against each other.
        covered = np.flatnonzero(
            close.notna().sum(axis=1).to_numpy() >= MIN_COVERAGE * close.shape[1]
        )
        if not len(covered):
            raise ValueError("no session has prices for most of the universe")
        row = int(covered[-1])
        asof_ts = close.index[row]
        asof = str(asof_ts.date())
        status["asof"] = asof
        if row != len(close.index) - 1:
            status["note"] = (
                f"newest session {close.index[-1].date()} has too few prices; forecasting {asof}"
            )
        status["grading"] = store.grade_due(state_dir, close, bench, now=now)
        age = (pd.Timestamp(now).tz_convert("UTC").normalize() - asof_ts).days
        if age > MAX_BAR_AGE_DAYS:
            status["errors"].append(f"price cache is stale: newest bar {asof} is {age} days old")
        elif store.has_forecasts(state_dir, asof, store.BASELINE):
            status["note"] = (status.get("note", "") + f" {asof} already forecast").strip()
        else:
            rows: list[dict[str, Any]] = []
            for horizon in HORIZONS:
                model_h = fit(ranked, excess[horizon], horizon=horizon, last_row=row)
                if model_h is None:
                    status["errors"].append(f"no baseline model for {horizon}d (history too short)")
                    continue
                for symbol, prob in forecast_row(model_h, ranked, row).items():
                    rows.append(
                        {
                            "asof": asof,
                            "symbol": symbol,
                            "horizon": horizon,
                            "source": store.BASELINE,
                            "prob": float(prob),
                            "note": f"trained through {model_h.trained_through}",
                        }
                    )
            status["baseline_written"] = store.save(state_dir, rows, now=now)
            if use_llm and rows:
                momentum = ranked["mom_6_1"].iloc[row].dropna().sort_values(ascending=False)
                forecastable = {r["symbol"] for r in rows}
                chosen = [s for s in momentum.index if s in forecastable][: max(0, int(top_n))]
                try:
                    headlines = (news_fn or _headlines)(chosen)
                except Exception as exc:
                    headlines = {}
                    status["errors"].append(f"headlines unavailable: {type(exc).__name__}")
                status["llm_asked"] = len(chosen)
                with_news = sum(1 for s in chosen if headlines.get(s))
                status["headline_coverage"] = f"{with_news}/{len(chosen)}"
                lenses: tuple[str, ...] = llm_forecaster.LENSES
                if chosen and with_news < MIN_HEADLINE_COVERAGE * len(chosen):
                    lenses = tuple(lens for lens in lenses if lens != "news")
                    status["errors"].append(
                        f"headlines reached only {with_news} of {len(chosen)} names; "
                        "the news forecaster was not asked"
                    )
                try:
                    _ask_llm(
                        state_dir,
                        asof,
                        _cards(chosen, raw, ranked, row, headlines),
                        status,
                        model=model,
                        use_batch=use_batch,
                        now=now,
                        llm_call=llm_call,
                        client=batch_client,
                        lenses=lenses,
                    )
                except Exception as exc:
                    status["errors"].append(
                        f"LLM request failed: {type(exc).__name__}: {str(exc)[:160]}"
                    )
        waited = 0.0
        while wait_s and store.batches(state_dir, state="pending") and waited < wait_s:
            sleep(30.0)
            waited += 30.0
            collect()
        status["llm_pending_batches"] = len(store.batches(state_dir, state="pending"))
    except Exception as exc:
        logger.bind(component="forecasts").exception("shadow forecast run failed")
        status["errors"].append(f"{type(exc).__name__}: {str(exc)[:200]}")
    try:
        # Computed here, once a day, and read from the status file by the
        # dashboard — which refreshes every minute and must not scan a
        # ledger that grows by a thousand rows a session.
        status["summary"] = store.summary(state_dir)
    except Exception as exc:
        status["errors"].append(f"summary failed: {type(exc).__name__}")
    store.write_status(state_dir, status)
    return status
