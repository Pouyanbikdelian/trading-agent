"""The daily run end to end, on a synthetic price cache."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from trading.forecasts import store
from trading.forecasts.job import run_daily

NOW = datetime(2026, 11, 10, 23, 30, tzinfo=timezone.utc)


def _fake_llm(seen):
    def call(system, prompt):
        payload = json.loads(prompt)
        seen.append(payload)
        return {
            "forecasts": [
                {"symbol": c["symbol"], "p5": 0.52, "p21": 0.47, "why": "test"}
                for c in payload["stocks"]
            ]
        }

    return call


def test_baseline_only_run_forecasts_the_universe_and_spends_nothing(tmp_path, data_dir, panel):
    close, _ = panel
    status = run_daily(tmp_path, data_dir, list(close.columns), now=NOW)
    assert status["asof"] == "2026-11-10" and not status["errors"]
    assert status["baseline_written"] == 2 * close.shape[1]  # 5d and 21d
    assert status["llm_written"] == 0 and status["llm_asked"] == 0
    assert store.read_status(tmp_path)["summary"]["total"] == 60


def test_llm_run_asks_the_top_momentum_names_without_showing_the_baseline(
    tmp_path, data_dir, panel
):
    close, _ = panel
    seen = []
    status = run_daily(
        tmp_path,
        data_dir,
        list(close.columns),
        top_n=12,
        use_llm=True,
        use_batch=False,
        model="claude-sonnet-5-5",
        now=NOW,
        llm_call=_fake_llm(seen),
        news_fn=lambda symbols: {symbols[0]: ["2026-11-09 Something was reported"]},
    )
    # Two forecasters (numbers, then news), each 12 names at ten per call.
    assert status["llm_asked"] == 12 and status["llm_written"] == 48
    assert len(seen) == 4
    assert all("recent_headlines" not in c for payload in seen[:2] for c in payload["stocks"])
    cards = [c for payload in seen[2:] for c in payload["stocks"]]
    # Exactly the twelve strongest six-month momentum names, strongest first.
    from trading.forecasts.features import raw_features

    expected = raw_features(close)["mom_6_1"].iloc[-1].sort_values(ascending=False).index[:12]
    assert [c["symbol"] for c in cards] == list(expected)
    assert cards[0]["recent_headlines"] == ["2026-11-09 Something was reported"]
    text = json.dumps(seen)
    assert "prob" not in text and "baseline" not in text  # no anchor to copy
    summary = store.read_status(tmp_path)["summary"]
    assert {s["source"] for s in summary["sources"]} == {
        "baseline",
        "llm:claude-sonnet-5-5:numbers",
        "llm:claude-sonnet-5-5:news",
    }


def test_a_second_run_for_the_same_session_adds_nothing(tmp_path, data_dir, panel):
    close, _ = panel
    seen = []
    kwargs = {"use_llm": True, "use_batch": False, "top_n": 5, "now": NOW, "news_fn": lambda s: {}}
    run_daily(tmp_path, data_dir, list(close.columns), llm_call=_fake_llm(seen), **kwargs)
    again = run_daily(tmp_path, data_dir, list(close.columns), llm_call=_fake_llm(seen), **kwargs)
    assert again["baseline_written"] == 0 and again["llm_written"] == 0
    assert len(seen) == 2  # one call per forecaster; not asked, or paid, twice


def test_a_stale_cache_is_reported_not_forecast(tmp_path, data_dir, panel):
    close, _ = panel
    later = datetime(2026, 11, 20, tzinfo=timezone.utc)
    status = run_daily(tmp_path, data_dir, list(close.columns), now=later)
    assert status["baseline_written"] == 0
    assert any("stale" in e for e in status["errors"])


def test_a_broken_run_still_leaves_a_status_that_says_so(tmp_path):
    status = run_daily(tmp_path, tmp_path / "no-data", ["AAA"], now=NOW)
    assert status["baseline_written"] == 0 and status["errors"]
    assert store.read_status(tmp_path)["errors"]


def test_headline_or_model_failures_do_not_cost_the_baseline(tmp_path, data_dir, panel):
    close, _ = panel

    def no_news(_symbols):
        raise ConnectionError("yahoo down")

    def no_model(_system, _prompt):
        raise RuntimeError("api down")

    status = run_daily(
        tmp_path,
        data_dir,
        list(close.columns),
        use_llm=True,
        use_batch=False,
        top_n=5,
        now=NOW,
        llm_call=no_model,
        news_fn=no_news,
    )
    assert status["baseline_written"] == 60 and status["llm_written"] == 0
    assert len(status["errors"]) == 3  # headlines, and one call per forecaster


def _settings(tmp_path, **over):
    base = {
        "forecast_shadow_enabled": True,
        "forecast_shadow_llm": False,
        "state_dir": tmp_path,
        "data_dir": tmp_path,
    }
    return SimpleNamespace(**{**base, **over})


class _Proc:
    def __init__(self, *, hang=False, returncode=0):
        self.hang, self.returncode, self.killed = hang, returncode, False

    async def communicate(self):
        if self.hang:
            await asyncio.sleep(30)
        return b"", b"boom"

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def test_the_runner_job_is_inert_until_switched_on(tmp_path, monkeypatch):
    from trading.runner import runner as mod

    spawned = []

    async def spawn(*cmd, **_kwargs):
        spawned.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    runner = object.__new__(mod.Runner)

    monkeypatch.setattr(mod, "settings", _settings(tmp_path, forecast_shadow_enabled=False))
    asyncio.run(runner._run_forecast_shadow_async())
    assert spawned == []

    monkeypatch.setattr(mod, "settings", _settings(tmp_path))
    asyncio.run(runner._run_forecast_shadow_async())
    assert spawned[0][1:5] == ["-m", "trading.cli", "forecasts", "run"]
    assert "--llm" not in spawned[0]  # tokens are a second, separate switch

    monkeypatch.setattr(mod, "settings", _settings(tmp_path, forecast_shadow_llm=True))
    asyncio.run(runner._run_forecast_shadow_async())
    assert spawned[1][-3:] == ["--llm", "--wait", "480"]


def test_a_hung_forecast_run_is_killed_and_never_raises_into_the_runner(tmp_path, monkeypatch):
    """It runs in its own process precisely so this is possible."""
    from trading.runner import runner as mod

    procs = []

    async def spawn(*_cmd, **_kwargs):
        procs.append(_Proc(hang=True))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(mod, "settings", _settings(tmp_path))
    runner = object.__new__(mod.Runner)
    runner.FORECAST_SHADOW_TIMEOUT_S = 0.05
    asyncio.run(runner._run_forecast_shadow_async())
    assert procs[0].killed

    async def crash(*_cmd, **_kwargs):
        return _Proc(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", crash)
    asyncio.run(runner._run_forecast_shadow_async())  # logged, not raised


def test_the_dashboard_block_is_a_file_read_and_the_panel_exists(tmp_path):
    from trading.dashboard.app import _PAGE, _forecasts_block

    assert _forecasts_block(tmp_path) == {}
    store.write_status(tmp_path, {"ran_at": "x", "summary": {"total": 3}})
    assert _forecasts_block(tmp_path)["summary"]["total"] == 3
    assert 'id="fcT"' in _PAGE and "function renderForecasts(){" in _PAGE
    assert "renderForecasts();renderLLM();" in _PAGE


def test_a_session_only_a_few_names_reached_is_not_the_one_forecast(tmp_path, data_dir, panel):
    """Seen on the real cache: 20 of 500 symbols had the newest bar. Ranking
    twenty stocks against each other is not a forecast of the universe."""
    import pandas as pd

    close, _ = panel
    for symbol in close.columns[3:]:
        path = data_dir / "equity" / symbol / "1D.parquet"
        pd.read_parquet(path).iloc[:-1].to_parquet(path)
    status = run_daily(tmp_path, data_dir, list(close.columns), now=NOW)
    assert status["asof"] == "2026-11-09"
    assert status["baseline_written"] == 2 * close.shape[1]
    assert "too few prices" in status["note"]


# ---------------------------------------------------------------------------
# Batch route: half price, answers later
# ---------------------------------------------------------------------------


class _Batches:
    """Stand-in for the Message Batches API."""

    def __init__(self, *, ended=True, bad=()):
        self.ended, self.bad = ended, set(bad)
        self.submitted: list[list[dict]] = []

    def submit(self, requests):
        self.submitted.append(requests)
        return f"msgbatch_{len(self.submitted):04d}"

    def poll(self, batch_id):
        if not self.ended:
            return {"processing_status": "in_progress", "results_url": None}
        return {"processing_status": "ended", "results_url": f"https://x/{batch_id}"}

    def results(self, url):
        requests = self.submitted[int(url[-4:]) - 1]
        lines = []
        for request in reversed(requests):  # the API promises no ordering
            cid = request["custom_id"]
            if cid in self.bad:
                lines.append({"custom_id": cid, "result": {"type": "errored", "error": {}}})
                continue
            stocks = json.loads(request["params"]["messages"][0]["content"])["stocks"]
            answer = {
                "forecasts": [{"symbol": c["symbol"], "p5": 0.51, "p21": 0.49} for c in stocks]
            }
            lines.append(
                {
                    "custom_id": cid,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {"type": "thinking", "thinking": "..."},
                                {"type": "text", "text": json.dumps(answer)},
                            ],
                            "usage": {"input_tokens": 900, "output_tokens": 300},
                        },
                    },
                }
            )
        return lines


def _batch_run(tmp_path, data_dir, close, client, **over):
    kwargs = {
        "use_llm": True,
        "top_n": 12,
        "now": NOW,
        "news_fn": lambda s: {},
        "batch_client": client,
    }
    return run_daily(tmp_path, data_dir, list(close.columns), **{**kwargs, **over})


def test_batch_route_queues_tonight_and_collects_on_the_next_run(
    tmp_path, data_dir, panel, monkeypatch
):
    from trading.forecasts import batch as batch_api

    close, _ = panel
    usage = []
    monkeypatch.setattr("trading.agents.llm._record_telemetry", lambda **kw: usage.append(kw))
    client = _Batches(ended=False)

    first = _batch_run(tmp_path, data_dir, close, client)
    assert first["llm_route"] == "batch" and first["baseline_written"] == 60
    assert first["llm_batch_submitted"] == 4  # two forecasters x two requests of <=10
    assert first["llm_written"] == 0 and first["llm_pending_batches"] == 1
    assert all(batch_api.CUSTOM_ID.match(r["custom_id"]) for r in client.submitted[0])
    assert client.submitted[0][0]["params"]["model"] == "claude-sonnet-5-5"

    # Same evening, run again: still processing, and NOT submitted twice.
    again = _batch_run(tmp_path, data_dir, close, client)
    assert len(client.submitted) == 1 and again["llm_pending_batches"] == 1

    client.ended = True
    later = _batch_run(tmp_path, data_dir, close, client)
    assert later["llm_written"] == 48 and later["llm_pending_batches"] == 0
    assert len(usage) == 4 and {u["tier"] for u in usage} == {"shadow-batch"}
    sources = {s["source"]: s for s in later["summary"]["sources"] if s["horizon"] == 5}
    news = sources["llm:claude-sonnet-5-5:news"]
    assert news["forecasts"] == 12 and news["first"] == "2026-11-10"  # the session asked about

    assert _batch_run(tmp_path, data_dir, close, client)["llm_written"] == 0  # idempotent


def test_a_run_can_wait_for_its_own_batch(tmp_path, data_dir, panel):
    close, _ = panel
    client = _Batches(ended=False)
    naps = []

    def nap(seconds):
        naps.append(seconds)
        client.ended = len(naps) >= 2

    status = _batch_run(tmp_path, data_dir, close, client, wait_s=300, sleep=nap)
    assert naps == [30.0, 30.0]
    assert status["llm_written"] == 48 and status["llm_pending_batches"] == 0


def test_one_bad_request_in_a_batch_loses_only_its_ten_stocks(tmp_path, data_dir, panel):
    close, _ = panel
    client = _Batches(bad={"20261110-news-00"})
    _batch_run(tmp_path, data_dir, close, client)
    status = _batch_run(tmp_path, data_dir, close, client)
    counts = {
        s["source"]: s["forecasts"] for s in status["summary"]["sources"] if s["horizon"] == 5
    }
    assert counts["llm:claude-sonnet-5-5:numbers"] == 12
    assert counts["llm:claude-sonnet-5-5:news"] == 2
    assert store.batches(tmp_path, state="collected")[0]["detail"] == "28 forecasts, 1 failed"


def test_a_batch_that_never_finishes_is_written_off_not_polled_forever(tmp_path, data_dir, panel):
    from datetime import timedelta

    close, _ = panel
    client = _Batches(ended=False)
    _batch_run(tmp_path, data_dir, close, client)
    status = _batch_run(tmp_path, data_dir, close, client, now=NOW + timedelta(hours=31))
    assert any("never finished" in e for e in status["errors"])
    assert store.batches(tmp_path, state="pending") == []
    assert store.batches(tmp_path, state="failed")


def test_a_failed_submission_costs_nothing_and_keeps_the_baseline(tmp_path, data_dir, panel):
    close, _ = panel

    class _Down(_Batches):
        def submit(self, requests):
            raise ConnectionError("api unreachable")

    status = _batch_run(tmp_path, data_dir, close, _Down())
    assert status["baseline_written"] == 60 and status["llm_batch_submitted"] == 0
    assert any("LLM request failed" in e for e in status["errors"])


def test_batch_requests_match_the_direct_call_and_reject_bad_ids(monkeypatch):
    from trading.forecasts import batch as batch_api

    params = batch_api.message_params("sys", "prompt", model="claude-sonnet-5-5", max_tokens=100)
    assert params["thinking"] == {"type": "adaptive"} and "effort" in params["output_config"]
    assert "stream" not in params
    monkeypatch.setattr(batch_api, "_headers", lambda: {})
    import pytest

    with pytest.raises(ValueError, match="custom_id"):
        batch_api.submit([{"custom_id": "2026-11-10|news|0", "params": params}])
    text, usage, kind = batch_api.message_text({"result": {"type": "expired"}})
    assert text is None and usage == {} and kind == "expired"


def test_the_pm_is_shown_against_the_mechanical_picks_not_only_spy(tmp_path):
    from trading.dashboard.app import _PAGE, _pm_shadow_block

    assert _pm_shadow_block(tmp_path) == {}
    folder = tmp_path / "agent_pm"
    folder.mkdir()
    records = [
        {"status": "graded", "horizon": 5, "attribution": {"pm_minus_mechanical": 0.02}},
        {"status": "graded", "horizon": 5, "attribution": {"pm_minus_mechanical": -0.01}},
        {"status": "pending", "horizon": 5, "attribution": {"pm_minus_mechanical": 9.0}},
        {"status": "graded", "horizon": 21, "attribution": {}},
    ]
    (folder / "shadow_status.json").write_text(
        json.dumps({"checked_at": "t", "counts": {"graded": 2, "pending": 1}, "records": records})
    )
    block = _pm_shadow_block(tmp_path)
    assert block["horizons"] == [
        {"horizon": 5, "cohorts": 2, "mean_pm_minus_mechanical": 0.005, "pm_ahead": 1}
    ]
    assert "function pmVsMechanical(){" in _PAGE and "+pmVsMechanical();" in _PAGE
