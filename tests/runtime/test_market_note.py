"""One daily market-risk note instead of four advisors (2026-09-26)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from trading.runtime import advisor, hmm_advisor, macro_monitor, market_note, options_monitor

NOW = datetime(2026, 9, 28, 20, 20, tzinfo=timezone.utc)


def _state(tmp: Path, **files: dict) -> None:
    for name, payload in files.items():
        (tmp / f"{name}.json").write_text(json.dumps(payload))


def test_the_monitors_defer_to_the_note_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MARKET_ALERTS", raising=False)
    sent: list[str] = []

    async def fake_send(text: str, **_k) -> bool:
        sent.append(text)
        return True

    import trading.bot.notifier as notifier

    monkeypatch.setattr(notifier, "send_message", fake_send)
    for mod in (advisor, hmm_advisor, options_monitor, macro_monitor):
        assert asyncio.run(mod._send_telegram("⚠️ *RISK signal* — `slow_grind`")) is False
    assert sent == []
    # An EXTREME SPY/VIX trigger still interrupts.
    assert asyncio.run(advisor._send_telegram("(severity: `EXTREME`)")) is True
    monkeypatch.setenv("MARKET_ALERTS", "instant")
    assert asyncio.run(macro_monitor._send_telegram("x")) is True


def test_a_calm_day_is_short_and_says_so(tmp_path: Path) -> None:
    _state(
        tmp_path,
        advisor={"active": [], "severities": {}},
        hmm_advisor={"label": "BULL", "p_bear": 0.05, "p_neutral": 0.15, "p_bull": 0.8},
        options_monitor={
            "active": [],
            "metrics": {"atm_iv": 0.14, "put_skew": 0.03, "term_slope": 0.01},
        },
        macro_monitor={"active": [], "readings": {"composite": 0.2}},
    )
    text, sigs = market_note.build_note(tmp_path, now=NOW)
    assert "🟢 Calm" in text and sigs == set()
    assert "no trigger" in text and "no stress" in text and "inside ±1.5σ" in text
    assert text.count("\n") <= 7


def test_new_and_cleared_signals_are_marked_against_yesterday(tmp_path: Path) -> None:
    _state(
        tmp_path,
        advisor={"active": ["slow_grind"], "severities": {"slow_grind": 1}},
        hmm_advisor={"label": "BEAR", "p_bear": 0.7, "p_neutral": 0.2, "p_bull": 0.1},
        options_monitor={"active": ["skew_stress"], "metrics": {"atm_iv": 0.28}},
        macro_monitor={"active": [], "readings": {"composite": 1.1}},
    )
    text, sigs = market_note.build_note(
        tmp_path, previous={"spy_vix:slow_grind", "macro:rates_shock"}, now=NOW
    )
    assert "🟠 Elevated" in text or "🔴 Stressed" in text
    assert "slow grind lower" in text and "🆕" in text  # BEAR regime and skew are new
    assert "Cleared since the last note* · rates_shock" in text
    assert "regime:BEAR" in sigs and "options:skew_stress" in sigs


def test_the_note_remembers_its_signals_for_tomorrow(tmp_path: Path) -> None:
    _state(tmp_path, advisor={"active": ["vol_spike"], "severities": {"vol_spike": 2}})
    first = market_note.compose_and_remember(tmp_path, now=NOW)
    second = market_note.compose_and_remember(tmp_path, now=NOW)
    assert "🆕" in first and "🆕" not in second
