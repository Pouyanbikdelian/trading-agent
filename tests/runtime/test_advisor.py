r"""Tests for the advisory alerter — must never auto-execute, only inform."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.runtime import advisor


def _make_spy(prices: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="1D", tz="UTC")
    return pd.Series(prices, index=idx, name="SPY", dtype=float)


@pytest.fixture
def captured(monkeypatch):
    """Capture would-be Telegram messages instead of sending them."""
    captured_messages: list[str] = []

    async def fake_send(text: str) -> bool:
        captured_messages.append(text)
        return True

    monkeypatch.setattr(advisor, "_send_telegram", fake_send)
    return captured_messages


def test_calm_market_sends_no_alert(captured, tmp_path: Path) -> None:
    spy = _make_spy(np.linspace(100, 200, 400).tolist())
    state_path = tmp_path / "advisor.json"
    out = asyncio.run(advisor.poll_and_alert(spy=spy, state_path=state_path))
    assert out["triggers"] == []
    assert captured == []  # no message sent


def test_first_trigger_sends_alert(captured, tmp_path: Path) -> None:
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    spy = _make_spy(up + down)
    out = asyncio.run(advisor.poll_and_alert(spy=spy, state_path=tmp_path / "advisor.json"))
    assert any(t["name"] == "slow_grind" for t in out["triggers"])
    assert "new_or_escalated" in out
    assert "slow_grind" in out["new_or_escalated"]
    assert len(captured) == 1
    assert "RISK signal" in captured[0]
    assert "advisory" in captured[0].lower()  # explicit "no auto-action" wording


def test_repeated_trigger_is_debounced(captured, tmp_path: Path) -> None:
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    spy = _make_spy(up + down)
    state_path = tmp_path / "advisor.json"

    asyncio.run(advisor.poll_and_alert(spy=spy, state_path=state_path))
    # Second poll on the same series — same trigger active, no new alert.
    asyncio.run(advisor.poll_and_alert(spy=spy, state_path=state_path))
    assert len(captured) == 1  # debounced


def test_recovery_sends_clear_alert(captured, tmp_path: Path) -> None:
    state_path = tmp_path / "advisor.json"

    # First poll: stress regime — fires
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    asyncio.run(advisor.poll_and_alert(spy=_make_spy(up + down), state_path=state_path))
    assert len(captured) == 1

    # Second poll: pure uptrend — all clear
    asyncio.run(
        advisor.poll_and_alert(
            spy=_make_spy(np.linspace(100, 250, 400).tolist()),
            state_path=state_path,
        )
    )
    assert len(captured) == 2
    assert "RECOVERY" in captured[1]


def test_state_file_is_persisted(captured, tmp_path: Path) -> None:
    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    spy = _make_spy(up + down)
    state_path = tmp_path / "advisor.json"
    asyncio.run(advisor.poll_and_alert(spy=spy, state_path=state_path))
    payload = json.loads(state_path.read_text())
    assert "active" in payload
    assert "last_polled_at" in payload


def test_advisor_never_writes_mode_file(captured, tmp_path: Path) -> None:
    """Hard invariant — the advisor MUST NOT modify mode.json."""
    state_path = tmp_path / "advisor.json"
    mode_path = tmp_path / "mode.json"
    mode_path.write_text('{"mode":"bull","set_at":"","set_by":"test","reason":""}')
    before = mode_path.read_text()

    up = np.linspace(100, 300, 250).tolist()
    down = np.linspace(300, 200, 200).tolist()
    asyncio.run(advisor.poll_and_alert(spy=_make_spy(up + down), state_path=state_path))
    after = mode_path.read_text()
    assert before == after  # advisory only — mode unchanged
