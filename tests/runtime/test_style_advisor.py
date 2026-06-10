"""Style-rotation advisor — hermetic tests (no network, no Telegram)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from trading.runtime import style_advisor


@pytest.fixture
def prices() -> pd.DataFrame:
    """~2 years of synthetic daily closes: one strong trender, one
    mean-reverter, one flat name — enough breadth for most strategies
    to emit weights without erroring."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-06-01", periods=520, tz="UTC")
    n = len(idx)
    trend = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, n)))
    revert = 100 + np.cumsum(rng.normal(0, 1.0, n)) * 0.1
    flat = np.full(n, 50.0) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"AAA": trend, "BBB": revert, "CCC": flat}, index=idx)


def test_rank_styles_returns_scored_table(prices: pd.DataFrame) -> None:
    table = style_advisor.rank_styles(
        prices, strategy_names=["donchian", "ema_cross"], windows_months=(3, 6)
    )
    assert not table.empty
    assert "score" in table.columns
    assert {"sharpe_3m", "sharpe_6m", "ret_3m", "ret_6m"} <= set(table.columns)
    # sorted descending by score
    scores = table["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_rank_styles_skips_broken_strategy(prices: pd.DataFrame) -> None:
    # Nonexistent names are silently skipped; result covers the rest.
    table = style_advisor.rank_styles(
        prices, strategy_names=["donchian", "no_such_strategy"], windows_months=(3,)
    )
    assert list(table.index) == ["donchian"]


def test_poll_alerts_only_on_leader_change(prices: pd.DataFrame, tmp_path, monkeypatch) -> None:
    sent: list[str] = []

    async def _fake_send(text: str) -> bool:
        sent.append(text)
        return True

    monkeypatch.setattr(style_advisor, "_send_telegram", _fake_send)
    state = tmp_path / "style_advisor.json"

    # First poll: no prior leader; deployed strategy differs from leader
    # in general → alert expected only when leader != current. Use a
    # deliberately-wrong "current" to force the alert.
    out1 = asyncio.run(
        style_advisor.poll_and_alert(
            prices=prices,
            current_strategy="definitely_not_the_leader",
            state_path=state,
            windows_months=(3, 6),
        )
    )
    assert out1["ranked"] >= 1
    assert out1["alert_sent"] is True
    assert "Style rotation check" in sent[0]

    # Second poll, same data: leader unchanged → silent.
    out2 = asyncio.run(
        style_advisor.poll_and_alert(
            prices=prices,
            current_strategy="definitely_not_the_leader",
            state_path=state,
            windows_months=(3, 6),
        )
    )
    assert out2["alert_sent"] is False
    assert len(sent) == 1

    payload = json.loads(state.read_text())
    assert payload["leader"] == out2["leader"]
