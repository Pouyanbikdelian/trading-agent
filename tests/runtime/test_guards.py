"""Guards — hermetic: injected positions/prices, env knobs, tmp state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.runtime.guards import check_guards, enabled

NOW = datetime(2026, 6, 12, 15, 0, tzinfo=timezone.utc)


def _pos(sym: str, qty: float = 10, avg: float = 100.0) -> dict:
    return {"symbol": sym, "qty": qty, "avg_price": avg}


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GUARDS_ENABLED", raising=False)
    assert enabled() is False


def test_trail_rises_and_never_sells_a_runner(tmp_path: Path, monkeypatch) -> None:
    """The NVDA case: +20% then +100% more — a trail must ride it."""
    monkeypatch.delenv("GUARD_TP_PCT", raising=False)
    for px in (110.0, 140.0, 200.0, 240.0):  # relentless runner
        r = check_guards(
            tmp_path,
            tmp_path,
            positions=[_pos("NVDA")],
            prices={"NVDA": px},
            equity=None,
            now=NOW,
        )
        assert r["exits"] == []  # never exited on the way up
    # pullback within the 8% floor: still held
    r = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("NVDA")],
        prices={"NVDA": 225.0},
        equity=None,
        now=NOW,
    )
    assert r["exits"] == []
    # real breakdown: below 240 * 0.92 = 220.8 -> exit, gains locked
    r = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("NVDA")],
        prices={"NVDA": 219.0},
        equity=None,
        now=NOW,
    )
    assert [e["symbol"] for e in r["exits"]] == ["NVDA"]
    assert r["exits"][0]["reason"] == "trailing_stop"


def test_hold_pin_blocks_guard_exit(tmp_path: Path) -> None:
    check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("MU")],
        prices={"MU": 200.0},
        equity=None,
        now=NOW,
    )
    r = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("MU")],
        prices={"MU": 100.0},
        equity=None,
        holds={"MU"},
        now=NOW,
    )
    assert r["exits"] == []  # operator pinned it


def test_exit_cooldown(tmp_path: Path) -> None:
    check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("STX")],
        prices={"STX": 200.0},
        equity=None,
        now=NOW,
    )
    first = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("STX")],
        prices={"STX": 100.0},
        equity=None,
        now=NOW,
    )
    assert first["exits"]
    again = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("STX")],
        prices={"STX": 95.0},
        equity=None,
        now=NOW + timedelta(hours=1),
    )
    assert again["exits"] == []  # within 24h cooldown


def test_static_tp_only_when_opted_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GUARD_TP_PCT", "30")
    r = check_guards(
        tmp_path,
        tmp_path,
        positions=[_pos("GLW", avg=100.0)],
        prices={"GLW": 131.0},
        equity=None,
        now=NOW,
    )
    assert r["exits"] and r["exits"][0]["reason"] == "take_profit"


def test_portfolio_ratchet_alerts_on_giveback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GUARD_LOCK_ARM_PCT", "20")
    monkeypatch.setenv("GUARD_LOCK_GIVEBACK_PCT", "40")
    # baseline 100k -> peak 130k (armed at +20%) -> fall to 115k:
    # kept 15k <= 18k (60% of 30k peak gain) -> alert
    for eq, expect in ((100_000.0, False), (130_000.0, False), (115_000.0, True)):
        r = check_guards(tmp_path, tmp_path, positions=[], prices={}, equity=eq, now=NOW)
        assert any("ratchet" in a.lower() for a in r["alerts"]) is expect
    # does not repeat-spam while still down
    r = check_guards(tmp_path, tmp_path, positions=[], prices={}, equity=114_000.0, now=NOW)
    assert r["alerts"] == []
