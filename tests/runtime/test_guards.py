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


def test_ratchet_off_by_default_keeps_classic_trail(tmp_path: Path, monkeypatch) -> None:
    """No ratchet env -> a +50% winner still gets the full 8% leash."""
    for var in ("GUARD_TRAIL_FLOOR", "GUARD_TRAIL_TIGHTEN", "GUARD_TP_PCT"):
        monkeypatch.delenv(var, raising=False)
    check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 150.0}, equity=None, now=NOW
    )
    # 144 > 150*0.92=138: inside the classic trail, must NOT exit.
    r = check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 144.0}, equity=None, now=NOW
    )
    assert r["exits"] == []


def test_ratchet_tightens_the_leash_on_winners(tmp_path: Path, monkeypatch) -> None:
    """floor=0.4, tighten=1.2, +50% gain -> distance 8% * 0.4 = 3.2%:
    a dip to 144 (< 150*0.968=145.2) now exits where classic held."""
    monkeypatch.setenv("GUARD_TRAIL_FLOOR", "0.4")
    monkeypatch.setenv("GUARD_TRAIL_TIGHTEN", "1.2")
    monkeypatch.delenv("GUARD_TP_PCT", raising=False)
    check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 150.0}, equity=None, now=NOW
    )
    r = check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 144.0}, equity=None, now=NOW
    )
    assert [e["symbol"] for e in r["exits"]] == ["XYZ"]
    assert r["exits"][0]["reason"] == "trailing_stop"


def test_ratchet_floor_bounds_the_tightening(tmp_path: Path, monkeypatch) -> None:
    """A +400% moonshot: distance clamps at floor (8*0.4=3.2%), never 0 —
    normal daily noise below that must not exit."""
    monkeypatch.setenv("GUARD_TRAIL_FLOOR", "0.4")
    monkeypatch.setenv("GUARD_TRAIL_TIGHTEN", "1.2")
    check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 500.0}, equity=None, now=NOW
    )
    # 490 > 500*0.968=484: inside the floored trail, held.
    r = check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 490.0}, equity=None, now=NOW
    )
    assert r["exits"] == []


def test_ratchet_stop_level_never_falls(tmp_path: Path, monkeypatch) -> None:
    """The published stop level is monotone: a pullback that survives the
    trail must not loosen it (state file is the contract /guards shows)."""
    import json as _json

    monkeypatch.setenv("GUARD_TRAIL_FLOOR", "0.4")
    monkeypatch.setenv("GUARD_TRAIL_TIGHTEN", "1.2")
    check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 150.0}, equity=None, now=NOW
    )
    lvl1 = _json.loads((tmp_path / "guards.json").read_text())["positions"]["XYZ"]["stop_level"]
    check_guards(
        tmp_path, tmp_path, positions=[_pos("XYZ")], prices={"XYZ": 146.0}, equity=None, now=NOW
    )
    lvl2 = _json.loads((tmp_path / "guards.json").read_text())["positions"]["XYZ"]["stop_level"]
    assert lvl2 >= lvl1


def test_ratchet_ignores_nonsense_knobs(tmp_path: Path, monkeypatch) -> None:
    """floor >= 1 or tighten <= 0 (or unparseable) -> classic behavior."""
    for floor, tighten in (("1.5", "1.2"), ("0.4", "0"), ("abc", "1.2")):
        monkeypatch.setenv("GUARD_TRAIL_FLOOR", floor)
        monkeypatch.setenv("GUARD_TRAIL_TIGHTEN", tighten)
        d = tmp_path / f"case_{floor}_{tighten}"
        d.mkdir()
        check_guards(d, d, positions=[_pos("XYZ")], prices={"XYZ": 150.0}, equity=None, now=NOW)
        r = check_guards(d, d, positions=[_pos("XYZ")], prices={"XYZ": 144.0}, equity=None, now=NOW)
        assert r["exits"] == [], (floor, tighten)
