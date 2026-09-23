"""Capital flows: a deposit is capital, not performance — however small."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading.runtime.capital_flows import flows_in_base, load_flows, record_flow

NOW = datetime(2026, 9, 23, 18, tzinfo=timezone.utc)


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    record_flow(
        tmp_path, amount=3000, currency="chf", day=date(2026, 8, 20), note="top-up", now=NOW
    )
    record_flow(tmp_path, amount=-500, currency="CHF", day=date(2026, 9, 1), now=NOW)
    flows = load_flows(tmp_path)
    assert [(f.day.isoformat(), f.amount, f.currency) for f in flows] == [
        ("2026-08-20", 3000.0, "CHF"),
        ("2026-09-01", -500.0, "CHF"),
    ]


def test_zero_and_future_flows_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        record_flow(tmp_path, amount=0, currency="CHF", day=date(2026, 8, 20), now=NOW)
    with pytest.raises(ValueError):
        record_flow(tmp_path, amount=10, currency="CHF", day=date(2026, 9, 24), now=NOW)


def test_a_corrupt_ledger_raises_instead_of_silently_forgetting(tmp_path: Path) -> None:
    (tmp_path / "capital_flows.json").write_text("{nope")
    with pytest.raises(ValueError):
        load_flows(tmp_path)


def test_flows_convert_to_base_and_unknown_rates_say_so(tmp_path: Path) -> None:
    record_flow(tmp_path, amount=1000, currency="USD", day=date(2026, 8, 20), now=NOW)
    record_flow(tmp_path, amount=2000, currency="CHF", day=date(2026, 8, 20), now=NOW)
    record_flow(tmp_path, amount=100, currency="GBP", day=date(2026, 8, 21), now=NOW)
    out = flows_in_base(load_flows(tmp_path), "CHF", {"USD": 0.8})
    assert out["2026-08-20"] == pytest.approx(2800.0)
    assert out["2026-08-21"] is None


# ------------------------------------------------------------ equity block


def _snapshots(tmp_path: Path, path: list[tuple[str, float, float]]) -> None:
    from trading.core.types import AccountSnapshot
    from trading.runner.state import RunnerStore

    rs = RunnerStore(tmp_path / "runner.db")
    for day, cash, equity in path:
        ts = datetime.fromisoformat(day + "T20:00:00+00:00")
        rs.save_snapshot(
            AccountSnapshot(ts=ts, cash=cash, equity=equity, positions={}, base_currency="CHF")
        )
    rs.close()


def test_equity_block_tracks_contributed_capital_and_flags_unrecorded_transfers(
    tmp_path: Path,
) -> None:
    from trading.dashboard.cockpit import equity_block

    _snapshots(
        tmp_path,
        [
            ("2026-08-18", 50_000, 88_000),
            ("2026-08-19", 50_000, 88_300),
            ("2026-08-20", 53_000, 91_300),  # +3k cash AND +3k equity: a deposit
            ("2026-08-21", 53_000, 90_900),
            ("2026-08-24", 51_000, 90_950),  # cash -2k, equity flat: a trade
        ],
    )
    out = equity_block(tmp_path / "runner.db", tmp_path)
    assert [c["t"] for c in out["flow_candidates"]] == ["2026-08-20"]
    assert out["flow_candidates"][0]["amount"] == pytest.approx(3000)

    record_flow(tmp_path, amount=3000, currency="CHF", day=date(2026, 8, 20), now=NOW)
    out = equity_block(tmp_path / "runner.db", tmp_path)
    assert out["flow_candidates"] == []
    by = {d["t"]: d for d in out["days"]}
    assert by["2026-08-20"]["flow"] == pytest.approx(3000)
    assert by["2026-08-24"]["net_flows"] == pytest.approx(3000)


def test_equity_block_reports_a_corrupt_ledger(tmp_path: Path) -> None:
    from trading.dashboard.cockpit import equity_block

    _snapshots(tmp_path, [("2026-08-18", 1, 2), ("2026-08-19", 1, 2)])
    (tmp_path / "capital_flows.json").write_text("{bad")
    out = equity_block(tmp_path / "runner.db", tmp_path)
    assert "unreadable" in out["note"] and len(out["days"]) == 2


# ---------------------------------------------------------------- the bot


def _bot(tmp_path: Path, monkeypatch):
    from trading.bot import telegram as tg

    monkeypatch.setattr(
        tg,
        "settings",
        SimpleNamespace(state_dir=tmp_path, trading_env="research", is_live_armed=lambda: False),
    )
    return tg


def test_deposit_and_withdraw_commands_record_signed_flows(tmp_path: Path, monkeypatch) -> None:
    tg = _bot(tmp_path, monkeypatch)
    today = datetime.now(tz=timezone.utc).date()
    assert "recorded deposit of 3,000.00 CHF on 2026-08-20" in tg._cmd_flow(
        ["3000", "CHF", "2026-08-20", "top-up"], +1
    )
    assert "withdrawal" in tg._cmd_flow(["1'500"], -1)
    raw = json.loads((tmp_path / "capital_flows.json").read_text())
    assert [(r["amount"], r["day"]) for r in raw] == [
        (3000.0, "2026-08-20"),
        (-1500.0, today.isoformat()),
    ]
    assert raw[0]["note"] == "top-up" and raw[0]["recorded_by"] == "telegram"
    listing = tg._cmd_flows()
    assert "+3,000.00 CHF" in listing and "net: +1,500.00 CHF" in listing


def test_flow_commands_refuse_bad_input(tmp_path: Path, monkeypatch) -> None:
    tg = _bot(tmp_path, monkeypatch)
    assert tg._cmd_flow([], +1).startswith("usage")
    assert "not an amount" in tg._cmd_flow(["abc"], +1)
    assert "positive amount" in tg._cmd_flow(["-5"], +1)
    future = (datetime.now(tz=timezone.utc).date() + timedelta(days=3)).isoformat()
    assert "future" in tg._cmd_flow(["100", "CHF", future], +1)
    assert not (tmp_path / "capital_flows.json").exists()
