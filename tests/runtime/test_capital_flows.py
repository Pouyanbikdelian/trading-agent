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


def test_equity_block_treats_a_manual_pinned_buy_as_a_transfer(tmp_path: Path) -> None:
    """Buying more of a pinned name by hand moves cash out of the desk and
    the same value into the pinned book. Neither is a return."""
    from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position
    from trading.dashboard.cockpit import equity_block
    from trading.runner.state import RunnerStore

    inst = Instrument(symbol="NVDA", asset_class=AssetClass.EQUITY, currency="CHF")
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["NVDA"]}))
    rs = RunnerStore(tmp_path / "runner.db")
    # day, cash, NVDA qty, NVDA mark
    for day, cash, qty, mark in [
        ("2026-08-18", 50_000, 100, 100.0),
        ("2026-08-19", 50_000, 100, 102.0),  # +200 pinned price move
        ("2026-08-20", 40_000, 200, 100.0),  # bought 100 @100 by hand, mark -2
        ("2026-08-21", 40_000, 200, 101.0),  # +200 pinned price move
    ]:
        pos = Position(
            instrument=inst, quantity=qty, avg_price=100.0, unrealized_pnl=qty * (mark - 100)
        )
        rs.save_snapshot(
            AccountSnapshot(
                ts=datetime.fromisoformat(day + "T20:00:00+00:00"),
                cash=cash,
                equity=cash + qty * mark,
                positions={inst.key: pos},
                base_currency="CHF",
            )
        )
    rs.close()
    by = {d["t"]: d for d in equity_block(tmp_path / "runner.db", tmp_path)["days"]}
    assert by["2026-08-18"]["pin_transfer"] == 0
    assert by["2026-08-20"]["pin_transfer"] == pytest.approx(10_000)
    # desk: 50k -> 40k, but 10k of that moved into NVDA: zero desk P&L.
    d0, d1 = by["2026-08-19"], by["2026-08-20"]
    desk_pnl = d1["desk"] - d0["desk"] + d1["pin_transfer"]
    pinned_pnl = d1["pinned"] - d0["pinned"] - d1["pin_transfer"]
    assert desk_pnl == pytest.approx(0)
    assert pinned_pnl == pytest.approx(-200)  # 100 shares fell 102 -> 100
    assert desk_pnl + pinned_pnl == pytest.approx(d1["account"] - d0["account"])
    assert "_pins" not in d1


def test_movers_value_each_position_in_the_base_currency(tmp_path: Path) -> None:
    """A dollar stock on a franc book: the line is francs, FX included."""
    from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position
    from trading.dashboard.cockpit import movers_block
    from trading.runner.state import RunnerStore

    nvda = Instrument(symbol="NVDA", asset_class=AssetClass.EQUITY, currency="USD")
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["NVDA"]}))
    rs = RunnerStore(tmp_path / "runner.db")
    # 10 NVDA: 100 -> 110 USD while USDCHF 0.80 -> 0.79; plus 1,000 USD cash.
    for ts, mark, fx in [
        ("2026-09-22T20:00:00+00:00", 100.0, 0.80),
        ("2026-09-23T15:00:00+00:00", 110.0, 0.79),
    ]:
        pos = Position(
            instrument=nvda, quantity=10, avg_price=100.0, unrealized_pnl=10 * (mark - 100)
        )
        rs.save_snapshot(
            AccountSnapshot(
                ts=datetime.fromisoformat(ts),
                cash=1_000 * fx,
                equity=1_000 * fx + 10 * mark * fx,
                positions={nvda.key: pos},
                base_currency="CHF",
                fx_rates={"USD": fx},
            )
        )
    rs.close()
    out = movers_block(tmp_path / "runner.db", tmp_path)
    (row,) = out["rows"]
    assert out["currency"] == "CHF"
    assert row["pnl"] == pytest.approx(1100 * 0.79 - 1000 * 0.80)  # 69 CHF, not 10 or 12.66
    assert row["pinned"] is True and row["traded"] is False
    # The USD cash lost 10 CHF to the dollar's move; no position owns it.
    assert out["residual"] == pytest.approx(-10.0)
    assert out["account_change"] == pytest.approx(row["pnl"] + out["residual"])
