"""Trading successfully must not lock the system out.

2026-08-11, minutes after the first live cycle filled: the account held
the PM's ten names, none of them in ``holds.json``, so
``check_unheld_positions`` reported all ten as unprotected and
``preflight_unheld`` raised. Under ``restart: unless-stopped`` that is a
loop — a live runner that can never start again, and three identical
CRITICAL alerts in the operator's chat.

The check was written for positions arriving from OUTSIDE the system
(the operator's WMT, the VST the guards sold). It had no way to tell
those from the desk's own book, so it inverted the moment the desk
started trading: the better the system worked, the louder it screamed.

The discriminator is the order ledger. If we placed the order, the
position is ours and the rebalance selling it later is the design.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from trading.runtime.broker_ready import check_unheld_positions, desk_opened_symbols


def _ledger(path: Path, symbols: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE orders (client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,"
        " side TEXT, quantity REAL, order_type TEXT, limit_price REAL, stop_price REAL,"
        " tif TEXT, created_at REAL, status TEXT, broker_order_id TEXT)"
    )
    for i, s in enumerate(symbols):
        conn.execute(
            "INSERT INTO orders VALUES (?,?,'BUY',1,'MARKET',NULL,NULL,'DAY',0,'FILLED',NULL)",
            (f"o{i}", json.dumps({"symbol": s, "asset_class": "equity"})),
        )
    conn.commit()
    conn.close()


def _broker(*symbols: str):
    return SimpleNamespace(
        get_positions=lambda: [
            SimpleNamespace(instrument=SimpleNamespace(symbol=s), quantity=10.0) for s in symbols
        ]
    )


class TestTheLockout:
    def test_the_desks_own_basket_is_not_flagged(self, tmp_path) -> None:
        """The regression, in the exact shape it occurred."""
        basket = ["CNC", "DELL", "GLD", "HUM", "JPM", "PM", "UNH", "URA", "V", "XLV"]
        _ledger(tmp_path / "orders.db", basket)

        out = check_unheld_positions(_broker(*basket), state_dir=tmp_path)

        assert out["ok"] is True
        assert out["unheld"] == []
        assert out["desk_opened"] == sorted(basket)

    def test_a_foreign_position_is_still_caught(self, tmp_path) -> None:
        """The whole point of the check must survive the fix — this is the
        VST case, and it is the one that cost real money."""
        _ledger(tmp_path / "orders.db", ["GLD", "XLV"])

        out = check_unheld_positions(_broker("GLD", "XLV", "VST"), state_dir=tmp_path)

        assert out["ok"] is False
        assert out["unheld"] == ["VST"]

    def test_a_held_position_is_still_reported_as_held(self, tmp_path) -> None:
        (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["WMT"]}))
        _ledger(tmp_path / "orders.db", ["GLD"])

        out = check_unheld_positions(_broker("GLD", "WMT"), state_dir=tmp_path)

        assert out["ok"] is True
        assert out["held"] == ["WMT"]
        assert out["desk_opened"] == ["GLD"]

    def test_a_mixed_account_flags_only_the_stranger(self, tmp_path) -> None:
        (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["WMT"]}))
        _ledger(tmp_path / "orders.db", ["GLD", "XLV"])

        out = check_unheld_positions(_broker("GLD", "XLV", "WMT", "TSLA"), state_dir=tmp_path)

        assert out["unheld"] == ["TSLA"]


class TestTheLedgerRead:
    def test_symbols_come_back_uppercased(self, tmp_path) -> None:
        _ledger(tmp_path / "orders.db", ["gld", "Xlv"])

        assert desk_opened_symbols(tmp_path) == {"GLD", "XLV"}

    def test_a_missing_ledger_is_an_empty_set(self, tmp_path) -> None:
        assert desk_opened_symbols(tmp_path) == set()

    def test_a_schemaless_ledger_degrades_cautiously(self, tmp_path) -> None:
        """A freshly-armed account has a zero-byte orders.db. Returning
        {} keeps everything FLAGGED rather than silently waving it
        through — the safe direction for this particular check."""
        (tmp_path / "orders.db").write_bytes(b"")

        assert desk_opened_symbols(tmp_path) == set()

    def test_a_corrupt_row_does_not_lose_the_others(self, tmp_path) -> None:
        db = tmp_path / "orders.db"
        _ledger(db, ["GLD"])
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO orders VALUES ('bad','not json','BUY',1,'MARKET',"
            "NULL,NULL,'DAY',0,'FILLED',NULL)"
        )
        conn.commit()
        conn.close()

        assert desk_opened_symbols(tmp_path) == {"GLD"}

    def test_orders_count_even_before_they_fill(self, tmp_path) -> None:
        """Reads the ORDER ledger, not fills: a partial fill still means
        the position is ours."""
        db = tmp_path / "orders.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE orders (client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,"
            " side TEXT, quantity REAL, order_type TEXT, limit_price REAL, stop_price REAL,"
            " tif TEXT, created_at REAL, status TEXT, broker_order_id TEXT)"
        )
        conn.execute(
            "INSERT INTO orders VALUES ('o1',?,'BUY',10,'MARKET',NULL,NULL,'DAY',0,"
            "'SUBMITTED',NULL)",
            (json.dumps({"symbol": "NVDA", "asset_class": "equity"}),),
        )
        conn.commit()
        conn.close()

        assert desk_opened_symbols(tmp_path) == {"NVDA"}


def test_an_unreachable_broker_does_not_block_startup(tmp_path) -> None:
    """A check that blocks startup on a network blip is a check that gets
    removed — this behaviour predates the fix and must survive it."""

    def boom():
        raise ConnectionError("gateway down")

    out = check_unheld_positions(SimpleNamespace(get_positions=boom), state_dir=tmp_path)

    assert out["ok"] is True
    assert "skipped" in out["reason"]
