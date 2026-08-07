"""A trailing stop must not fire on a price two sources disagree about.

``last_prices()`` reads yfinance — a free feed that occasionally serves
an unadjusted price across a split, a stale bar, or a bad tick. The
guards compared that number straight to the stop level and answered with
a full-position market sell. A 2:1 split not yet adjusted reads as a 50%
crash, and the exit is irreversible.

The broker already provides an independent mark for every position:
``avg_price + unrealized_pnl / quantity``. Disagreement between the two
means we do not know the price — and "we do not know" must never be a
reason to sell.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trading.runtime.guards import check_guards

NOW = datetime(2026, 8, 7, 18, 30, tzinfo=timezone.utc)


def _pos(symbol: str, qty: float, avg: float, mark: float | None = None) -> dict:
    p = {"symbol": symbol, "qty": qty, "avg_price": avg}
    if mark is not None:
        p["mark"] = mark
    return p


def _run(tmp_path, positions, prices, **kw):
    return check_guards(
        tmp_path,
        tmp_path,
        positions=positions,
        prices=prices,
        equity=88_000.0,
        now=NOW,
        **kw,
    )


def _prime(tmp_path, symbol, high, mark):
    """Establish a high-water mark so a later drop breaches the stop."""
    _run(tmp_path, [_pos(symbol, 63, 100.0, mark)], {symbol: high})


class TestASplitLooksLikeACrash:
    def test_a_halved_quote_with_an_intact_mark_does_not_sell(self, tmp_path) -> None:
        """The 2:1 split case: yfinance says 70, the broker says 140."""
        _prime(tmp_path, "VST", 140.0, 140.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 140.0)], {"VST": 70.0})

        assert out["exits"] == []

    def test_and_it_says_so_loudly(self, tmp_path) -> None:
        _prime(tmp_path, "VST", 140.0, 140.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 140.0)], {"VST": 70.0})

        joined = " ".join(out["alerts"])
        assert "price looks wrong" in joined
        assert "SKIPPED" in joined


class TestRealMovesStillExit:
    def test_a_genuine_drop_both_sources_agree_on_still_stops_out(self, tmp_path) -> None:
        """The guard must still do its job — both feeds down together."""
        _prime(tmp_path, "VST", 140.0, 140.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 100.0)], {"VST": 100.0})

        assert [e["symbol"] for e in out["exits"]] == ["VST"]
        assert out["exits"][0]["reason"] == "trailing_stop"

    def test_a_small_divergence_is_tolerated(self, tmp_path) -> None:
        """Quotes and marks are never identical — a 15-min delay is normal."""
        _prime(tmp_path, "VST", 140.0, 140.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 101.0)], {"VST": 100.0})

        assert [e["symbol"] for e in out["exits"]] == ["VST"]


class TestDegradingSafely:
    def test_no_mark_available_does_not_disable_the_guards(self, tmp_path) -> None:
        """A freshly opened position has no meaningful unrealized PnL.
        Half a safety net beats none."""
        _prime(tmp_path, "VST", 140.0, None)

        out = _run(tmp_path, [_pos("VST", 63, 100.0)], {"VST": 100.0})

        assert [e["symbol"] for e in out["exits"]] == ["VST"]

    def test_a_zero_mark_is_treated_as_unknown(self, tmp_path) -> None:
        _prime(tmp_path, "VST", 140.0, 0.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 0.0)], {"VST": 100.0})

        assert [e["symbol"] for e in out["exits"]] == ["VST"]

    def test_the_tolerance_is_tunable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("GUARD_PRICE_SANITY_PCT", "0.5")
        _prime(tmp_path, "VST", 140.0, 140.0)

        out = _run(tmp_path, [_pos("VST", 63, 100.0, 101.0)], {"VST": 100.0})

        assert out["exits"] == []


def test_a_suspect_quote_blocks_take_profit_too(tmp_path, monkeypatch) -> None:
    """Both exit reasons run off the same number."""
    monkeypatch.setenv("GUARD_TP_PCT", "10")
    _prime(tmp_path, "VST", 100.0, 100.0)

    out = _run(tmp_path, [_pos("VST", 63, 100.0, 100.0)], {"VST": 300.0})

    assert out["exits"] == []


def test_other_symbols_are_unaffected_by_one_bad_quote(tmp_path) -> None:
    _run(
        tmp_path,
        [_pos("VST", 63, 100.0, 140.0), _pos("WMT", 10, 90.0, 140.0)],
        {"VST": 140.0, "WMT": 140.0},
    )

    out = _run(
        tmp_path,
        [_pos("VST", 63, 100.0, 140.0), _pos("WMT", 10, 90.0, 100.0)],
        {"VST": 70.0, "WMT": 100.0},
    )

    assert [e["symbol"] for e in out["exits"]] == ["WMT"]
