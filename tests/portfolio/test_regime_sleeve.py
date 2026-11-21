"""regime sleeve merge: off at 0, adds not overwrites, zeros sell exits."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.portfolio.regime_sleeve import (
    check_fresh,
    format_note,
    key_resolver,
    merge_regime_targets,
    read_state,
    write_state,
)

UNI = ("GLD", "IEF", "SPY")


def test_off_at_zero_adds_no_keys() -> None:
    book = {"equity:AAPL": 0.05}
    out = merge_regime_targets(book, {"SPY": 0.5}, 0.0, universe=UNI, key_for=lambda s: f"etf:{s}")
    assert out == book


def test_scales_adds_and_zeroes_exits() -> None:
    book = {"equity:AAPL": 0.05, "etf:GLD": 0.03}  # PM already holds GLD
    out = merge_regime_targets(
        book, {"SPY": 0.4, "GLD": 0.2}, 0.25, universe=UNI, key_for=lambda s: f"etf:{s}"
    )
    assert out["etf:SPY"] == pytest.approx(0.10)
    assert out["etf:GLD"] == pytest.approx(0.03 + 0.05)  # added, not overwritten
    assert out["etf:IEF"] == 0.0  # dropped leg gets an explicit zero -> sold
    assert out["equity:AAPL"] == 0.05
    assert book == {"equity:AAPL": 0.05, "etf:GLD": 0.03}  # input untouched


def test_key_resolver_prefers_existing_key() -> None:
    k = key_resolver({"equity:SPY": object(), "equity:AAPL": object()})
    assert k("SPY") == "equity:SPY"
    assert k("GLD") == "etf:GLD"


def test_stale_prices_refused() -> None:
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    check_fresh("2026-11-19", now)
    with pytest.raises(ValueError, match="days old"):
        check_fresh("2026-11-01", now)


def test_state_roundtrip_and_note(tmp_path) -> None:
    assert "no allocation yet" in format_note(read_state(tmp_path))
    write_state(
        tmp_path,
        {
            "regime": "bear",
            "legs": {"gtaa5": 1.0},
            "weights": {"IEF": 0.2},
            "sleeve_pct": 0.25,
            "effective_pct": 0.25,
            "asof": "2026-11-20",
        },
    )
    note = format_note(read_state(tmp_path))
    assert "🔴 bear" in note and "GTAA-5 100%" in note and "IEF 5.0%" in note
