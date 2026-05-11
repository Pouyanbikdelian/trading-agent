"""Tests for the canonical bar-frame schema and the DataSource Protocol."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.data.base import (
    BAR_COLUMNS,
    CANONICAL_FREQUENCIES,
    DataSource,
    empty_bars_frame,
    validate_bars_frame,
)
from tests.data.conftest import make_bars


def test_empty_frame_has_canonical_schema() -> None:
    df = empty_bars_frame()
    assert list(df.columns) == list(BAR_COLUMNS)
    assert df.index.name == "ts"
    assert str(df.index.tz) == "UTC"
    assert df.empty


def test_validate_requires_tz_aware_index() -> None:
    df = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3).tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_bars_frame(df)


def test_validate_normalizes_tz_and_orders_columns() -> None:
    base = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3)
    # Put columns out of order and convert to a non-UTC tz on purpose.
    scrambled = base[["close", "volume", "adj_close", "open", "high", "low"]].tz_convert(
        "America/New_York"
    )
    out = validate_bars_frame(scrambled)
    assert list(out.columns) == list(BAR_COLUMNS)
    assert str(out.index.tz) == "UTC"


def test_validate_drops_duplicate_timestamps() -> None:
    base = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3)
    dup = pd.concat([base, base.iloc[[0]]])
    out = validate_bars_frame(dup)
    assert not out.index.has_duplicates


def test_datasource_protocol_runtime_check() -> None:
    class _Stub:
        name = "stub"

        def get_bars(self, instrument, start, end, freq):  # type: ignore[no-untyped-def]
            return empty_bars_frame()

    assert isinstance(_Stub(), DataSource)


def test_canonical_freqs_are_pandas_aliases() -> None:
    # Every canonical freq must parse as a pandas offset alias.
    for f in CANONICAL_FREQUENCIES:
        pd.tseries.frequencies.to_offset(f)


def test_instrument_unused_in_base() -> None:
    # Sanity import — keeps the test file honest about deps.
    Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
