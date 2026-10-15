"""A network request returning cached bars is not a successful refresh."""

import asyncio
from types import SimpleNamespace

import pandas as pd

from trading.core.types import AssetClass, Instrument
from trading.runner import runner as mod


def test_refresh_reports_stale_symbols_even_with_fresh_spy(tmp_path, monkeypatch):
    today = pd.Timestamp.now(tz="UTC").normalize()
    symbols = [f"S{i}" for i in range(10)]
    universe = [Instrument(symbol=s, asset_class=AssetClass.EQUITY) for s in symbols]
    monkeypatch.setattr("trading.core.universes.load_universe", lambda _: universe)
    monkeypatch.setattr(mod, "_add_reference_targets", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "trading.memory.store.default_store",
        lambda: SimpleNamespace(scorecard_backfill_targets=lambda **_: []),
    )
    monkeypatch.setattr(mod, "settings", SimpleNamespace(data_dir=tmp_path))

    def bars(_self, _source, ins, *_args):
        end = today - pd.Timedelta(days=9) if ins.symbol == "S9" else today
        return pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex([end]))

    monkeypatch.setattr(mod.ParquetCache, "get_bars", bars)
    monkeypatch.setattr(
        "trading.runtime.portfolio_stats._read_close",
        lambda *_: pd.Series([100.0], index=pd.DatetimeIndex([today])),
    )
    warnings = []
    runner = object.__new__(mod.Runner)
    runner.alerts = SimpleNamespace(warning=warnings.append, info=warnings.append)
    asyncio.run(runner._refresh_price_cache_async())
    assert len(warnings) == 1
    assert "9/10 current" in warnings[0]
    assert "S9" in warnings[0]
    asyncio.run(runner._refresh_price_cache_async())
    assert len(warnings) == 1
