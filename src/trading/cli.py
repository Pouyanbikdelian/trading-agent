"""Single CLI entry point. New subcommands are added as the system grows.

The CLI is the primary way humans interact with the system:
  - ``trading status``        : print env, broker state, risk limits.
  - ``trading data fetch``    : backfill historical data into the Parquet cache.
  - ``trading backtest``      : run a strategy or ensemble over a date range.
  - ``trading paper``         : start the paper-trading runner.
  - ``trading live``          : start the live runner (gated by ALLOW_LIVE_TRADING).
"""

from __future__ import annotations

from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from trading import __version__
from trading.core.config import settings
from trading.core.logging import configure_logging, logger
from trading.core.types import AssetClass, Instrument
from trading.core.universes import available_universes, load_universe
from trading.data.base import CANONICAL_FREQUENCIES, DataSource, Frequency
from trading.data.cache import ParquetCache

app = typer.Typer(
    name="trading",
    help="Trading Agent — research and execution CLI.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logs to console")) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"trading-agent [bold]{__version__}[/bold]")


@app.command()
def status() -> None:
    """Print current environment, broker config, and risk limits."""
    settings.ensure_dirs()

    t = Table(title="Trading Agent — Status", show_header=False, box=None)
    t.add_column("key", style="cyan", no_wrap=True)
    t.add_column("value")

    t.add_row("version", __version__)
    t.add_row("trading_env", settings.trading_env)
    t.add_row("live armed?", "[red bold]YES[/red bold]" if settings.is_live_armed() else "no")
    t.add_row("IBKR", f"{settings.ibkr_host}:{settings.ibkr_port} (client_id={settings.ibkr_client_id})")
    t.add_row("data_dir", str(settings.data_dir))
    t.add_row("log_dir", str(settings.log_dir))
    t.add_row("state_dir", str(settings.state_dir))
    t.add_row("max_gross_exposure", f"{settings.max_gross_exposure:.2%}")
    t.add_row("max_position_pct", f"{settings.max_position_pct:.2%}")
    t.add_row("max_daily_loss_pct", f"{settings.max_daily_loss_pct:.2%}")
    t.add_row("max_drawdown_pct", f"{settings.max_drawdown_pct:.2%}")

    console.print(t)
    logger.debug("status command completed")


# ---------------------------------------------------------------------------
# Placeholders for upcoming subcommand groups. Filled in by later phases.
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="Data backfills and inspection.")
backtest_app = typer.Typer(help="Backtesting and walk-forward analysis.")
paper_app = typer.Typer(help="Paper-trading runner.")
live_app = typer.Typer(help="Live trading runner — gated by ALLOW_LIVE_TRADING.")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(paper_app, name="paper")
app.add_typer(live_app, name="live")


@data_app.command("universes")
def _data_universes() -> None:
    """List the universes defined in config/universes.yaml."""
    names = available_universes()
    for n in names:
        console.print(f"- {n}")


def _source_for(instrument: Instrument) -> DataSource:
    """Pick the right DataSource for an instrument's asset class.

    Equities/ETFs -> yfinance (free, fine for research).
    Crypto       -> ccxt (Binance public, free).
    FX           -> IBKR (needs IB Gateway running).
    """
    cls = instrument.asset_class
    if cls in (AssetClass.EQUITY, AssetClass.ETF):
        from trading.data.yfinance_source import YFinanceSource
        return YFinanceSource()
    if cls == AssetClass.CRYPTO:
        from trading.data.ccxt_source import CcxtSource
        return CcxtSource(exchange_id=instrument.exchange or "binance")
    if cls == AssetClass.FX:
        from trading.data.ibkr_source import IbkrSource
        return IbkrSource()
    raise typer.BadParameter(f"no DataSource configured for asset_class={cls.value}")


def _parse_iso_date(s: str) -> datetime:
    """Accept ``YYYY-MM-DD`` or full ISO 8601. Assume UTC for date-only input."""
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise typer.BadParameter(f"invalid ISO date: {s!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@data_app.command("fetch")
def _data_fetch(
    universe: str = typer.Argument(..., help="Universe name from config/universes.yaml."),
    from_: str = typer.Option(..., "--from", help="Start date (ISO 8601, e.g. 2020-01-01)."),
    to: str = typer.Option(
        datetime.now(tz=timezone.utc).date().isoformat(),
        "--to",
        help="End date (default: today UTC).",
    ),
    freq: str = typer.Option("1D", "--freq", help=f"One of {list(CANONICAL_FREQUENCIES)}."),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Ignore cached bars."),
) -> None:
    """Backfill bars for a universe into the Parquet cache.

    Picks the appropriate adapter per asset_class (yfinance / ccxt / IBKR).
    Cache layout: ``data/parquet/<asset_class>/<symbol>/<freq>.parquet``.
    """
    if freq not in CANONICAL_FREQUENCIES:
        raise typer.BadParameter(f"freq={freq!r} not in {list(CANONICAL_FREQUENCIES)}")

    start = _parse_iso_date(from_)
    end = _parse_iso_date(to)
    instruments = load_universe(universe)
    cache = ParquetCache(settings.data_dir)

    table = Table(title=f"Fetching {len(instruments)} symbols [{universe}] @ {freq}")
    table.add_column("symbol")
    table.add_column("rows", justify="right")
    table.add_column("status")

    for ins in instruments:
        try:
            source = _source_for(ins)
            df = cache.get_bars(source, ins, start, end, freq, force_refresh=force_refresh)  # type: ignore[arg-type]
            table.add_row(ins.symbol, str(len(df)), "ok")
            logger.bind(symbol=ins.symbol).info(f"cached {len(df)} bars [{freq}] via {source.name}")
        except Exception as e:  # noqa: BLE001 — surface, don't crash whole backfill
            table.add_row(ins.symbol, "-", f"[red]{type(e).__name__}: {e}[/red]")
            logger.bind(symbol=ins.symbol).exception("fetch failed")

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
