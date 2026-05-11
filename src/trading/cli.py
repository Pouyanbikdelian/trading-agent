"""Single CLI entry point. New subcommands are added as the system grows.

The CLI is the primary way humans interact with the system:
  - ``trading status``        : print env, broker state, risk limits.
  - ``trading data fetch``    : backfill historical data into the Parquet cache.
  - ``trading backtest``      : run a strategy or ensemble over a date range.
  - ``trading paper``         : start the paper-trading runner.
  - ``trading live``          : start the live runner (gated by ALLOW_LIVE_TRADING).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from trading import __version__
from trading.core.config import settings
from trading.core.logging import configure_logging, logger

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


@data_app.command("hello")
def _data_hello() -> None:
    """Smoke command to verify the subcommand group is wired."""
    console.print("data subcommand reachable.")


if __name__ == "__main__":  # pragma: no cover
    app()
