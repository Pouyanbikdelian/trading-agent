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

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trading import __version__
from trading.backtest import compute_metrics, run_vectorized
from trading.backtest.costs import CostModel
from trading.core.config import settings
from trading.core.logging import configure_logging, logger
from trading.core.types import AssetClass, Instrument
from trading.core.universes import available_universes, load_universe
from trading.data.base import CANONICAL_FREQUENCIES, DataSource
from trading.data.cache import ParquetCache
from trading.runner import Runner, RunnerConfig
from trading.strategies import available_strategies, get_strategy

app = typer.Typer(
    name="trading",
    help="Trading Agent — research and execution CLI.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logs to console"),
) -> None:
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
    t.add_row(
        "IBKR", f"{settings.ibkr_host}:{settings.ibkr_port} (client_id={settings.ibkr_client_id})"
    )
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
        except Exception as e:
            table.add_row(ins.symbol, "-", f"[red]{type(e).__name__}: {e}[/red]")
            logger.bind(symbol=ins.symbol).exception("fetch failed")

    console.print(table)


# ---------------------------------------------------------------------------
# Backtest subcommands
# ---------------------------------------------------------------------------


@app.command("signals")
def _signals(
    universe: str = typer.Argument(..., help="Universe from config/universes.yaml."),
    strategy: list[str] = typer.Option(
        ["donchian"],
        "--strategy",
        "-s",
        help="Strategy name(s); repeat for multi-strategy view.",
    ),
    freq: str = typer.Option("1D", "--freq"),
    history_bars: int = typer.Option(
        252, "--bars", help="Bars of price history to feed the strategies."
    ),
    price_column: str = typer.Option("adj_close", "--price-column"),
    show_flat: bool = typer.Option(
        False, "--show-flat", help="Also print symbols the strategies want to leave flat."
    ),
) -> None:
    """Read-only: print what each strategy WOULD buy/sell today. No broker, no state.

    Useful as a sanity check before letting a runner go autonomous, or to
    compare what multiple strategies want on the same universe at the
    same instant.
    """
    if freq not in CANONICAL_FREQUENCIES:
        raise typer.BadParameter(f"freq={freq!r} not in {list(CANONICAL_FREQUENCIES)}")

    instruments = load_universe(universe)
    cache = ParquetCache(settings.data_dir)
    series: dict[str, pd.Series] = {}
    for ins in instruments:
        df = cache.read(ins, freq)  # type: ignore[arg-type]
        if df.empty or price_column not in df.columns:
            continue
        s = df[price_column].dropna()
        if not s.empty:
            series[ins.symbol] = s.iloc[-history_bars:]
    if not series:
        raise typer.BadParameter(f"no cached bars for {universe!r}; run `trading data fetch` first")
    prices = pd.DataFrame(series).sort_index().dropna(how="any")
    if prices.empty:
        raise typer.BadParameter("inner-join on dates produced an empty frame")

    for name in strategy:
        cls = get_strategy(name)
        s = cls()
        w = s.generate(prices)
        last = w.iloc[-1]
        longs = sorted([(sym, float(v)) for sym, v in last.items() if v > 0], key=lambda x: -x[1])
        shorts = sorted([(sym, float(v)) for sym, v in last.items() if v < 0], key=lambda x: x[1])
        flats = [sym for sym, v in last.items() if v == 0]

        t = Table(title=f"{name} on {universe} @ {prices.index[-1].date()}")
        t.add_column("symbol", style="cyan")
        t.add_column("side", style="green")
        t.add_column("weight", justify="right")
        for sym, v in longs:
            t.add_row(sym, "long", f"{v:+.3f}")
        for sym, v in shorts:
            t.add_row(sym, "short", f"{v:+.3f}")
        if show_flat:
            for sym in flats:
                t.add_row(sym, "flat", "0.000")
        console.print(t)
        console.print(
            f"  {len(longs)} long, {len(shorts)} short, {len(flats)} flat (total: {len(last)})"
        )


@backtest_app.command("strategies")
def _backtest_strategies() -> None:
    """List built-in strategies."""
    for n in available_strategies():
        console.print(f"- {n}")


def _parse_params(pairs: list[str]) -> dict[str, str]:
    """Parse ``--param key=value`` flags into a plain dict.

    The pydantic Params class on each strategy handles type coercion, so we
    don't try to be clever about int/float/bool parsing here.
    """
    out: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise typer.BadParameter(f"--param expects key=value, got {raw!r}")
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_prices_from_cache(
    universe: str,
    freq: str,
    start: datetime,
    end: datetime,
    price_column: str,
) -> pd.DataFrame:
    """Read each instrument's cached parquet and align into one wide frame."""
    instruments = load_universe(universe)
    cache = ParquetCache(settings.data_dir)
    series: dict[str, pd.Series] = {}
    for ins in instruments:
        df = cache.read(ins, freq)  # type: ignore[arg-type]
        if df.empty:
            logger.bind(symbol=ins.symbol).warning("no cached bars; run `trading data fetch` first")
            continue
        if price_column not in df.columns:
            raise typer.BadParameter(f"price_column={price_column!r} not in cache schema")
        s = df[price_column].dropna()
        s = s[(s.index >= start) & (s.index <= end)]
        if not s.empty:
            series[ins.symbol] = s
    if not series:
        raise typer.BadParameter(
            f"no cached prices for any symbol in universe {universe!r}; "
            "run `trading data fetch` first."
        )
    prices = pd.DataFrame(series).sort_index()
    # Inner-align: a strategy needs prices for the same dates across symbols.
    prices = prices.dropna(how="any")
    return prices


@backtest_app.command("run")
def _backtest_run(
    strategy: str = typer.Argument(..., help="Strategy name (see `trading backtest strategies`)."),
    universe: str = typer.Argument(..., help="Universe name from config/universes.yaml."),
    from_: str = typer.Option(..., "--from", help="Start date (ISO 8601)."),
    to: str = typer.Option(
        datetime.now(tz=timezone.utc).date().isoformat(),
        "--to",
        help="End date (default: today UTC).",
    ),
    freq: str = typer.Option("1D", "--freq", help=f"One of {list(CANONICAL_FREQUENCIES)}."),
    price_column: str = typer.Option(
        "adj_close",
        "--price-column",
        help="Which OHLCV column to use as the price series (close, adj_close, ...).",
    ),
    param: list[str] = typer.Option(
        [], "--param", "-p", help="Strategy override, e.g. -p lookback=20."
    ),
    commission_bps: float = typer.Option(1.0, "--commission-bps"),
    slippage_bps: float = typer.Option(2.0, "--slippage-bps"),
    periods_per_year: int = typer.Option(
        252,
        "--periods-per-year",
        help="Used to annualize Sharpe/CAGR. 252 daily equities, 365 crypto.",
    ),
) -> None:
    """Backtest a single strategy over a universe.

    Reads prices from the Parquet cache (populated by `trading data fetch`),
    materializes target weights via the strategy, runs the vectorized
    engine, and prints headline metrics.
    """
    if freq not in CANONICAL_FREQUENCIES:
        raise typer.BadParameter(f"freq={freq!r} not in {list(CANONICAL_FREQUENCIES)}")

    start = _parse_iso_date(from_)
    end = _parse_iso_date(to)

    overrides = _parse_params(param)
    strategy_cls = get_strategy(strategy)
    # Pydantic coerces string overrides ("True", "55", "0.2") into the right types.
    params = strategy_cls.Params(**overrides) if overrides else strategy_cls.Params()
    instance = strategy_cls(params=params)

    prices = _load_prices_from_cache(universe, freq, start, end, price_column)
    if len(prices) < 2:
        raise typer.BadParameter("need at least 2 aligned price rows to run a backtest")

    weights = instance.generate(prices)
    costs = CostModel(commission_bps=commission_bps, slippage_bps=slippage_bps)
    result = run_vectorized(prices, weights, costs=costs)
    metrics = compute_metrics(result, periods_per_year=periods_per_year)

    t = Table(title=f"Backtest — {strategy} on {universe} [{freq}]")
    t.add_column("metric", style="cyan")
    t.add_column("value", justify="right")
    for k, v in metrics.items():
        if k == "n_trades":
            t.add_row(k, f"{int(v):d}")
        elif (
            "return" in k
            or "drawdown" in k
            or "exposure" in k
            or "rate" in k
            or "vol" in k
            or "turnover" in k
            or "cagr" in k
        ):
            t.add_row(k, f"{v:.2%}")
        else:
            t.add_row(k, f"{v:.3f}")
    t.add_row("bars", f"{len(prices):d}")
    t.add_row("symbols", f"{prices.shape[1]:d}")
    console.print(t)


# ---------------------------------------------------------------------------
# Paper / live runner subcommands
# ---------------------------------------------------------------------------


def _build_runner_config(
    universe: str,
    strategies: list[str],
    freq: str,
    schedule_cron: str,
    schedule_tz: str,
    vol_target_value: float | None,
    initial_cash: float,
    use_simulator: bool,
) -> RunnerConfig:
    if freq not in CANONICAL_FREQUENCIES:
        raise typer.BadParameter(f"freq={freq!r} not in {list(CANONICAL_FREQUENCIES)}")
    return RunnerConfig(
        universe=universe,
        strategies=strategies,
        freq=freq,  # type: ignore[arg-type]
        schedule_cron=schedule_cron,
        schedule_tz=schedule_tz,
        vol_target=vol_target_value,
        initial_cash=initial_cash,
        use_simulator=use_simulator,
    )


def _print_report(report) -> None:
    t = Table(title=f"Cycle @ {report.ts.isoformat()}")
    t.add_column("field", style="cyan")
    t.add_column("value", justify="right")
    t.add_row("status", report.status)
    t.add_row("orders_submitted", str(report.orders_submitted))
    t.add_row("fills_received", str(report.fills_received))
    t.add_row("decisions", str(len(report.decisions)))
    t.add_row("duration_ms", f"{report.duration_ms:.1f}")
    if report.error:
        t.add_row("error", f"[red]{report.error}[/red]")
    console.print(t)


@paper_app.command("run")
def _paper_run(
    universe: str = typer.Argument(..., help="Universe from config/universes.yaml."),
    strategy: list[str] = typer.Option(
        ["donchian"], "--strategy", "-s", help="Strategy name(s); repeat for multi."
    ),
    freq: str = typer.Option("1D", "--freq"),
    cron: str = typer.Option("0 16 * * MON-FRI", "--cron", help="Crontab schedule."),
    tz: str = typer.Option("UTC", "--tz"),
    vol_target_value: float | None = typer.Option(
        None, "--vol-target", help="Annualized portfolio vol target (e.g. 0.10)."
    ),
    initial_cash: float = typer.Option(100_000.0, "--cash"),
    once: bool = typer.Option(False, "--once", help="Run one cycle and exit."),
) -> None:
    """Paper-trading runner (uses the in-memory Simulator)."""
    cfg = _build_runner_config(
        universe=universe,
        strategies=strategy,
        freq=freq,
        schedule_cron=cron,
        schedule_tz=tz,
        vol_target_value=vol_target_value,
        initial_cash=initial_cash,
        use_simulator=True,
    )
    runner = Runner.from_config(cfg)

    if once:
        report = runner.run_once()
        _print_report(report)
        return

    import asyncio

    try:
        asyncio.run(runner.run_forever())
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")


@live_app.command("run")
def _live_run(
    universe: str = typer.Argument(..., help="Universe from config/universes.yaml."),
    strategy: list[str] = typer.Option(["donchian"], "--strategy", "-s"),
    freq: str = typer.Option("1D", "--freq"),
    cron: str = typer.Option("0 16 * * MON-FRI", "--cron"),
    tz: str = typer.Option("UTC", "--tz"),
    vol_target_value: float | None = typer.Option(None, "--vol-target"),
    initial_cash: float = typer.Option(100_000.0, "--cash"),
) -> None:
    """Live-trading runner. Refuses unless both .env flags say live."""
    if not settings.is_live_armed():
        raise typer.BadParameter(
            "live trading requires BOTH TRADING_ENV=live AND ALLOW_LIVE_TRADING=true in .env"
        )
    # Build with the IBKR broker.
    from trading.execution.ibkr import IbkrBroker

    broker = IbkrBroker()
    broker.connect()
    cfg = _build_runner_config(
        universe=universe,
        strategies=strategy,
        freq=freq,
        schedule_cron=cron,
        schedule_tz=tz,
        vol_target_value=vol_target_value,
        initial_cash=initial_cash,
        use_simulator=False,
    )
    runner = Runner.from_config(cfg, broker=broker)
    import asyncio

    try:
        asyncio.run(runner.run_forever())
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


report_app = typer.Typer(help="Daily / mark-to-market reports.")
app.add_typer(report_app, name="report")


@report_app.command("daily")
def _report_daily(
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write the Markdown report to this file."
    ),
    no_news: bool = typer.Option(False, "--no-news", help="Skip news fetching."),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip the LLM executive summary."),
    no_vix: bool = typer.Option(False, "--no-vix", help="Skip the VIX regime fetch."),
) -> None:
    """Generate the daily report.

    Reads from the runner / order stores under settings.state_dir,
    optionally pulls news for held symbols, generates an executive
    summary via the Anthropic API if ANTHROPIC_API_KEY is set (otherwise
    falls back to a deterministic bullet-point summary), and writes
    Markdown to stdout or ``--output``.
    """
    from pathlib import Path

    from trading.reporting import (
        fetch_news_for_symbols,
        gather_daily_report,
        render_markdown,
        summarise,
    )

    report = gather_daily_report(fetch_vix=not no_vix)
    if not no_news and report.positions:
        report.news_by_symbol = fetch_news_for_symbols(
            list(report.positions.keys()),
            max_per_symbol=3,
        )
    summary = None if no_summary else summarise(report)
    md = render_markdown(report, executive_summary=summary)
    if output:
        Path(output).write_text(md)
        console.print(f"wrote {output}")
    else:
        console.print(md)


@report_app.command("weekly")
def _report_weekly(
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write the Markdown report to this file."
    ),
    no_news: bool = typer.Option(False, "--no-news", help="Skip news fetching."),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip the LLM executive summary."),
    no_vix: bool = typer.Option(False, "--no-vix", help="Skip the VIX regime fetch."),
) -> None:
    """Generate the weekly report.

    Same content as the daily report but the trade window is widened to
    7 days and more news per symbol is included — better suited to a
    slow-rebalance strategy where daily reports are mostly noise.
    """
    from pathlib import Path

    from trading.reporting import (
        fetch_news_for_symbols,
        gather_weekly_report,
        render_markdown,
        summarise,
    )

    report = gather_weekly_report(fetch_vix=not no_vix)
    if not no_news and report.positions:
        report.news_by_symbol = fetch_news_for_symbols(
            list(report.positions.keys()),
            max_per_symbol=5,
        )
    summary = None if no_summary else summarise(report)
    md = render_markdown(report, executive_summary=summary)
    if output:
        Path(output).write_text(md)
        console.print(f"wrote {output}")
    else:
        console.print(md)


if __name__ == "__main__":  # pragma: no cover
    app()
