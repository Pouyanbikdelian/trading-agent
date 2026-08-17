"""An unarmed live runner must refuse without becoming a crash loop.

``trader-live`` runs under ``restart: unless-stopped``, which restarts on
ANY exit code. ``trading live run`` refused instantly when not armed, so
the container respawned as fast as Python could start — hundreds of
restarts a minute, logs unreadable, autoheal thrashing.

It went unnoticed because the arming flag was hardcoded ``"true"`` in
docker-compose, so the refusal path never ran in production. The moment
that became settable from ``.env`` (2026-08-07), disarming a running
container spun it hot.

Refusing is right; refusing instantly under a restart policy is not.
"""

from __future__ import annotations

import types

import pytest
import typer

from trading import cli


@pytest.fixture
def unarmed(monkeypatch):
    monkeypatch.setattr(
        cli,
        "settings",
        types.SimpleNamespace(
            trading_env="live",
            allow_live_trading=False,
            is_live_armed=lambda: False,
        ),
    )
    slept: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", lambda s: slept.append(s))
    return slept


def _run_live() -> None:
    cli._live_run.__wrapped__("sp500") if hasattr(cli._live_run, "__wrapped__") else cli._live_run(
        universe="sp500",
        strategy=["donchian"],
        freq="1D",
        cron="5 21 * * FRI",
        tz="UTC",
        vol_target_value=None,
        initial_cash=100_000.0,
        param=[],
    )


def test_it_refuses_to_start(unarmed) -> None:
    with pytest.raises((typer.Exit, typer.BadParameter, SystemExit)):
        _run_live()


def test_it_waits_before_exiting(unarmed) -> None:
    """The whole point: a delay turns a restart storm into one quiet line
    a minute, without weakening the refusal."""
    with pytest.raises((typer.Exit, typer.BadParameter, SystemExit)):
        _run_live()

    assert unarmed, "exited immediately — this respawns as fast as it can under docker"
    assert unarmed[0] >= 1.0


def test_the_delay_is_tunable_for_tests_and_ci(monkeypatch, unarmed) -> None:
    monkeypatch.setenv("UNARMED_EXIT_DELAY_S", "0")

    with pytest.raises((typer.Exit, typer.BadParameter, SystemExit)):
        _run_live()

    assert unarmed[0] == 0.0


def test_it_never_reaches_the_broker(unarmed, monkeypatch) -> None:
    """The refusal must happen before any connection attempt."""

    def boom() -> None:
        raise AssertionError("connected to the broker while unarmed")

    monkeypatch.setattr("trading.execution.ibkr.IbkrBroker.connect", lambda self: boom())

    with pytest.raises((typer.Exit, typer.BadParameter, SystemExit)):
        _run_live()


def test_armed_cli_defers_broker_connection_to_the_safe_runner_bootstrap(monkeypatch) -> None:
    """The CLI must construct state/config before it attempts a live login.

    ``Runner.run_forever`` owns the bounded no-orders bootstrap loop, so a
    2FA outage remains observable instead of preventing the watchdog from
    coming up at all.
    """
    monkeypatch.setattr(cli, "settings", types.SimpleNamespace(is_live_armed=lambda: True))
    calls: list[str] = []

    class _Broker:
        def connect(self) -> None:
            calls.append("connect")
            raise AssertionError("CLI connected before Runner bootstrap")

    class _Runner:
        async def run_forever(self) -> None:
            calls.append("run_forever")

    monkeypatch.setattr("trading.execution.ibkr.IbkrBroker", _Broker)
    monkeypatch.setattr(
        cli,
        "Runner",
        types.SimpleNamespace(from_config=lambda _cfg, *, broker: _Runner()),
    )

    live_fn = cli._live_run.__wrapped__ if hasattr(cli._live_run, "__wrapped__") else cli._live_run
    live_fn(
        universe="sp500",
        strategy=["donchian"],
        freq="1D",
        cron="5 21 * * FRI",
        tz="UTC",
        vol_target_value=None,
        initial_cash=100_000.0,
        param=[],
    )

    assert calls == ["run_forever"]
