r"""Test for the `--use-ibkr` safety check in `trading paper run`.

We can't easily test the full runner construction here (it'd need a
real broker connection), but we CAN verify the safety check rejects
IBKR_PORT=4001 — the worst-case footgun.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from trading.cli import app


def test_use_ibkr_refuses_live_port(monkeypatch) -> None:
    """If IBKR_PORT=4001 (live) the --use-ibkr flag must refuse to start.

    This is the defense-in-depth check: a user who flips ports without
    flipping the command shouldn't accidentally route paper-runner
    fills to a LIVE gateway.
    """
    monkeypatch.setenv("IBKR_PORT", "4001")
    # Force settings to re-read by clearing the cached singleton.
    from trading.core import config as _cfg

    _cfg.settings.__class__.__init__(_cfg.settings)  # type: ignore[misc]

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "paper",
            "run",
            "us_large_cap",
            "--strategy",
            "donchian",
            "--use-ibkr",
            "--once",
        ],
    )
    # Typer returns non-zero exit code on BadParameter.
    assert result.exit_code != 0
    assert "IBKR_PORT=4001" in result.output or "live" in result.output.lower()


def test_paper_run_help_shows_use_ibkr_flag() -> None:
    """The flag must be visible in --help so operators discover it."""
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "run", "--help"])
    assert result.exit_code == 0
    assert "--use-ibkr" in result.output


def test_paper_run_help_shows_param_flag() -> None:
    """The -p / --param flag must be visible — that's how operators
    override strategy params per-deploy (e.g. tighter rebalance for
    paper-testing)."""
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "run", "--help"])
    assert result.exit_code == 0
    assert "--param" in result.output or "-p" in result.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
