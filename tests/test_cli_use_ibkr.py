r"""Test for the `--use-ibkr` safety check in `trading paper run`.

We can't easily test the full runner construction here (it'd need a
real broker connection), but we CAN verify the safety check rejects
IBKR_PORT=4001 — the worst-case footgun.

CI quirk: GitHub Actions runs with a narrow terminal (~80 cols) and
Typer/Rich wraps the --help output. Substring checks like ``"--use-ibkr"
in result.output`` fail when the flag straddles a line break. We use
a wider TERMINAL_WIDTH and strip whitespace before matching.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from trading.cli import app


def _normalize(text: str) -> str:
    """Strip ANSI escapes + collapse whitespace so substring checks
    don't break on terminal line-wraps."""
    # Strip CSI/SGR ANSI sequences
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    # Collapse all whitespace (including newlines + zero-width box chars)
    return re.sub(r"\s+", "", text)


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
    out = _normalize(result.output)
    assert "IBKR_PORT=4001" in out or "live" in out.lower()


def test_paper_run_help_shows_use_ibkr_flag() -> None:
    """The flag must be visible in --help so operators discover it.

    Use ``_normalize`` so a narrow terminal that wraps ``--use-ibkr``
    across a line break doesn't break the test (CI's default ~80 cols
    triggers this).
    """
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "run", "--help"])
    assert result.exit_code == 0
    assert "--use-ibkr" in _normalize(result.output)


def test_paper_run_help_shows_param_flag() -> None:
    """The -p / --param flag must be visible — that's how operators
    override strategy params per-deploy (e.g. tighter rebalance for
    paper-testing)."""
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "run", "--help"])
    assert result.exit_code == 0
    normalized = _normalize(result.output)
    assert "--param" in normalized or "-p" in result.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
