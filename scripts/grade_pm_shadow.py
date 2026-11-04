"""Grade immutable PM cohorts from local caches, with no broker or network calls."""

import json
from pathlib import Path

import typer

from trading.agents.pm import _save
from trading.core.clock import UtcClock
from trading.reporting.pm_shadow import _format_shadow_status, _grade_saved_shadows


def main(
    state_dir: Path = typer.Option(..., help="State containing recorded PM decisions"),
    data_dir: Path = typer.Option(..., help="Parquet root with equity/etf/fx subdirectories"),
) -> None:
    report = _grade_saved_shadows(state_dir, data_dir, UtcClock().now())
    _save(state_dir / "agent_pm", "shadow_status.json", report)
    print(_format_shadow_status(report))
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
