"""Everything that touches state must follow STATE_DIR.

Live moves ``STATE_DIR`` to ``/app/state/live`` so the kill switch gets a
clean equity baseline. Anything still pointing at the hardcoded
``/app/state`` silently stops applying at exactly the moment it starts
mattering — and every instance of it is invisible:

* the **healthcheck** baked into the image reads
  ``/app/state/heartbeat.json``. With STATE_DIR moved, the runner writes
  its heartbeat somewhere the check never looks, so the container reports
  unhealthy forever regardless of how well it is running. The one signal
  that says "cycles have stopped" becomes a permanent red light.
* the **ofelia hygiene jobs** guard with ``[ -f "$db" ]``, so on live they
  find nothing and exit 0. Disk grows, nothing complains.

These are compose-level assertions rather than unit tests because the
defect lives in the YAML, which nothing else covers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"

#: Services that run the trading image and therefore write a heartbeat.
RUNNER_SERVICES = ("trader", "trader-live")


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text()


@pytest.fixture(scope="module")
def compose_doc() -> dict:
    # Compose interpolation (${VAR:-default}) is not YAML-significant, so
    # a plain safe_load is enough to inspect structure.
    return yaml.safe_load(COMPOSE.read_text())


class TestHeartbeatHealthcheck:
    @pytest.mark.parametrize("service", RUNNER_SERVICES)
    def test_service_overrides_the_baked_in_healthcheck(
        self, compose_doc: dict, service: str
    ) -> None:
        svc = compose_doc["services"][service]
        assert "healthcheck" in svc, (
            f"{service} inherits the Dockerfile healthcheck, which hardcodes "
            "/app/state/heartbeat.json and breaks as soon as STATE_DIR moves"
        )

    @pytest.mark.parametrize("service", RUNNER_SERVICES)
    def test_the_healthcheck_path_follows_state_dir(
        self, compose_doc: dict, service: str
    ) -> None:
        test = " ".join(compose_doc["services"][service]["healthcheck"]["test"])
        assert "STATE_DIR" in test, f"{service} healthcheck does not follow STATE_DIR: {test}"
        assert "heartbeat.json" in test

    @pytest.mark.parametrize("service", RUNNER_SERVICES)
    def test_it_still_defaults_to_the_paper_path(self, compose_doc: dict, service: str) -> None:
        """An unset STATE_DIR must keep working — the default deployment
        does not set one, and a healthcheck resolving to
        '/heartbeat.json' would be worse than the bug it replaces."""
        test = " ".join(compose_doc["services"][service]["healthcheck"]["test"])
        assert "${STATE_DIR:-/app/state}" in test


class TestOfeliaHygieneJobs:
    """These guard with `[ -f "$db" ]`, so a wrong path is a silent no-op."""

    def test_no_ofelia_command_hardcodes_the_paper_state_dir(self, compose_text: str) -> None:
        # The `/app/state` inside `${STATE_DIR:-/app/state}` is the fallback,
        # not a hardcode — drop the interpolations before looking for bare
        # paths, or the fix looks identical to the bug.
        interpolation = re.compile(r"\$\{STATE_DIR(:-[^}]*)?\}")
        offenders = [
            line.strip()
            for line in compose_text.splitlines()
            if "ofelia.job-exec" in line and "/app/state" in interpolation.sub("", line)
        ]
        assert not offenders, (
            "ofelia job(s) hardcode /app/state and become no-ops on live:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_vacuum_job_targets_both_databases(self, compose_text: str) -> None:
        line = next(ln for ln in compose_text.splitlines() if "sqlite-vacuum.command" in ln)
        assert "runner.db" in line and "orders.db" in line


class TestMemorySpineLivesUnderStateDir:
    """The reason Gate 3 of the go-live runbook exists: moving STATE_DIR
    without copying memory/ silently resets lessons, predictions, source
    trust and the shadow ledger to zero."""

    def test_the_production_store_is_rooted_at_state_dir(self) -> None:
        import inspect

        from trading.memory import store

        src = inspect.getsource(store)
        assert 'state_dir) / "memory"' in src.replace("'", '"'), (
            "memory store no longer resolves under settings.state_dir — the "
            "go-live runbook's Gate 3 copy step is now wrong"
        )
