"""IB Gateway container settings that decide whether it can boot unattended.

IBC drives a real GUI. Several of its defaults assume a human is sitting in
front of it, and on a headless VPS those defaults mean "block forever":
the container stays `up`, the API port is never opened, and the runner
crash-loops on connection refused while `docker compose ps` looks fine.

Every assertion here corresponds to a way that has actually happened or
would have:

* ``EXISTING_SESSION_DETECTED_ACTION`` defaults to ``manual``. Hit on
  2026-08-07 just by recreating the gateway twice in a row, before IBKR
  had released the previous session.
* ``TWOFA_TIMEOUT_ACTION`` defaults to exiting, which turns a missed 2FA
  tap into a dead container instead of a retry.
* the healthcheck port must follow the trading mode, or autoheal
  restart-loops the gateway on live day.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def gateway() -> dict:
    return yaml.safe_load(COMPOSE.read_text())["services"]["ib-gateway"]


class TestUnattendedBoot:
    def test_existing_session_is_taken_over_not_left_to_a_human(self, gateway: dict) -> None:
        """IBC's default is `manual`: it raises a modal and waits. Nobody
        is there to click it, so the gateway never opens its API port."""
        val = gateway["environment"]["EXISTING_SESSION_DETECTED_ACTION"]
        assert "primary" in val, (
            f"gateway would block on the 'Existing session detected' modal; got {val!r}"
        )

    def test_a_missed_2fa_prompt_retries_rather_than_dying(self, gateway: dict) -> None:
        assert gateway["environment"]["TWOFA_TIMEOUT_ACTION"] == "restart"

    def test_the_startup_warning_is_bypassed(self, gateway: dict) -> None:
        assert gateway["environment"]["BYPASS_WARNING"] == "yes"

    def test_the_api_is_not_read_only(self, gateway: dict) -> None:
        """READ_ONLY_API=yes accepts connections and silently refuses every
        order — a failure that looks like a strategy deciding to do
        nothing."""
        assert gateway["environment"]["READ_ONLY_API"] == "no"


class TestModeFollowsConfiguration:
    def test_credentials_and_mode_come_from_env(self, gateway: dict) -> None:
        env = gateway["environment"]
        assert env["TWS_USERID"] == "${IBKR_USERNAME}"
        assert env["TWS_PASSWORD"] == "${IBKR_PASSWORD}"
        assert "IBKR_TRADING_MODE" in env["TRADING_MODE"]

    def test_healthcheck_port_follows_the_trading_mode(self, gateway: dict) -> None:
        """Hardcoded 4002 meant a gateway in live mode reported unhealthy
        forever — and autoheal watches this container, so it would have
        been restarted in a loop on live day."""
        probe = " ".join(gateway["healthcheck"]["test"])
        assert "IBKR_PORT" in probe, f"healthcheck pins a port: {probe}"

    def test_autoheal_watches_the_gateway(self, gateway: dict) -> None:
        assert "autoheal=true" in gateway["labels"]

    def test_start_period_covers_the_ibc_login_dance(self, gateway: dict) -> None:
        """Login + auth + config dialog takes 60-120s. A short start_period
        marks it unhealthy mid-boot and autoheal restarts it into a loop
        it can never finish."""
        start = gateway["healthcheck"]["start_period"]
        assert start.endswith("s")
        assert int(start.rstrip("s")) >= 120


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load(COMPOSE.read_text())["services"]


class TestTheArmingFlagIsOperable:
    """`trader-live` hardcoded ``ALLOW_LIVE_TRADING: "true"``.

    A compose ``environment:`` value overrides the ``env_file:`` above it,
    so on 2026-08-07 the operator set ``ALLOW_LIVE_TRADING=false`` in
    ``.env``, recreated the container, and the service stayed armed —
    while ``grep ALLOW_LIVE_TRADING .env`` reported false. They believed
    they had two layers between them and the market (disarmed + halted)
    and had one.

    A gate you cannot read from the file you edit is not a gate.
    """

    def test_arming_is_governed_by_the_env_file(self, services: dict) -> None:
        value = str(services["trader-live"]["environment"]["ALLOW_LIVE_TRADING"])
        assert "${ALLOW_LIVE_TRADING" in value, (
            "hardcoding this makes .env a lie — interpolate it instead"
        )

    def test_the_default_is_disarmed(self, services: dict) -> None:
        """A missing value must mean 'not armed', never 'armed'."""
        value = str(services["trader-live"]["environment"]["ALLOW_LIVE_TRADING"])
        assert ":-false}" in value

    def test_the_live_service_still_reads_the_env_file(self, services: dict) -> None:
        assert ".env" in services["trader-live"]["env_file"]

    def test_no_service_hardcodes_arming_on(self, services: dict) -> None:
        """Belt and braces across every service, present and future."""
        for name, svc in services.items():
            raw = str((svc.get("environment") or {}).get("ALLOW_LIVE_TRADING", ""))
            assert raw.lower() not in ("true", "1", "yes"), f"{name} hardcodes arming ON"
