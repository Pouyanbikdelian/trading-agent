"""Releasing the IBKR session from a phone.

IBC runs with ``ExistingSessionDetectedAction=primary``, so the gateway
container holds the account's ONE session and the operator cannot log
into TWS or the mobile app while it is up. Taking manual control meant
an SSH client on the phone — at exactly the moment someone wants to
close a position by hand.

``trader-live`` carried ``group_add: DOCKER_GID`` — permission to use
the docker socket — and never mounted the socket. So this, and the
gateway auto-restart that recovers a wedged API session, were both dead
on LIVE while working on paper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trading.runtime.docker_ctl import (
    ACTIONS,
    ALLOWED_CONTAINERS,
    GATEWAY_CONTAINER,
    docker_action,
)

COMPOSE = Path("docker-compose.yml").read_text()
PROCESSOR = Path("src/trading/runtime/command_processor.py").read_text()
BOT = Path("src/trading/bot/telegram.py").read_text()


class TestTheBlastRadius:
    def test_only_the_gateway_may_be_controlled(self) -> None:
        """An operator who can name any container from Telegram is one
        typo from stopping the VPS."""
        assert frozenset({GATEWAY_CONTAINER}) == ALLOWED_CONTAINERS

    @pytest.mark.parametrize("bad", ["trader-live", "trader-bot", "scheduler", "../../etc"])
    def test_another_container_is_refused(self, bad) -> None:
        with pytest.raises(ValueError, match="not a container"):
            docker_action(bad, "stop")

    def test_an_unknown_action_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            docker_action(GATEWAY_CONTAINER, "kill")

    def test_the_action_list_is_deliberately_small(self) -> None:
        assert set(ACTIONS) == {"stop", "start", "restart"}


class TestFailuresAreLoud:
    def test_an_unreachable_socket_raises(self, tmp_path) -> None:
        """Silence would leave the operator believing the session was
        freed; they would learn otherwise from a rejected login."""
        with pytest.raises(RuntimeError, match="docker socket unreachable"):
            docker_action(GATEWAY_CONTAINER, "stop", socket_path=str(tmp_path / "nope.sock"))

    def test_state_never_raises(self, tmp_path) -> None:
        from trading.runtime.docker_ctl import container_state

        assert container_state(socket_path=str(tmp_path / "nope.sock")).startswith("unknown")


class TestStoppingHaltsFirst:
    def test_the_handler_halts_before_releasing_the_session(self) -> None:
        """Order matters: the runner must not spend the manual session
        firing orders at a dead socket."""
        body = PROCESSOR[
            PROCESSOR.index("def _h_gateway_stop") : PROCESSOR.index("def _h_gateway_start")
        ]

        assert body.index("set_halted(") < body.index('docker_action(GATEWAY_CONTAINER, "stop")')

    def test_starting_does_not_resume_trading(self) -> None:
        """The operator has just traded by hand — the desk's view of the
        book is stale until the next snapshot. Resuming is deliberate."""
        body = PROCESSOR[PROCESSOR.index("def _h_gateway_start") :][:900]

        assert "set_halted" not in body
        assert "/resume" in body

    def test_both_handlers_are_registered(self) -> None:
        assert "CommandType.GATEWAY_STOP: _h_gateway_stop," in PROCESSOR
        assert "CommandType.GATEWAY_START: _h_gateway_start," in PROCESSOR

    def test_they_are_not_order_submitting(self) -> None:
        """Outside the execution lock on purpose: the point is to act
        when the desk cannot."""
        from trading.runtime.command_processor import _ORDER_SUBMITTING_COMMANDS
        from trading.runtime.commands import CommandType

        assert CommandType.GATEWAY_STOP not in _ORDER_SUBMITTING_COMMANDS
        assert CommandType.GATEWAY_START not in _ORDER_SUBMITTING_COMMANDS


class TestTheBotSide:
    def test_the_command_is_dispatched(self) -> None:
        assert 'if cmd == "/gateway":' in BOT
        assert "return _cmd_gateway(args)" in BOT

    def test_it_is_in_the_registry(self) -> None:
        reg = Path("src/trading/bot/registry.py").read_text()

        assert '"/gateway"' in reg
        assert "/gateway stop|start|status" in reg

    def test_a_bare_call_explains_rather_than_acting(self) -> None:
        """`/gateway` alone must not stop anything."""
        from trading.bot.telegram import _cmd_gateway

        out = _cmd_gateway([])

        assert "usage" in out.lower()
        assert "frees your IBKR session" in out

    def test_an_unknown_subcommand_is_refused(self) -> None:
        from trading.bot.telegram import _cmd_gateway

        assert "usage" in _cmd_gateway(["restart"]).lower()


class TestTheComposeMount:
    """Parsed, not string-matched: the whole defect was a line present in
    one service block and absent from another, which a substring search
    over the whole file cannot tell apart."""

    @staticmethod
    def _volumes(service: str) -> list[str]:
        import yaml

        spec = yaml.safe_load(COMPOSE)
        return [str(v) for v in (spec["services"][service].get("volumes") or [])]

    def test_trader_live_now_has_the_docker_socket(self) -> None:
        assert "/var/run/docker.sock:/var/run/docker.sock" in self._volumes("trader-live")

    def test_the_paper_trader_still_has_it(self) -> None:
        assert "/var/run/docker.sock:/var/run/docker.sock" in self._volumes("trader")

    def test_group_add_without_the_socket_is_the_bug(self) -> None:
        """Either both or neither — group_add alone is what shipped, and
        it made the gateway auto-restart dead code on live."""
        import yaml

        spec = yaml.safe_load(COMPOSE)
        for name in ("trader", "trader-live"):
            svc = spec["services"][name]
            has_sock = any("docker.sock" in str(v) for v in (svc.get("volumes") or []))
            has_group = bool(svc.get("group_add"))
            assert has_sock == has_group, f"{name}: socket={has_sock} group_add={has_group}"


class TestTheBotActsDirectly:
    """The runner cannot be the executor of a command whose whole purpose
    is to work when the runner is down.

    2026-08-11: /gateway stop worked (runner alive), the gateway went
    away, and cli.py connects the broker BEFORE starting the scheduler —
    so the runner crash-looped, the command processor never ran, and two
    /gateway stop commands sat in the queue waiting for a process that
    could not come back. /gateway start was unreachable in the exact
    situation it exists for.
    """

    def test_the_bot_does_not_queue_gateway_commands(self) -> None:
        body = BOT[BOT.index("def _cmd_gateway") :][:3000]

        assert '_queue_command("gateway_stop"' not in body
        assert '_queue_command("gateway_start"' not in body

    def test_the_bot_calls_docker_directly(self) -> None:
        body = BOT[BOT.index("def _cmd_gateway") :][:3000]

        assert 'docker_action(GATEWAY_CONTAINER, "stop")' in body
        assert 'docker_action(GATEWAY_CONTAINER, "start")' in body

    def test_stopping_still_halts_first(self) -> None:
        body = BOT[BOT.index("def _cmd_gateway") :][:3000]

        assert body.index("set_halted(") < body.index('docker_action(GATEWAY_CONTAINER, "stop")')

    def test_a_docker_failure_is_reported_with_the_manual_fallback(self) -> None:
        """If the socket is not mounted the operator needs the SSH line,
        not a traceback."""
        body = BOT[BOT.index("def _cmd_gateway") :][:3000]

        assert "could not" in body
        assert "docker compose up -d ib-gateway" in body

    def test_the_bot_has_the_docker_socket(self) -> None:
        import yaml

        svc = yaml.safe_load(COMPOSE)["services"]["bot"]

        assert any("docker.sock" in str(v) for v in (svc.get("volumes") or []))
        assert svc.get("group_add")
