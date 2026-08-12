"""Inline keyboards and callback data.

The safety-critical property under test is that a button cannot act on
something the operator did not look at. Telegram keeps every keyboard
alive in the chat history forever, so an approval prompt from three weeks
ago is one tap away at all times.
"""

from __future__ import annotations

import pytest

from trading.bot import keyboards


def _all_callback_data(markup: dict) -> list[str]:
    return [b["callback_data"] for row in markup["inline_keyboard"] for b in row]


def _all_labels(markup: dict) -> list[str]:
    return [b["text"] for row in markup["inline_keyboard"] for b in row]


class TestCallbackCodec:
    def test_round_trip(self) -> None:
        assert keyboards.decode(keyboards.encode("a", "abc123")) == ("a", "abc123")

    def test_round_trip_without_a_token(self) -> None:
        assert keyboards.decode(keyboards.encode("mx")) == ("mx", "")

    @pytest.mark.parametrize(
        "bad",
        ["", "garbage", "1|", "|a|b", "9|a|b", "1|a", "1|a|b|c"],
    )
    def test_malformed_or_wrong_version_is_rejected(self, bad: str) -> None:
        assert keyboards.decode(bad) is None

    def test_oversized_data_raises_rather_than_truncating(self) -> None:
        """A silently truncated token yields a button that always looks
        stale — fail at build time instead."""
        with pytest.raises(ValueError):
            keyboards.encode("a", "x" * 100)

    def test_every_shipped_button_fits_telegrams_64_byte_cap(self) -> None:
        markups = [
            keyboards.approval_keyboard("abcdef1234567890"),
            keyboards.sentinel_keyboard(),
            keyboards.mode_keyboard("neutral"),
            keyboards.desk_change_keyboard("dc-abcdef1234"),
            keyboards.read_only_keyboard("/positions", "/orders", "/health"),
        ]
        for m in markups:
            assert m is not None
            for data in _all_callback_data(m):
                assert len(data.encode("utf-8")) <= keyboards.MAX_CALLBACK_BYTES


class TestKeyboardsBindToTheirSubject:
    def test_approval_buttons_carry_the_cycle_id(self) -> None:
        data = _all_callback_data(keyboards.approval_keyboard("abcdef1234"))
        acting = [d for d in data if keyboards.decode(d)[0] != keyboards.ACT_RUN]
        assert acting, "expected state-changing buttons"
        for d in acting:
            _action, token = keyboards.decode(d)
            assert token.startswith("abcdef12")  # truncated to the displayed 8

    def test_mode_buttons_carry_the_mode(self) -> None:
        for d in _all_callback_data(keyboards.mode_keyboard("defense")):
            assert keyboards.decode(d)[1] == "defense"

    def test_desk_change_buttons_carry_the_full_proposal_id(self) -> None:
        proposal_id = "dc-abcdef1234"
        for d in _all_callback_data(keyboards.desk_change_keyboard(proposal_id)):
            action, token = keyboards.decode(d)
            assert action in {keyboards.ACT_DESK_APPROVE, keyboards.ACT_DESK_CANCEL}
            assert token == proposal_id

    def test_scale_button_encodes_its_percentage(self) -> None:
        data = _all_callback_data(keyboards.approval_keyboard("abc12345"))
        scaled = [d for d in data if keyboards.decode(d)[0] == keyboards.ACT_APPROVE_SCALED]
        assert scaled and scaled[0].endswith(":80")


class TestNoOrderPathHasAButton:
    """Rule #4 in keyboard form: a mis-tap must not move money."""

    FORBIDDEN = ("/buy", "/sell", "/close", "/flatten", "/halt", "/cancel_order")

    def test_allowlist_contains_no_order_commands(self) -> None:
        for cmd in self.FORBIDDEN:
            assert cmd not in keyboards.SAFE_COMMANDS

    def test_no_shipped_keyboard_offers_an_order_command(self) -> None:
        markups = [
            keyboards.approval_keyboard("abc12345"),
            keyboards.sentinel_keyboard(),
            keyboards.mode_keyboard("bear"),
            keyboards.desk_change_keyboard("dc-abcdef1234"),
        ]
        for m in markups:
            blob = " ".join(_all_callback_data(m)) + " " + " ".join(_all_labels(m))
            for cmd in self.FORBIDDEN:
                assert cmd not in blob

    def test_sentinel_offers_only_read_only_and_debate(self) -> None:
        """The alert says 'consider trimming' — but trimming stays typed."""
        for d in _all_callback_data(keyboards.sentinel_keyboard()):
            action, token = keyboards.decode(d)
            assert action == keyboards.ACT_RUN
            assert token in keyboards.SAFE_COMMANDS or token == "/committee"

    def test_read_only_keyboard_drops_commands_not_on_the_allowlist(self) -> None:
        m = keyboards.read_only_keyboard("/positions", "/flatten")
        assert m is not None
        assert len(_all_callback_data(m)) == 1
        assert "/flatten" not in " ".join(_all_callback_data(m))

    def test_read_only_keyboard_is_none_when_nothing_survives(self) -> None:
        assert keyboards.read_only_keyboard("/buy", "/sell") is None
