"""The command table, and the typo matcher built on it.

The coverage test is the important one: it walks the real ``_dispatch``
source and fails if a command can be typed but has no registry entry.
Without it the registry silently rots and ``/postions`` goes back to
"unknown command".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trading.bot import registry


class TestSuggest:
    @pytest.mark.parametrize(
        ("typo", "expected"),
        [
            ("/postions", "/positions"),
            ("/positons", "/positions"),
            ("/aprove", "/approve"),
            ("/approv", "/approve"),
            ("/balance", "/balances"),
            ("/statuss", "/status"),
            ("/halr", "/halt"),
            ("/reprot", "/report"),
        ],
    )
    def test_near_misses_are_matched(self, typo: str, expected: str) -> None:
        assert expected in registry.suggest(typo)

    def test_prose_is_not_matched(self) -> None:
        # "/whats up with intc" must fall through to the copilot, not be
        # bent into some command that trades.
        assert registry.suggest("/whats") == []
        assert registry.suggest("/tellmeabouttheportfolio") == []

    def test_suggestions_are_canonical_not_aliases(self) -> None:
        # /corr is an alias of /correlation; the reply should name the
        # canonical command so /help stays consistent with it.
        for hit in registry.suggest("/correlatio"):
            assert registry.find(hit) is not None
            assert registry.find(hit).name == hit

    def test_suggestion_count_is_bounded(self) -> None:
        assert len(registry.suggest("/c", limit=2)) <= 2

    def test_missing_leading_slash_still_matches(self) -> None:
        assert "/positions" in registry.suggest("postions")


class TestSpecs:
    def test_lookup_by_alias(self) -> None:
        assert registry.find("/corr") is registry.find("/correlation")
        assert registry.find("/cycle_now") is registry.find("/cycle")

    def test_lookup_is_case_insensitive(self) -> None:
        assert registry.find("/POSITIONS") is registry.find("/positions")

    def test_unknown_lookup_is_none(self) -> None:
        assert registry.find("/nope") is None

    def test_usage_line_includes_an_example_when_it_has_one(self) -> None:
        line = registry.usage_for("/buy")
        assert "/buy SYM QTY" in line and "AAPL" in line

    def test_usage_for_unknown_points_at_help(self) -> None:
        assert "/help" in registry.usage_for("/nope")

    def test_every_command_taking_arguments_documents_them(self) -> None:
        for spec in registry.REGISTRY:
            if spec.usage and spec.usage != spec.name:
                assert spec.example, f"{spec.name} shows a usage but no example"

    def test_no_duplicate_names_or_aliases(self) -> None:
        seen: set[str] = set()
        for spec in registry.REGISTRY:
            for token in (spec.name, *spec.aliases):
                assert token not in seen, f"{token} registered twice"
                seen.add(token)

    def test_all_tokens_start_with_a_slash(self) -> None:
        assert all(t.startswith("/") for t in registry.all_names())


def _dispatched_commands() -> set[str]:
    """Every literal command string ``_dispatch`` compares against."""
    src = (
        Path(__file__).resolve().parents[2] / "src" / "trading" / "bot" / "telegram.py"
    ).read_text()
    body = src.split("async def _dispatch(")[1].split("\nasync def _cmd_unknown(")[0]
    found: set[str] = set()
    for match in re.finditer(r"cmd (?:==|in) (\(?)([^\n:]+)", body):
        found.update(re.findall(r'"(/[a-z_\-]+)"', match.group(2)))
    return found


class TestRegistryCoversDispatch:
    def test_every_dispatched_command_is_registered(self) -> None:
        dispatched = _dispatched_commands()
        assert len(dispatched) > 30, "parser found suspiciously few commands"
        missing = sorted(c for c in dispatched if registry.find(c) is None)
        assert not missing, f"dispatched but unregistered: {missing}"

    def test_every_registered_command_is_dispatched(self) -> None:
        dispatched = _dispatched_commands()
        missing = sorted(t for t in registry.all_names() if t not in dispatched)
        assert not missing, f"registered but unreachable: {missing}"


class TestStrictMode:
    """Trailing words mean a sentence, so the bar goes up."""

    def test_weak_match_is_dropped_when_words_follow(self) -> None:
        # "/hows the book" is a question; /holds is a 0.73 match.
        assert "/holds" in registry.suggest("/hows")
        assert registry.suggest("/hows", strict=True) == []

    def test_real_typos_survive_strict_mode(self) -> None:
        # "/aprove 80" is a typo that carries a legitimate argument.
        assert "/approve" in registry.suggest("/aprove", strict=True)
        assert "/positions" in registry.suggest("/postions", strict=True)
