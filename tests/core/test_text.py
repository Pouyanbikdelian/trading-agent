"""Clipping rules for operator-facing text.

The regressions pinned here come from a real 2026-07-29 Telegram
transcript: ``…across six consecut_`` and ``…consider trimmin_``. Both
were raw ``text[:N]`` slices — a severed word plus, in those cases, a
Markdown delimiter orphaned by the cut.
"""

from __future__ import annotations

from trading.core.text import balance_markdown, clip


class TestClipBoundaries:
    def test_short_text_is_untouched(self) -> None:
        assert clip("all good", 100) == "all good"

    def test_exact_limit_is_untouched(self) -> None:
        assert clip("abcde", 5) == "abcde"

    def test_never_cuts_mid_word(self) -> None:
        # The literal shape of the bug: "consider trimming" must not
        # become "consider trimmin".
        text = "treat as correlated tech selling and consider trimming the position"
        out = clip(text, 50)
        assert "trimmin…" not in out
        for word in out.rstrip("…").split():
            assert word in text.split()

    def test_appends_ellipsis_when_truncated(self) -> None:
        assert clip("one two three four five", 12).endswith("…")

    def test_prefers_a_sentence_end_near_the_limit(self) -> None:
        text = "The book is flat and quiet. Semis look extended and the tape is thin."
        out = clip(text, 40)
        # A complete thought beats a fragment plus an ellipsis.
        assert out == "The book is flat and quiet."

    def test_ignores_a_sentence_end_that_wastes_the_budget(self) -> None:
        # A period at 40% of the limit would throw away more than half the
        # room available — better to fill it and mark the cut.
        text = "Flat. Semis look extended and the tape is thin into the close."
        out = clip(text, 40)
        assert out != "Flat."
        assert out.endswith("…")

    def test_ignores_a_sentence_end_that_is_far_too_early(self) -> None:
        text = "Hi. " + "word " * 60
        out = clip(text, 100)
        assert out != "Hi."
        assert out.endswith("…")

    def test_handles_a_single_unbroken_token(self) -> None:
        out = clip("x" * 200, 20)
        assert len(out) <= 21
        assert out.endswith("…")

    def test_zero_and_negative_limits_are_empty(self) -> None:
        assert clip("anything", 0) == ""
        assert clip("anything", -5) == ""

    def test_accepts_non_string_input(self) -> None:
        assert clip(1234, 10) == "1234"
        assert clip(None, 10) == "None"

    def test_trailing_punctuation_is_trimmed_before_the_ellipsis(self) -> None:
        assert "  " not in clip("alpha, beta, gamma delta", 13)


class TestMarkdownBalance:
    def test_orphan_italic_marker_is_dropped(self) -> None:
        # The transcript's "consecut_": an italic closer left without its
        # opener once the text ahead of it was cut away.
        assert balance_markdown("across six consecut_") == "across six consecut"

    def test_orphan_bold_marker_is_dropped(self) -> None:
        assert balance_markdown("*Suggested: trim") == "Suggested: trim"

    def test_balanced_markers_survive(self) -> None:
        text = "*Tripped:* SPY -1.6% _held_"
        assert balance_markdown(text) == text

    def test_clip_does_not_leave_an_unpaired_marker(self) -> None:
        # Cutting inside an italic span must not strip the rest of the
        # message into italics — or make Telegram reject it outright.
        out = clip("_Suggested: watch INTC into the close for volume_", 30)
        assert out.count("_") % 2 == 0

    def test_markers_inside_code_spans_are_literal(self) -> None:
        text = "run `a_b_c` now"
        assert balance_markdown(text) == text

    def test_unclosed_code_span_is_repaired(self) -> None:
        assert balance_markdown("see `orders").count("`") == 0
