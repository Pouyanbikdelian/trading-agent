"""Text helpers for operator-facing messages.

Why this module exists
----------------------
Telegram alerts were being cut mid-word by raw ``text[:N]`` slices
scattered across the formatters (sentinel assessment, sentinel suggested
action, the PM rationale in ``/pm``). A 2026-07-29 transcript shows the
damage: ``…across six consecut_``, ``…consider trimmin_``, and a sentinel
assessment that simply stopped at ``…modest put skew and a flat``. The
suggested action — the single most useful line in a risk alert — was the
worst affected, capped at 200 characters.

Two distinct bugs hide in a raw slice:

1. **Mid-word cuts.** The message reads like a dropped call and the
   operator cannot tell whether the thought was finished or the process
   died.
2. **Unbalanced Markdown.** Telegram's legacy Markdown parser treats
   ``*`` and ``_`` as paired delimiters. Slicing between a pair leaves an
   orphan marker, and the *whole message* is then either rejected with
   ``can't parse entities`` or rendered with the rest of the text
   swallowed into italics. That trailing ``_`` in ``consecut_`` is an
   italic closer colliding with a severed word.

``clip`` fixes both. Everything that shortens operator-facing text should
go through it rather than slicing.
"""

from __future__ import annotations

ELLIPSIS = "…"


def balance_markdown(text: str) -> str:
    """Drop unpaired Telegram Markdown delimiters from ``text``.

    Legacy Markdown (what ``notifier.send_message`` sends) pairs ``*`` for
    bold and ``_`` for italic. An odd count means a delimiter lost its
    partner — usually to truncation — and Telegram rejects the message
    rather than rendering it. Removing the last orphan is the honest fix:
    losing one asterisk beats losing the alert.

    Backtick-quoted spans are left alone: inside code the characters are
    literal, so they neither need nor have partners.
    """
    out = list(text)
    in_code = False
    # Track the positions of unmatched delimiters outside code spans.
    open_at: dict[str, list[int]] = {"*": [], "_": []}
    for i, ch in enumerate(out):
        if ch == "`":
            in_code = not in_code
            continue
        if in_code or ch not in open_at:
            continue
        if open_at[ch]:
            open_at[ch].pop()
        else:
            open_at[ch].append(i)
    orphans = sorted(
        [i for positions in open_at.values() for i in positions],
        reverse=True,
    )
    for i in orphans:
        del out[i]
    # An unclosed code span would swallow the tail the same way.
    if in_code:
        last = "".join(out).rfind("`")
        if last != -1:
            del out[last]
    return "".join(out)


def clip(text: object, limit: int) -> str:
    """Shorten ``text`` to ``limit`` chars without cutting mid-word.

    Trims back to the last space, appends an ellipsis, and repairs any
    Markdown delimiter left unpaired by the cut. Returns the input
    unchanged when it already fits, so the common case costs nothing.

    ``limit`` counts the characters kept from the source, not the
    ellipsis — a caller sizing a Telegram message against the 4096-char
    budget is off by one character, which never matters, whereas a
    caller reasoning about how much *content* survives is exact.
    """
    s = str(text)
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return balance_markdown(s)
    cut = s[:limit].rstrip()
    # Prefer a sentence end when one sits reasonably close to the limit:
    # a complete thought beats a fragment plus an ellipsis.
    for stop in (". ", "! ", "? "):
        idx = cut.rfind(stop)
        if idx >= limit * 0.6:
            return balance_markdown(cut[: idx + 1])
    head = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return balance_markdown((head or cut).rstrip(" ,;:-")) + ELLIPSIS
