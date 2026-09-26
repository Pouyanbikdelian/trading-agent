"""Telegram alerts — verify behavior without hitting the network."""

from __future__ import annotations

from trading.runner import NullAlerts, TelegramAlerts


def test_null_alerts_is_disabled() -> None:
    a = NullAlerts()
    assert a.enabled is False
    a.info("hi")
    a.critical("oh no")
    # All sends recorded in memory but never networked.
    assert [s[0] for s in a.sent] == ["info", "critical"]


def test_missing_credentials_disables() -> None:
    a = TelegramAlerts(token=None, chat_id="123", enabled=True)
    assert a.enabled is False
    b = TelegramAlerts(token="abc", chat_id=None, enabled=True)
    assert b.enabled is False


def test_explicit_disable_bypasses_creds() -> None:
    a = TelegramAlerts(token="abc", chat_id="123", enabled=False)
    assert a.enabled is False


def test_sent_log_in_order() -> None:
    a = NullAlerts()
    a.info("first")
    a.warning("second")
    a.error("third")
    a.critical("fourth")
    levels = [s[0] for s in a.sent]
    assert levels == ["info", "warning", "error", "critical"]


def test_critical_prefix_added() -> None:
    a = NullAlerts()
    a.critical("disk full")
    msg = a.sent[-1][1]
    assert msg.startswith("🚨 ")


def test_send_swallows_network_errors(monkeypatch) -> None:
    """An enabled alerter must not raise when the network is unreachable."""
    a = TelegramAlerts(token="t", chat_id="c", enabled=True)

    def boom(*args, **kwargs):
        raise OSError("no route to host")

    import urllib.request as _ur

    monkeypatch.setattr(_ur, "urlopen", boom)
    # Should NOT raise — only logs.
    a.error("hello")
    assert ("error", "hello") in a.sent


def _capture(monkeypatch, *, reject_markdown: bool = False):
    import io
    import urllib.error
    import urllib.parse
    import urllib.request as _ur

    posts: list[dict[str, str]] = []

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        fields = dict(urllib.parse.parse_qsl(req.data.decode()))
        posts.append(fields)
        if reject_markdown and "parse_mode" in fields:
            raise urllib.error.HTTPError(req.full_url, 400, "can't parse entities", {}, None)
        return _Resp(b"{}")

    monkeypatch.setattr(_ur, "urlopen", fake)
    return posts


def test_alerts_render_markdown(monkeypatch) -> None:
    """2026-09-26: plain text showed every *bold* and `code` literally."""
    posts = _capture(monkeypatch)
    TelegramAlerts(token="t", chat_id="c").info("*Desk online*")
    assert posts[0]["parse_mode"] == "Markdown"


def test_a_formatting_rejection_is_resent_as_plain_text(monkeypatch) -> None:
    posts = _capture(monkeypatch, reject_markdown=True)
    TelegramAlerts(token="t", chat_id="c").info("AGENT_PM_SLEEVE_PCT is 1.0")
    assert len(posts) == 2 and "parse_mode" not in posts[1]
    assert posts[1]["text"] == "AGENT_PM_SLEEVE_PCT is 1.0"


def test_a_long_message_is_split_not_lost(monkeypatch) -> None:
    """Over 4,096 chars Telegram rejects the whole message."""
    posts = _capture(monkeypatch)
    body = "\n".join(f"line {i} " + "x" * 80 for i in range(120))  # ~10k chars
    TelegramAlerts(token="t", chat_id="c").warning(body, buttons={"inline_keyboard": []})
    assert len(posts) >= 3 and all(len(p["text"]) <= 3_800 for p in posts)
    assert "reply_markup" in posts[-1] and "reply_markup" not in posts[0]
