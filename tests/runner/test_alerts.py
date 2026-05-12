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
