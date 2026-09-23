"""External dead-man ping: never raises, disabled without a URL."""

from __future__ import annotations

from types import SimpleNamespace

from trading.runtime.deadman import ping


class _Client:
    def __init__(self, status: int | None = 200, exc: Exception | None = None) -> None:
        self.status, self.exc, self.calls = status, exc, []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        if self.exc:
            raise self.exc
        return SimpleNamespace(status_code=self.status)


def test_disabled_without_url() -> None:
    client = _Client()
    assert ping(None, client=client) is False
    assert ping("", client=client) is False
    assert client.calls == []


def test_ok_on_2xx() -> None:
    client = _Client(200)
    assert ping("https://hc.example/abc", client=client) is True
    assert client.calls[0][0] == "https://hc.example/abc"


def test_http_error_and_exceptions_are_swallowed() -> None:
    assert ping("https://hc.example/abc", client=_Client(500)) is False
    assert ping("https://hc.example/abc", client=_Client(exc=OSError("down"))) is False
