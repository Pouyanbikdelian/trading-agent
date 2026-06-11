"""Dashboard — summary builder + auth, hermetic."""

from __future__ import annotations

import base64
import json
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from trading.dashboard.app import _Handler, build_summary
from trading.memory import MemoryStore
from trading.runner.state import RunnerStore


@pytest.fixture
def populated_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    store = RunnerStore(state / "runner.db")
    from trading.core.types import AccountSnapshot

    for i, eq in enumerate([100_000.0, 101_200.0, 100_700.0]):
        store.save_snapshot(
            AccountSnapshot(ts=datetime(2026, 6, 9 + i, tzinfo=timezone.utc), cash=eq, equity=eq)
        )
    mem = MemoryStore(state / "memory")
    mem.journal("cycle", {"status": "ok"})
    mem.add_lesson("test lesson")
    (state / "last_committee.json").write_text(json.dumps({"ok": False, "reason": "none yet"}))
    return state, tmp_path / "data"


def test_build_summary_sections(populated_state) -> None:
    state, data = populated_state
    out = build_summary(state, data)
    assert len(out["equity_curve"]) == 3
    assert out["equity_curve"][-1]["v"] == pytest.approx(100_700.0)
    assert out["memory"]["stats"]["lessons"] == 1
    assert out["committee"]["ok"] is False
    assert "context" in out and "generated_at" in out


def test_build_summary_degrades_on_empty_dirs(tmp_path) -> None:
    out = build_summary(tmp_path / "nope", tmp_path / "nada")
    assert out["equity_curve"] == []
    # memory store auto-creates; stats present but empty
    assert out["memory"]["stats"]["journal"] == 0


def test_http_auth_and_endpoints(populated_state) -> None:
    state, data = populated_state
    _Handler.state_dir = state
    _Handler.data_dir = data
    _Handler.auth_token = base64.b64encode(b"yan:secret").decode()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # No auth -> 401
        req = urllib.request.Request(f"http://127.0.0.1:{port}/")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req)
        assert ei.value.code == 401

        # With auth -> page and JSON
        hdr = {"Authorization": "Basic " + base64.b64encode(b"yan:secret").decode()}
        page = urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/", headers=hdr)
        ).read()
        assert b"Trading Agent" in page
        api = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{port}/api/summary", headers=hdr)
            ).read()
        )
        assert len(api["equity_curve"]) == 3
    finally:
        server.shutdown()
