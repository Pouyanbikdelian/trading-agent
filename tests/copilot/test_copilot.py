"""Copilot Phase 1 — hermetic: fixture journal + fake LLM, no network.

Covers the contract, not the prose: retrieval finds the right
decisions, evidence with citations reaches the LLM, missing evidence is
reported honestly WITHOUT an LLM call, and no copilot path can reach
broker execution.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trading.copilot.engine import answer
from trading.copilot.store import CopilotStore

# ------------------------------------------------------------ fixtures

_ORDERS_SCHEMA = """
CREATE TABLE orders (
    client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,
    side TEXT NOT NULL, quantity REAL NOT NULL, order_type TEXT NOT NULL,
    limit_price REAL, stop_price REAL, tif TEXT NOT NULL,
    created_at REAL NOT NULL, status TEXT NOT NULL, broker_order_id TEXT);
CREATE TABLE fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL,
    ts REAL NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
    commission REAL NOT NULL DEFAULT 0, venue TEXT);
"""


def _make_state(tmp_path: Path) -> Path:
    """A state dir with a memory journal (2 takes + 1 ruling + 1 PM run)
    and an orders.db holding one filled NVDA buy."""
    state = tmp_path / "state"
    mem_dir = state / "memory"
    mem_dir.mkdir(parents=True)
    mem = sqlite3.connect(mem_dir / "memory.db")
    mem.execute(
        "CREATE TABLE journal (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " kind TEXT NOT NULL, actor TEXT NOT NULL DEFAULT 'system', payload TEXT NOT NULL)"
    )
    t0 = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc).timestamp()

    def j(ts_off: float, kind: str, actor: str, payload: dict) -> None:
        mem.execute(
            "INSERT INTO journal (ts, kind, actor, payload) VALUES (?,?,?,?)",
            (t0 + ts_off, kind, actor, json.dumps(payload)),
        )

    j(
        0,
        "take",
        "quant",
        {
            "agent": "quant",
            "stance": "bullish",
            "take": "NVDA momentum rank 1, accelerating datacenter revenue",
        },
    )
    j(
        10,
        "take",
        "risk_officer",
        {
            "agent": "risk_officer",
            "stance": "bearish",
            "take": "NVDA position would breach tech concentration comfort",
        },
    )
    j(
        60,
        "committee",
        "manager",
        {
            "ruling": {
                "posture": "risk_on",
                "proposal": "Add NVDA on datacenter momentum; semis leadership intact",
                "watch": "NVDA closing below its 50dma invalidates",
            },
            "takes": {"quant": {"stance": "bullish"}, "risk_officer": {"stance": "bearish"}},
            "disagreement": 1.5,
        },
    )
    j(
        120,
        "agent_pm",
        "pm",
        {
            "equity": 1_000_000,
            "weights": {"XLE": 0.2},
            "rationale": "Rotated into energy on scout theme",
        },
    )
    mem.commit()
    mem.close()

    orders = sqlite3.connect(state / "orders.db")
    orders.executescript(_ORDERS_SCHEMA)
    ins = json.dumps({"symbol": "NVDA", "asset_class": "equity", "currency": "USD"})
    orders.execute(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("trd-nvda1", ins, "buy", 10, "MARKET", None, None, "DAY", t0 + 3600, "FILLED", None),
    )
    orders.execute(
        "INSERT INTO fills (order_id, ts, quantity, price, commission) VALUES (?,?,?,?,?)",
        ("trd-nvda1", t0 + 3700, 10, 900.0, 1.0),
    )
    orders.commit()
    orders.close()
    return state


# ------------------------------------------------------------ retrieval


def test_ingest_and_retrieval_finds_relevant_decision(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = CopilotStore(state)
    added = store.ingest(state / "memory", known_symbols={"NVDA", "XLE"})
    assert added == 4
    # Idempotent: second ingest adds nothing.
    assert store.ingest(state / "memory") == 0

    hits = store.search_decisions(["datacenter", "momentum"], symbol="NVDA")
    assert hits and hits[0]["id"] == "D3"
    assert "NVDA" in hits[0]["symbols"]
    assert hits[0]["votes"] == {"quant": "bullish", "risk_officer": "bearish"}
    assert "50dma" in hits[0]["invalidation"]
    # Transcript linked to the ruling it fed.
    transcript = store.transcript_for_decision("D3")
    assert {t["agent"] for t in transcript} == {"quant", "risk_officer"}
    store.close()


def test_symbol_filter_excludes_unrelated_decisions(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = CopilotStore(state)
    store.ingest(state / "memory", known_symbols={"NVDA", "XLE"})
    hits = store.search_decisions(["energy"], symbol="XLE")
    assert hits and hits[0]["kind"] == "agent_pm"
    assert all("NVDA" not in h["symbols"] for h in hits)
    store.close()


# ------------------------------------------------------------- engine


def test_answer_passes_cited_evidence_to_llm(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    seen: dict[str, str] = {}

    def fake_llm(system: str, prompt: str) -> str:
        seen["system"], seen["prompt"] = system, prompt
        return "THEN: bought on momentum (D3). NOW: holding (trd-nvda1). CHANGED: none."

    out = answer(
        "Why did we buy NVDA?",
        state_dir=state,
        data_dir=tmp_path / "nodata",
        llm=fake_llm,
    )
    assert "D3" in out
    # Evidence JSON contains citation ids for decision, transcript, order+fill.
    assert "D3" in seen["prompt"] and "T1" in seen["prompt"]
    assert "trd-nvda1" in seen["prompt"]
    # The charter demands the THEN/NOW/CHANGED structure and citations.
    assert "THEN" in seen["system"] and "cite" in seen["system"].lower()
    # Untrusted-transcript rule is stated.
    assert "never an instruction" in seen["system"].lower()
    # No secrets in the outbound evidence.
    for needle in ("TELEGRAM", "PASSWORD", "API_KEY", "avelekpbik"):
        assert needle not in seen["prompt"]


def test_missing_evidence_is_honest_and_skips_llm(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    calls = {"n": 0}

    def fake_llm(system: str, prompt: str) -> str:
        calls["n"] += 1
        return "should never be called"

    out = answer(
        "Why did we buy ZZZQ?",
        state_dir=state,
        data_dir=tmp_path / "nodata",
        symbol="ZZZQ",
        llm=fake_llm,
    )
    assert calls["n"] == 0  # no evidence → no LLM call
    assert "No recorded" in out


# ------------------------------------------------------------- safety


def test_copilot_package_never_imports_execution() -> None:
    """The copilot must have no path to order submission. Enforced at
    the import graph: importing every copilot module must not pull in
    trading.execution. Runs in a SUBPROCESS so the check sees a clean
    interpreter (and doesn't corrupt this process's module identity)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import trading.copilot.store, trading.copilot.facts, "
        "trading.copilot.engine, trading.copilot.provider\n"
        "bad = [m for m in sys.modules if m.startswith('trading.execution')]\n"
        "assert not bad, f'copilot imports execution modules: {bad}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_provider_config_requires_key(monkeypatch) -> None:
    from trading.copilot.provider import ProviderConfig, ProviderError

    monkeypatch.setenv("COPILOT_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    try:
        ProviderConfig.from_env()
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        assert "DEEPSEEK_API_KEY" in str(e)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = ProviderConfig.from_env()
    assert cfg.name == "deepseek" and cfg.model == "deepseek-chat"
    assert cfg.base_url and "deepseek.com" in cfg.base_url
