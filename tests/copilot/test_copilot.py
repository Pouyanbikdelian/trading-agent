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

from trading.copilot.engine import answer, evidence_scope
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


def test_explicit_book_questions_select_the_correct_evidence_boundary() -> None:
    assert evidence_scope("Explain our live trading account holdings") == "live_account"
    assert evidence_scope("Why did we buy NVDA?") == "live_account"
    assert evidence_scope("Explain the PM simulated book") == "pm_book"
    assert (
        evidence_scope("Compare the PM simulated book with the live account") == "book_comparison"
    )
    assert evidence_scope("What is this cycle proposing?") == "active_proposal"
    assert evidence_scope("How big is the PM sleeve?") == "configuration"
    assert evidence_scope("Tell me more about our lessons") == "lessons"


def test_slash_why_prompt_shape_is_scoped_to_the_real_account() -> None:
    assert (
        evidence_scope("Why did we buy, sell, or hold NVDA? What was the thesis?") == "live_account"
    )


def test_live_account_explanation_excludes_the_pm_simulation(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    pm = state / "agent_pm"
    pm.mkdir()
    (pm / "portfolio.json").write_text(
        json.dumps({"cash": 1_000.0, "holdings": {"XLE": 3.0}, "history": []})
    )
    captured: dict[str, object] = {}

    def fake_llm(_system: str, prompt: str) -> str:
        captured.update(json.loads(prompt))
        return "The live account is separate from the PM book."

    answer(
        "Explain the live trading account and why did we buy NVDA?",
        state_dir=state,
        data_dir=tmp_path / "nodata",
        llm=fake_llm,
    )

    assert captured["answer_scope"] == "live_account"
    assert captured["authoritative_now_sources"] == [
        "NOW_real_trading_account",
        "NOW_trading_account_orders",
        "NOW_market",
    ]
    assert "NOW_real_trading_account" in captured
    assert "NOW_trading_account_orders" in captured
    assert "NOW_agent_pm_simulated_book" not in captured
    assert "NOW_configuration" not in captured


def test_pm_explanation_excludes_the_real_trading_account(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    pm = state / "agent_pm"
    pm.mkdir()
    (pm / "portfolio.json").write_text(
        json.dumps({"cash": 1_000.0, "holdings": {"XLE": 3.0}, "history": []})
    )
    captured: dict[str, object] = {}

    def fake_llm(_system: str, prompt: str) -> str:
        captured.update(json.loads(prompt))
        return "The PM simulation is separate."

    answer(
        "Explain the PM simulated book",
        state_dir=state,
        data_dir=tmp_path / "nodata",
        llm=fake_llm,
    )

    assert captured["answer_scope"] == "pm_book"
    assert captured["authoritative_now_sources"] == ["NOW_agent_pm_simulated_book"]
    assert "NOW_agent_pm_simulated_book" in captured
    assert "NOW_real_trading_account" not in captured
    assert "NOW_trading_account_orders" not in captured


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


# ------------------------------------------------- two-books arithmetic


def _pm_fixture(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    pm = state / "agent_pm"
    pm.mkdir(parents=True)
    (pm / "portfolio.json").write_text(
        json.dumps(
            {
                "cash": 40_000.0,
                # fractional SHARE quantities — these read like weights
                # and the LLM once presented them as such (2026-07-16)
                "holdings": {"JPM": 100.0, "GLD": 50.0},
                "start_equity": 100_000.0,
                "history": [{"t": "2026-07-13T20:00:00+00:00", "equity": 99_000.0}],
            }
        )
    )
    return state


def test_pm_book_precomputes_weights_and_deployed(tmp_path: Path, monkeypatch) -> None:
    """With prices available the book carries weight_pct / deployed_pct so
    the model never derives percentages from raw share counts."""
    from trading.copilot import facts

    state = _pm_fixture(tmp_path)
    px = {"JPM": 300.0, "GLD": 400.0}  # values: 30k + 20k, equity 90k

    monkeypatch.setattr(
        facts,
        "last_close",
        lambda data_dir, sym: {"available": True, "close": px[sym], "as_of": "2026-07-15"},
    )
    book = facts.pm_book(state, tmp_path / "data")
    assert book["available"]
    assert book["marked_equity_now"] == 90_000.0
    assert book["deployed_pct"] == round((1 - 40_000 / 90_000) * 100, 1)
    assert book["holdings"]["JPM"]["weight_pct"] == round(30_000 / 90_000 * 100, 1)
    assert book["holdings"]["JPM"]["shares"] == 100.0
    assert "NOT weights" in book["note"]
    # raw share dict must not leak alongside the marked one
    assert "holdings_share_quantities_NOT_weights" not in book


def test_pm_book_without_prices_labels_shares_loudly(tmp_path: Path) -> None:
    from trading.copilot import facts

    state = _pm_fixture(tmp_path)
    book = facts.pm_book(state)  # no data_dir → no marks
    assert book["available"]
    assert book["holdings_share_quantities_NOT_weights"] == {"JPM": 100.0, "GLD": 50.0}
    assert "deployed_pct" not in book  # never a made-up percentage


def test_charter_pins_two_books_and_no_arithmetic() -> None:
    """The prompt rules the 2026-07-16 transcript bugs regressed on."""
    from trading.copilot.engine import CHARTER

    assert "NO ARITHMETIC" in CHARTER
    assert "never write underscores" in CHARTER
    assert "share quantities, not weights" in CHARTER.lower()


def test_charter_bans_the_restatement_opener() -> None:
    """Observed 2026-07-30: every reply opened with 'I'm taking your
    message to mean', which reads as a machine echoing its input."""
    from trading.copilot.engine import CHARTER

    c = CHARTER.lower()
    assert "taking your message to mean" in c  # named as forbidden
    assert "never open with a formula" in c


def test_charter_handles_questions_about_its_own_capabilities() -> None:
    """'Can you consider my conviction on a stock?' asks about the desk's
    process, not for a fact — answering it with 'I cannot see the source
    code' is the failure this rule exists to prevent."""
    from trading.copilot.engine import CHARTER

    c = CHARTER.lower()
    assert "questions about you and the desk's process" in c
    assert "cannot see the source code" in c


def test_charter_requires_a_closing_so_what() -> None:
    from trading.copilot.engine import CHARTER

    assert "close with the so-what" in CHARTER.lower()


def test_copilot_defaults_to_opus_5_5_with_room_to_think(monkeypatch) -> None:
    """2026-09-26: every agent on Opus 5.5. It always thinks, and max_tokens
    caps thinking plus the reply — the old 900-token Haiku budget could be
    spent entirely on thinking and return an empty answer."""
    from trading.copilot import provider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("COPILOT_MODEL", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER", raising=False)
    cfg = provider.ProviderConfig.from_env()
    assert cfg.model == "claude-opus-5-5"
    body, timeout_s = provider._anthropic_body("sys", "q", cfg.model)
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] >= 4_000 and timeout_s >= 60
    assert "thinking" not in body and "temperature" not in body


def test_haiku_is_still_sent_without_an_effort_field() -> None:
    """Haiku 4.5 does not support effort; sending it would be a 400."""
    from trading.copilot import provider

    body, timeout_s = provider._anthropic_body("sys", "q", "claude-haiku-4-5-20251001")
    assert "output_config" not in body
    assert body["max_tokens"] == provider.MAX_TOKENS and timeout_s == provider.TIMEOUT_S


class TestEvidenceBudget:
    """2026-09-26: a raw 14k slice with NOW_* last — positions, risk and the
    PM book were cut first on any 'why' question, and the JSON was invalid."""

    @staticmethod
    def _payload() -> dict:
        return {
            "question": "why did we sell NVDA?",
            "CHAT_recent_turns": [{"role": "user", "text": "t" * 1_500} for _ in range(10)],
            "THEN_decisions_matching_question": [
                {"id": f"d{i}", "x": "d" * 3_000} for i in range(8)
            ],
            "THEN_transcript_hits": [{"id": f"t{i}", "x": "h" * 3_000} for i in range(8)],
            "NOW_positions": {"NVDA": 40},
            "NOW_risk_state": {"drawdown": -0.01},
        }

    def test_current_state_survives_and_the_json_is_valid(self) -> None:
        from trading.copilot.engine import _budgeted_evidence

        out = json.loads(_budgeted_evidence(self._payload(), budget=20_000))
        assert out["NOW_positions"] == {"NVDA": 40} and out["NOW_risk_state"]
        assert out["question"].startswith("why")
        assert any(o.startswith("THEN_transcript_hits") for o in out["_evidence_omissions"])

    def test_a_normal_question_fits_the_default_budget_untouched(self) -> None:
        from trading.copilot.engine import MAX_EVIDENCE_CHARS, _budgeted_evidence

        p = self._payload()
        p["THEN_decisions_matching_question"] = p["THEN_decisions_matching_question"][:4]
        p["THEN_transcript_hits"] = p["THEN_transcript_hits"][:4]
        out = json.loads(_budgeted_evidence(p))  # ~40k: the old 14k slice cut it
        assert "_evidence_omissions" not in out and MAX_EVIDENCE_CHARS >= 60_000


def test_a_cut_off_copilot_answer_says_so(monkeypatch) -> None:
    from trading.copilot import provider

    class R:
        status_code = 200

        def json(self):
            return {
                "content": [{"type": "text", "text": "NVDA was sold because"}],
                "stop_reason": "max_tokens",
            }

    monkeypatch.setattr(provider.httpx, "post", lambda *a, **k: R())
    cfg = provider.ProviderConfig(
        name="anthropic", model="claude-opus-5-5", base_url=None, api_key="k"
    )
    assert "answer cut off" in provider._anthropic("s", "q", cfg)
