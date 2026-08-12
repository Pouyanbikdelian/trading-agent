"""The copilot must see the exact plan that is currently awaiting approval."""

from __future__ import annotations

import json
from pathlib import Path

from trading.copilot import facts
from trading.copilot.engine import answer


def _write_pending(state_dir: Path) -> None:
    (state_dir / "cycle_approval_pending.json").write_text(
        json.dumps(
            {
                "id": "20260812-abcdef",
                "phase": "initial",
                "selectable_symbols": ["AMD", "CNC"],
                "candidates": [{"symbol": "SNDK", "score": 5.39, "in_basket": False}],
                "plan": {
                    "base_currency": "CHF",
                    "order_count": 2,
                    "gross_turnover": 1_200.0,
                    "turnover_pct": 1.4,
                    "net_cash_change": -400.0,
                    "orders": [
                        {
                            "symbol": "AMD",
                            "side": "BUY",
                            "quantity": 1,
                            "notional_base": 400.0,
                            "context": "new position",
                        },
                        {
                            "symbol": "CNC",
                            "side": "SELL",
                            "quantity": 12,
                            "notional_base": 800.0,
                            "context": "closes 12 held",
                        },
                    ],
                },
            }
        )
    )


def test_active_cycle_plan_is_a_first_class_now_fact(tmp_path: Path) -> None:
    _write_pending(tmp_path)

    plan = facts.active_cycle_plan(tmp_path)

    assert plan["active"] is True and plan["readable"] is True
    assert plan["cycle_id"] == "20260812"
    assert plan["plan"]["orders"][0]["symbol"] == "AMD"
    assert plan["candidate_alternatives"][0]["symbol"] == "SNDK"


def test_plain_question_receives_the_active_plan_as_evidence(tmp_path: Path) -> None:
    _write_pending(tmp_path)
    captured: dict[str, object] = {}

    def fake_llm(_system: str, prompt: str) -> str:
        captured.update(json.loads(prompt))
        return "Buy 1 AMD; sell 12 CNC. No order has been submitted yet."

    out = answer(
        "What is this cycle proposing to buy and sell?",
        state_dir=tmp_path,
        data_dir=tmp_path / "data",
        llm=fake_llm,
    )

    assert "AMD" in out and "CNC" in out
    active = captured["NOW_active_cycle_proposal"]
    assert isinstance(active, dict) and active["active"] is True
    assert active["plan"]["orders"][1]["context"] == "closes 12 held"
