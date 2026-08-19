"""Bot boundary tests for a halted, non-executable account review."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import trading.bot.telegram as telegram_module
from trading.bot.telegram import _cmd_approve, _cmd_cycle_now, _cmd_review


def _settings(state_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(state_dir=state_dir, require_cycle_approval=True)


def _halt(state_dir: Path, reason: str = "daily loss limit") -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "halt.json").write_text(json.dumps({"halted": True, "reason": reason}))


def test_cycle_while_halted_asks_the_runner_and_opens_no_approval(
    tmp_path: Path, monkeypatch
) -> None:
    """The bot never decides what a halt means; only the runner sees the book.

    ``/cycle`` therefore always requests the execute-capable path. The
    runner strips the plan down to reduce-only orders (or falls back to a
    read-only card) because it is the process holding the live positions.
    What the bot must guarantee is that it opens no approval window itself.
    """
    monkeypatch.setattr(telegram_module, "settings", _settings(tmp_path))
    _halt(tmp_path)

    reply = _cmd_cycle_now()

    trigger = json.loads((tmp_path / "trigger_now.flag").read_text())
    assert trigger["mode"] == "execute"
    assert "halted" in reply.lower()
    assert "lower risk" in reply.lower()
    assert not (tmp_path / "cycle_approval_pending.json").exists()


def test_explicit_review_command_still_requests_a_review_only_run(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)

    _cmd_review()

    trigger = json.loads((tmp_path / "trigger_now.flag").read_text())
    assert trigger["mode"] == "review"


def test_approve_refuses_an_ordinary_pending_cycle_while_halted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings(tmp_path))
    _halt(tmp_path, "drawdown limit")
    (tmp_path / "cycle_approval_pending.json").write_text(
        json.dumps({"id": "live-cycle-123", "phase": "initial"})
    )

    reply = _cmd_approve([])

    assert "execution is currently halted" in reply.lower()
    assert "drawdown limit" in reply
    assert not (tmp_path / "cycle_approval_decision.json").exists()


def test_approve_accepts_a_runner_verified_reduce_only_basket_while_halted(
    tmp_path: Path, monkeypatch
) -> None:
    """The one approval a halt permits: a basket that can only shrink the book.

    The flag is written by the runner, which verified every order against
    the live positions and re-verifies against a fresh snapshot before
    submitting. Without it the same window is refused (test above).
    """
    monkeypatch.setattr(telegram_module, "settings", _settings(tmp_path))
    _halt(tmp_path, "daily loss -1.09% breaches limit -0.60%")
    (tmp_path / "cycle_approval_pending.json").write_text(
        json.dumps(
            {
                "id": "live-cycle-456",
                "phase": "initial",
                "defensive_reduce_only": True,
            }
        )
    )

    reply = _cmd_approve([])

    decision = json.loads((tmp_path / "cycle_approval_decision.json").read_text())
    assert decision["action"] == "approve"
    assert decision["id"] == "live-cycle-456"
    assert "halted" not in reply.lower() or "approved" in reply.lower()


def test_review_repeats_real_account_artifact_without_approval_controls(
    tmp_path: Path, monkeypatch
) -> None:
    """The bot renders the runner's exact CHF plan; it does not rebuild orders."""
    monkeypatch.setattr(telegram_module, "settings", _settings(tmp_path))
    _halt(tmp_path)
    artifact = {
        "schema_version": 1,
        "kind": "halted_review",
        "id": "review-12345678",
        "halted": True,
        "halt_reason": "daily loss -1.09% breaches limit -0.60%",
        "account": {"base_currency": "CHF", "equity": 86087.26, "position_count": 2},
        "plan": {
            "base_currency": "CHF",
            "equity": 86087.26,
            "order_count": 2,
            "gross_turnover": 1200.0,
            "turnover_pct": 1.39,
            "net_cash_change": -400.0,
            "orders": [
                {
                    "symbol": "GLD",
                    "side": "BUY",
                    "quantity": 1.0,
                    "notional_base": 400.0,
                    "context": "new position",
                },
                {
                    "symbol": "AMD",
                    "side": "SELL",
                    "quantity": 2.0,
                    "notional_base": 800.0,
                    "context": "reduces 8 held",
                },
            ],
            "unchanged_symbols": ["CEG"],
            "fx_fallback_symbols": [],
        },
        "will_never_auto_execute": True,
    }
    (tmp_path / "cycle_halted_review.json").write_text(json.dumps(artifact))

    reply = _cmd_review()

    assert "CHF 86,087 equity" in reply
    assert "CHF 1,200 turnover" in reply
    assert "`GLD` buy 1" in reply
    assert "`AMD` sell 2" in reply
    assert "0 orders submitted" in reply
    assert "cannot be approved, queued or executed" in reply
    assert "✅ Approve as shown" not in reply
    assert not (tmp_path / "cycle_approval_pending.json").exists()
    assert not (tmp_path / "cycle_approval_decision.json").exists()
