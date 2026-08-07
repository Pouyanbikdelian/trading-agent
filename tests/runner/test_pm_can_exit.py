"""The PM must be able to SELL what it bought.

A target weight of zero is how every book in this system says "get out".
But a name the PM has dropped does not appear in its weights at all — it
is absent, not zero. And ``_add_pm_targets`` prices only the PM's
CURRENT picks, so a dropped name is not even in ``instruments_by_key``.

Absent from the merged weights means ``signal_to_orders`` never computes
a delta for it, which means no order, which means the position stays.

For S&P names the mechanical strategy covers the gap by accident: its
weight vector spans the whole universe, so a name it does not pick
carries an explicit 0.0. The PM's ETF shelf — GLD, XLV, XLK, IBB, URA —
is outside that universe, and those are exactly what the PM uses to
express a hedge. Its first real live decision was GLD 0.15 + XLV 0.12.

So the PM's book could ratchet: add, never remove. The position keeps
counting toward equity and gross, silently eating the sleeve.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.agents.pm_signal import (
    load_pm_signal,
    load_previous_targets,
    pm_decision_path,
    save_targets,
)

DECIDED = datetime(2026, 8, 7, 20, 20, tzinfo=timezone.utc)
NOW = DECIDED + timedelta(hours=1)
NOW_ISO = DECIDED.isoformat()


def _write_decision(state_dir: Path, weights: dict[str, float]) -> None:
    path = pm_decision_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": True, "ts": NOW_ISO, "weights": weights}))


def test_a_dropped_name_is_not_in_the_weights_at_all(tmp_path: Path) -> None:
    """The property that makes the bug possible — pinned so the fix is
    understood as 'reinstate the zero', not 'the PM emits zeros'."""
    _write_decision(tmp_path, {"AAPL": 0.10, "GLD": 0.0})
    r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.11, max_age_h=6)
    assert r.signal is not None
    assert "etf:GLD" not in r.signal.target_weights


def test_previous_targets_round_trip(tmp_path: Path) -> None:
    save_targets(tmp_path, {"etf:GLD", "equity:AAPL"})
    assert load_previous_targets(tmp_path) == {"etf:GLD", "equity:AAPL"}


def test_no_previous_targets_is_an_empty_set(tmp_path: Path) -> None:
    assert load_previous_targets(tmp_path) == set()


def test_a_corrupt_previous_file_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "agent_pm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent_pm" / "last_targets.json").write_text("{not json")
    assert load_previous_targets(tmp_path) == set()


def test_exiting_keys_are_the_ones_dropped_since_last_cycle(tmp_path: Path) -> None:
    save_targets(tmp_path, {"etf:GLD", "etf:XLV", "equity:AAPL"})
    _write_decision(tmp_path, {"AAPL": 0.10, "MSFT": 0.08})

    r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.11, max_age_h=6)

    assert r.signal is not None
    assert r.exiting == {"etf:GLD", "etf:XLV"}
    # The names it still wants are not exits.
    assert "equity:AAPL" not in r.exiting


def test_a_refused_decision_does_not_report_exits(tmp_path: Path) -> None:
    """Staleness must not be read as 'the PM wants out of everything'."""
    save_targets(tmp_path, {"etf:GLD"})
    _write_decision(tmp_path, {"AAPL": 0.10})

    r = load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.11, max_age_h=0.5)

    assert r.signal is None
    assert r.exiting == set()


def test_targets_are_saved_only_on_a_tradeable_signal(tmp_path: Path) -> None:
    """Otherwise a refused cycle overwrites the record of what is held,
    and the next good cycle has nothing left to exit."""
    save_targets(tmp_path, {"etf:GLD"})
    _write_decision(tmp_path, {"AAPL": 0.10})

    load_pm_signal(tmp_path, now=NOW, sleeve_pct=0.11, max_age_h=0.5)  # stale, refused

    assert load_previous_targets(tmp_path) == {"etf:GLD"}
