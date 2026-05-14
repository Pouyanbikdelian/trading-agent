"""Regime-driven playbook — dynamically switch strategy / universe / vol target.

Concept
-------
You don't want to run the same strategies in every market environment. A
trend system bleeds in chop; a mean-reversion system is run over in a
crisis; a risk-parity ETF basket is the only thing that should be alive
when volatility explodes. The playbook is a YAML file that maps a regime
label to "what to do":

.. code-block:: yaml

    classifier: vix          # which regime classifier provides the label
    default_rule: mid_vol    # rule to use when label is unknown / warm-up
    rules:
      low_vol:
        strategies: [donchian, xsec_momentum]
        universe: nasdaq100
        vol_target: 0.15
      mid_vol:
        strategies: [donchian, risk_parity]
        universe: sp500
        vol_target: 0.10
      high_vol:
        strategies: [risk_parity]
        universe: us_macro_etfs
        vol_target: 0.05
      crisis:
        strategies: []
        force_flatten: true

The runner's cycle, on each tick, asks the playbook "what regime are we
in?" → "what rule applies?" → overrides ``strategies``, ``universe``,
``vol_target`` on the working ``RunnerConfig`` before running.

What the playbook does NOT do
-----------------------------
* Doesn't change the broker or the risk-manager limits — those stay
  constant. The playbook is about *strategy mix*, not risk appetite.
  (You absolutely can lower ``max_position_pct`` for risk-off regimes;
  do it via ``config/risk.yaml`` and accept the constraint applies to
  every regime.)
* Doesn't auto-fit anything. The YAML is hand-authored — that's
  intentional. A regime-switching system that auto-tunes itself is the
  fastest way to overfit yourself into a hole.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PlaybookRule(BaseModel):
    """One row of the playbook — what to do in a given regime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategies: list[str] = Field(default_factory=list)
    """Strategy names from STRATEGY_REGISTRY. Empty list = stay flat
    (no orders generated; existing positions held)."""

    universe: str | None = None
    """Universe to trade in this regime. None = inherit from RunnerConfig."""

    vol_target: float | None = None
    """Annualized portfolio vol target. None = inherit from RunnerConfig."""

    force_flatten: bool = False
    """If True, the runner closes all open positions on entry to this regime.
    Use for ``crisis`` rules where you want to be in cash, period."""

    strategy_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-strategy overrides keyed by strategy name."""


class Playbook(BaseModel):
    """The whole playbook — classifier name + rules + default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classifier: str = "vix"
    """Which classifier produces the regime label this playbook keys on.
    Only ``vix`` is wired today; ``realized_vol`` and ``hmm`` are also
    valid identifiers reserved for later wiring."""

    default_rule: str | None = None
    """Rule name to use when the classifier returns no label (warm-up)
    or returns a label not in ``rules``. If None, the runner inherits
    its static config."""

    rules: dict[str, PlaybookRule] = Field(default_factory=dict)


def load_playbook(path: str | Path) -> Playbook:
    """Read a Playbook from a YAML file. Validates via pydantic."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"playbook not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Playbook.model_validate(raw)


def rule_for(playbook: Playbook, label: str) -> PlaybookRule | None:
    """Look up the rule for a regime label, falling back to ``default_rule``.
    Returns None if neither resolves — caller should keep its static config."""
    if label in playbook.rules:
        return playbook.rules[label]
    if playbook.default_rule and playbook.default_rule in playbook.rules:
        return playbook.rules[playbook.default_rule]
    return None
