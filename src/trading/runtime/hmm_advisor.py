r"""HMM-based regime classifier for the advisor.

Replaces the SMA-crossover trigger in ``risk_monitor`` with a smarter
probabilistic signal: fit a 3-state Gaussian HMM on SPY returns, then
ask the model "what's the probability we're in the bear state today?"

Why HMM instead of more SMA rules
---------------------------------
SMA-based regime triggers (e.g. "SPY < SMA(200) for 5 days") have a
single fixed threshold. They flip on/off and have no notion of
*confidence*. The HMM gives back a posterior distribution over states:

    P(bull | today)   P(neutral | today)   P(bear | today)

That's a much richer signal. The advisor can fire a "soft" alert when
P(bear) > 0.4, a "hard" alert when P(bear) > 0.7. No magic threshold
on a single moving average.

The model
---------
* Gaussian-emission HMM (existing ``trading.regime.HmmRegime``).
* 3 states — Bear / Neutral / Bull. After fitting we sort by emission
  mean so ``state=0`` is always Bear and ``state=2`` is always Bull.
* Trained on SPY daily log-returns over the last ~5 years of history.
  The model needs sufficient history to see at least one bear regime
  to learn its parameters — 5 years usually has 2008 or 2020 visible.

This module is *advisory only* — it never writes ``mode.json``. It just
produces a Telegram-ready message about the current regime. The
operator decides what to do.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger
from trading.regime.hmm import HmmParams, HmmRegime

STATE_FILENAME = "hmm_advisor.json"

# Labels chosen by mean-rank: state 0 = lowest emission mean = bear.
STATE_LABELS = {0: "BEAR", 1: "NEUTRAL", 2: "BULL"}


@dataclass(frozen=True)
class RegimeSnapshot:
    state: int  # 0 = bear, 1 = neutral, 2 = bull
    label: str
    p_bear: float
    p_neutral: float
    p_bull: float
    as_of: str  # ISO 8601 UTC


def fit_and_classify(
    spy_returns: pd.Series,
    *,
    n_states: int = 3,
    random_state: int = 42,
) -> tuple[HmmRegime, RegimeSnapshot]:
    r"""Fit an HMM on ``spy_returns`` and return both the fitted model
    and the latest-bar regime snapshot (state + posterior probabilities).

    ``spy_returns`` should be daily log-returns (or simple returns —
    both work at this frequency). We need at least ~500 observations to
    estimate a 3-state model reliably.
    """
    if len(spy_returns) < 300:
        raise ValueError(
            f"need at least 300 SPY return observations to fit a {n_states}-state HMM; "
            f"got {len(spy_returns)}"
        )

    model = HmmRegime(HmmParams(n_states=n_states, random_state=random_state))
    model.fit(spy_returns)
    snapshot = _snapshot_from_model(model, spy_returns)
    return model, snapshot


def _snapshot_from_model(model: HmmRegime, spy_returns: pd.Series) -> RegimeSnapshot:
    """Compute the posterior probability over states for the LAST bar."""
    if model._model is None or model._state_order is None:  # type: ignore[attr-defined]
        raise RuntimeError("model has not been fit")
    # hmmlearn's predict_proba returns shape (T, n_components) in HMM-internal
    # state order; we remap to mean-sorted order so state=0 is always bear.
    x = spy_returns.values.reshape(-1, 1)
    raw_proba = model._model.predict_proba(x)  # type: ignore[attr-defined]
    # Remap columns: raw column i -> mean-sorted column self._state_order[i]
    order = model._state_order  # type: ignore[attr-defined]
    sorted_proba = np.zeros_like(raw_proba)
    for raw_i, sorted_i in enumerate(order):
        sorted_proba[:, sorted_i] = raw_proba[:, raw_i]
    last = sorted_proba[-1]
    state = int(np.argmax(last))
    return RegimeSnapshot(
        state=state,
        label=STATE_LABELS.get(state, f"STATE_{state}"),
        p_bear=float(last[0]),
        p_neutral=float(last[1]) if len(last) > 1 else 0.0,
        p_bull=float(last[-1]),
        as_of=datetime.now(tz=timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Advisory alerter
# ---------------------------------------------------------------------------


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _format_message(snap: RegimeSnapshot, prior_label: str | None) -> str:
    """Telegram-ready, severity-graded.

    A regime *change* is the alert-worthy event. Stable regimes are
    quiet so we don't spam the operator. The probabilities are always
    included so the operator can see how confident the model is.
    """
    if prior_label and prior_label == snap.label:
        return ""  # no change, no alert
    severity_emoji = {"BEAR": "🛑", "NEUTRAL": "🟡", "BULL": "🟢"}.get(snap.label, "ℹ️")
    if snap.label == "BEAR":
        suggestion = "consider `/mode defense` or `/mode bear`"
    elif snap.label == "BULL":
        suggestion = "no defensive action needed"
    else:
        suggestion = "watch — neither aggressive nor defensive"
    return (
        f"{severity_emoji} *HMM regime change* → `{snap.label}`\n"
        f"P(bear): `{snap.p_bear:.0%}`  "
        f"P(neutral): `{snap.p_neutral:.0%}`  "
        f"P(bull): `{snap.p_bull:.0%}`\n\n"
        f"_advisory only_ — {suggestion}"
    )


async def poll_and_alert(
    *,
    spy_returns: pd.Series,
    state_path: Path | None = None,
    n_states: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    r"""Fit / update the HMM, compute the latest regime snapshot, and
    push a Telegram alert when the *labeled* state changes.

    Caller is expected to pass a recent slice of SPY returns (e.g. last
    5 years). The HMM is fit fresh every poll — at our cadence (hourly)
    this is cheap (~50ms) and avoids state-drift problems from old fits.
    """
    state_path = state_path or (settings.state_dir / STATE_FILENAME)
    prior = _read_state(state_path)
    prior_label = prior.get("label")

    try:
        _, snap = fit_and_classify(spy_returns, n_states=n_states, random_state=random_state)
    except Exception as e:
        logger.warning(f"HMM advisor poll failed: {e}")
        return {"error": str(e)}

    text = _format_message(snap, prior_label)
    sent = False
    if text:
        sent = await _send_telegram(text)

    _write_state(
        state_path,
        {
            "state": snap.state,
            "label": snap.label,
            "p_bear": snap.p_bear,
            "p_neutral": snap.p_neutral,
            "p_bull": snap.p_bull,
            "as_of": snap.as_of,
        },
    )
    return {
        "snapshot": {
            "state": snap.state,
            "label": snap.label,
            "p_bear": snap.p_bear,
            "p_neutral": snap.p_neutral,
            "p_bull": snap.p_bull,
        },
        "regime_changed": bool(text),
        "alert_sent": sent,
    }


async def _send_telegram(text: str) -> bool:
    try:
        from trading.bot.notifier import send_message
    except Exception:
        logger.warning("HMM advisor: cannot import telegram notifier; alert dropped")
        return False
    return await send_message(text)
