"""One daily market-risk note instead of four advisors pinging separately.

Why (2026-09-26). Four monitors each sent their own Telegram message when a
trigger appeared or cleared: the SPY/VIX advisor (hourly, and by its own
docstring it "fires far too often"), the HMM regime model, the options
vol-surface monitor and the macro conditions dial. On a choppy week that
was a stream of partial views, each ending "advisory only", none saying
how they added up.

They still run on their own schedules and still write their state files —
the committee, the PM and the dashboard read those. What changed is who
speaks: the monitors stay quiet (``instant_alerts()`` is False) and this
module reads their files once each weekday after the close and writes one
note: an overall read, one line per source, and what is new or cleared
since yesterday's note.

Two things still interrupt immediately, because they are the reason to
look at a phone mid-session: the intraday sentinel (SPY/VIX/held-name
tripwires, unchanged) and an EXTREME SPY/VIX trigger.

Advisory only. Nothing here reads or writes anything on the order path.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading.core.logging import logger

NOTE_STATE_FILE = "market_note.json"

#: Plain-language names for the SPY/VIX advisor's trigger ids.
_ADVISOR_NAMES = {
    "slow_grind": "slow grind lower (SPY below its trend)",
    "fast_crash": "fast drop in SPY",
    "vol_spike": "VIX spike",
    "combined_extreme": "fast drop and VIX spike together",
}


def instant_alerts() -> bool:
    """False (the default): monitors defer to the daily note.

    ``MARKET_ALERTS=instant`` restores the old per-monitor messages.
    """
    return os.getenv("MARKET_ALERTS", "digest").strip().lower() == "instant"


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, path)


def _active(raw: dict[str, Any]) -> list[str]:
    return sorted(str(x) for x in raw.get("active") or [])


def snapshot(state_dir: Path) -> dict[str, Any]:
    """The four monitors' current readings, from their own state files."""
    sd = Path(state_dir)
    adv = _read(sd / "advisor.json")
    hmm = _read(sd / "hmm_advisor.json")
    opt = _read(sd / "options_monitor.json")
    mac = _read(sd / "macro_monitor.json")
    return {
        "advisor": {"active": _active(adv), "severities": adv.get("severities") or {}},
        "hmm": {
            "label": hmm.get("label"),
            "p_bear": hmm.get("p_bear"),
            "p_neutral": hmm.get("p_neutral"),
            "p_bull": hmm.get("p_bull"),
            "as_of": hmm.get("as_of"),
        },
        "options": {"active": _active(opt), "metrics": opt.get("metrics") or {}},
        "macro": {"active": _active(mac), "readings": mac.get("readings") or {}},
    }


def _signals(snap: dict[str, Any]) -> set[str]:
    """Every flagged thing, as stable ids, for the new/cleared diff."""
    out = {f"spy_vix:{n}" for n in snap["advisor"]["active"]}
    out |= {f"options:{n}" for n in snap["options"]["active"]}
    out |= {f"macro:{n}" for n in snap["macro"]["active"]}
    if str(snap["hmm"].get("label") or "").upper() == "BEAR":
        out.add("regime:BEAR")
    return out


def _verdict(snap: dict[str, Any]) -> tuple[str, str]:
    sev = [int(v) for v in (snap["advisor"]["severities"] or {}).values() if str(v).isdigit()]
    n = len(_signals(snap))
    bear = str(snap["hmm"].get("label") or "").upper() == "BEAR"
    if (sev and max(sev) >= 3) or n >= 4:
        return "🔴", "Stressed"
    if bear or (sev and max(sev) >= 2) or n >= 2:
        return "🟠", "Elevated"
    if n == 1:
        return "🟡", "Watch"
    return "🟢", "Calm"


def _tag(sig: str, new: set[str]) -> str:
    return " 🆕" if sig in new else ""


def build_note(
    state_dir: Path,
    *,
    previous: set[str] | None = None,
    temperature: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[str, set[str]]:
    """The note text, and the signal set to remember for tomorrow's diff."""
    now = now or datetime.now(tz=timezone.utc)
    snap = snapshot(state_dir)
    sigs = _signals(snap)
    prev = set(previous or ())
    new, cleared = sigs - prev, prev - sigs
    icon, word = _verdict(snap)
    day = now.astimezone(ZoneInfo("America/New_York"))
    lines = [f"📊 *Market risk — {day:%a %d %b}* · {icon} {word}"]

    if temperature and temperature.get("label"):
        t = temperature.get("temperature")
        dist = temperature.get("spy_vs_200d")
        extra = f": SPY {dist * 100:+.1f}% vs its 200-day" if isinstance(dist, (int, float)) else ""
        score = f" ({t:+.2f})" if isinstance(t, (int, float)) else ""
        lines.append(f"*Temperature* · {temperature['label']}{score}{extra}")

    hmm = snap["hmm"]
    if hmm.get("label"):
        probs = [
            f"{name} {float(hmm[key]):.0%}"
            for name, key in (("bear", "p_bear"), ("neutral", "p_neutral"), ("bull", "p_bull"))
            if isinstance(hmm.get(key), (int, float))
        ]
        lines.append(
            f"*Regime* · {hmm['label']}{_tag('regime:BEAR', new)}"
            + (f" — {' · '.join(probs)}" if probs else "")
        )

    adv = snap["advisor"]["active"]
    lines.append(
        "*SPY/VIX* · "
        + (
            "; ".join(f"{_ADVISOR_NAMES.get(n, n)}{_tag('spy_vix:' + n, new)}" for n in adv)
            if adv
            else "no trigger"
        )
    )

    m = snap["options"]["metrics"]
    opt_bits = []
    if isinstance(m.get("atm_iv"), (int, float)):
        opt_bits.append(f"IV {m['atm_iv']:.0%}")
    if isinstance(m.get("put_skew"), (int, float)):
        opt_bits.append(f"put skew {m['put_skew'] * 100:+.1f}")
    if isinstance(m.get("term_slope"), (int, float)):
        opt_bits.append(f"term slope {m['term_slope'] * 100:+.1f}")
    opt_act = snap["options"]["active"]
    flagged = ", ".join(f"`{n}`{_tag('options:' + n, new)}" for n in opt_act)
    if opt_bits or opt_act:
        lines.append(
            "*Options* · "
            + " · ".join(opt_bits)
            + (f" — {flagged}" if opt_act else (" — no stress" if opt_bits else ""))
        )

    r = snap["macro"]["readings"]
    mac_act = snap["macro"]["active"]
    if r or mac_act:
        comp = r.get("composite")
        head = f"composite {comp:+.1f}σ" if isinstance(comp, (int, float)) else "dial"
        flags = ", ".join(f"`{n}`{_tag('macro:' + n, new)}" for n in mac_act)
        lines.append(f"*Macro* · {head}" + (f" — {flags}" if mac_act else " — inside ±1.5σ"))

    if cleared:
        pretty = ", ".join(s.split(":", 1)[1] for s in sorted(cleared))
        lines.append(f"*Cleared since the last note* · {pretty}")
    if word in ("Elevated", "Stressed"):
        lines.append("_Worth a look: `/regime` for detail, `/mode defense` to preview a posture._")
    lines.append("_Advisory only — nothing trades on this note._")
    return "\n".join(lines), sigs


def compose_and_remember(
    state_dir: Path,
    *,
    temperature: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    """Build today's note and store its signal set for tomorrow's diff."""
    path = Path(state_dir) / NOTE_STATE_FILE
    prev = set(_read(path).get("signals") or [])
    text, sigs = build_note(state_dir, previous=prev, temperature=temperature, now=now)
    try:
        _write_atomic(
            path,
            {
                "signals": sorted(sigs),
                "sent_at": (now or datetime.now(tz=timezone.utc)).isoformat(),
            },
        )
    except Exception:
        logger.bind(component="market_note").exception("could not store the market-note state")
    return text
