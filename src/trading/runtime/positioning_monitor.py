"""Archive actual observations and expose gaps. Optional feed failures never authorize trades."""

from __future__ import annotations

import fcntl
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from trading.core.clock import UtcClock
from trading.core.types import PositioningObservation
from trading.data.positioning import NY, _identity, expected_cot_date, fetch_cot, fetch_gamma
from trading.runtime.options_monitor import _read_state, _write_state

_LOCK = threading.Lock()


def _archive(root: Path, obs: PositioningObservation) -> PositioningObservation:
    root.mkdir(parents=True, exist_ok=True)
    path = root / (obs.id.replace(":", "-") + ".json")
    fd, temp = tempfile.mkstemp(dir=root, prefix=".observation-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(obs.model_dump_json(indent=2))
        try:
            os.link(temp, path)
        except FileExistsError:
            # Immutable archives are published atomically; retain actual first seen.
            existing = PositioningObservation.model_validate_json(path.read_text())
            if existing.facts != obs.facts or existing.observed_at != obs.observed_at:
                raise ValueError("report revision needs a new original report ID") from None
            return existing
    finally:
        os.unlink(temp)
    return obs


def import_goldman(state_dir: Path, report: dict[str, Any], now: datetime | None = None) -> str:
    """Import operator-checked public evidence; don't manufacture a free Prime Services API."""
    now = now or UtcClock().now()
    url = str(report["source_url"])
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "www.goldmansachs.com",
        "goldmansachs.com",
    }:
        raise ValueError("public Goldman HTTPS URL required")
    if report.get("verified_against_source") is not True:
        raise ValueError("verify dates, quotation and metrics against the original public report")
    publication = datetime.fromisoformat(report["published_at"])
    covered = datetime.fromisoformat(report["observed_at"])
    if any(t.tzinfo is None for t in (publication, covered)) or not covered <= publication <= now:
        raise ValueError("invalid publication/covered-period times")
    if not str(report.get("report_id", "")).strip() or not str(report.get("excerpt", "")).strip():
        raise ValueError("original report ID and checked excerpt required")
    scope = str(report["scope"])
    obs = PositioningObservation(
        id="goldman:" + _identity([report["report_id"], scope]),
        source="goldman_public",
        kind="reported",
        scope=scope,
        symbols=tuple(report.get("symbols", [])),
        observed_at=covered,
        available_at=now,
        expires_at=min(publication + timedelta(days=14), covered + timedelta(days=21)),
        source_url=url,
        facts=report.get("facts", {}),
        note=f"Goldman client sample, not whole-market flow. Published {publication.isoformat()}. Units: {report['units']}. Source-checked excerpt: {report['excerpt']}",
    )
    _archive(state_dir / "positioning" / "reports", obs)
    return obs.id


def read_positioning(state_dir: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or UtcClock().now()
    state = _read_state(state_dir / "positioning.json")
    active = []
    rejected = 0
    for row in state.get("observations", []):
        try:
            obs = PositioningObservation.model_validate(row)
            if not obs.observed_at <= now or not obs.available_at <= now < obs.expires_at:
                raise ValueError("expired or future evidence")
            if obs.source == "cot" and obs.observed_at.date().isoformat() < expected_cot_date(now):
                raise ValueError("awaiting normally expected release (holiday delays possible)")
            active.append(obs.model_dump(mode="json"))
        except (ValueError, TypeError):
            rejected += 1
    last = state.get("last_attempt_at")
    try:
        stamp = datetime.fromisoformat(str(last))
        grace = now.astimezone(NY) - timedelta(minutes=30)
        deadline = grace.replace(hour=15, minute=46, second=0, microsecond=0)
        if deadline > grace:
            deadline -= timedelta(days=1)
        while deadline.weekday() >= 5:
            deadline -= timedelta(days=1)
        watchdog = "ok" if deadline <= stamp <= now else "collector_overdue"
    except (ValueError, TypeError):
        watchdog = "never_completed"
    coverage = {
        s: sum(r["source"] == s for r in active)
        for s in ("cot", "goldman_public", "gamma_estimate")
    }
    return {
        "version": 1,
        "mode": "advisory_and_shadow",
        "observations": active,
        "coverage": coverage,
        "errors": state.get("errors", {}),
        "last_attempt_at": last,
        "watchdog": watchdog,
        "rejected": rejected,
        "expected_cot_report": expected_cot_date(now),
        "release_note": "Normal Friday expectation; holiday/delayed releases remain unavailable until observed. No synthetic release timestamp.",
        "policy": "Missing means unknown; not bearish. Macro futures are not individual-stock flows. Gamma is estimated with unknown OI age. No automatic live positioning overlay.",
    }


def refresh_positioning(state_dir: Path, now: datetime | None = None) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK, (state_dir / ".positioning.lock").open("a") as lock_file:
        # CLI and runner are separate processes sharing the state volume.
        # Serialize their read/fetch/write transaction, not only Python threads.
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        now = now or UtcClock().now()
        prior = _read_state(state_dir / "positioning.json")
        observations = []
        errors = {}
        success = dict(prior.get("last_success", {}))
        jobs: dict[str, Callable[[], list[PositioningObservation]]] = {
            "cot": lambda: fetch_cot(now),
            "gamma_SPY": lambda: [fetch_gamma("SPY", now)],
            "gamma_QQQ": lambda: [fetch_gamma("QQQ", now)],
        }
        for source, fetch in jobs.items():
            old = [
                r
                for r in prior.get("observations", [])
                if (
                    r.get("source") == "cot"
                    if source == "cot"
                    else r.get("source") == "gamma_estimate" and source[6:] in r.get("symbols", [])
                )
            ]
            try:
                age = (now - datetime.fromisoformat(str(success.get(source)))).total_seconds()
            except (ValueError, TypeError):
                age = float("inf")
            # COT rechecked at decision time when a new normal release is due.
            cot_current = old and all(r["observed_at"][:10] >= expected_cot_date(now) for r in old)
            cached = 0 <= age < (3600 if source == "cot" and cot_current else 600)
            if cached:
                observations.extend(old)
                continue
            try:
                fresh = fetch()
                if not fresh:
                    raise ValueError("empty response")
                observations.extend(
                    _archive(state_dir / "positioning" / "archive", o).model_dump(mode="json")
                    for o in fresh
                )
                success[source] = now.isoformat()
            except Exception as exc:
                errors[source] = type(exc).__name__ + ": " + str(exc)[:180]
                observations.extend(old)
        for path in sorted((state_dir / "positioning" / "reports").glob("*.json")):
            try:
                observations.append(
                    PositioningObservation.model_validate_json(path.read_text()).model_dump(
                        mode="json"
                    )
                )
            except ValueError:
                errors[path.name] = "invalid imported report"
        state = {
            "last_attempt_at": now.isoformat(),
            "last_success": success,
            "observations": observations,
            "errors": errors,
        }
        _write_state(state_dir / "positioning.json", state)
        return read_positioning(state_dir, now)
