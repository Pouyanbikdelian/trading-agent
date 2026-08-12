"""Approval-gated operator changes for the Telegram desk copilot.

The copilot may understand a request such as "add NVDA to our list", but
understanding must not itself mutate desk state.  This module owns the small,
deterministic write boundary: it stages a concrete change, binds approval to
that proposal, applies it once, and records what changed.  None of these
operations reach the broker or the trading universe.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

from trading.core.clock import Clock, UtcClock
from trading.core.logging import logger
from trading.memory.store import MemoryStore

_LOG = logger.bind(component="desk_copilot")

PROPOSALS_FILE = "desk_change_proposals.json"
AUDIT_FILE = "desk_change_audit.jsonl"
WATCHLIST_OVERRIDES_FILE = "operator_watchlist.json"
PROPOSAL_TTL = timedelta(minutes=10)
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,7}$")

ChangeKind = Literal[
    "watchlist_add",
    "watchlist_remove",
    "watchlist_add_many",
    "watchlist_remove_many",
    "lesson_create",
    "lesson_supersede",
    "lesson_status",
    "lesson_archive",
]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _lesson_snapshot(mem: MemoryStore, lesson_id: str) -> dict[str, str] | None:
    """The small stable lesson shape needed by proposal validation/audit."""
    row = mem.conn.execute(
        "SELECT id, statement, status, tags FROM lessons WHERE id = ?", (lesson_id,)
    ).fetchone()
    if row is None:
        return None
    return {
        "id": str(row["id"]),
        "statement": str(row["statement"]),
        "status": str(row["status"]),
        "tags": str(row["tags"] or ""),
    }


class WatchlistStore:
    """Effective dashboard-only watchlist: static config plus operator overrides.

    ``config/watchlist.yaml`` remains a versioned, read-only baseline inside
    the bot container.  Telegram changes live in the shared state volume as
    a compact add/remove overlay; the dashboard uses this same resolver.
    That matters on the VPS: making ``config/`` writable for a chat bot
    would broaden the bot's authority far beyond the operator watchlist.
    """

    def __init__(self, config_path: Path, overrides_path: Path) -> None:
        self.config_path = Path(config_path)
        self.overrides_path = Path(overrides_path)

    @staticmethod
    def _symbol(symbol: str) -> str:
        clean = str(symbol).strip().upper()
        if not _SYMBOL_RE.fullmatch(clean):
            raise ValueError("use a ticker such as NVDA, BRK.B, or RDS-A")
        return clean

    def _base_items(self) -> list[str]:
        if not self.config_path.exists():
            return []
        try:
            payload = yaml.safe_load(self.config_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"watchlist YAML is invalid: {e}") from e
        raw = payload.get("watchlist", []) if isinstance(payload, dict) else []
        if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
            raise ValueError("watchlist.yaml must contain a string list named 'watchlist'")
        return list(dict.fromkeys(self._symbol(s) for s in raw))

    def baseline_items(self) -> list[str]:
        """Versioned watchlist before any operator-state overlay is applied."""
        return self._base_items()

    def _overrides(self) -> tuple[list[str], list[str]]:
        """``(added, removed)`` with strict validation at the write boundary."""
        try:
            raw = json.loads(self.overrides_path.read_text())
        except FileNotFoundError:
            return [], []
        except json.JSONDecodeError as e:
            raise ValueError(f"operator watchlist state is invalid: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("operator watchlist state must be an object")
        added, removed = raw.get("added", []), raw.get("removed", [])
        if not isinstance(added, list) or not isinstance(removed, list):
            raise ValueError("operator watchlist state must have string added/removed lists")
        if not all(isinstance(s, str) for s in [*added, *removed]):
            raise ValueError("operator watchlist state must have string added/removed lists")
        return (
            list(dict.fromkeys(self._symbol(s) for s in added)),
            list(dict.fromkeys(self._symbol(s) for s in removed)),
        )

    def _save_overrides(self, added: list[str], removed: list[str]) -> None:
        _atomic_write(
            self.overrides_path,
            json.dumps({"added": added, "removed": removed}, indent=2) + "\n",
        )

    def items(self) -> list[str]:
        base = self._base_items()
        added, removed = self._overrides()
        omitted = set(removed)
        return list(dict.fromkeys(symbol for symbol in [*base, *added] if symbol not in omitted))

    @classmethod
    def _symbols(cls, symbols: list[str]) -> list[str]:
        clean = list(dict.fromkeys(cls._symbol(symbol) for symbol in symbols))
        if not clean:
            raise ValueError("give at least one ticker")
        return clean

    def add_many(self, symbols: list[str]) -> tuple[bool, list[str]]:
        """Atomically add every symbol, or none when one is already present."""
        symbols = self._symbols(symbols)
        items = set(self.items())
        already_present = [symbol for symbol in symbols if symbol in items]
        if already_present:
            return False, already_present

        base = self._base_items()
        added, removed = self._overrides()
        for symbol in symbols:
            removed = [item for item in removed if item != symbol]
            if symbol not in base:
                added.append(symbol)
        self._save_overrides(list(dict.fromkeys(added)), list(dict.fromkeys(removed)))
        return True, []

    def remove_many(self, symbols: list[str]) -> tuple[bool, list[str]]:
        """Atomically remove every symbol, or none when one is already absent."""
        symbols = self._symbols(symbols)
        items = set(self.items())
        absent = [symbol for symbol in symbols if symbol not in items]
        if absent:
            return False, absent

        base = self._base_items()
        added, removed = self._overrides()
        for symbol in symbols:
            if symbol in base:
                removed.append(symbol)
            else:
                added = [item for item in added if item != symbol]
        self._save_overrides(list(dict.fromkeys(added)), list(dict.fromkeys(removed)))
        return True, []

    def add(self, symbol: str) -> bool:
        return self.add_many([symbol])[0]

    def remove(self, symbol: str) -> bool:
        return self.remove_many([symbol])[0]


def _format_symbols(symbols: list[str]) -> str:
    return ", ".join(f"`{symbol}`" for symbol in symbols)


def _watchlist_payload(symbols: list[str], *, undo_of: str | None = None) -> dict[str, str]:
    symbols = WatchlistStore._symbols(symbols)
    payload = {"symbol": symbols[0]} if len(symbols) == 1 else {"symbols": ",".join(symbols)}
    if undo_of:
        payload["undo_of"] = undo_of
    return payload


def _proposal_symbols(payload: dict[str, str]) -> list[str]:
    if "symbols" in payload:
        return WatchlistStore._symbols(payload["symbols"].split(","))
    return WatchlistStore._symbols([payload["symbol"]])


@dataclass(frozen=True)
class DeskProposal:
    """One reviewable mutation.  Its id is the callback's safety binding."""

    id: str
    kind: ChangeKind
    payload: dict[str, str]
    requested_at: str
    expires_at: str
    requested_by: str = "telegram"
    status: str = "pending"

    def is_expired(self, now: datetime) -> bool:
        try:
            return datetime.fromisoformat(self.expires_at) <= now
        except (TypeError, ValueError):
            return True

    @classmethod
    def from_dict(cls, raw: object) -> DeskProposal | None:
        if not isinstance(raw, dict):
            return None
        try:
            kind = str(raw["kind"])
            if kind not in {
                "watchlist_add",
                "watchlist_remove",
                "watchlist_add_many",
                "watchlist_remove_many",
                "lesson_create",
                "lesson_supersede",
                "lesson_status",
                "lesson_archive",
            }:
                return None
            payload = raw.get("payload", {})
            if not isinstance(payload, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in payload.items()
            ):
                return None
            required = {
                "watchlist_add": {"symbol"},
                "watchlist_remove": {"symbol"},
                "watchlist_add_many": {"symbols"},
                "watchlist_remove_many": {"symbols"},
                "lesson_create": {"statement", "strength", "status"},
                "lesson_supersede": {
                    "lesson_id",
                    "old_statement",
                    "statement",
                    "strength",
                    "status",
                },
                "lesson_status": {"lesson_id", "old_status", "status"},
                "lesson_archive": {"lesson_id", "statement"},
            }
            if not required[kind].issubset(payload):
                return None
            if (
                kind in {"watchlist_add_many", "watchlist_remove_many"}
                and len(_proposal_symbols(payload)) < 2
            ):
                return None
            return cls(
                id=str(raw["id"]),
                kind=kind,  # type: ignore[arg-type]
                payload=dict(payload),
                requested_at=str(raw["requested_at"]),
                expires_at=str(raw["expires_at"]),
                requested_by=str(raw.get("requested_by", "telegram")),
                status=str(raw.get("status", "pending")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ChangeResult:
    proposal: DeskProposal | None
    message: str


class DeskChangeStore:
    """Proposal state, audit history, and the narrow desk-write boundary."""

    def __init__(
        self,
        state_dir: Path,
        watchlist_path: Path,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / PROPOSALS_FILE
        self.audit_path = self.state_dir / AUDIT_FILE
        self.watchlist = WatchlistStore(watchlist_path, self.state_dir / WATCHLIST_OVERRIDES_FILE)
        self.clock = clock or UtcClock()

    def _all(self) -> list[DeskProposal]:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            raise ValueError(f"desk-change proposal state is invalid: {e}") from e
        if not isinstance(raw, list):
            raise ValueError("desk-change proposal state must be a list")
        proposals: list[DeskProposal] = []
        for row in raw:
            proposal = DeskProposal.from_dict(row)
            if proposal is None:
                raise ValueError("desk-change proposal state contains an invalid proposal")
            proposals.append(proposal)
        return proposals

    def _save(self, proposals: list[DeskProposal]) -> None:
        _atomic_write(self.path, json.dumps([asdict(p) for p in proposals], indent=2) + "\n")

    def _audit(self, event: str, proposal: DeskProposal, **detail: Any) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": self.clock.now().isoformat(),
            "event": event,
            "proposal": asdict(proposal),
            **detail,
        }
        try:
            with self.audit_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            _LOG.exception("could not write desk copilot audit")

    def _stage(self, kind: ChangeKind, payload: dict[str, str]) -> DeskProposal:
        now = self.clock.now()
        proposal = DeskProposal(
            id=f"dc-{uuid.uuid4().hex[:10]}",
            kind=kind,
            payload=payload,
            requested_at=now.isoformat(),
            expires_at=(now + PROPOSAL_TTL).isoformat(),
        )
        rows = self._all()
        self._save([*rows, proposal])
        self._audit("proposed", proposal)
        return proposal

    def propose_watchlist_add(self, symbol: str) -> ChangeResult:
        symbol = self.watchlist._symbol(symbol)
        if symbol in self.watchlist.items():
            return ChangeResult(None, f"`{symbol}` is already on the operator watchlist.")
        return ChangeResult(self._stage("watchlist_add", {"symbol": symbol}), "")

    def propose_watchlist_add_many(self, symbols: list[str]) -> ChangeResult:
        symbols = self.watchlist._symbols(symbols)
        if len(symbols) == 1:
            return self.propose_watchlist_add(symbols[0])
        already_present = [symbol for symbol in symbols if symbol in self.watchlist.items()]
        if already_present:
            return ChangeResult(
                None,
                f"Already on the operator watchlist: {_format_symbols(already_present)}. No change staged.",
            )
        return ChangeResult(self._stage("watchlist_add_many", _watchlist_payload(symbols)), "")

    def propose_watchlist_remove(self, symbol: str) -> ChangeResult:
        symbol = self.watchlist._symbol(symbol)
        if symbol not in self.watchlist.items():
            return ChangeResult(None, f"`{symbol}` is not on the operator watchlist.")
        return ChangeResult(self._stage("watchlist_remove", {"symbol": symbol}), "")

    def propose_watchlist_remove_many(self, symbols: list[str]) -> ChangeResult:
        symbols = self.watchlist._symbols(symbols)
        if len(symbols) == 1:
            return self.propose_watchlist_remove(symbols[0])
        absent = [symbol for symbol in symbols if symbol not in self.watchlist.items()]
        if absent:
            return ChangeResult(
                None,
                f"Not on the operator watchlist: {_format_symbols(absent)}. No change staged.",
            )
        return ChangeResult(self._stage("watchlist_remove_many", _watchlist_payload(symbols)), "")

    def propose_undo_last_watchlist_change(self) -> ChangeResult:
        """Stage the inverse of the last applied watchlist change.

        Undo is another proposal, not a magical direct rollback.  That keeps
        a stray "undo" message from changing the list and leaves a normal
        audit event linking the reversal to the original proposal.
        """
        try:
            lines = self.audit_path.read_text().splitlines()
        except FileNotFoundError:
            lines = []
        original: DeskProposal | None = None
        for line in reversed(lines):
            try:
                record = json.loads(line)
                proposal = DeskProposal.from_dict(record.get("proposal"))
            except (AttributeError, json.JSONDecodeError):
                continue
            if (
                record.get("event") == "applied"
                and proposal is not None
                and proposal.kind
                in {
                    "watchlist_add",
                    "watchlist_remove",
                    "watchlist_add_many",
                    "watchlist_remove_many",
                }
            ):
                original = proposal
                break
        if original is None:
            return ChangeResult(None, "There is no applied watchlist change to undo.")
        symbols = _proposal_symbols(original.payload)
        items = set(self.watchlist.items())
        if original.kind in {"watchlist_add", "watchlist_add_many"}:
            absent = [symbol for symbol in symbols if symbol not in items]
            if absent:
                return ChangeResult(
                    None,
                    f"Already absent: {_format_symbols(absent)}; that change cannot be undone.",
                )
            kind: ChangeKind = "watchlist_remove" if len(symbols) == 1 else "watchlist_remove_many"
        else:
            present = [symbol for symbol in symbols if symbol in items]
            if present:
                return ChangeResult(
                    None,
                    f"Already present: {_format_symbols(present)}; that change cannot be undone.",
                )
            kind = "watchlist_add" if len(symbols) == 1 else "watchlist_add_many"
        return ChangeResult(
            self._stage(kind, _watchlist_payload(symbols, undo_of=original.id)),
            "",
        )

    def propose_lesson_create(self, statement: str) -> ChangeResult:
        statement = " ".join(statement.split())
        if len(statement) < 15:
            return ChangeResult(
                None, "That is too short to be a lesson — say what to do and when it applies."
            )
        if len(statement) > 1_500:
            return ChangeResult(None, "A lesson must be 1,500 characters or fewer.")
        from trading.copilot.mandates import STRONG, grade_strength

        strength = grade_strength(statement)
        status = "established" if strength == STRONG else "candidate"
        return ChangeResult(
            self._stage(
                "lesson_create",
                {"statement": statement, "strength": strength, "status": status},
            ),
            "",
        )

    def _operator_lesson(self, lesson_id: str) -> dict[str, str] | None:
        mem = MemoryStore(self.state_dir / "memory")
        try:
            row = _lesson_snapshot(mem, lesson_id)
            if row is None or "operator" not in row["tags"]:
                return None
            return row
        finally:
            mem.close()

    def propose_lesson_supersede(self, lesson_id: str, statement: str) -> ChangeResult:
        old = self._operator_lesson(lesson_id)
        if old is None or old["status"] == "retired":
            return ChangeResult(None, f"`{lesson_id}` is not an active operator lesson.")
        statement = " ".join(statement.split())
        if len(statement) < 15:
            return ChangeResult(None, "The replacement lesson is too short.")
        if len(statement) > 1_500:
            return ChangeResult(None, "A lesson must be 1,500 characters or fewer.")
        from trading.copilot.mandates import STRONG, grade_strength

        strength = grade_strength(statement)
        status = "established" if strength == STRONG else "candidate"
        return ChangeResult(
            self._stage(
                "lesson_supersede",
                {
                    "lesson_id": old["id"],
                    "old_statement": old["statement"],
                    "statement": statement,
                    "strength": strength,
                    "status": status,
                },
            ),
            "",
        )

    def propose_lesson_status(self, lesson_id: str, status: str) -> ChangeResult:
        if status not in {"candidate", "established"}:
            return ChangeResult(None, "Lesson status must be candidate or established.")
        mem = MemoryStore(self.state_dir / "memory")
        try:
            row = _lesson_snapshot(mem, lesson_id)
            if row is None or row["status"] == "retired":
                return ChangeResult(None, f"No live lesson `{lesson_id}`.")
            if row["status"] == status:
                return ChangeResult(None, f"`{lesson_id}` is already {status}.")
            old_status = str(row["status"])
        finally:
            mem.close()
        return ChangeResult(
            self._stage(
                "lesson_status",
                {"lesson_id": lesson_id, "old_status": old_status, "status": status},
            ),
            "",
        )

    def propose_lesson_archive(self, lesson_id: str) -> ChangeResult:
        old = self._operator_lesson(lesson_id)
        if old is None or old["status"] == "retired":
            return ChangeResult(None, f"`{lesson_id}` is not an active operator lesson.")
        return ChangeResult(
            self._stage(
                "lesson_archive",
                {"lesson_id": old["id"], "statement": old["statement"]},
            ),
            "",
        )

    def pending(self, proposal_id: str | None = None) -> DeskProposal | None:
        now = self.clock.now()
        rows = self._all()
        changed = False
        live: list[DeskProposal] = []
        for p in rows:
            if p.status == "pending" and p.is_expired(now):
                p = replace(p, status="expired")
                self._audit("expired", p)
                changed = True
            live.append(p)
        if changed:
            self._save(live)
        candidates = [p for p in live if p.status == "pending"]
        if proposal_id is not None:
            return next((p for p in candidates if p.id == proposal_id), None)
        return max(candidates, key=lambda p: p.requested_at, default=None)

    def cancel(self, proposal_id: str | None = None) -> ChangeResult:
        try:
            proposal = self.pending(proposal_id)
            if proposal is None:
                return ChangeResult(None, "No pending desk change to cancel.")
            self._replace(proposal, "cancelled")
            self._audit("cancelled", replace(proposal, status="cancelled"))
            return ChangeResult(None, f"Cancelled proposed change `{proposal.id}`.")
        except (OSError, ValueError) as e:
            _LOG.warning(f"could not cancel desk change: {e}")
            return ChangeResult(
                None, f"Could not cancel the desk change: {e}. No change was applied."
            )

    def approve(self, proposal_id: str | None = None) -> ChangeResult:
        try:
            proposal = self.pending(proposal_id)
        except (OSError, ValueError, sqlite3.Error) as e:
            _LOG.warning(f"could not read desk change for approval: {e}")
            return ChangeResult(
                None, f"Could not read the desk change: {e}. No change was applied."
            )
        if proposal is None:
            return ChangeResult(None, "That desk-change proposal is no longer pending.")

        try:
            message, detail = self._apply(proposal)
        except (OSError, ValueError, sqlite3.Error) as e:
            # A lesson operation is backed by both SQLite and a markdown
            # card, so a storage error can arrive after part of that write.
            # Do not invite a blind retry: it could make a duplicate lesson.
            _LOG.warning(f"desk change {proposal.id} could not be completed: {e}")
            return ChangeResult(
                None,
                f"Could not complete `{proposal.id}`: {e}. It may have changed state; do not approve it again.",
            )

        try:
            applied = replace(proposal, status="applied")
            self._replace(proposal, "applied")
        except (OSError, ValueError) as e:
            # The mutation has returned successfully, but the durable
            # proposal status has not.  State this plainly rather than
            # claiming the mutation failed and risking a duplicate retry.
            _LOG.error(f"desk change {proposal.id} applied but could not finalize ledger: {e}")
            return ChangeResult(
                None,
                f"`{proposal.id}` was applied, but its proposal ledger could not be finalized: {e}. Do not approve it again.",
            )

        self._audit("applied", applied, **detail)
        return ChangeResult(None, message)

    def _replace(self, proposal: DeskProposal, status: str) -> None:
        rows = [replace(p, status=status) if p.id == proposal.id else p for p in self._all()]
        self._save(rows)

    def _apply(self, proposal: DeskProposal) -> tuple[str, dict[str, Any]]:
        p = proposal.payload
        if proposal.kind in {"watchlist_add", "watchlist_add_many"}:
            symbols = _proposal_symbols(p)
            before = self.watchlist.items()
            changed, already_present = self.watchlist.add_many(symbols)
            after = self.watchlist.items()
            if not changed:
                return f"Already on the operator watchlist: {_format_symbols(already_present)}.", {
                    "old_value": before,
                    "new_value": after,
                }
            verb = "Restored" if p.get("undo_of") else "Added"
            undo = f" Reverted `{p['undo_of']}`." if p.get("undo_of") else ""
            if len(symbols) > 1:
                return (
                    f"✅ {verb} {len(symbols)} names to the operator watchlist: "
                    f"{_format_symbols(symbols)}.{undo} No trading action was taken.",
                    {"old_value": before, "new_value": after},
                )
            return (
                f"✅ {verb} `{symbols[0]}` on the operator watchlist.{undo} "
                "No trading action was taken.",
                {"old_value": before, "new_value": after},
            )
        if proposal.kind in {"watchlist_remove", "watchlist_remove_many"}:
            symbols = _proposal_symbols(p)
            before = self.watchlist.items()
            changed, absent = self.watchlist.remove_many(symbols)
            after = self.watchlist.items()
            if not changed:
                return f"Already absent from the operator watchlist: {_format_symbols(absent)}.", {
                    "old_value": before,
                    "new_value": after,
                }
            verb = "Removed to undo" if p.get("undo_of") else "Removed"
            undo = f" `{p['undo_of']}` was reverted." if p.get("undo_of") else ""
            if len(symbols) > 1:
                return (
                    f"✅ {verb} {len(symbols)} names from the operator watchlist: "
                    f"{_format_symbols(symbols)}.{undo} No trading action was taken.",
                    {"old_value": before, "new_value": after},
                )
            return (
                f"✅ {verb} `{symbols[0]}` from the operator watchlist.{undo} "
                "No trading action was taken.",
                {"old_value": before, "new_value": after},
            )

        mem = MemoryStore(self.state_dir / "memory")
        try:
            if proposal.kind == "lesson_create":
                lid = mem.add_lesson(
                    p["statement"], tags=f"operator {p['strength']}", status=p["status"]
                )
                return (
                    f"✅ Lesson `{lid}` added as *{p['status']}*.",
                    {"old_value": None, "new_value": {"id": lid, "statement": p["statement"]}},
                )
            if proposal.kind == "lesson_supersede":
                old = _lesson_snapshot(mem, p["lesson_id"])
                if old is None or "operator" not in old["tags"] or old["status"] == "retired":
                    raise ValueError(f"`{p['lesson_id']}` is no longer an active operator lesson")
                lid = mem.add_lesson(
                    p["statement"], tags=f"operator {p['strength']}", status=p["status"]
                )
                mem.retire_lesson(p["lesson_id"], f"Superseded by {lid} via Telegram approval")
                return (
                    f"✅ Lesson `{p['lesson_id']}` superseded by `{lid}` ({p['status']}).",
                    {
                        "old_value": {"id": p["lesson_id"], "statement": p["old_statement"]},
                        "new_value": {"id": lid, "statement": p["statement"]},
                    },
                )
            if proposal.kind == "lesson_status":
                lesson_before = _lesson_snapshot(mem, p["lesson_id"])
                if lesson_before is None or lesson_before["status"] == "retired":
                    raise ValueError(f"`{p['lesson_id']}` is no longer a live lesson")
                if not mem.set_lesson_status(p["lesson_id"], p["status"]):
                    raise ValueError(f"could not update `{p['lesson_id']}`")
                return (
                    f"✅ Lesson `{p['lesson_id']}` is now *{p['status']}*.",
                    {"old_value": lesson_before["status"], "new_value": p["status"]},
                )
            if proposal.kind == "lesson_archive":
                old = _lesson_snapshot(mem, p["lesson_id"])
                if old is None or "operator" not in old["tags"] or old["status"] == "retired":
                    raise ValueError(f"`{p['lesson_id']}` is no longer an active operator lesson")
                mem.retire_lesson(p["lesson_id"], "Archived by operator via Telegram approval")
                return (
                    f"✅ Archived operator lesson `{p['lesson_id']}`; its history remains in memory.",
                    {
                        "old_value": {"id": old["id"], "statement": old["statement"]},
                        "new_value": "retired",
                    },
                )
        finally:
            mem.close()
        raise ValueError(f"unsupported desk change {proposal.kind}")


def describe_proposal(proposal: DeskProposal) -> str:
    """Human preview of the exact change, before any mutation happens."""
    p = proposal.payload
    if proposal.kind in {"watchlist_add", "watchlist_add_many"}:
        symbols = _proposal_symbols(p)
        action = "Restore" if p.get("undo_of") else "Add"
        prior = f" This reverts `{p['undo_of']}`." if p.get("undo_of") else ""
        if len(symbols) == 1:
            body = f"{action} `{symbols[0]}` to the operator watchlist.{prior}\nNo trading action will be taken."
        else:
            body = (
                f"{action} {len(symbols)} names to the operator watchlist: "
                f"{_format_symbols(symbols)}.{prior}\nNo trading action will be taken."
            )
    elif proposal.kind in {"watchlist_remove", "watchlist_remove_many"}:
        symbols = _proposal_symbols(p)
        action = "Remove to undo" if p.get("undo_of") else "Remove"
        prior = f" This reverts `{p['undo_of']}`." if p.get("undo_of") else ""
        if len(symbols) == 1:
            body = f"{action} `{symbols[0]}` from the operator watchlist.{prior}\nNo trading action will be taken."
        else:
            body = (
                f"{action} {len(symbols)} names from the operator watchlist: "
                f"{_format_symbols(symbols)}.{prior}\nNo trading action will be taken."
            )
    elif proposal.kind == "lesson_create":
        body = f"Create an operator lesson as *{p['status']}*:\n{p['statement']}"
    elif proposal.kind == "lesson_supersede":
        body = (
            f"Supersede operator lesson `{p['lesson_id']}` with a new *{p['status']}* lesson:\n"
            f"{p['statement']}\n\nThe prior lesson will be retired, not overwritten."
        )
    elif proposal.kind == "lesson_status":
        body = f"Change lesson `{p['lesson_id']}` from {p['old_status']} to *{p['status']}*."
    else:
        body = f"Archive operator lesson `{p['lesson_id']}`. Its historical card and audit trail remain."
    return (
        f"📋 Proposed desk change `{proposal.id}`\n{body}\n\n"
        "Tap Approve or Cancel. This proposal expires in 10 minutes."
    )


__all__ = [
    "ChangeResult",
    "DeskChangeStore",
    "DeskProposal",
    "WatchlistStore",
    "describe_proposal",
]
