#!/usr/bin/env python3
"""Catch stale service images without importing trading code or contacting IBKR.

The September 2026 baseline repair existed in the VPS checkout but not in
the running bot. A Git SHA or the mutable ``latest`` tag cannot prove which
code is serving commands. This stdlib-only host script reads narrowly scoped
Docker metadata and hashes the actual runtime files in every selected service.

Usage: python3 scripts/verify_deployment.py [REPO [CONTAINER ...]]
Defaults: this checkout; trader-live, trader-bot, trader-dashboard.
JSON goes to stdout. Exit 0: matching deployment; 1: drift; 2: unverifiable.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

_DEFAULT_CONTAINERS = ("trader-live", "trader-bot", "trader-dashboard")
_CONTAINER_FORMAT = (
    '{"id":{{json .Id}},"image_id":{{json .Image}},'
    '"image_ref":{{json .Config.Image}},"status":{{json .State.Status}},'
    '"started_at":{{json .State.StartedAt}},'
    '"restarts":{{json .RestartCount}},'
    # Containers started with --no-healthcheck omit Health entirely. Docker's
    # template engine rejects a missing dotted key; index returns an empty value.
    '"health":{{with (index .State "Health")}}{{json .Status}}{{else}}"none"{{end}}}'
)
_HEX256 = re.compile(r"[a-f0-9]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}\Z")


def _manifest(root: Path) -> dict[str, str]:
    """Use the same inclusion rules on the host and in the runtime image."""
    result: dict[str, str] = {}
    for name in ("src", "config", "docker", "pyproject.toml"):
        base = root / name
        if base.is_symlink() or not base.exists():
            raise ValueError("required runtime input missing or symlinked")
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or path.name == ".DS_Store":
                continue
            if path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink():
                raise ValueError("symlinked runtime input cannot be verified")
            if path.is_file():
                result[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "src/trading/__init__.py" not in result:
        raise ValueError("trading package absent from runtime inputs")
    return result


def _fingerprint(manifest: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _probe() -> str:
    # find_spec on this TOP-LEVEL package resolves its origin without executing
    # __init__.py. Importing Settings, CLI, or trading itself is unnecessary.
    return (
        "from pathlib import Path\nimport hashlib, importlib.util, json\n"
        + inspect.getsource(_manifest)
        + '\nspec = importlib.util.find_spec("trading")\n'
        + 'print(json.dumps({"manifest": _manifest(Path("/app")), '
        + '"origin": spec.origin if spec else None, '
        + '"locations": list(spec.submodule_search_locations or []) if spec else []}))\n'
    )


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # Docker errors can include command arguments or arbitrary daemon text;
        # never echo stderr, resolved Compose config, or container environments.
        raise RuntimeError("Docker read failed or timed out") from exc


def _read_metadata(container: str) -> dict[str, object]:
    value = json.loads(_run(["docker", "inspect", "--format", _CONTAINER_FORMAT, container]))
    if not isinstance(value, dict):
        raise ValueError("invalid container metadata")
    if not isinstance(value.get("id"), str) or not _HEX256.fullmatch(value["id"]):
        raise ValueError("invalid container identity")
    if not isinstance(value.get("image_id"), str) or not _IMAGE_ID.fullmatch(value["image_id"]):
        raise ValueError("invalid image identity")
    if not isinstance(value.get("image_ref"), str) or not value["image_ref"]:
        raise ValueError("missing image reference")
    return value


def _read_manifest(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("invalid runtime manifest")
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or not _HEX256.fullmatch(digest)
            or Path(key).is_absolute()
            or ".." in Path(key).parts
            or Path(key).parts[0] not in ("src", "config", "docker", "pyproject.toml")
        ):
            raise ValueError("invalid runtime manifest entry")
    return value


def _verify_container(container: str, expected: dict[str, str]) -> dict[str, object]:
    report: dict[str, object] = {"container": container, "status": "ERROR"}
    stage = "inspect container"
    try:
        before = _read_metadata(container)
        report.update(before)
        if before["status"] != "running":
            raise ValueError("container is not running")
        stage = "resolve image reference"
        tag_id = _run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", str(before["image_ref"])]
        ).strip()
        if not _IMAGE_ID.fullmatch(tag_id):
            raise ValueError("invalid image reference")
        report["current_tag_image_id"] = tag_id
        stage = "hash runtime files"
        # Pin by immutable CONTAINER ID so a concurrent recreation cannot
        # silently redirect exec from the inspected instance to its successor.
        runtime = json.loads(
            _run(["docker", "exec", str(before["id"]), "python", "-B", "-c", _probe()])
        )
        actual = _read_manifest(runtime["manifest"])
        stage = "recheck container identity"
        after = _read_metadata(container)
        if before != after:
            raise ValueError("container changed during verification")
        report["runtime_sha256"] = _fingerprint(actual)
        report["missing"] = sorted(expected.keys() - actual.keys())
        report["unexpected"] = sorted(actual.keys() - expected.keys())
        report["changed"] = sorted(
            k for k in expected.keys() & actual.keys() if expected[k] != actual[k]
        )
        report["package_origin_matches"] = runtime.get(
            "origin"
        ) == "/app/src/trading/__init__.py" and runtime.get("locations") == ["/app/src/trading"]
        report["image_tag_matches"] = before["image_id"] == tag_id
        report["status"] = (
            "PASS"
            if not any(report[k] for k in ("missing", "unexpected", "changed"))
            and report["package_origin_matches"]
            and report["image_tag_matches"]
            else "DRIFT"
        )
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
        report["status"] = "ERROR"
        report["error"] = f"Could not verify deployment: {stage}"
    return report


def verify(repo: Path, containers: tuple[str, ...]) -> tuple[dict[str, object], int]:
    """A changing checkout or unavailable service must never become a pass."""
    report: dict[str, object] = {
        "status": "ERROR",
        "scope": "Runtime src/, config/, docker/, pyproject.toml; container and image identity",
        "limits": (
            "Does not verify installed dependency versions, uv.lock, broker readiness, "
            "risk baselines, or in-memory imported code. Health is reported, not certified. "
            "config/ is a bind mount and reflects the currently visible configuration."
        ),
        "services": [],
    }
    try:
        expected = _manifest(repo)
        report["checkout_sha256"] = _fingerprint(expected)
        report["file_count"] = len(expected)
        services = [_verify_container(name, expected) for name in containers]
        report["services"] = services
        if _manifest(repo) != expected:
            raise ValueError("checkout changed during verification")
    except (OSError, ValueError):
        report["error"] = "Checkout runtime inputs unavailable or changed during verification"
        return report, 2
    if not services or any(s["status"] == "ERROR" for s in services):
        return report, 2
    same_image = len({s["image_id"] for s in services}) == 1
    report["services_share_image"] = same_image
    ok = same_image and all(s["status"] == "PASS" for s in services)
    report["status"] = "PASS" if ok else "DRIFT"
    return report, 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--help"] or args == ["-h"]:
        print(__doc__)
        return 0
    repo = Path(args.pop(0)).resolve() if args else Path(__file__).resolve().parents[1]
    containers = tuple(args) if args else _DEFAULT_CONTAINERS
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", c) for c in containers):
        print(json.dumps({"status": "ERROR", "error": "Invalid container name"}))
        return 2
    report, code = verify(repo, containers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
