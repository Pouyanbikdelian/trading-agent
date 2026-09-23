"""The live bot once served old accounting code while the checkout looked fixed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_deployment.py"
_SPEC = importlib.util.spec_from_file_location("verify_deployment", _PATH)
assert _SPEC and _SPEC.loader
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)

_IMAGE = "sha256:" + "a" * 64
_OLD_IMAGE = "sha256:" + "b" * 64
_ID = "c" * 64


@pytest.fixture
def checkout(tmp_path):
    for directory in ("src/trading", "config", "docker"):
        (tmp_path / directory).mkdir(parents=True)
    for path, contents in {
        "src/trading/__init__.py": "# package\n",
        "src/trading/bot.py": "# corrected accounting\n",
        "config/risk.yaml": "max_drawdown_pct: 0.02\n",
        "docker/healthcheck.py": "# probe\n",
        "pyproject.toml": "[project]\n",
    }.items():
        (tmp_path / path).write_text(contents)
    return tmp_path


def _fake_docker(monkeypatch, checkout, *, changed=None, image_id=_IMAGE, origin=None):
    manifest = verifier._manifest(checkout)
    if changed:
        changed(manifest)
    calls = []
    metadata = {
        "id": _ID,
        "image_id": image_id,
        "image_ref": "trading-agent:latest",
        "status": "running",
        "started_at": "2026-09-20T00:00:00Z",
        "restarts": 0,
        "health": "healthy",
    }

    def run(args):
        calls.append(args)
        if args[:2] == ["docker", "inspect"]:
            return json.dumps(metadata)
        if args[:3] == ["docker", "image", "inspect"]:
            return _IMAGE
        assert args[:6] == ["docker", "exec", _ID, "python", "-B", "-c"]
        return json.dumps(
            {
                "manifest": manifest,
                "origin": origin or "/app/src/trading/__init__.py",
                "locations": ["/app/src/trading"],
            }
        )

    monkeypatch.setattr(verifier, "_run", run)
    return calls, metadata


def test_matching_services_pass_without_environment_or_trading_imports(checkout, monkeypatch):
    calls, _ = _fake_docker(monkeypatch, checkout)
    report, code = verifier.verify(checkout, verifier._DEFAULT_CONTAINERS)
    assert code == 0
    assert report["status"] == "PASS"
    assert len(report["services"]) == 3
    assert all(".Env" not in " ".join(c) for c in calls)
    assert all(c[1] in {"inspect", "exec", "image"} for c in calls)
    assert "import trading" not in verifier._probe()
    assert report["services_share_image"]


def test_stale_bot_source_fails_even_when_latest_tag_matches(checkout, monkeypatch):
    _fake_docker(
        monkeypatch, checkout, changed=lambda m: m.update({"src/trading/bot.py": "0" * 64})
    )
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 1
    assert report["services"][0]["changed"] == ["src/trading/bot.py"]


def test_changed_tag_is_drift_even_with_matching_source(checkout, monkeypatch):
    _fake_docker(monkeypatch, checkout, image_id=_OLD_IMAGE)
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 1
    assert not report["services"][0]["image_tag_matches"]


def test_missing_and_extra_runtime_files_are_detected(checkout, monkeypatch):
    def change(manifest):
        del manifest["docker/healthcheck.py"]
        manifest["src/trading/old.py"] = "1" * 64

    _fake_docker(monkeypatch, checkout, changed=change)
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 1
    assert report["services"][0]["missing"] == ["docker/healthcheck.py"]
    assert report["services"][0]["unexpected"] == ["src/trading/old.py"]


def test_different_python_import_origin_is_drift(checkout, monkeypatch):
    _fake_docker(monkeypatch, checkout, origin="/opt/venv/site-packages/trading/__init__.py")
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 1
    assert not report["services"][0]["package_origin_matches"]


def test_unavailable_service_is_unverifiable(checkout, monkeypatch):
    _, metadata = _fake_docker(monkeypatch, checkout)
    metadata["status"] = "exited"
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 2
    assert report["services"][0]["status"] == "ERROR"


def test_container_recreation_during_probe_is_not_a_pass(checkout, monkeypatch):
    _fake_docker(monkeypatch, checkout)
    run = verifier._run
    count = 0

    def changing(args):
        nonlocal count
        result = run(args)
        if args[:2] == ["docker", "inspect"]:
            count += 1
            if count == 2:
                result = json.dumps({**json.loads(result), "id": "d" * 64})
        return result

    monkeypatch.setattr(verifier, "_run", changing)
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 2
    assert "recheck container identity" in report["services"][0]["error"]


def test_checkout_change_during_verification_is_not_a_pass(checkout, monkeypatch):
    _fake_docker(monkeypatch, checkout)
    run = verifier._run

    def changing(args):
        (checkout / "src/trading/bot.py").write_text("# concurrent edit\n")
        return run(args)

    monkeypatch.setattr(verifier, "_run", changing)
    report, code = verifier.verify(checkout, ("trader-bot",))
    assert code == 2
    assert report["status"] == "ERROR"


def test_manifest_ignores_caches_and_rejects_symlinks(checkout):
    original = verifier._manifest(checkout)
    cache = checkout / "src/trading/__pycache__"
    cache.mkdir()
    (cache / "bot.cpython-312.pyc").write_bytes(b"cache")
    (checkout / "src/.DS_Store").write_bytes(b"macOS")
    assert verifier._manifest(checkout) == original
    (checkout / "src/trading/external.py").symlink_to(checkout / "pyproject.toml")
    with pytest.raises(ValueError):
        verifier._manifest(checkout)


def test_docker_errors_do_not_print_daemon_output_or_credentials(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="PASSWORD=do-not-print")

    monkeypatch.setattr(verifier.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as exc:
        verifier._run(["docker", "inspect", "missing"])
    assert "PASSWORD" not in str(exc.value)


def test_generated_probe_uses_same_manifest_and_never_executes_package(
    checkout, monkeypatch, capsys
):
    # Execute the exact program sent to docker, remapping only /app to a fixture.
    (checkout / "src/trading/__init__.py").write_text("raise AssertionError('imported trading')\n")
    monkeypatch.syspath_prepend(str(checkout / "src"))
    fake_spec = type(
        "Spec",
        (),
        {
            "origin": "/app/src/trading/__init__.py",
            "submodule_search_locations": ["/app/src/trading"],
        },
    )()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: fake_spec)
    code = verifier._probe().replace('Path("/app")', f"Path({str(checkout)!r})")
    exec(compile(code, "<probe>", "exec"), {})
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"] == verifier._manifest(checkout)


def test_cli_rejects_option_injection_before_docker(monkeypatch, capsys, checkout):
    monkeypatch.setattr(verifier, "_run", lambda _: pytest.fail("Docker must not run"))
    assert verifier.main([str(checkout), "--privileged"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"
