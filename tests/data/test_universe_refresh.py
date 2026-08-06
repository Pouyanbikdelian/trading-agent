"""Universe refresh — importable, and writable where it actually runs.

Two bugs are pinned here, both of the same family: code that looks
scheduled but cannot possibly work in the environment it runs in.

1. The generated file was written to ``config/``, which is bind-mounted
   READ-ONLY into the container. Two months stale, silently.
2. The fix put the refresh in ``scripts/`` and had the runner shell out
   to ``/app/scripts/refresh_universes.py`` — a path that does not exist
   in the image, because the Dockerfile copies ``src/`` and not
   ``scripts/``. It would have raised FileNotFoundError every Sunday
   inside a broad ``except``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading.data import universe_refresh


def test_module_is_importable_from_src() -> None:
    """The whole point: the runner must be able to import this, because
    only ``src/`` ships in the Docker image."""
    from trading.data.universe_refresh import refresh

    assert callable(refresh)


class TestOutPath:
    def test_defaults_into_state_not_config(self, monkeypatch) -> None:
        """``config/`` is read-only in the container; ``state/`` is a
        writable volume."""
        path = universe_refresh.out_path()
        assert path.name == "universes.generated.yaml"
        assert "config" not in path.parts

    def test_env_override_wins(self, monkeypatch, tmp_path: Path) -> None:
        target = tmp_path / "custom.yaml"
        monkeypatch.setenv("UNIVERSES_GENERATED_PATH", str(target))
        assert universe_refresh.out_path() == target


class TestNormalize:
    def test_dots_become_dashes_for_yfinance(self) -> None:
        assert universe_refresh._normalize(["BRK.B", "BF.B"]) == ["BF-B", "BRK-B"]

    def test_deduplicates_and_sorts(self) -> None:
        assert universe_refresh._normalize(["msft", "MSFT", "aapl"]) == ["AAPL", "MSFT"]

    def test_blanks_dropped(self) -> None:
        assert universe_refresh._normalize(["", "  ", "AAPL"]) == ["AAPL"]


class TestRefresh:
    def test_writes_a_loadable_file(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(universe_refresh, "fetch_sp500", lambda: [f"S{i}" for i in range(450)])
        monkeypatch.setattr(
            universe_refresh, "fetch_nasdaq100", lambda: [f"N{i}" for i in range(90)]
        )
        monkeypatch.setattr(
            universe_refresh, "fetch_russell1000", lambda: [f"R{i}" for i in range(800)]
        )
        target = tmp_path / "universes.generated.yaml"

        result = universe_refresh.refresh(target)

        assert result["ok"] is True
        assert result["counts"] == {"sp500": 450, "nasdaq100": 90, "russell1000": 800}
        doc = yaml.safe_load(target.read_text())
        assert set(doc["universes"]) == {"sp500", "nasdaq100", "russell1000"}
        assert doc["universes"]["sp500"]["asset_class"] == "equity"

    def test_refuses_to_write_a_truncated_index(self, monkeypatch, tmp_path: Path) -> None:
        """Better to keep last week's list than to ship a five-symbol
        'S&P 500' because Wikipedia renamed a column."""
        monkeypatch.setattr(universe_refresh, "fetch_sp500", lambda: ["AAPL", "MSFT"])
        monkeypatch.setattr(
            universe_refresh, "fetch_nasdaq100", lambda: [f"N{i}" for i in range(90)]
        )
        target = tmp_path / "universes.generated.yaml"

        result = universe_refresh.refresh(target)

        assert result["ok"] is False and "refusing" in result["reason"]
        assert not target.exists()  # the old file, wherever it is, survives

    def test_russell_is_best_effort(self, monkeypatch, tmp_path: Path) -> None:
        """Its Wikipedia page is less rigorously maintained; its absence
        must not block the primary indices."""
        monkeypatch.setattr(universe_refresh, "fetch_sp500", lambda: [f"S{i}" for i in range(450)])
        monkeypatch.setattr(
            universe_refresh, "fetch_nasdaq100", lambda: [f"N{i}" for i in range(90)]
        )

        def boom() -> list[str]:
            raise RuntimeError("table moved")

        monkeypatch.setattr(universe_refresh, "fetch_russell1000", boom)
        target = tmp_path / "universes.generated.yaml"

        result = universe_refresh.refresh(target)

        assert result["ok"] is True
        assert set(yaml.safe_load(target.read_text())["universes"]) == {"sp500", "nasdaq100"}

    def test_network_failure_returns_not_raises(self, monkeypatch, tmp_path: Path) -> None:
        """A scheduled job must degrade, and say so."""

        def boom() -> list[str]:
            raise OSError("wikipedia unreachable")

        monkeypatch.setattr(universe_refresh, "fetch_sp500", boom)
        result = universe_refresh.refresh(tmp_path / "x.yaml")
        assert result["ok"] is False and "fetch failed" in result["reason"]


def test_loader_prefers_state_over_the_legacy_config_copy(monkeypatch, tmp_path: Path) -> None:
    """Migration safety: an un-migrated checkout keeps working, but once
    state/ exists it wins."""
    from trading.core import universes as u

    legacy = tmp_path / "config.yaml"
    fresh = tmp_path / "state.yaml"
    legacy.write_text(
        yaml.safe_dump({"universes": {"sp500": {"asset_class": "equity", "symbols": ["OLD"]}}})
    )
    fresh.write_text(
        yaml.safe_dump({"universes": {"sp500": {"asset_class": "equity", "symbols": ["NEW"]}}})
    )

    monkeypatch.setattr(u, "GENERATED_UNIVERSES_PATH", legacy)
    monkeypatch.setattr(u, "_generated_paths", lambda: (fresh, legacy))
    monkeypatch.setattr(u, "DEFAULT_UNIVERSES_PATH", tmp_path / "absent.yaml")
    u.clear_cache()
    try:
        assert [i.symbol for i in u.load_universe("sp500")] == ["NEW"]
    finally:
        u.clear_cache()


@pytest.mark.slow
def test_live_fetch_smoke() -> None:
    """Network. Only meaningful as a canary that Wikipedia's tables have
    not moved — run deliberately, never in CI."""
    assert len(universe_refresh.fetch_sp500()) >= universe_refresh.MIN_SP500
