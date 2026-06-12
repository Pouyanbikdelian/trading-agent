"""Econ watch — hermetic tests for CSV parsing, transforms, staleness."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.runtime.econ_watch import _parse_csv, _transform, latest_block


def test_parse_csv_skips_missing_observations() -> None:
    text = "DATE,CPIAUCSL\n2024-01-01,300.5\n2024-02-01,.\n2024-03-01,302.1\nbad,row,extra\n"
    rows = _parse_csv(text)
    assert rows == [("2024-01-01", 300.5), ("2024-03-01", 302.1)]


def test_yoy_transform() -> None:
    rows = [("2024-01-01", 100.0), ("2025-01-01", 103.5)]
    pts = _transform(rows, "yoy")
    assert pts == [{"t": "2025-01-01", "v": 3.5}]


def test_tn_transform_scales_millions() -> None:
    pts = _transform([("2025-01-01", 7_500_000.0)], "tn")
    assert pts[0]["v"] == 7.5


def test_latest_block_staleness(tmp_path: Path) -> None:
    payload = {
        "t": datetime.now(tz=timezone.utc).isoformat(),
        "series": {"cpi_yoy": {"label": "CPI", "unit": "%", "latest": 2.9, "points": []}},
    }
    p = tmp_path / "econ_watch.json"
    p.write_text(json.dumps(payload))
    assert latest_block(tmp_path)["cpi_yoy"]["v"] == 2.9
    payload["t"] = (datetime.now(tz=timezone.utc) - timedelta(hours=100)).isoformat()
    p.write_text(json.dumps(payload))
    assert latest_block(tmp_path) == {}
