from datetime import datetime, timedelta, timezone

import pytest

from trading.agents.positioning import check_positioning, shadow_positioning
from trading.data.positioning import cot_features, expected_cot_date, gamma_features
from trading.runtime.positioning_monitor import (
    import_goldman,
    read_positioning,
    refresh_positioning,
)

NOW = datetime(2026, 11, 6, 21, tzinfo=timezone.utc)


def cot_rows(count=30):
    return [
        {
            "cftc_contract_market_code": "13874A",
            "report_date_as_yyyy_mm_dd": (datetime(2026, 11, 3) - timedelta(weeks=i))
            .date()
            .isoformat(),
            "open_interest_all": 1000,
            "asset_mgr_positions_long": 600 - i,
            "asset_mgr_positions_short": 100,
            "lev_money_positions_long": 50,
            "lev_money_positions_short": 400,
        }
        for i in range(count)
    ]


def test_cot_categories_deltas_history_and_real_availability():
    am, lm = cot_features(cot_rows(), "13874A", NOW)
    assert am.facts["net_oi"] == 0.5
    assert lm.facts["net_oi"] == -0.35
    assert am.facts["delta_1w"] == pytest.approx(0.001)
    assert am.facts["delta_4w"] == pytest.approx(0.004)
    assert am.facts["percentile"] == 1
    assert am.available_at == NOW > am.observed_at
    assert "MPC" not in am.symbols
    rows = cot_rows(2)
    rows.pop(1)
    only = cot_features(rows, "13874A", NOW)[0]
    assert only.facts["delta_1w"] is None and only.facts["percentile"] is None


@pytest.mark.parametrize("change", ["duplicate", "mixed", "nan", "negative"])
def test_bad_cot_rejected(change):
    rows = cot_rows()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "mixed":
        rows[0]["cftc_contract_market_code"] = "OTHER"
    elif change == "nan":
        rows[0]["open_interest_all"] = "nan"
    else:
        rows[0]["asset_mgr_positions_long"] = -1
    with pytest.raises(ValueError):
        cot_features(rows, "13874A", NOW)


def test_expected_release_dst_and_before_publication():
    assert expected_cot_date(NOW.replace(hour=20, minute=29)) == "2026-10-27"
    assert expected_cot_date(NOW.replace(hour=20, minute=30)) == "2026-11-03"
    assert expected_cot_date(datetime(2026, 10, 30, 19, 30, tzinfo=timezone.utc)) == "2026-10-27"


def option_rows():
    return [
        {
            "strike": 95 + i,
            "impliedVolatility": 0.2,
            "openInterest": 100,
            "expiry": "2026-11-13",
            "lastTradeDate": NOW.isoformat(),
            "contractSize": "REGULAR",
            "side": side,
        }
        for side in ("call", "put")
        for i in range(11)
    ]


def test_gamma_sensitivity_is_not_dealer_inventory():
    obs = gamma_features("SPY", 100, option_rows(), NOW)
    assert obs.kind == "estimated"
    assert obs.facts["unsigned_gex_per_1pct"] > 0
    assert obs.facts["call_positive_put_negative"] == pytest.approx(0)
    assert obs.facts["all_dealers_short"] == -obs.facts["all_dealers_long"]
    assert "OI as-of unknown" in obs.note
    assert obs.expires_at - NOW == timedelta(minutes=30)


@pytest.mark.parametrize(
    "field,value",
    [
        ("openInterest", None),
        ("lastTradeDate", "2026-10-01T00:00:00+00:00"),
        ("expiry", "2026-11-06"),
        ("contractSize", "MINI"),
    ],
)
def test_bad_gamma_coverage_rejected(field, value):
    with pytest.raises(ValueError):
        gamma_features("SPY", 100, [{**r, field: value} for r in option_rows()], NOW)


def mock_sources(monkeypatch):
    monkeypatch.setattr(
        "trading.runtime.positioning_monitor.fetch_cot",
        lambda now: cot_features(cot_rows(), "13874A", now),
    )
    monkeypatch.setattr(
        "trading.runtime.positioning_monitor.fetch_gamma",
        lambda symbol, now: gamma_features(symbol, 100, option_rows(), now),
    )


def test_refresh_archive_first_seen_expiry_failure_and_watchdog(tmp_path, monkeypatch):
    mock_sources(monkeypatch)
    first = refresh_positioning(tmp_path, NOW)
    assert first["coverage"] == {"cot": 2, "goldman_public": 0, "gamma_estimate": 2}
    refresh_positioning(tmp_path, NOW + timedelta(hours=2))
    cot = [
        r
        for r in read_positioning(tmp_path, NOW + timedelta(hours=2))["observations"]
        if r["source"] == "cot"
    ]
    assert all(r["available_at"] == NOW.isoformat().replace("+00:00", "Z") for r in cot)

    def fail(now):
        raise ValueError("offline")

    monkeypatch.setattr("trading.runtime.positioning_monitor.fetch_cot", fail)
    failed = refresh_positioning(tmp_path, NOW + timedelta(days=7))
    assert not failed["coverage"]["cot"] and "cot" in failed["errors"]
    assert read_positioning(tmp_path, NOW + timedelta(days=11))["watchdog"] == "collector_overdue"
    assert not read_positioning(tmp_path, NOW - timedelta(seconds=1))["observations"]


def report():
    return {
        "source_url": "https://www.goldmansachs.com/insights/example",
        "verified_against_source": True,
        "report_id": "original-note-20261103",
        "published_at": "2026-11-05T12:00:00+00:00",
        "observed_at": "2026-11-03T12:00:00+00:00",
        "scope": "US technology client sample",
        "symbols": ["QQQ"],
        "excerpt": "Sample observation",
        "units": "ordinal rank",
        "facts": {"rank": 1},
    }


def test_goldman_dedupes_original_note_and_expires(tmp_path, monkeypatch):
    mock_sources(monkeypatch)
    original = report()
    key = import_goldman(tmp_path, original, NOW)
    assert (
        import_goldman(
            tmp_path, {**original, "source_url": original["source_url"] + "-repeat"}, NOW
        )
        == key
    )
    assert len(list((tmp_path / "positioning/reports").glob("*.json"))) == 1
    assert refresh_positioning(tmp_path, NOW)["coverage"]["goldman_public"] == 1
    assert read_positioning(tmp_path, NOW + timedelta(days=20))["coverage"]["goldman_public"] == 0
    with pytest.raises(ValueError):
        import_goldman(
            tmp_path, {**original, "source_url": "https://goldmansachs.com.evil.com/a"}, NOW
        )
    with pytest.raises(ValueError):
        import_goldman(tmp_path, {**original, "verified_against_source": False}, NOW)


def test_contract_missing_wrong_scope_and_available_evidence():
    rows = [
        {
            "symbol": "MPC",
            "source": source,
            "assessment": "unavailable",
            "effect": "none",
            "reason": "No mapped source",
            "evidence_ids": [],
        }
        for source in ("cot", "goldman_public", "gamma_estimate")
    ]
    assert not check_positioning({"positioning_assessments": rows}, {"MPC"}, {})
    assert check_positioning({}, {"MPC"}, {})
    catalog = {
        "cot:x": {"positioning_source": "cot", "symbols": ["USO"], "requires_revalidation": False}
    }
    rows[0].update(assessment="support", evidence_ids=["cot:x"], effect="rank")
    assert any(
        "wrong-scope" in i
        for i in check_positioning({"positioning_assessments": rows}, {"MPC"}, catalog)
    )
    catalog["cot:x"]["symbols"] = ["MPC"]
    assert not check_positioning({"positioning_assessments": rows}, {"MPC"}, catalog)
    rows[0]["assessment"] = "unavailable"
    assert check_positioning({"positioning_assessments": rows}, {"MPC"}, catalog)


def test_shadow_bounded_no_stock_proxy_or_forced_replacement():
    obs = cot_features(cot_rows(), "13874A", NOW)
    weights = {"SPY": 0.2, "MPC": 0.1}
    shadow = shadow_positioning(weights, {"observations": [r.model_dump(mode="json") for r in obs]})
    assert shadow["arms"]["experimental_positioning_cot"] == {
        "SPY": pytest.approx(0.18),
        "MPC": 0.1,
    }
    assert weights == {"SPY": 0.2, "MPC": 0.1}
    assert shadow_positioning(weights, {})["arms"]["experimental_positioning_cot"] == weights


def test_market_context_must_be_reviewed_but_cannot_be_stock_flow():
    rows = [
        {
            "symbol": "TMO",
            "source": s,
            "assessment": "unavailable",
            "effect": "none",
            "reason": "Absent",
            "evidence_ids": [],
        }
        for s in ("cot", "goldman_public", "gamma_estimate")
    ]
    catalog = {
        "cot:market": {
            "positioning_source": "cot",
            "symbols": ["SPY"],
            "requires_revalidation": False,
        }
    }
    assert check_positioning({"positioning_assessments": rows}, {"TMO"}, catalog)
    rows[0].update(
        assessment="neutral", effect="timing", scope="market_context", evidence_ids=["cot:market"]
    )
    assert not check_positioning({"positioning_assessments": rows}, {"TMO"}, catalog)
    rows[0]["effect"] = "rank"
    assert any(
        "stock-specific" in issue
        for issue in check_positioning({"positioning_assessments": rows}, {"TMO"}, catalog)
    )


def test_goldman_revision_cannot_silently_overwrite(tmp_path):
    original = report()
    import_goldman(tmp_path, original, NOW)
    with pytest.raises(ValueError, match="revision"):
        import_goldman(tmp_path, {**original, "facts": {"rank": 2}}, NOW)


def test_malformed_assessment_does_not_crash_review():
    assert check_positioning(
        {"positioning_assessments": [{"symbol": [], "source": "cot"}]}, {"TMO"}, {}
    )


def test_watchdog_does_not_expect_weekend_poll(tmp_path, monkeypatch):
    mock_sources(monkeypatch)
    refresh_positioning(tmp_path, NOW)
    assert read_positioning(tmp_path, NOW + timedelta(days=2))["watchdog"] == "ok"


def test_refresh_holds_shared_process_lock_and_releases_it(tmp_path, monkeypatch):
    import fcntl

    mock_sources(monkeypatch)

    def checked_fetch(now):
        with (tmp_path / ".positioning.lock").open("a") as competing, pytest.raises(BlockingIOError):
            fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return cot_features(cot_rows(), "13874A", now)

    monkeypatch.setattr("trading.runtime.positioning_monitor.fetch_cot", checked_fetch)
    assert refresh_positioning(tmp_path, NOW)["coverage"]["cot"] == 2
    with (tmp_path / ".positioning.lock").open("a") as competing:
        fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
