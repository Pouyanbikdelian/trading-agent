"""The Macro, Rotation and Economy views restored 2026-09-27.

Dashboard v2 shrank the v1 Macro tab to sparkline tiles, folded the
Economy tab into one table and dropped the animated rotation graph, the
money map, the investment clock and the radar. Yan used them; they are
back. These pins keep them from quietly disappearing again, and pin the
two data fixes that came with them:

* the regime box read a sample-data shape, so on live data it showed no
  colours and no playbook;
* the macro dial and vol surface came only through the agents' context,
  which drops readings older than 36 h — blank every weekend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trading.dashboard.app import _PAGE as PAGE
from trading.dashboard.app import build_summary


class TestTheTabsAreBack:
    def test_macro_rotation_and_economy_tabs(self) -> None:
        assert '<button data-tab="market">Macro</button>' in PAGE
        assert '<button data-tab="rotation">Rotation</button>' in PAGE
        assert '<button data-tab="economy">Economy</button>' in PAGE
        assert "rotation:renderRotation,economy:renderEconomy" in PAGE


class TestMacroCharts:
    def test_four_history_charts_with_readings(self) -> None:
        for chart_id in ("curveCh", "vixCh", "brCh", "ratioCh"):
            assert f'id="{chart_id}"' in PAGE
            assert f"mline('{chart_id}'" in PAGE
        for read_id in ("curveInt", "vixInt", "brInt", "ratioInt", "macroInt", "volInt"):
            assert f'id="{read_id}"' in PAGE

    def test_dial_and_vol_surface_read_the_monitors_with_their_age(self) -> None:
        assert "(D.monitors||{}).macro" in PAGE
        assert "(D.monitors||{}).options" in PAGE
        assert "const monAge=" in PAGE


class TestRotation:
    def test_animated_rrg_with_controls(self) -> None:
        for el in ('id="rrg"', 'id="rrgPlay"', 'id="rrgScrub"', 'id="rrgRange"', 'id="rrgTip"'):
            assert el in PAGE
        assert "requestAnimationFrame(rrgTick)" in PAGE

    def test_a_refresh_does_not_reset_playback(self) -> None:
        assert "if(RR.playing)return;" in PAGE

    def test_money_map_clock_and_radar(self) -> None:
        assert "function renderMoneyMap(){" in PAGE
        assert "function renderRegimeClock(){" in PAGE
        assert "function renderRadar(){" in PAGE
        assert 'class="qboard"' in PAGE

    def test_regime_box_reads_the_real_playbook_shape(self) -> None:
        assert "const play=pb[cur.r]||{};const favors=(play.favors||[])" in PAGE
        assert "(pb[h.r]||{}).color" in PAGE


class TestEconomy:
    def test_seven_fred_panels_with_a_range(self) -> None:
        for chart_id in ("ecInf", "ecIcx", "ecPol", "ecHou", "ecLab", "ecCr", "ecCo"):
            assert f'id="{chart_id}"' in PAGE
        assert 'id="ecRange"' in PAGE


def test_summary_carries_monitor_readings_even_when_stale(tmp_path) -> None:
    state, data = tmp_path / "state", tmp_path / "data"
    state.mkdir()
    data.mkdir()
    old = datetime(2026, 9, 25, 20, 0, tzinfo=timezone.utc).isoformat()
    (state / "macro_monitor.json").write_text(
        json.dumps({"readings": {"composite": -0.4, "rates_shock_z": -1.3}, "last_polled_at": old})
    )
    (state / "options_monitor.json").write_text(
        json.dumps({"metrics": {"atm_iv": 0.16}, "last_polled_at": old, "active": ["skew"]})
    )
    out = build_summary(state, data)
    assert out["monitors"]["macro"]["values"]["composite"] == -0.4
    assert out["monitors"]["macro"]["as_of"] == old
    assert out["monitors"]["options"]["values"]["atm_iv"] == 0.16
    assert out["monitors"]["options"]["active"] == ["skew"]


def test_summary_monitors_degrade_to_empty(tmp_path) -> None:
    out = build_summary(tmp_path / "nope", tmp_path / "nada")
    assert out["monitors"] == {"macro": {}, "options": {}}
