"""Two dashboard panels added 2026-09-26.

* Agent PM vs SPY — the question the operator actually asks: if every PM
  decision had been executed, would it beat the index, and is the lead more
  than luck? (The statistics are cross-checked against Python in review.)
* Does selection add edge? — the row labelled "ladder" measures the
  MECHANICAL momentum cut (its own top picks vs the ranks below, recorded
  before the PM merge), not the PM; and repeated daily logging makes the
  pick count overstate the sample. Both are now said on the panel.
"""

from __future__ import annotations

from trading.dashboard.app import _PAGE as PAGE


class TestThePmVsSpyPanel:
    def test_the_panel_exists_on_the_portfolio_tab(self) -> None:
        assert 'id="pmVsChart"' in PAGE and 'id="rdPmVs"' in PAGE
        assert "renderPM();renderPmVsSpy();" in PAGE

    def test_it_measures_lead_weeks_tracking_error_and_luck(self) -> None:
        assert "function pmVsSpyStats(rows){" in PAGE
        assert "const t=sd>0?mean/sd*Math.sqrt(n):0;" in PAGE
        assert "pLuck:1-normCdf(t)" in PAGE
        assert "te:sd*Math.sqrt(252)" in PAGE

    def test_todays_intraday_mark_is_excluded(self) -> None:
        assert "if(t<todayISO&&h.equity>0&&h.spy>0)" in PAGE

    def test_it_says_the_simulation_is_frictionless(self) -> None:
        assert "Frictionless simulation" in PAGE


class TestTheEdgePanelSaysWhatItMeasures:
    def test_the_ladder_row_is_named_as_the_momentum_ranking_not_the_pm(self) -> None:
        assert "ladder:['Momentum ranking'" in PAGE
        assert "not traded; tests the ranking" in PAGE
        assert "pm_selection:['Agent PM'" in PAGE

    def test_the_sample_is_counted_in_different_stocks(self) -> None:
        assert "n_taken_symbols" in PAGE
        assert "the number of different stocks is the honest sample size" in PAGE
        assert "small sample" in PAGE
