"""The "strategy race" was comparing three things that were not comparable.

Reported by the operator on 2026-08-19 — "this part is showing stupid and
bullshit" — and he was right on every count:

  * each series was rebased to 100 at its OWN first datapoint, so SPY and
    the PM sim started at 100 in June while the account line started at
    100 in August and the gaps between them measured nothing;
  * the account line is CHF and the other two are USD, unconverted, so
    every USDCHF move rendered as strategy alpha;
  * it was labelled "momentum top-k (live)" while ``STRATEGY_SLEEVE_PCT``
    is 0.0 — the momentum book places no orders, so the line is the whole
    account's NetLiq including the operator's personal positions;
  * capital flows were plotted as performance.

Two of those (common window, FX) were fixed on the Live tab in July 2026
and recorded as done in GO_LIVE.md. The Portfolio tab held an independent
copy that nobody went back for. Hence the shared-helper tests below: the
point is not only that the chart is right today, it is that there is one
place left to get it wrong.

The page is a JS string inlined in Python, so the chart maths cannot be
executed here. Python-side behaviour is tested directly; for the JS the
tests pin the invariants whose absence caused each defect.
"""

from __future__ import annotations

import re

from trading.dashboard.app import _PAGE as APP
from trading.dashboard.app import account_curve_usd

CURVE = [
    {"t": "2026-08-17", "v": 86_000.0},
    {"t": "2026-08-18", "v": 86_500.0},
]


class TestTheAccountCurveReachesTheChartInUSD:
    def test_a_chf_curve_is_converted(self) -> None:
        converted, ok = account_curve_usd(CURVE, "CHF", {"2026-08-17": 0.80, "2026-08-18": 0.80})

        assert ok is True
        assert [round(p["v"], 2) for p in converted] == [107_500.0, 108_125.0]

    def test_a_usd_account_is_passed_through_untouched(self) -> None:
        converted, ok = account_curve_usd(CURVE, "USD", {})

        assert ok is True
        assert converted == CURVE

    def test_a_missing_fx_series_returns_the_raw_curve_and_says_so(self) -> None:
        """Flagged, not silently dropped.

        `convert_curve_to_usd` returns nothing when it has no rate old
        enough, and an account line that quietly vanishes reads as flat.
        Showing francs is acceptable only if the page admits it — that is
        what the second return value drives.
        """
        converted, ok = account_curve_usd(CURVE, "CHF", {})

        assert ok is False
        assert converted == CURVE

    def test_an_fx_series_that_starts_after_the_curve_is_also_flagged(self) -> None:
        converted, ok = account_curve_usd(CURVE, "CHF", {"2099-01-01": 0.8})

        assert ok is False
        assert converted == CURVE

    def test_the_payload_carries_the_converted_curve_and_its_status(self, tmp_path) -> None:
        from trading.dashboard.app import build_summary

        out = build_summary(tmp_path, tmp_path)

        assert "equity_curve_usd" in out
        assert "equity_usd_ok" in out
        assert "equity_currency" in out


class TestThereIsExactlyOneRebase:
    """Three copies is why the July fix landed on one chart out of three."""

    def test_the_shared_helper_exists(self) -> None:
        assert "const rebase100=(series)=>" in APP

    def test_all_three_normalized_charts_go_through_it(self) -> None:
        """Sleeve race, strategy race, holdings — one call each."""
        uses = re.findall(r"rebase100\(", APP)
        assert len(uses) == 3, uses

    def test_no_chart_still_rebases_on_its_own_first_point(self) -> None:
        """The literal defect, in each of its three spellings."""
        assert "const base=pts[0][key]" not in APP
        assert "const base=tser[sym][0].v" not in APP
        assert "base=s.pts[0]" not in APP

    def test_the_common_start_is_the_latest_inception(self) -> None:
        assert "const start=usable.map(s=>s.pts.map(p=>p.t).sort()[0]).sort().slice(-1)[0];" in APP

    def test_a_zero_base_drops_the_series_instead_of_dividing(self) -> None:
        """A wedged IBKR session writes equity=0; one Infinity kills the axis."""
        assert "if(!(base>0)){dropped.push(s.label);return null;}" in APP

    def test_the_page_states_the_rebase_date_and_what_was_dropped(self) -> None:
        """A series used to vanish from the legend with nothing to say it had."""
        assert "const rebaseNote=" in APP
        assert "rebased to 100 at " in APP
        assert 'id="raceNote"' in APP
        assert 'id="lvRaceNote"' in APP
        assert 'id="tickNote"' in APP


class TestTheSeriesSayWhatTheyAre:
    def test_the_account_line_is_not_called_a_strategy(self) -> None:
        """Only the comment explaining the old name may still contain it."""
        code = [line for line in APP.splitlines() if not line.strip().startswith("//")]
        assert not [line for line in code if "momentum top-k" in line]
        assert "account NetLiq (" in APP

    def test_the_race_heading_names_the_currency_and_the_caveats(self) -> None:
        assert "Strategy race · USD, normalized to 100" in APP
        assert "not a strategy sleeve" in APP

    def test_an_unconverted_account_line_is_labelled_on_the_chart(self) -> None:
        assert "UNCONVERTED" in APP


class TestCapitalFlowsAreNotPlottedAsReturns:
    def test_the_race_curves_are_flow_adjusted(self) -> None:
        """A deposit used to read as a vertical rally, +979.92% once."""
        assert "const flowAdjustedCurve=(pts)=>" in APP
        assert "flowAdjustedCurve(acct)" in APP
        assert "flowAdjustedCurve(pm)" in APP

    def test_spy_is_not_flow_adjusted(self) -> None:
        """It is a price index. Flattening a real -25% session would be a lie."""
        assert "flowAdjustedCurve(spy)" not in APP
        assert "{label:'SPY',pts:spy," in APP

    def test_the_threshold_still_matches_the_server(self) -> None:
        from trading.dashboard.live import FLOW_THRESHOLD

        assert f"const FLOW_THRESHOLD={FLOW_THRESHOLD}" in APP


class TestTheSeriesEndOnTheSameDay:
    def test_todays_intraday_point_is_excluded_from_the_pm_marks(self) -> None:
        """The account curve already stopped at yesterday; PM/SPY did not.

        The final gap between them was then an artefact of one curve
        carrying an extra point priced intraday rather than at a close.
        """
        assert "const todayISO=new Date().toISOString().slice(0,10);" in APP
        assert ".filter(h=>h.t<todayISO)" in APP


class TestOneFxFetchPerPayload:
    def test_the_live_tab_is_handed_the_rate_the_portfolio_tab_used(self) -> None:
        """Two tabs fetching seconds apart is its own way to invent alpha."""
        assert "build_live(state_dir, data_dir, fx=fx)" in _app_source()


def _app_source() -> str:
    from pathlib import Path

    import trading.dashboard.app as mod

    return Path(mod.__file__).read_text()
