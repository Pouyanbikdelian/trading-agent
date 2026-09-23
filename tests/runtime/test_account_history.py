"""IBKR Flex history: NAV since account opening, deposits as the broker lists them."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading.runtime.account_history import (
    FlexError,
    fetch_flex,
    load_history,
    merge_and_save,
    parse_flex,
)

STATEMENT = b"""<FlexQueryResponse queryName="history" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U1234567" fromDate="20260102" toDate="20260806" period="Custom" whenGenerated="20260923;100000">
<EquitySummaryInBase>
<EquitySummaryByReportDateInBase accountId="U1234567" currency="CHF" reportDate="20260102" cash="88000" stock="0" total="88000"/>
<EquitySummaryByReportDateInBase accountId="U1234567" currency="CHF" reportDate="20260420" cash="1000" stock="90000" total="91000"/>
<EquitySummaryByReportDateInBase accountId="U1234567" currency="CHF" reportDate="2026-08-06" total="88500.5"/>
</EquitySummaryInBase>
<CashTransactions>
<CashTransaction accountId="U1234567" currency="CHF" fxRateToBase="1" type="Deposits/Withdrawals" amount="88000" dateTime="20260102;093000" description="CASH RECEIPTS" transactionID="t1"/>
<CashTransaction accountId="U1234567" currency="USD" fxRateToBase="0.8" type="Deposits/Withdrawals" amount="2500" dateTime="20260420;101500" description="WIRE" transactionID="t2"/>
<CashTransaction accountId="U1234567" currency="USD" fxRateToBase="0.8" type="Dividends" amount="12.5" dateTime="20260501" transactionID="t3"/>
</CashTransactions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>"""


def test_parse_reads_nav_and_only_transfers() -> None:
    h = parse_flex(STATEMENT)
    assert h.account == "U1234567" and h.base_currency == "CHF"
    assert h.nav == {"2026-01-02": 88000.0, "2026-04-20": 91000.0, "2026-08-06": 88500.5}
    assert [(f["day"], f["amount"], f["currency"], f["amount_base"]) for f in h.flows] == [
        ("2026-01-02", 88000.0, "CHF", 88000.0),
        ("2026-04-20", 2500.0, "USD", 2000.0),
    ]  # the dividend is income, not a transfer


def test_an_error_response_is_not_an_empty_account() -> None:
    err = b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012</ErrorCode><ErrorMessage>Token has expired.</ErrorMessage></FlexStatementResponse>"
    with pytest.raises(FlexError, match="Token has expired"):
        parse_flex(err)


def test_a_query_without_the_sections_says_which_to_add() -> None:
    with pytest.raises(FlexError, match="Net Asset Value"):
        parse_flex(
            b"<FlexQueryResponse><FlexStatements><FlexStatement accountId='U1'/></FlexStatements></FlexQueryResponse>"
        )


def test_merge_is_idempotent_and_later_imports_win(tmp_path: Path) -> None:
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    first = merge_and_save(tmp_path, [parse_flex(STATEMENT)], source="file", now=now)
    again = merge_and_save(tmp_path, [parse_flex(STATEMENT)], source="file", now=now)
    assert (first["days"], first["new_days"], first["new_flows"]) == (3, 3, 2)
    assert (again["new_days"], again["new_flows"]) == (0, 0)
    revised = parse_flex(STATEMENT.replace(b'total="88500.5"', b'total="88600"'))
    merge_and_save(tmp_path, [revised], source="file", now=now)
    assert load_history(tmp_path)["nav"]["2026-08-06"] == 88600.0


def test_fetch_polls_until_the_statement_is_ready() -> None:
    calls: list[tuple[str, dict]] = []
    replies = [
        b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>42</ReferenceCode><Url>https://x/GetStatement</Url></FlexStatementResponse>",
        b"<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode><ErrorMessage>in progress</ErrorMessage></FlexStatementResponse>",
        STATEMENT,
    ]

    def get(url, params):
        calls.append((url, params))
        return SimpleNamespace(content=replies[len(calls) - 1])

    out = fetch_flex("tok", "q1", send_url="https://x/SendRequest", get=get, sleep=lambda s: None)
    assert out == STATEMENT
    assert calls[0] == ("https://x/SendRequest", {"t": "tok", "q": "q1", "v": "3"})
    assert calls[1][0] == "https://x/GetStatement" and calls[1][1]["q"] == "42"


def test_fetch_stops_on_a_hard_error() -> None:
    replies = [
        b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>42</ReferenceCode></FlexStatementResponse>",
        b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015</ErrorCode><ErrorMessage>Token is invalid.</ErrorMessage></FlexStatementResponse>",
    ]
    it = iter(replies)
    with pytest.raises(FlexError, match="1015"):
        fetch_flex(
            "tok", "q", get=lambda u, p: SimpleNamespace(content=next(it)), sleep=lambda s: None
        )


def test_dashboard_prepends_history_and_counts_broker_deposits(tmp_path: Path) -> None:
    from trading.core.types import AccountSnapshot
    from trading.dashboard.cockpit import equity_block
    from trading.runner.state import RunnerStore

    merge_and_save(tmp_path, [parse_flex(STATEMENT)], source="file")
    rs = RunnerStore(tmp_path / "runner.db")
    for day, eq in [("2026-08-07", 88700.0), ("2026-08-10", 88100.0)]:
        rs.save_snapshot(
            AccountSnapshot(
                ts=datetime.fromisoformat(day + "T20:00:00+00:00"),
                cash=50_000,
                equity=eq,
                positions={},
                base_currency="CHF",
            )
        )
    rs.close()

    out = equity_block(tmp_path / "runner.db", tmp_path)
    days = out["days"]

    assert [d["t"] for d in days] == [
        "2026-01-02",
        "2026-04-20",
        "2026-08-06",
        "2026-08-07",
        "2026-08-10",
    ]
    assert days[0]["source"] == "ibkr_flex" and days[-1]["source"] == "runner"
    assert days[1]["flow"] == pytest.approx(2000.0)
    assert days[-1]["net_flows"] == pytest.approx(90_000.0)  # 88k opening + 2k (USD 2.5k)
    assert out["history"]["flex_first"] == "2026-01-02"
    assert out["flow_candidates"] == []


# ------------------------------------------ queries without the Type column

UNTYPED = b"""<FlexQueryResponse queryName="q" type="AF"><FlexStatements count="1">
<FlexStatement accountId="U0000000" fromDate="20260205" toDate="20260605">
<EquitySummaryInBase>
<EquitySummaryByReportDateInBase currency="CHF" reportDate="20260206" total="13.13" />
<EquitySummaryByReportDateInBase currency="CHF" reportDate="20260209" total="52181.48" />
<EquitySummaryByReportDateInBase currency="CHF" reportDate="20260603" total="54216.58" />
<EquitySummaryByReportDateInBase currency="CHF" reportDate="20260604" total="89027.85" />
</EquitySummaryInBase>
<CashTransactions>
<CashTransaction currency="USD" amount="68088" dateTime="20260209" reportDate="20260209" />
<CashTransaction currency="USD" amount="107.46" dateTime="20260603" reportDate="20260603" />
<CashTransaction currency="USD" amount="-21.49" dateTime="20260603" reportDate="20260604" />
<CashTransaction currency="CHF" amount="35000" dateTime="20260604" reportDate="20260604" />
</CashTransactions></FlexStatement></FlexStatements></FlexQueryResponse>"""


def test_untyped_cash_rows_are_transfers_only_when_nav_moves_by_their_size() -> None:
    """The operator's real query (2026-09-23) had no Type column: interest,
    withholding tax and both deposits all arrived as bare amounts."""
    h = parse_flex(UNTYPED)
    got = {(f["day"], f["currency"]): f for f in h.flows}
    assert set(got) == {("2026-02-09", "USD"), ("2026-06-04", "CHF")}
    assert got[("2026-06-04", "CHF")]["amount_base"] == 35_000
    assert all(f["inferred"] for f in h.flows)
    assert any("no Type column" in n for n in h.notes)


def test_a_dollar_deposit_is_never_booked_as_francs() -> None:
    """No fxRateToBase used to mean rate 1.0: 68,088 USD as 68,088 CHF."""
    h = parse_flex(UNTYPED)
    usd = next(f for f in h.flows if f["currency"] == "USD")
    assert usd["amount_base"] == pytest.approx(52_168.35, abs=0.01)  # IBKR's own NAV jump
    assert usd["basis"] == "NAV jump"
    h2 = parse_flex(UNTYPED, usdchf={"2026-02-06": 0.77})
    usd2 = next(f for f in h2.flows if f["currency"] == "USD")
    assert usd2["amount_base"] == pytest.approx(68_088 * 0.77)
    assert usd2["basis"] == "USDCHF close"


def test_dormant_months_before_the_first_deposit_are_not_charted(tmp_path: Path) -> None:
    from trading.dashboard.cockpit import equity_block

    merge_and_save(tmp_path, [parse_flex(UNTYPED)], source="file")
    days = equity_block(tmp_path / "runner.db", tmp_path)["days"]
    assert days[0]["t"] == "2026-02-09"
    assert days[0]["flow"] == pytest.approx(52_168.35, abs=0.01)
    assert days[-1]["net_flows"] == pytest.approx(87_168.35, abs=0.01)


def test_a_statement_alone_keeps_its_base_currency(tmp_path: Path) -> None:
    """Before the first runner snapshot the page labelled a CHF account USD."""
    from trading.dashboard.cockpit import equity_block

    merge_and_save(tmp_path, [parse_flex(UNTYPED)], source="file")
    assert equity_block(tmp_path / "runner.db", tmp_path)["currency"] == "CHF"
