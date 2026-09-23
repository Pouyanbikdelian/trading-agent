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
