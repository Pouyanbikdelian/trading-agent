# Charter + context edits — APPLIED 2026-07-02 (A + B; C was already live in context.py)

These three changes address the same root cause the committee output exposed:
a **"near-highs + low-IV = safe"** prior, plus **correlation-blindness**, living in
the LLM personas rather than in any tunable parameter. They are prompt/feature
edits, not model retuning.

Already applied separately (tested): the PPI series fix (`econ_watch.py`) and the
`sector_map` wiring that switches on the existing sector cap (`cycle.py`).

> **CORRECTION 2026-07-15:** the sector-cap "wiring" above was code-path only —
> in production it never bound because no fundamentals cache existed on the VPS
> and the failure was silent. Fixed for real on 2026-07-14: default cache path,
> loud alert when disabled, and a `trading data fundamentals` CLI to populate
> it. See docs/incidents.md. Lesson: "wired" is not the same as "firing".

The edits below are left for your review because they change agent behaviour at
the prompt level.

---

## Change A — Quant charter (`src/trading/agents/committee.py`)

**Why:** the quant treats `now_pctile_52w ≈ 1.0` as a bullish "donchian breakout"
— but that 1.0 is the same datum the bears read as exhaustion. It also discounts
the `macro_dial`/`economy` it is actually shown ("trust the numbers over any
story"), and counts five correlated semis as five confirmations.

**Before**

```python
    "quant": (
        "You are the Quant on a small systematic trading desk. You trust the "
        "numbers in the context block (momentum ranks, regime, vol surface, "
        "macro dial) over any story. Be terse and specific. " + _TAKE_SCHEMA
    ),
```

**After**

```python
    "quant": (
        "You are the Quant on a small systematic trading desk. You reason from "
        "the numbers in the context block — momentum ranks, regime, vol surface, "
        "macro dial — not from stories. But the regime and macro dial ARE "
        "numbers: weight them. A bullish trend signal in a hostile regime is a "
        "smaller bet, not a full one. Two hard rules: "
        "(1) A high 52-week percentile (now_pctile_52w near 1.0) tells you WHERE "
        "price is, not whether reward-to-risk is good. Never cite 'at/near the "
        "high' as bullish confirmation by itself — at the 100th percentile an "
        "entry is maximally far from any trend stop. If you call a name a "
        "hold/add, say where the stop sits and the resulting reward-to-risk. "
        "(2) Correlated holdings are ONE bet. If several positions share a "
        "sector/theme (e.g. semis), do not count them as independent "
        "confirmations; conviction reflects the single underlying factor. "
        "Be terse and specific. " + _TAKE_SCHEMA
    ),
```

---

## Change B — Sentinel charter (`src/trading/runtime/sentinel.py`)

**Why:** the charter already says real alarms are "correlated selling" — good —
but the false-alarm call leaned on "near highs, big gains, IV 15.7% subdued" as
reassurance and labelled a simultaneous GLW+SNDK drop "idiosyncratic."

**Before** (the two middle sentences)

```python
    "trips are noise: a single name gapping on earnings, a stale print. "
    "Real alarms are systemic: correlated selling, vol spike with breadth "
    "collapse, credit cracking. Be decisive and terse. Respond ONLY with "
```

**After**

```python
    "trips are noise: a single name gapping on earnings, a stale print. "
    "Real alarms are systemic: correlated selling, vol spike with breadth "
    "collapse, credit cracking. Do NOT treat any of these as reassurance: a "
    "name near its 52-week high, large unrealized gains, or a subdued SPY IV — "
    "those describe a crowded, extended long with the most to give back, and "
    "quiet IV often precedes repricing, not safety. When two or more held names "
    "in the SAME sector trip together, your default is correlated selling (a "
    "real alarm), NOT 'idiosyncratic' — only call it idiosyncratic if you can "
    "name a stock-specific catalyst for each. Be decisive and terse. Respond "
    "ONLY with "
```

---

## Change C — Give both personas sector tags (`src/trading/agents/context.py`)

**Why (enabling change):** Rules A-2 and B above only work if the agents can SEE
that the book is concentrated. Position rows currently carry no sector, so the
quant and sentinel have to guess from tickers. Add `sector` to each position row
from the fundamentals cache.

**Sketch** — in `build_context`, where position rows are assembled:

```python
# near the top of build_context, load the fundamentals cache once:
funds = {}
try:
    from trading.data.fundamentals_source import read_fundamentals_cache
    from trading.core.config import settings as _s
    fpath = _s.data_dir / "fundamentals.parquet"
    if fpath.exists():
        funds = read_fundamentals_cache(fpath)
except Exception:
    funds = {}

# ...then inside the per-position loop, after `row = {...}`:
f = funds.get(sym)
if f is not None and f.sector:
    row["sector"] = f.sector
```

With sectors visible, the quant's "five semis = one bet" rule and the sentinel's
"same-sector trip = correlated selling" rule have the data they need. This also
shares the same fundamentals cache the `sector_map` risk fix now relies on.

---

## Apply + verify

```bash
uv run pytest tests/agents tests/runtime/test_sentinel.py -q   # personas still parse
uv run pytest tests/risk tests/runner/test_cycle.py -q         # already green here
```

Charters are prose, so there is no numeric test for "did the reasoning improve" —
the check is qualitative: re-run a committee/sentinel cycle and confirm the quant
stops citing bare 52-week highs as confirmation and the sentinel stops waving off
correlated same-sector drops. Worth eyeballing a few cycles before paper-committing.
