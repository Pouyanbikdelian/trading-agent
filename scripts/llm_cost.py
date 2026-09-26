"""What the agent layer costs per weekly cycle, from state/llm_usage.jsonl.

Run inside the live container (the log lives in its state dir):

    docker compose exec -T trader-live python - < scripts/llm_cost.py

Prices are Anthropic list prices per million tokens as of 2026-09-26
(platform.claude.com/docs/en/models/overview). Cache reads/writes are
approximated (writes 1.25x input, reads 10% of input; 5% on Opus 5.5).
"If on Opus 5.5" re-prices the frontier calls actually made at Opus 5.5
rates with the same token counts; 5.5 may use fewer or more tokens.
The Telegram copilot has its own provider and is not in this log.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from trading.core.config import settings

PRICE = {
    "claude-opus-5-5": (4.0, 20.0, 0.05),
    "claude-opus-5": (5.0, 25.0, 0.10),
    "claude-sonnet-5": (2.0, 10.0, 0.10),
    "claude-fable-5-1": (10.0, 50.0, 0.10),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}


def price(model):
    for k in sorted(PRICE, key=len, reverse=True):
        if str(model).startswith(k):
            return PRICE[k]
    return None


def cost(r, model=None):
    p = price(model or r.get("model"))
    if p is None:
        return None
    pin, pout, cache_share = p
    i = r.get("input_tokens") or 0
    o = r.get("output_tokens") or 0
    cw = r.get("cache_creation_input_tokens") or 0
    cr = r.get("cache_read_input_tokens") or 0
    return (i * pin + cw * pin * 1.25 + cr * pin * cache_share + o * pout) / 1e6


path = Path(settings.state_dir) / "llm_usage.jsonl"
rows = []
for line in path.read_text(errors="replace").splitlines():
    try:
        r = json.loads(line)
        r["_ts"] = datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
        rows.append(r)
    except Exception:
        pass
if not rows:
    raise SystemExit(f"no rows in {path}")
rows.sort(key=lambda r: r["_ts"])
print(f"log: {path}  {len(rows)} calls  {rows[0]['_ts']:%Y-%m-%d} -> {rows[-1]['_ts']:%Y-%m-%d}")
unknown = sorted({r.get("model") for r in rows if price(r.get("model")) is None})
if unknown:
    print("no price for (left out of costs):", unknown)

print("\nBY MODEL AND TIER (all time)")
agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
for r in rows:
    a = agg[(r.get("model"), r.get("tier"))]
    a[0] += 1
    a[1] += 1 if (r.get("error_type") or (r.get("http_status") or 200) >= 400) else 0
    a[2] += r.get("input_tokens") or 0
    a[3] += r.get("output_tokens") or 0
    a[4] += (r.get("cache_read_input_tokens") or 0) + (r.get("cache_creation_input_tokens") or 0)
    a[5] += cost(r) or 0.0
print(
    f"{'model':22} {'tier':9} {'calls':>6} {'errors':>6} {'in_tok':>11} {'out_tok':>10} {'cache_tok':>10} {'usd':>8}"
)
for (m, t), a in sorted(agg.items(), key=lambda x: -x[1][5]):
    print(f"{m!s:22} {t!s:9} {a[0]:6d} {a[1]:6d} {a[2]:11,d} {a[3]:10,d} {a[4]:10,d} {a[5]:8.2f}")


def cycle_week(ts):
    d = ts.date()
    return d + timedelta(days=(4 - d.weekday()) % 7)


print("\nPER CYCLE WEEK (Saturday..Friday, labelled by the Friday)")
print(
    f"{'friday':10} {'calls':>6} {'usd_as_billed':>13} {'usd_if_opus_5_5':>15} {'frontier_calls':>14} {'standard_calls':>14}"
)
weeks = defaultdict(list)
for r in rows:
    weeks[cycle_week(r["_ts"])].append(r)
billed_all, whatif_all = [], []
for wk in sorted(weeks):
    rs = weeks[wk]
    billed = sum(cost(r) or 0.0 for r in rs)
    whatif = sum(
        (cost(r, "claude-opus-5-5") if r.get("tier") == "frontier" else cost(r)) or 0.0 for r in rs
    )
    fr = sum(1 for r in rs if r.get("tier") == "frontier")
    print(f"{wk} {len(rs):6d} {billed:13.2f} {whatif:15.2f} {fr:14d} {len(rs) - fr:14d}")
    billed_all.append(billed)
    whatif_all.append(whatif)

print("\nBY WEEKDAY (all time): what runs daily vs on Fridays")
wd = defaultdict(lambda: [0, 0.0])
for r in rows:
    k = r["_ts"].strftime("%a")
    wd[k][0] += 1
    wd[k][1] += cost(r) or 0.0
for k in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    if k in wd:
        print(f"{k}  calls {wd[k][0]:5d}  usd {wd[k][1]:7.2f}")

wk_sorted = sorted(weeks)
today = datetime.now(tz=rows[-1]["_ts"].tzinfo).date()
keep = [
    i
    for i, wk in enumerate(wk_sorted)
    if not (i == 0 and rows[0]["_ts"].date() > wk - timedelta(days=6))
    and not (i == len(wk_sorted) - 1 and wk >= today)
]
full = [billed_all[i] for i in keep] or billed_all
fullw = [whatif_all[i] for i in keep] or whatif_all
if full:
    print(
        f"\nTYPICAL FULL WEEK: as billed ${sum(full) / len(full):.2f}  on Opus 5.5 ${sum(fullw) / len(fullw):.2f}  (partial first/current weeks left out; {len(full)} weeks)"
    )
