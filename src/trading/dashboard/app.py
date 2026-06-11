"""Dashboard server — stdlib HTTP, basic auth, read-only over state.

Endpoints:
  GET /             single-page UI (inline HTML + Chart.js CDN)
  GET /api/summary  everything the page renders, one JSON blob

Auth: HTTP Basic. Credentials from DASHBOARD_USER / DASHBOARD_PASS env
vars (default user 'yan', no default password — server refuses to start
without DASHBOARD_PASS so an open dashboard can't happen by accident).

Reads ONLY: runner.db (equity curve, snapshot), monitor state JSONs,
holds/k override, last committee digest, the memory store, and the
parquet cache for 52-week percentiles. Writes nothing. The order path
is untouched and untouchable from here.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def build_summary(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    """One JSON blob with everything the page shows. Defensive: each
    section degrades to empty rather than failing the whole payload."""
    from trading.agents.context import build_context
    from trading.memory.store import MemoryStore
    from trading.runner.state import RunnerStore

    out: dict[str, Any] = {"generated_at": datetime.now(tz=timezone.utc).isoformat()}

    # Book, monitors, holds — reuse the committee's context pass.
    try:
        out["context"] = build_context(state_dir, data_dir)
    except Exception as e:
        logger.bind(component="dashboard").warning(f"context failed: {e}")
        out["context"] = {}

    # Equity curve (downsampled to <=500 points for the chart).
    try:
        curve = RunnerStore(state_dir / "runner.db").equity_curve()
        step = max(1, len(curve) // 500)
        out["equity_curve"] = [{"t": ts.isoformat(), "v": float(eq)} for ts, eq in curve[::step]]
    except Exception as e:
        logger.bind(component="dashboard").warning(f"equity curve failed: {e}")
        out["equity_curve"] = []

    # Last committee digest (already persisted for /detail).
    try:
        p = state_dir / "last_committee.json"
        out["committee"] = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        out["committee"] = {}

    # Memory vitals.
    try:
        mem = MemoryStore(state_dir / "memory")
        out["memory"] = {
            "stats": mem.stats(),
            "calibration": mem.calibration(),
            "trust": mem.trust_table(min_graded=1)[:10],
            "lessons": [
                {"id": r["id"], "status": r["status"], "statement": r["statement"]}
                for r in mem.lessons()[:8]
            ],
            "journal_tail": [
                {"ts": e["ts"].isoformat(), "kind": e["kind"], "actor": e["actor"]}
                for e in mem.journal_tail(12)
            ],
        }
    except Exception as e:
        logger.bind(component="dashboard").warning(f"memory failed: {e}")
        out["memory"] = {}

    return out


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Agent</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
 body{background:#0f1419;color:#d7dde4;font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 14px} h2{font-size:13px;color:#8b98a5;margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em}
 .grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
 .card{background:#161d26;border:1px solid #232d39;border-radius:10px;padding:14px}
 .big{grid-column:1/-1} table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{padding:4px 6px;text-align:right;border-bottom:1px solid #232d39} td:first-child,th:first-child{text-align:left}
 .pos{color:#4cc38a}.neg{color:#e5534b}.muted{color:#8b98a5}
 .tile{display:inline-block;margin:4px 10px 4px 0}.tile b{font-size:17px}
 .pill{display:inline-block;padding:1px 8px;border-radius:9px;background:#232d39;font-size:12px;margin:2px}
</style></head><body>
<h1>📈 Trading Agent <span class="muted" id="asof"></span></h1>
<div class="grid">
 <div class="card big"><h2>Equity (paper)</h2><canvas id="eq" height="84"></canvas></div>
 <div class="card"><h2>Account</h2><div id="account"></div><h2 style="margin-top:12px">Positions</h2><div id="positions"></div></div>
 <div class="card"><h2>Macro dial</h2><div id="macro"></div><h2 style="margin-top:12px">Vol surface</h2><div id="vol"></div></div>
 <div class="card"><h2>Committee (latest)</h2><div id="committee"></div></div>
 <div class="card"><h2>Memory</h2><div id="memory"></div></div>
</div>
<script>
const pct=(x,d=1)=>x==null?'–':((x>=0?'+':'')+(100*x).toFixed(d)+'%');
const num=(x)=>x==null?'–':Number(x).toLocaleString(undefined,{maximumFractionDigits:0});
fetch('api/summary').then(r=>r.json()).then(d=>{
 document.getElementById('asof').textContent=' · '+new Date(d.generated_at).toLocaleString();
 // equity chart + drawdown color
 const c=d.equity_curve||[];
 new Chart(document.getElementById('eq'),{type:'line',data:{labels:c.map(p=>p.t.slice(0,10)),
  datasets:[{data:c.map(p=>p.v),borderColor:'#4cc38a',borderWidth:1.6,pointRadius:0,tension:.2,fill:false}]},
  options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b98a5',maxTicksLimit:8}},y:{ticks:{color:'#8b98a5'}}}}});
 const ctx=d.context||{};
 const a=ctx.account||{};
 document.getElementById('account').innerHTML=
  `<span class="tile"><b>${num(a.equity)}</b><br><span class="muted">${a.base_currency||''} equity</span></span>`+
  `<span class="tile"><b>${num(a.cash)}</b><br><span class="muted">cash</span></span>`+
  `<span class="tile"><b>${(ctx.holds||[]).length}</b><br><span class="muted">pinned</span></span>`;
 const rows=(ctx.positions||[]).map(p=>`<tr><td>${p.symbol}</td>
   <td class="${(p.unrealized_pct||0)>=0?'pos':'neg'}">${pct(p.unrealized_pct)}</td>
   <td>${p.entry_pctile_52w!=null?Math.round(p.entry_pctile_52w*100)+'%':'–'}</td>
   <td>${p.now_pctile_52w!=null?Math.round(p.now_pctile_52w*100)+'%':'–'}</td></tr>`).join('');
 document.getElementById('positions').innerHTML=rows?
  `<table><tr><th>sym</th><th>uP&L</th><th>entry@52w</th><th>now@52w</th></tr>${rows}</table>`:
  '<span class="muted">flat</span>';
 const m=ctx.macro_dial||{};
 document.getElementById('macro').innerHTML=['composite','rates_shock_z','dollar_z','energy_z','btc_confirm_z']
  .map(k=>m[k]!=null?`<span class="tile"><b>${Number(m[k]).toFixed(1)}σ</b><br><span class="muted">${k.replace('_z','').replace('_',' ')}</span></span>`:'').join('')||'<span class="muted">no reading yet</span>';
 const v=ctx.vol_surface||{};
 document.getElementById('vol').innerHTML=v.atm_iv!=null?
  `<span class="tile"><b>${(100*v.atm_iv).toFixed(0)}%</b><br><span class="muted">ATM IV</span></span>
   <span class="tile"><b>${(100*(v.put_skew||0)).toFixed(1)}</b><br><span class="muted">put skew</span></span>
   <span class="tile"><b>${(100*(v.term_slope||0)).toFixed(1)}</b><br><span class="muted">term slope</span></span>
   <span class="tile"><b>${(v.pc_oi_ratio||0).toFixed(2)}</b><br><span class="muted">P/C OI</span></span>`:
  '<span class="muted">no reading yet</span>';
 const co=d.committee||{};
 if(co.ok){const r=co.ruling||{};
  const takes=Object.entries(co.takes||{}).map(([n,t])=>`<span class="pill">${{'bullish':'🟢','neutral':'⚪','bearish':'🔴'}[t.stance]||'⚪'} ${n}</span>`).join('');
  document.getElementById('committee').innerHTML=
   `<div><b>${(r.posture||'neutral').replace('_',' ').toUpperCase()}</b> <span class="muted">dissent ${(co.disagreement_index??0).toFixed(1)}</span></div>
    <div style="margin:6px 0">${takes}</div><div>${r.proposal||''}</div>
    <div class="muted" style="margin-top:6px">watching: ${r.watch||'–'}</div>`;
 } else document.getElementById('committee').innerHTML='<span class="muted">no committee run yet — weekdays 14:00 UTC</span>';
 const me=d.memory||{};const s=me.stats||{};
 const cal=(me.calibration||[]).map(c=>`<tr><td>${c.agent}</td><td>${c.n}</td><td>${pct(c.hit_rate,0)}</td><td>${(c.brier??0).toFixed(2)}</td></tr>`).join('');
 const les=(me.lessons||[]).slice(0,4).map(l=>`<div class="pill">${l.status}: ${l.statement.slice(0,70)}</div>`).join('');
 document.getElementById('memory').innerHTML=
  `<span class="tile"><b>${s.journal??0}</b><br><span class="muted">journal</span></span>
   <span class="tile"><b>${s.episodes??0}</b><br><span class="muted">episodes</span></span>
   <span class="tile"><b>${s.lessons??0}</b><br><span class="muted">lessons</span></span>
   <span class="tile"><b>${s.predictions??0}</b><br><span class="muted">predictions</span></span>`+
  (cal?`<table style="margin-top:8px"><tr><th>agent</th><th>n</th><th>hit</th><th>brier</th></tr>${cal}</table>`:'')+
  (les?`<div style="margin-top:8px">${les}</div>`:'');
});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    state_dir: Path
    data_dir: Path
    auth_token: str  # base64 of user:pass

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Basic {self.auth_token}"

    def do_GET(self) -> None:
        if not self._authorized():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="trading"')
            self.end_headers()
            return
        if self.path.startswith("/api/summary"):
            try:
                body = json.dumps(
                    build_summary(self.state_dir, self.data_dir), default=str
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
        elif self.path == "/" or self.path.startswith("/index"):
            body = _PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.bind(component="dashboard").debug(fmt % args)


def serve(host: str = "0.0.0.0", port: int = 8787) -> None:
    """Run the dashboard until interrupted. Refuses to start without
    DASHBOARD_PASS — an unauthenticated dashboard must be impossible."""
    from trading.core.config import settings

    user = os.getenv("DASHBOARD_USER", "yan")
    password = os.getenv("DASHBOARD_PASS", "")
    if not password:
        raise SystemExit("set DASHBOARD_PASS in .env — refusing to serve without auth")

    _Handler.state_dir = Path(settings.state_dir)
    _Handler.data_dir = Path(settings.data_dir)
    _Handler.auth_token = base64.b64encode(f"{user}:{password}".encode()).decode()

    server = ThreadingHTTPServer((host, port), _Handler)
    logger.bind(component="dashboard").info(f"dashboard listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
