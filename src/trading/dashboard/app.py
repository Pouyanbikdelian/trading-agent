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

    # Equity curves: one point per day for range views, plus today's
    # intraday points (snapshots land every 60s) for the "today" view.
    try:
        curve = RunnerStore(state_dir / "runner.db").equity_curve()
        daily: dict[str, float] = {}
        for ts, eq in curve:
            daily[ts.date().isoformat()] = float(eq)  # last point of each day wins
        out["equity_curve"] = [{"t": k, "v": v} for k, v in sorted(daily.items())]
        today = datetime.now(tz=timezone.utc).date().isoformat()
        intraday = [(ts, eq) for ts, eq in curve if ts.date().isoformat() == today]
        step = max(1, len(intraday) // 300)
        out["equity_today"] = [{"t": ts.isoformat(), "v": float(eq)} for ts, eq in intraday[::step]]
    except Exception as e:
        logger.bind(component="dashboard").warning(f"equity curve failed: {e}")
        out["equity_curve"] = []
        out["equity_today"] = []

    # Last committee digest (already persisted for /detail).
    try:
        p = state_dir / "last_committee.json"
        out["committee"] = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        out["committee"] = {}

    # Market watch history (macro tab).
    try:
        mw = state_dir / "market_watch.json"
        out["market_watch"] = json.loads(mw.read_text()) if mw.exists() else {}
    except Exception:
        out["market_watch"] = {}

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
 h1{font-size:18px;margin:0 0 10px} h2{font-size:13px;color:#8b98a5;margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em}
 .grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
 .card{background:#161d26;border:1px solid #232d39;border-radius:10px;padding:14px}
 .big{grid-column:1/-1} table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{padding:4px 6px;text-align:right;border-bottom:1px solid #232d39} td:first-child,th:first-child{text-align:left}
 .pos{color:#4cc38a}.neg{color:#e5534b}.muted{color:#8b98a5}
 .tile{display:inline-block;margin:4px 10px 4px 0}.tile b{font-size:17px}
 .pill{display:inline-block;padding:1px 8px;border-radius:9px;background:#232d39;font-size:12px;margin:2px}
 .tabs{margin:0 0 14px}.tabs button{background:#161d26;border:1px solid #232d39;color:#8b98a5;
  padding:7px 16px;border-radius:8px;margin-right:6px;cursor:pointer;font-size:14px}
 .tabs button.on{color:#d7dde4;border-color:#4cc38a}
 .tab{display:none}.tab.on{display:block}
 .warn{color:#e5a54b}
</style></head><body>
<h1>📈 Trading Agent <span class="muted" id="asof"></span></h1>
<div class="tabs">
 <button data-t="portfolio" class="on">Portfolio</button>
 <button data-t="macro">Macro</button>
 <button data-t="memory">Memory</button>
</div>

<div class="tab on" id="tab-portfolio"><div class="grid">
 <div class="card big"><h2>Equity (paper) <span id="eqret" style="float:right"></span></h2>
  <div class="tabs" id="ranges" style="margin:4px 0 8px">
   <button data-r="today">Today</button><button data-r="1w">1W</button>
   <button data-r="1m">1M</button><button data-r="3m">3M</button>
   <button data-r="6m">6M</button><button data-r="ytd">YTD</button>
   <button data-r="1y" class="on">1Y</button><button data-r="all">All</button>
  </div><canvas id="eq" height="84"></canvas></div>
 <div class="card"><h2>Account</h2><div id="account"></div><h2 style="margin-top:12px">Positions</h2><div id="positions"></div></div>
 <div class="card"><h2>Committee (latest)</h2><div id="committee"></div></div>
</div></div>

<div class="tab" id="tab-macro"><div class="grid">
 <div class="card"><h2>Yield curve · 10y − 3m</h2><div id="curveTile"></div><canvas id="curveCh" height="110"></canvas></div>
 <div class="card"><h2>VIX term structure</h2><div id="vixTile"></div><canvas id="vixCh" height="110"></canvas></div>
 <div class="card"><h2>Breadth · % above SMA</h2><div id="brTile"></div><canvas id="brCh" height="110"></canvas></div>
 <div class="card"><h2>Credit & risk appetite (1y base = 100)</h2><div id="ratioTile"></div><canvas id="ratioCh" height="110"></canvas></div>
 <div class="card"><h2>Macro dial (z-scores)</h2><div id="macro"></div></div>
 <div class="card"><h2>Vol surface (SPY options)</h2><div id="vol"></div></div>
</div></div>

<div class="tab" id="tab-memory"><div class="grid">
 <div class="card"><h2>Vitals</h2><div id="memstats"></div></div>
 <div class="card"><h2>Agent calibration</h2><div id="memcal"></div></div>
 <div class="card"><h2>Source trust</h2><div id="memtrust"></div></div>
 <div class="card big"><h2>Lessons</h2><div id="memlessons"></div></div>
</div></div>

<script>
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));
 document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');document.getElementById('tab-'+b.dataset.t).classList.add('on');});
const pct=(x,d=1)=>x==null?'–':((x>=0?'+':'')+(100*x).toFixed(d)+'%');
const num=(x)=>x==null?'–':Number(x).toLocaleString(undefined,{maximumFractionDigits:0});
const line=(el,labels,sets)=>new Chart(document.getElementById(el),{type:'line',
 data:{labels,datasets:sets.map(s=>({...s,borderWidth:1.6,pointRadius:0,tension:.2,fill:false}))},
 options:{plugins:{legend:{labels:{color:'#8b98a5',boxWidth:10}}},
  scales:{x:{ticks:{color:'#8b98a5',maxTicksLimit:7}},y:{ticks:{color:'#8b98a5'}}}}});
fetch('api/summary').then(r=>r.json()).then(d=>{
 document.getElementById('asof').textContent=' · '+new Date(d.generated_at).toLocaleString();
 const daily=d.equity_curve||[],today=d.equity_today||[];let eqChart=null;
 function drawEq(range){
  let pts;const now=new Date();
  if(range==='today'){pts=today.map(p=>({t:p.t.slice(11,16),v:p.v}));}
  else{
   const days={'1w':7,'1m':31,'3m':92,'6m':183,'1y':365}[range];
   let cut=null;
   if(days)cut=new Date(now-days*864e5);
   if(range==='ytd')cut=new Date(now.getFullYear(),0,1);
   pts=daily.filter(p=>!cut||new Date(p.t)>=cut).map(p=>({t:p.t,v:p.v}));
  }
  const ret=pts.length>1?(pts[pts.length-1].v/pts[0].v-1):null;
  document.getElementById('eqret').innerHTML=ret==null?'':
   `<span class="${ret>=0?'pos':'neg'}">${pct(ret,2)}</span>`;
  if(eqChart)eqChart.destroy();
  eqChart=line('eq',pts.map(p=>p.t),[{data:pts.map(p=>p.v),
   borderColor:(ret??0)>=0?'#4cc38a':'#e5534b'}]);
 }
 document.querySelectorAll('#ranges button').forEach(b=>b.onclick=(e)=>{
  e.stopPropagation();
  document.querySelectorAll('#ranges button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');drawEq(b.dataset.r);});
 drawEq('1y');
 const ctx=d.context||{};const a=ctx.account||{};
 document.getElementById('account').innerHTML=
  `<span class="tile"><b>${num(a.equity)}</b><br><span class="muted">${a.base_currency||''} equity</span></span>`+
  `<span class="tile"><b>${num(a.cash)}</b><br><span class="muted">cash</span></span>`+
  `<span class="tile"><b>${(ctx.holds||[]).length}</b><br><span class="muted">pinned</span></span>`;
 const rows=(ctx.positions||[]).map(p=>`<tr><td>${p.symbol}</td>
   <td class="${(p.unrealized_pct||0)>=0?'pos':'neg'}">${pct(p.unrealized_pct)}</td>
   <td>${p.entry_pctile_52w!=null?Math.round(p.entry_pctile_52w*100)+'%':'–'}</td>
   <td>${p.now_pctile_52w!=null?Math.round(p.now_pctile_52w*100)+'%':'–'}</td></tr>`).join('');
 document.getElementById('positions').innerHTML=rows?
  `<table><tr><th>sym</th><th>uP&L</th><th>entry@52w</th><th>now@52w</th></tr>${rows}</table>`:'<span class="muted">flat</span>';
 const co=d.committee||{};
 if(co.ok){const r=co.ruling||{};
  const takes=Object.entries(co.takes||{}).map(([n,t])=>`<span class="pill">${{'bullish':'🟢','neutral':'⚪','bearish':'🔴'}[t.stance]||'⚪'} ${n}</span>`).join('');
  document.getElementById('committee').innerHTML=
   `<div><b>${(r.posture||'neutral').replace('_',' ').toUpperCase()}</b> <span class="muted">dissent ${(co.disagreement_index??0).toFixed(1)}</span></div>
    <div style="margin:6px 0">${takes}</div><div>${r.proposal||''}</div>
    <div class="muted" style="margin-top:6px">watching: ${r.watch||'–'}</div>`;
 } else document.getElementById('committee').innerHTML='<span class="muted">no committee run yet — weekdays 14:00 UTC</span>';

 // ---- MACRO TAB
 const mh=(d.market_watch||{}).history||[];const last=(d.market_watch||{}).latest||{};
 const L=mh.map(h=>String(h.t).slice(0,10));
 if(mh.length){
  const cur=last.curve_10y3m;
  document.getElementById('curveTile').innerHTML=
   `<span class="tile"><b class="${cur<0?'neg':cur<0.3?'warn':'pos'}">${cur==null?'–':cur.toFixed(2)+'pp'}</b><br>
    <span class="muted">${cur<0?'INVERTED':'spread'} · 10y ${last.y_10y??'–'} / 3m ${last.y_3m??'–'}</span></span>`;
  line('curveCh',L,[{data:mh.map(h=>h.curve_10y3m),borderColor:'#5ab0f6',label:'10y−3m'}]);
  const vr=last.vix_ratio;
  document.getElementById('vixTile').innerHTML=
   `<span class="tile"><b>${last.vix??'–'}</b><br><span class="muted">VIX</span></span>
    <span class="tile"><b class="${vr>1?'neg':'pos'}">${vr??'–'}</b><br><span class="muted">VIX/VIX3M ${vr>1?'⚠ backwardation':''}</span></span>`;
  line('vixCh',L,[{data:mh.map(h=>h.vix),borderColor:'#e5a54b',label:'VIX'},
                  {data:mh.map(h=>h.vix3m),borderColor:'#8b98a5',label:'VIX3M'}]);
  document.getElementById('brTile').innerHTML=
   `<span class="tile"><b>${pct(last.pct_above_50,0)}</b><br><span class="muted">above 50dma</span></span>
    <span class="tile"><b>${pct(last.pct_above_200,0)}</b><br><span class="muted">above 200dma</span></span>`;
  line('brCh',L,[{data:mh.map(h=>100*(h.pct_above_50??null)),borderColor:'#4cc38a',label:'%>50d'},
                 {data:mh.map(h=>100*(h.pct_above_200??null)),borderColor:'#5ab0f6',label:'%>200d'}]);
  document.getElementById('ratioTile').innerHTML=['hyg_ief','spy_tlt','qqq_spy','gld_dbc']
   .map(k=>last[k]!=null?`<span class="tile"><b>${last[k].toFixed(0)}</b><br><span class="muted">${k.replace('_','/').toUpperCase()}</span></span>`:'').join('');
  line('ratioCh',L,[{data:mh.map(h=>h.hyg_ief),borderColor:'#e5534b',label:'HYG/IEF credit'},
                    {data:mh.map(h=>h.spy_tlt),borderColor:'#4cc38a',label:'SPY/TLT'},
                    {data:mh.map(h=>h.qqq_spy),borderColor:'#5ab0f6',label:'QQQ/SPY'}]);
 } else {
  document.getElementById('curveTile').innerHTML='<span class="muted">collector runs daily after the close — first reading tonight</span>';
 }
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

 // ---- MEMORY TAB
 const me=d.memory||{};const st=me.stats||{};
 document.getElementById('memstats').innerHTML=['journal','episodes','lessons','dossiers','predictions','sources']
  .map(k=>`<span class="tile"><b>${st[k]??0}</b><br><span class="muted">${k}</span></span>`).join('');
 const cal=(me.calibration||[]).map(c=>`<tr><td>${c.agent}</td><td>${c.n}</td><td>${pct(c.hit_rate,0)}</td><td>${(c.brier??0).toFixed(2)}</td></tr>`).join('');
 document.getElementById('memcal').innerHTML=cal?`<table><tr><th>agent</th><th>n</th><th>hit</th><th>brier</th></tr>${cal}</table>`:'<span class="muted">no graded predictions yet</span>';
 const tr=(me.trust||[]).map(t=>`<tr><td>${t.source}</td><td>${t.trust.toFixed(2)}</td><td>${t.graded}</td></tr>`).join('');
 document.getElementById('memtrust').innerHTML=tr?`<table><tr><th>source</th><th>trust</th><th>graded</th></tr>${tr}</table>`:'<span class="muted">no sources graded yet</span>';
 document.getElementById('memlessons').innerHTML=(me.lessons||[]).map(l=>`<div class="pill">${l.status}: ${l.statement.slice(0,110)}</div>`).join('')||'<span class="muted">no lessons yet</span>';
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
