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

    # Agent PM (simulated sleeve): daily-marked equity history + book.
    try:
        pm_path = state_dir / "agent_pm" / "portfolio.json"
        pm = json.loads(pm_path.read_text()) if pm_path.exists() else {}
        last_path = state_dir / "agent_pm" / "last_run.json"
        out["agent_pm"] = {
            "history": pm.get("history", []),
            "holdings": pm.get("holdings", {}),
            "cash": pm.get("cash"),
            "start_equity": pm.get("start_equity"),
            "last_run": json.loads(last_path.read_text()) if last_path.exists() else {},
        }
    except Exception as e:
        logger.bind(component="dashboard").warning(f"agent_pm failed: {e}")
        out["agent_pm"] = {}

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

    # Economy (FRED) — full trimmed history for the Economy tab.
    try:
        ec = state_dir / "econ_watch.json"
        out["econ"] = json.loads(ec.read_text()) if ec.exists() else {}
    except Exception:
        out["econ"] = {}

    # News + sector momentum (the scout's inputs — worth eyeballing raw).
    try:
        nw = state_dir / "news.json"
        news = json.loads(nw.read_text()) if nw.exists() else {}
        out["news"] = {
            "t": news.get("t"),
            "headlines": (news.get("headlines") or [])[:12],
            "sector_momentum": news.get("sector_momentum") or {},
        }
    except Exception:
        out["news"] = {}

    # Committee posture history — the debate's trajectory over time.
    try:
        from trading.memory.store import MemoryStore

        hist = MemoryStore(state_dir / "memory").journal_tail(30, kind="committee")
        out["committee_history"] = [
            {
                "t": e["ts"].isoformat(),
                "posture": (e["payload"].get("ruling") or {}).get("posture", "neutral"),
                "dissent": e["payload"].get("disagreement", 0),
            }
            for e in reversed(hist)
        ]
    except Exception:
        out["committee_history"] = []

    # Ops: halt state + data freshness (age in minutes per state file).
    try:
        now = datetime.now(tz=timezone.utc).timestamp()

        def _age(p: Path) -> int | None:
            return int((now - p.stat().st_mtime) / 60) if p.exists() else None

        halt = {}
        hp = state_dir / "halt.json"
        if hp.exists():
            halt = json.loads(hp.read_text())
        out["ops"] = {
            "halted": bool(halt.get("halted")),
            "halt_reason": halt.get("reason", ""),
            "ages_min": {
                "news": _age(state_dir / "news.json"),
                "market_watch": _age(state_dir / "market_watch.json"),
                "committee": _age(state_dir / "last_committee.json"),
                "pm_book": _age(state_dir / "agent_pm" / "portfolio.json"),
                "snapshot": _age(state_dir / "runner.db"),
            },
        }
    except Exception:
        out["ops"] = {}

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
 :root{--bg:#0b0f14;--card:#11161d;--edge:#1d2632;--ink:#dce3ea;--mut:#7e8b99;
       --up:#3fcf8e;--dn:#f0556d;--warn:#e8a54b;--acc:#8b7cf6;--blue:#58a6ff}
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--ink);margin:0;padding:18px 20px;
  font:14px/1.5 ui-sans-serif,-apple-system,'Segoe UI',Inter,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
 h1{font-size:17px;font-weight:650;margin:0 0 14px;display:flex;align-items:baseline;gap:8px}
 h1 .muted{font-size:12px;font-weight:400}
 h2{font-size:11px;color:var(--mut);margin:0 0 10px;text-transform:uppercase;letter-spacing:.09em;font-weight:600}
 .grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
 /* Macro + Economy: exactly 3 columns on wide screens so 5-6 cards tile
    without orphan gaps; charts get the extra width. */
 @media(min-width:1200px){#tab-macro .grid,#tab-economy .grid{grid-template-columns:repeat(3,1fr)}}
 .card{background:linear-gradient(180deg,#131922 0%,var(--card) 100%);border:1px solid var(--edge);
  border-radius:14px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.35)}
 .big{grid-column:1/-1}
 table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
 td,th{padding:5px 6px;text-align:right;border-bottom:1px solid var(--edge)}
 td:first-child,th:first-child{text-align:left} th{color:var(--mut);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
 tr:last-child td{border-bottom:none}
 .pos{color:var(--up)}.neg{color:var(--dn)}.muted{color:var(--mut)}.warn{color:var(--warn)}
 .tile{display:inline-block;margin:4px 16px 4px 0}.tile b{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums}
 .pill{display:inline-block;padding:2px 9px;border-radius:99px;background:#1a2230;border:1px solid var(--edge);font-size:12px;margin:2px}
 .tabs{margin:0 0 16px}.tabs button{background:transparent;border:1px solid var(--edge);color:var(--mut);
  padding:7px 18px;border-radius:99px;margin-right:6px;cursor:pointer;font-size:13px;transition:all .15s}
 .tabs button:hover{color:var(--ink)}
 .tabs button.on{color:var(--ink);border-color:var(--up);background:rgba(63,207,142,.08)}
 .tab{display:none}.tab.on{display:block}
 #ranges{display:inline-flex;background:var(--bg);border:1px solid var(--edge);border-radius:99px;padding:2px;margin:2px 0 12px}
 #ranges button{background:transparent;border:none;color:var(--mut);padding:4px 12px;border-radius:99px;margin:0;font-size:12px;cursor:pointer}
 #ranges button.on{background:var(--up);color:#08110c;font-weight:600}
 .srow{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12.5px}
 .srow .nm{width:118px;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .srow .val{width:52px;text-align:right;font-variant-numeric:tabular-nums}
 .track{flex:1;height:7px;border-radius:4px;background:#1a2230;position:relative;overflow:hidden}
 .track i{position:absolute;top:0;bottom:0;border-radius:4px}
 .hl{margin:5px 0;font-size:12.5px;line-height:1.4}
 .hl .src{color:var(--mut);font-size:11px}
 .dot{display:inline-block;width:8px;height:8px;border-radius:99px;margin-right:6px}
 .ev{display:flex;justify-content:space-between;margin:4px 0;font-size:12.5px}
 .ok{color:var(--up)}.stale{color:var(--warn)}
</style></head><body>
<h1>📈 Trading Agent <span class="muted" id="asof"></span></h1>
<div class="tabs">
 <button data-t="portfolio" class="on">Portfolio</button>
 <button data-t="macro">Macro</button>
 <button data-t="economy">Economy</button>
 <button data-t="memory">Memory</button>
</div>

<div class="tab on" id="tab-portfolio"><div class="grid">
 <div class="card big"><h2>Equity (paper) <span id="eqret" style="float:right"></span></h2>
  <div id="ranges">
   <button data-r="today">Today</button><button data-r="1w">1W</button>
   <button data-r="1m">1M</button><button data-r="3m">3M</button>
   <button data-r="6m">6M</button><button data-r="ytd">YTD</button>
   <button data-r="1y" class="on">1Y</button><button data-r="all">All</button>
  </div><canvas id="eq" height="84"></canvas></div>
 <div class="card big"><h2>Strategy race · normalized to 100
  <span class="muted" style="text-transform:none;letter-spacing:0">— click legend entries to toggle series</span></h2>
  <div id="raceEmpty" class="muted" style="display:none;padding:18px 0"></div>
  <canvas id="race" height="84"></canvas><div id="pmline" class="muted" style="margin-top:8px"></div></div>
 <div class="card"><h2>Account</h2><div id="account"></div><h2 style="margin-top:12px">Positions</h2><div id="positions"></div></div>
 <div class="card"><h2>Committee (latest)</h2><div id="committee"></div>
  <h2 style="margin-top:14px">Posture history <span class="muted" style="text-transform:none;letter-spacing:0">— dot color = posture, height = dissent</span></h2>
  <canvas id="postCh" height="56"></canvas></div>
 <div class="card"><h2>Sector momentum · 1M vs SPY</h2><div id="sectors"></div></div>
 <div class="card"><h2>Headlines feeding the scout</h2><div id="heads"></div></div>
 <div class="card"><h2>Next up (UTC)</h2><div id="sched"></div>
  <h2 style="margin-top:14px">System</h2><div id="ops"></div></div>
</div></div>

<div class="tab" id="tab-macro"><div class="grid">
 <div class="card"><h2>Yield curve · 10y − 3m</h2><div id="curveTile"></div><canvas id="curveCh" height="110"></canvas></div>
 <div class="card"><h2>VIX term structure</h2><div id="vixTile"></div><canvas id="vixCh" height="110"></canvas></div>
 <div class="card"><h2>Breadth · % above SMA</h2><div id="brTile"></div><canvas id="brCh" height="110"></canvas></div>
 <div class="card"><h2>Credit & risk appetite (1y base = 100)</h2><div id="ratioTile"></div><canvas id="ratioCh" height="110"></canvas></div>
 <div class="card"><h2>Macro dial (z-scores)</h2><div id="macro"></div></div>
 <div class="card"><h2>Vol surface (SPY options)</h2><div id="vol"></div></div>
</div></div>

<div class="tab" id="tab-economy"><div class="grid">
 <div class="card"><h2>Inflation</h2><div id="ecInfT"></div><canvas id="ecInf" height="110"></canvas></div>
 <div class="card"><h2>Housing</h2><div id="ecHouT"></div><canvas id="ecHou" height="110"></canvas></div>
 <div class="card"><h2>Labor</h2><div id="ecLabT"></div><canvas id="ecLab" height="110"></canvas></div>
 <div class="card"><h2>Credit & liquidity</h2><div id="ecCrT"></div><canvas id="ecCr" height="110"></canvas></div>
 <div class="card"><h2>Consumer</h2><div id="ecCoT"></div><canvas id="ecCo" height="110"></canvas></div>
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
const fx=(x,d=1)=>x==null?'–':Number(x).toFixed(d);
const num=(x)=>x==null?'–':Number(x).toLocaleString(undefined,{maximumFractionDigits:0});
const line=(el,labels,sets)=>{
 const scales={x:{ticks:{color:'#8b98a5',maxTicksLimit:7}},y:{ticks:{color:'#8b98a5'}}};
 if(sets.some(s=>s.yAxisID==='y1'))scales.y1={position:'right',ticks:{color:'#8b98a5'},grid:{drawOnChartArea:false}};
 return new Chart(document.getElementById(el),{type:'line',
  data:{labels,datasets:sets.map(s=>({...s,borderWidth:1.6,pointRadius:s.pointRadius??0,tension:.2,fill:false}))},
  options:{plugins:{legend:{display:sets.some(s=>s.label),labels:{color:'#8b98a5',boxWidth:10}}},scales}});};
setTimeout(()=>location.reload(),300e3); // fresh data every 5 minutes
fetch('api/summary').then(r=>r.json()).then(d=>{
 document.getElementById('asof').textContent=' · '+new Date(d.generated_at).toLocaleString()+' · auto-refreshes';
 const daily=d.equity_curve||[],today=d.equity_today||[];let eqChart=null,raceChart=null;
 const cutFor=(range)=>{
  const now=new Date();
  const days={'1w':7,'1m':31,'3m':92,'6m':183,'1y':365}[range];
  if(days)return new Date(now-days*864e5);
  if(range==='ytd')return new Date(now.getFullYear(),0,1);
  return null;};
 function drawRace(range){
  const pmH=((d.agent_pm||{}).history||[]).map(h=>({t:String(h.t).slice(0,10),v:+h.equity,spy:h.spy?+h.spy:null}));
  // The race only makes sense over the sim's lifetime: the paper book's
  // history includes capital injections that wreck any longer-window
  // normalization. Clamp the window to PM inception.
  let cut=range==='today'?null:cutFor(range);
  if(pmH.length){const pmStart=new Date(pmH[0].t);if(!cut||cut<pmStart)cut=pmStart;}
  const inWin=p=>!cut||new Date(p.t)>=cut;
  const paper=daily.filter(inWin);
  const pm=pmH.filter(inWin);
  const spy=pmH.filter(h=>h.spy!=null).filter(inWin);
  const dates=[...new Set([...paper.map(p=>p.t),...pm.map(p=>p.t),...spy.map(p=>p.t)])].sort();
  const mk=(pts,key,label,color)=>{
   if(pts.length<2)return null;
   const m={};pts.forEach(p=>m[p.t]=p[key]);  // last same-day point wins
   const base=pts[0][key];
   return {label,data:dates.map(dt=>m[dt]!=null?100*m[dt]/base:null),borderColor:color,spanGaps:true};};
  const sets=[mk(paper,'v','momentum top-k (paper)','#4cc38a'),
              mk(pm,'v','agent PM (sim)','#b07cf6'),
              mk(spy,'spy','SPY','#8b98a5')].filter(Boolean);
  if(raceChart){raceChart.destroy();raceChart=null;}
  const ok=sets.length&&dates.length>1;
  document.getElementById('race').style.display=ok?'':'none';
  const em=document.getElementById('raceEmpty');
  em.style.display=ok?'none':'';
  em.textContent=pmH.length?
   'sim started today — the race plots from its second daily mark (21:15 UTC tomorrow); both curves rebase to 100 at sim inception':
   'no sim book yet — /pm run in Telegram starts one';
  if(ok)raceChart=line('race',dates,sets);
 }
 function drawEq(range){
  let pts;
  if(range==='today'){pts=today.map(p=>({t:p.t.slice(11,16),v:p.v}));}
  else{
   const cut=cutFor(range);
   pts=daily.filter(p=>!cut||new Date(p.t)>=cut).map(p=>({t:p.t,v:p.v}));
  }
  const ret=pts.length>1?(pts[pts.length-1].v/pts[0].v-1):null;
  document.getElementById('eqret').innerHTML=ret==null?'':
   `<span class="${ret>=0?'pos':'neg'}">${pct(ret,2)}</span>`;
  if(eqChart)eqChart.destroy();
  eqChart=line('eq',pts.map(p=>p.t),[{data:pts.map(p=>p.v),label:'',
   borderColor:(ret??0)>=0?'#4cc38a':'#e5534b'}]);
 }
 document.querySelectorAll('#ranges button').forEach(b=>b.onclick=(e)=>{
  e.stopPropagation();
  document.querySelectorAll('#ranges button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');drawEq(b.dataset.r);drawRace(b.dataset.r);});
 drawEq('1y');drawRace('1y');
 const apm=d.agent_pm||{};
 if(apm.history&&apm.history.length){
  const h0=apm.history[0],hN=apm.history[apm.history.length-1];
  const base=+(apm.start_equity||h0.equity);
  const simRet=(+hN.equity/base-1);
  const hold=Object.keys(apm.holdings||{}).sort().join(', ')||'all cash';
  const lr=apm.last_run||{};
  document.getElementById('pmline').innerHTML=
   `🧪 sim started <b>${String(h0.t).slice(0,10)}</b> at <b>$${num(base)}</b> · now ${num(hN.equity)} `+
   `(<span class="${simRet>=0?'pos':'neg'}">${pct(simRet,2)}</span>) · holds: <b>${hold}</b> · cash ${num(apm.cash)}`+
   (lr.rationale?`<br>last rationale: ${String(lr.rationale).slice(0,220)}…`:'');
 } else document.getElementById('pmline').textContent='agent PM has no book yet — /pm run in Telegram starts one';
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

 // Posture history: dissent line, dots colored by posture.
 const ch=d.committee_history||[];
 if(ch.length>1){
  const pcol={'risk_on':'#3fcf8e','neutral':'#7e8b99','risk_off':'#f0556d'};
  line('postCh',ch.map(h=>String(h.t).slice(5,10)),[{data:ch.map(h=>h.dissent),
   borderColor:'#58a6ff',pointRadius:4,
   pointBackgroundColor:ch.map(h=>pcol[h.posture]||'#7e8b99')}]);
 }

 // Sector momentum bars (1m vs SPY, sorted).
 const sm=Object.entries((d.news||{}).sector_momentum||{})
  .map(([n,v])=>({n,etf:v.etf,m:v.ret_1m_vs_spy}))
  .filter(s=>s.m!=null).sort((a,b)=>b.m-a.m);
 if(sm.length){
  const mx=Math.max(...sm.map(s=>Math.abs(s.m)),.1);
  document.getElementById('sectors').innerHTML=sm.map(s=>{
   const w=50*Math.abs(s.m)/mx, left=s.m<0?50-w:50;
   return `<div class="srow"><span class="nm">${s.n.replace(/_/g,' ')} <span class="muted">${s.etf}</span></span>
    <div class="track"><i style="left:${left}%;width:${w}%;background:${s.m>=0?'var(--up)':'var(--dn)'}"></i></div>
    <span class="val ${s.m>=0?'pos':'neg'}">${s.m>=0?'+':''}${s.m.toFixed(1)}%</span></div>`;}).join('');
 } else document.getElementById('sectors').innerHTML='<span class="muted">news watch hasn\\'t collected yet — weekdays 13:40 UTC</span>';

 // Headlines.
 const hl=((d.news||{}).headlines||[]);
 document.getElementById('heads').innerHTML=hl.length?hl.slice(0,9).map(h=>
  `<div class="hl"><span class="pill" style="font-size:10px">${h.topic||''}</span> ${h.title}
   <span class="src">— ${h.source||'?'}</span></div>`).join(''):
  '<span class="muted">no headlines collected yet</span>';

 // Schedule: next occurrences of the recurring jobs (computed in UTC).
 const jobs=[
  {n:'🏛 committee',dow:[1,2,3,4,5],h:14,m:0},
  {n:'📰 news watch',dow:[1,2,3,4,5],h:13,m:40},
  {n:'🧪 PM rebalance',dow:[1],h:14,m:30},
  {n:'📊 PM daily mark',dow:[1,2,3,4,5],h:21,m:15},
  {n:'⚖️ rebalance (paper)',dow:[5],h:21,m:5},
  {n:'🎓 prediction grading',dow:[0,1,2,3,4,5,6],h:22,m:30},
  {n:'📜 historian',dow:[5],h:22,m:45}];
 const nowU=new Date();
 const nextOf=j=>{for(let i=0;i<8;i++){
   const c=new Date(Date.UTC(nowU.getUTCFullYear(),nowU.getUTCMonth(),nowU.getUTCDate()+i,j.h,j.m));
   if(c>nowU&&j.dow.includes(c.getUTCDay()))return c;}return null;};
 document.getElementById('sched').innerHTML=jobs.map(j=>({j,c:nextOf(j)})).filter(x=>x.c)
  .sort((a,b)=>a.c-b.c).slice(0,5).map(({j,c})=>{
   const mins=Math.round((c-nowU)/6e4);
   const rel=mins<60?mins+'m':mins<2880?Math.round(mins/60)+'h':Math.round(mins/1440)+'d';
   return `<div class="ev"><span>${j.n}</span><span class="muted">${String(c.getUTCHours()).padStart(2,'0')}:${String(c.getUTCMinutes()).padStart(2,'0')} · in ${rel}</span></div>`;}).join('');

 // Ops: halt + freshness.
 const ops=d.ops||{};const ages=ops.ages_min||{};
 const fr=(k,lbl,limit)=>{const a=ages[k];
  return `<div class="ev"><span>${lbl}</span><span class="${a==null?'muted':a<=limit?'ok':'stale'}">${a==null?'never':a<60?a+'m ago':Math.round(a/60)+'h ago'}</span></div>`;};
 document.getElementById('ops').innerHTML=
  `<div class="ev"><span>trading</span><span class="${ops.halted?'neg':'ok'}">${ops.halted?'⛔ HALTED '+(ops.halt_reason||''):'● active'}</span></div>`+
  fr('snapshot','broker snapshot',90)+fr('news','news watch',1560)+
  fr('market_watch','macro collector',1560)+fr('committee','last committee',1560)+fr('pm_book','PM book',1560);

 // ---- MACRO TAB
 const mh=(d.market_watch||{}).history||[];const last=(d.market_watch||{}).latest||{};
 const L=mh.map(h=>String(h.t).slice(0,10));
 if(mh.length){
  const cur=last.curve_10y3m;
  document.getElementById('curveTile').innerHTML=
   `<span class="tile"><b class="${cur<0?'neg':cur<0.3?'warn':'pos'}">${cur==null?'–':cur.toFixed(2)+'pp'}</b><br>
    <span class="muted">${cur<0?'INVERTED':'spread'} · 10y ${fx(last.y_10y,2)} / 3m ${fx(last.y_3m,2)}</span></span>`;
  line('curveCh',L,[{data:mh.map(h=>h.curve_10y3m),borderColor:'#5ab0f6',label:'10y−3m'}]);
  const vr=last.vix_ratio;
  document.getElementById('vixTile').innerHTML=
   `<span class="tile"><b>${fx(last.vix,1)}</b><br><span class="muted">VIX</span></span>
    <span class="tile"><b class="${vr>1?'neg':'pos'}">${fx(vr,2)}</b><br><span class="muted">VIX/VIX3M ${vr>1?'⚠ backwardation':''}</span></span>`;
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

 // ---- ECONOMY TAB (FRED)
 const ec=(d.econ||{}).series||{};
 const ecChart=(el,tEl,keys)=>{
  const ss=keys.map(([k,color,ax])=>({s:ec[k],color,ax})).filter(x=>x.s&&x.s.points&&x.s.points.length>1);
  document.getElementById(tEl).innerHTML=ss.map(x=>
   `<span class="tile"><b>${x.s.latest}${x.s.unit==='%'?'%':''}</b><br>
    <span class="muted">${x.s.label}${x.s.unit&&x.s.unit!=='%'?' ('+x.s.unit+')':''}</span></span>`).join('')||
   '<span class="muted">collector runs weekdays 11:00 UTC — first reading soon</span>';
  if(!ss.length)return;
  const dates=[...new Set(ss.flatMap(x=>x.s.points.map(p=>p.t)))].sort();
  line(el,dates,ss.map(x=>{const m=Object.fromEntries(x.s.points.map(p=>[p.t,p.v]));
   return {label:x.s.label,data:dates.map(dt=>m[dt]??null),borderColor:x.color,spanGaps:true,yAxisID:x.ax||'y'};}));
 };
 ecChart('ecInf','ecInfT',[['cpi_yoy','#f0556d'],['core_cpi_yoy','#e8a54b'],['breakeven_10y','#58a6ff']]);
 ecChart('ecHou','ecHouT',[['mortgage_30y','#58a6ff'],['case_shiller_yoy','#3fcf8e'],['housing_starts','#7e8b99','y1']]);
 ecChart('ecLab','ecLabT',[['unemployment','#f0556d'],['claims','#58a6ff','y1']]);
 ecChart('ecCr','ecCrT',[['hy_oas','#f0556d'],['curve_2s10s','#3fcf8e'],['fed_bs','#7e8b99','y1']]);
 ecChart('ecCo','ecCoT',[['retail_yoy','#3fcf8e'],['sentiment','#8b7cf6','y1']]);

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
