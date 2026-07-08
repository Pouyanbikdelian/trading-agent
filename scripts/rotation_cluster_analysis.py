"""Rotation assessment: defensive (XLV/XLRE/IBB) vs risk-on (XLK/SMH/URA).

Data: IBKR monthly closes, 2021-07 .. 2026-07 (61 monthly obs after dropping a
duplicate partial July bar). Reproduces the dashboard's RS-Ratio / RS-Momentum,
tests the cluster anti-correlation Yan spotted, lead-lag, and a monthly rotation
backtest. Small sample (61 obs) — treat backtest stats as directional.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# --- monthly closes (IBKR, contract-resolved US primary listings) ---
C = {
"SPY":[438.51,451.56,429.14,459.25,455.56,474.96,449.91,436.63,451.64,412,412.93,377.25,411.99,395.18,357.18,386.21,407.68,382.43,406.48,396.26,409.39,415.93,417.85,443.28,457.79,450.35,427.48,418.2,456.4,475.31,482.88,508.08,523.07,501.98,527.37,544.22,550.81,563.68,573.76,568.64,602.55,586.08,601.82,594.18,559.39,554.54,589.39,617.85,632.08,645.05,666.18,682.06,683.39,681.92,691.97,685.99,650.34,718.66,756.48,746.77,751.28,748.9],
"XLV":[132.15,135.21,127.3,133.82,129.73,140.89,131.23,129.96,136.99,130.29,132.23,128.24,132.4,124.76,121.11,132.75,139.02,135.85,133.36,127.17,129.46,133.53,127.83,132.73,134.15,133.21,128.74,124.54,131.31,136.38,140.38,144.82,147.73,140.33,143.7,145.75,149.63,157.2,154.02,146.87,147.41,137.57,146.87,148.93,146.01,140.47,132.64,134.79,130.43,137.43,139.17,144.25,157.65,154.8,154.74,160.2,146.61,145.99,149.47,158.66,161.96,164.07],
"XLRE":[46.38,47.68,44.45,47.82,47.39,51.81,47.34,45.07,48.32,46.6,44.22,40.86,44.34,41.85,36.01,36.73,39.24,36.93,40.59,38.21,37.38,37.75,36.04,37.69,38.19,37.02,34.07,33.1,37.23,40.06,38.13,39.11,39.53,36.19,38.05,38.41,41.19,43.55,44.67,43.2,45,40.67,41.42,43.15,41.85,41.3,41.73,41.42,41.41,42.31,42.13,40.9,41.67,40.35,41.43,43.84,40.83,44.4,43.99,44.03,44.29,45],
"IBB":[165.78,172.28,161.68,159.6,154.22,152.62,131.86,126.07,130.3,116.65,116.67,117.63,124.08,121.46,116.96,128.59,136.53,131.29,136.55,127.18,129.16,130.51,126.28,126.96,129.31,128.71,122.29,113.68,120.93,135.85,134.38,136.84,137.22,126.92,133.88,137.26,147.97,148.17,145.6,140.29,141.55,132.21,138.66,137.03,127.9,126.58,121.72,126.51,132.76,138.22,144.37,159.38,173.83,168.77,172.43,175.37,168.85,168.69,172.18,190.19,195.31,198.37],
"XLK":[76.7,79.43,74.66,80.765,84.36,86.935,80.985,77.035,79.465,70.71,70.225,63.56,72.11,67.63,59.39,63.935,67.98,62.22,67.98,68.26,75.505,75.415,82.14,86.93,89.175,87.83,81.965,82.005,92.58,96.24,98.84,103.49,104.135,98.135,105.08,113.115,109.4,110.165,112.88,111.12,116.865,116.26,115.405,112.765,103.24,104.985,115.455,126.615,131.37,131.225,140.93,150.34,143.11,143.97,143.88,138.76,132.9,159.5,191.02,190.52,183.57,180.14],
"SMH":[131.57,135.415,128.135,136.82,152.45,154.395,137.7,134.075,134.895,114.925,122.29,101.865,118.58,107.285,92.56,94.61,113.865,101.47,118.55,119.7,131.595,123.625,144.33,152.25,160.62,156.22,144.98,138.95,160.48,174.87,185.87,211.95,224.99,214.09,240.48,260.7,246.99,243.46,245.45,241.68,242.13,242.17,243.62,232.77,211.47,211.28,239.75,278.88,288.78,290.29,326.36,363.02,352.28,360.13,403.46,406.37,383.4,506.72,598.93,655.89,604.3,582.35],
"URA":[20.01,20.95,23.79,26.86,25.04,22.82,20.66,24.11,26.16,23.19,21.94,18.56,21.58,23.34,19.82,20.2,21.49,20.08,23.03,20.93,19.93,20,19.89,21.7,22.7,24.01,27.04,26.84,28.96,27.69,30.32,27.46,28.83,28.77,32.24,28.95,28.34,25.9,28.61,30.73,32.35,26.78,28.31,24.8,22.92,25,32.01,38.81,39.36,40.67,47.67,55.12,45.28,42.73,54.99,54.34,48.43,56.42,50.76,43.7,43.88,42.02],
}
for k,v in C.items():
    assert len(v)==62, (k,len(v))
idx = pd.period_range("2021-07","2026-07",freq="M").to_timestamp()   # 61
px = pd.DataFrame({k:v[:61] for k,v in C.items()}, index=idx)
DEF=["XLV","XLRE","IBB"]; RISK=["XLK","SMH","URA"]; THEMES=DEF+RISK

print("="*72); print("WINDOW", px.index[0].date(),"→",px.index[-1].date(), f"({len(px)} monthly obs)")
tot=(px.iloc[-1]/px.iloc[0]-1)*100
print("\nTotal return over window (%):"); print(tot.round(1).to_string())

# --- Relative strength vs SPY and monthly relative returns ---
rs = px[THEMES].div(px["SPY"],axis=0)               # relative-strength line
rel = rs.pct_change().dropna()                       # monthly relative return (beta-stripped)
absr = px[THEMES+["SPY"]].pct_change().dropna()      # absolute monthly return

Dr, Rr = rel[DEF].mean(axis=1), rel[RISK].mean(axis=1)   # equal-weight cluster composites

print("\n"+"="*72); print("1) CLUSTER ANTI-CORRELATION  (monthly RELATIVE-to-SPY returns)")
print(f"   corr(DEF composite, RISK composite) = {Dr.corr(Rr):+.2f}")
print(f"   corr on ABSOLUTE returns            = {absr[DEF].mean(1).corr(absr[RISK].mean(1)):+.2f}  (beta not stripped)")
print("\n   Full pairwise corr of relative returns (DEF rows vs RISK cols):")
pc = pd.DataFrame({r:{d:rel[d].corr(rel[r]) for d in DEF} for r in RISK})
print(pc.round(2).to_string())
print("\n   Within-cluster corr (should be POSITIVE if clusters are real):")
print(f"     DEF internal avg  = {np.mean([rel[a].corr(rel[b]) for a in DEF for b in DEF if a<b]):+.2f}")
print(f"     RISK internal avg = {np.mean([rel[a].corr(rel[b]) for a in RISK for b in RISK if a<b]):+.2f}")

print("\n"+"="*72); print("2) LEAD-LAG  corr(DEF_rel(t), RISK_rel(t+k))  [k<0: RISK leads DEF]")
for k in range(-3,4):
    print(f"     k={k:+d} : {Dr.corr(Rr.shift(-k)):+.2f}")

# --- RRG reproduction (monthly analogue of dashboard 63/21 daily) ---
print("\n"+"="*72); print("3) RRG QUADRANTS  (RS-Ratio 6m-mean, RS-Momentum 3m; dashboard logic)")
ratio = 100.0*rs/rs.rolling(6).mean()
mom = 100.0*ratio/ratio.shift(3)
def quad(x,y): return ("leading" if x>=100 and y>=100 else "weakening" if x>=100 else "improving" if y>=100 else "lagging")
Q = pd.DataFrame({c:[quad(ratio[c].iloc[i],mom[c].iloc[i]) for i in range(len(px))] for c in THEMES}, index=idx).iloc[8:]
def side(row):
    d=sum(row[c]=="leading" for c in DEF)-sum(row[c]=="lagging" for c in DEF)
    r=sum(row[c]=="leading" for c in RISK)-sum(row[c]=="lagging" for c in RISK)
    return d,r
opp=sum(1 for _,row in Q.iterrows() if (side(row)[0]>0 and side(row)[1]<0) or (side(row)[0]<0 and side(row)[1]>0))
print(f"   months where clusters sit on OPPOSITE sides (one leading-tilt, other lagging-tilt): {opp}/{len(Q)} = {opp/len(Q)*100:.0f}%")
print("   latest quadrants:"); print(Q.iloc[-1].to_string())

# --- Backtest: monthly rotation using 3m relative-momentum spread ---
print("\n"+"="*72); print("4) BACKTEST  monthly rotation, signal = 3m relative momentum spread")
mom3 = rs.pct_change(3)                       # 3-month relative momentum per ETF
sig = (mom3[RISK].mean(1) - mom3[DEF].mean(1)).dropna()   # +ve => risk-on cluster leading
retA = absr.reindex(sig.index)                # aligned absolute returns
def cluster_ret(names): return retA[names].mean(1)
spy = retA["SPY"]
# long-only rotation: hold whichever cluster's momentum leads (decided at t-1, realized t)
pos = np.where(sig.shift(1)>0, "RISK","DEF")
rot = pd.Series([cluster_ret(RISK).iloc[i] if pos[i]=="RISK" else cluster_ret(DEF).iloc[i] for i in range(len(sig))], index=sig.index)
# market-neutral: long leader cluster, short laggard cluster (exploits the anti-corr directly)
mn = pd.Series([ (cluster_ret(RISK).iloc[i]-cluster_ret(DEF).iloc[i]) if sig.shift(1).iloc[i]>0 else (cluster_ret(DEF).iloc[i]-cluster_ret(RISK).iloc[i]) for i in range(len(sig))], index=sig.index).dropna()
def stats(r,label):
    r=r.dropna(); n=len(r); ann=12
    cagr=((1+r).prod())**(ann/n)-1; vol=r.std()*np.sqrt(ann); shp=(r.mean()*ann)/(r.std()*np.sqrt(ann)) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    print(f"   {label:22s} CAGR {cagr*100:6.1f}%  vol {vol*100:5.1f}%  Sharpe {shp:5.2f}  maxDD {dd*100:6.1f}%  hit {100*(r>0).mean():4.0f}%")
print(f"   (n={len(rot.dropna())} months, signal flips to RISK {100*(pd.Series(pos)=='RISK').mean():.0f}% of months)")
stats(spy,"SPY (benchmark)"); stats(cluster_ret(DEF),"Always DEF"); stats(cluster_ret(RISK),"Always RISK")
stats(rot,"Rotation (long-only)"); stats(mn,"Long leader/short lag")

# --- Regime split ---
print("\n"+"="*72); print("5) SUB-PERIOD corr(DEF_rel,RISK_rel)")
for lab,a,b in [("2021-07..2022-12 (bear/infl)","2021-07","2022-12"),("2023-01..2024-12 (AI bull)","2023-01","2024-12"),("2025-01..2026-07 (recent)","2025-01","2026-07")]:
    m=(Dr.index>=a)&(Dr.index<=b); print(f"   {lab:30s} corr={Dr[m].corr(Rr[m]):+.2f}  n={m.sum()}")
print("\n"+"="*72); print("6) PERSISTENCE  rolling 12m corr(DEF_rel, RISK_rel)")
rc=Dr.rolling(12).corr(Rr).dropna()
print(f"   mean {rc.mean():+.2f}  min {rc.min():+.2f}  max {rc.max():+.2f}  share of windows <0: {100*(rc<0).mean():.0f}%")

print("\n"+"="*72); print("7) MEAN-REVERSION variant (buy the LAGGING cluster; fade the spread)")
posM=np.where(sig.shift(1)>0,"DEF","RISK")   # if RISK led last month, buy DEF (laggard)
rotM=pd.Series([cluster_ret(RISK).iloc[i] if posM[i]=="RISK" else cluster_ret(DEF).iloc[i] for i in range(len(sig))],index=sig.index)
mnM=pd.Series([ (cluster_ret(DEF).iloc[i]-cluster_ret(RISK).iloc[i]) if sig.shift(1).iloc[i]>0 else (cluster_ret(RISK).iloc[i]-cluster_ret(DEF).iloc[i]) for i in range(len(sig))],index=sig.index).dropna()
stats(rotM,"MeanRev long-only"); stats(mnM,"MeanRev long/short")

print("\n"+"="*72); print("8) CURRENT STATE (as of last obs)")
print(f"   3m rel-momentum spread (RISK-DEF) = {sig.iloc[-1]:+.3f}  -> momentum favors {'RISK' if sig.iloc[-1]>0 else 'DEF'}")
print(f"   latest 1m relative move: DEF {Dr.iloc[-1]*100:+.1f}%  RISK {Rr.iloc[-1]*100:+.1f}%")
print(f"   RRG tilt now: DEF leading-count {sum(Q.iloc[-1][c]=='leading' for c in DEF)}/3, RISK leading-count {sum(Q.iloc[-1][c]=='leading' for c in RISK)}/3")
print("\nDONE")
