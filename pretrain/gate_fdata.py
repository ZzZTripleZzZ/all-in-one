"""GO/NO-GO GATE (falsification-first, per the 2026-07-16 adversarial pre-mortem):
Does F-DATA job-duration have CONDITIONAL tail signal that a sequence model can exploit, and do
per-job TABULAR COVARIATES dominate the univariate-duration history? We compare quantile pinball loss
(overall + tail) of three predictors of log-duration:
  (1) UNCONDITIONAL  : constant = train empirical q-quantile (no model, no context)
  (2) LAGGED-ONLY    : quantile-GBM on {lag1,lag2,lag3,running-mean} = what the sequence model sees
  (3) FULL-COVARIATES: quantile-GBM on lags + {nodes,cores,freq,priority,wait,hour,dow}
Interpretation:  (2)<<(1) => conditional signal exists (GO).  (2)~=(1) => aleatoric tail (NO-GO).
                 (3)<<(2) => covariates dominate => the model MUST ingest covariates, not a fancier head.
CPU-only, minutes. Run as a compute job.  Data: ~/nfm/data/fdata/*.parquet
"""
import glob, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
np.random.seed(0)
D=os.path.expanduser("~/nfm/data/fdata")
cols=['usr','adt','duration','nnumr','cnumr','freq_req','pri','qdt','sdt']
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(D+"/*.parquet"))],ignore_index=True)
d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce'); d=d.dropna(subset=['adt']).sort_values(['usr','adt'])
for c in ['qdt','sdt']: d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
d['wait']=np.log1p((d['sdt']-d['qdt']).dt.total_seconds().clip(lower=0)).fillna(0)
d['hour']=d['adt'].dt.hour; d['dow']=d['adt'].dt.dayofweek
for c in ['nnumr','cnumr','freq_req','pri']: d[c]=np.log1p(pd.to_numeric(d[c],errors='coerce').fillna(0).clip(lower=0))
d['y']=np.log1p(pd.to_numeric(d['duration'],errors='coerce').clip(lower=0)); d=d.dropna(subset=['y'])
g=d.groupby('usr')['y']
d['lag1']=g.shift(1); d['lag2']=g.shift(2); d['lag3']=g.shift(3)
d['umean']=g.transform(lambda s: s.shift(1).expanding().mean())
d=d.dropna(subset=['lag1','lag2','lag3','umean'])
cut=d['adt'].quantile(0.8); tr=d[d['adt']<=cut].copy(); te=d[d['adt']>cut].copy()   # temporal split
def pinball(y,p,q): e=np.asarray(y,float)-np.asarray(p,float); return float(np.mean(np.maximum(q*e,(q-1)*e)))
lagf=['lag1','lag2','lag3','umean']; fullf=lagf+['nnumr','cnumr','freq_req','pri','wait','hour','dow']
print(f"train={len(tr):,}  test={len(te):,}  (temporal split by arrival time)",flush=True)
def run(name, y_eval,Xte):
    print(f"\n--- {name} (n={len(y_eval):,}) ---",flush=True)
    for q in [0.5,0.9,0.99]:
        p1=pinball(y_eval,np.full(len(y_eval),np.quantile(tr['y'],q)),q)
        m2=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=250,learning_rate=0.08).fit(tr[lagf],tr['y'])
        p2=pinball(y_eval,m2.predict(Xte[lagf]),q)
        m3=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=250,learning_rate=0.08).fit(tr[fullf],tr['y'])
        p3=pinball(y_eval,m3.predict(Xte[fullf]),q)
        print(f"  q={q}: (1)uncond={p1:.4f}  (2)lagged={p2:.4f}  (3)full-cov={p3:.4f}  |  cond-signal (1->2)={(p1-p2)/p1:+.1%}  covariate-gain (2->3)={(p2-p3)/p2:+.1%}",flush=True)
run("ALL test",te['y'],te)
thr=np.quantile(te['y'],0.9); tail=te[te['y']>=thr]                                  # top-10% longest jobs
run("TAIL (top-10% duration)",tail['y'],tail)
print("\nGATE_DONE",flush=True)
