"""Strong walltime-estimator baselines for the backfill replay, emitted as MODEL_EST npz files:
  tsafrir   average of the user's last TWO actual runtimes (Tsafrir TPDS'07, the canonical cheap
            estimator reviewers ask for); fallback = user limit when <2 prior jobs.
  gbm-qXX   HistGradientBoosting QUANTILE regression on causal tabular features (log nodes, log
            limit, hour-of-day, day-of-week, per-user rolling stats of past log-durations:
            count/mean/median/std/last/last2avg). Trained on pre-window rows only (t<t0); at
            inference the rolling features are grown in submit order — same causality contract as
            hist_quantile_estimates and the NFM model arm. This is the fair non-sequence learned
            baseline (fixes the earlier under-featured-GBM criticism).
Alignment: est arrays are aligned to the FULL eval window (before the too-wide fit filter), same as
run_backfill_model.py. Env: NFILE DAYS SKIPD MAXJ QS(0.6,0.9) OUT(~/nfm/data/backfill_base).
"""
import os, glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

NFILE=int(os.environ.get('NFILE',0)); DAYS=float(os.environ.get('DAYS',14)); SKIPD=float(os.environ.get('SKIPD',7))
MAXJ=int(os.environ.get('MAXJ',300000))
QS=[float(x) for x in os.environ.get('QS','0.6,0.9').split(',')]
OUT=os.path.expanduser(os.environ.get('OUT','~/nfm/data/backfill_base')); os.makedirs(OUT,exist_ok=True)

def load():
    f=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))[NFILE]
    d=pd.read_parquet(f,columns=['usr','adt','nnumr','elpl','duration'])
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce')
    d=d.dropna(subset=['adt']).sort_values('adt')
    d['t']=d['adt'].astype('datetime64[ns, UTC]').astype(np.int64)//10**9
    d['dur']=pd.to_numeric(d['duration'],errors='coerce').clip(lower=1)
    d['lim']=pd.to_numeric(d['elpl'],errors='coerce')
    d['nn']=pd.to_numeric(d['nnumr'],errors='coerce').fillna(1).clip(lower=1).astype(int)
    d=d.dropna(subset=['dur','lim']); d['lim']=np.maximum(d['lim'],d['dur'])
    t0=d['t'].iloc[0]+SKIPD*86400
    w=d[(d['t']>=t0)&(d['t']<t0+DAYS*86400)].head(MAXJ).reset_index(drop=True)
    return d,w,t0,os.path.basename(f)

lg=lambda a: np.log1p(np.asarray(a,float))

def features_causal(d,w,t0):
    """walk ALL jobs in submit order; per row emit features from the user's history BEFORE it.
    Returns (F_all, y_all, is_train, w_row_index)."""
    hist={}
    wt=w['t'].values; wu=w['usr'].values
    # window rows are exactly the first len(w) d-rows in [t0,t0+DAYS) in the same sort order,
    # so sequential matching while walking d recovers the w-row index for each
    F=[];Y=[];TR=[];WI=[]
    wi=0
    tend=w['t'].iloc[-1] if len(w) else t0
    for u,t,nn,lim,dur in zip(d['usr'].values,d['t'].values,d['nn'].values,d['lim'].values,d['dur'].values):
        inwin = (t>=t0) and (wi<len(w)) and (t<=tend)
        h=hist.get(u,[])
        k=len(h)
        last=h[-1] if k>0 else 0.0
        last2=(h[-1]+h[-2])/2 if k>1 else last
        arr=np.array(h[-50:],float) if k>0 else np.zeros(1)
        hour=(t%86400)/3600.0; dow=((t//86400)%7)
        F.append([lg([nn])[0],lg([lim])[0],hour,dow,float(k),arr.mean(),np.median(arr),arr.std(),last,last2])
        Y.append(lg([dur])[0])
        if t<t0: TR.append(len(F)-1); WI.append(-1)
        elif inwin and wu[wi]==u and wt[wi]==t: WI.append(wi); wi+=1
        else: WI.append(-1)
        hist.setdefault(u,[]).append(lg([dur])[0])
        if len(hist[u])>100: hist[u]=hist[u][-100:]
    return np.array(F,np.float32),np.array(Y,np.float32),np.array(TR,int),np.array(WI,int)

if __name__=='__main__':
    d,w,t0,fn=load()
    print(f"file={fn} pre(train)={int((d['t']<t0).sum()):,} window(eval)={len(w):,} QS={QS}",flush=True)
    wu=w['usr'].values; wd=w['dur'].values; wl=w['lim'].values
    # ---- tsafrir last-2 (causal walk, mirrors hist_quantile_estimates) ----
    hist={}
    pre=d[d['t']<t0]
    for u,g in pre.groupby('usr'): hist[u]=list(g['dur'].values[-2:])
    est=np.empty(len(w))
    for i,(u,dur,lim) in enumerate(zip(wu,wd,wl)):
        h=hist.get(u,[])
        est[i]=np.mean(h[-2:]) if len(h)>=2 else lim
        hist.setdefault(u,[]).append(dur); hist[u]=hist[u][-2:]
    est=np.clip(est,60.0,np.maximum(wl,60.0))
    p=os.path.join(OUT,"tsafrir.npz"); np.savez(p,est=est)
    cov=(est>=wd).mean(); print(f"  tsafrir: median est/dur={np.median(est/np.maximum(wd,1)):.2f} coverage={cov:.3f} -> {p}",flush=True)
    # ---- GBM quantile regression on causal features ----
    F,Y,TR,WI=features_causal(d,w,t0)
    wmask=WI>=0
    print(f"  gbm features: train rows={len(TR):,} window rows matched={int(wmask.sum()):,}/{len(w):,}",flush=True)
    for q in QS:
        g=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=300,learning_rate=0.06,
                                        max_depth=None,min_samples_leaf=40,random_state=0)
        g.fit(F[TR],Y[TR])
        pr=np.full(len(w),np.nan)
        pr[WI[wmask]]=np.expm1(g.predict(F[wmask]))
        miss=np.isnan(pr)
        if miss.any(): pr[miss]=wl[miss]                                       # unmatched rows fall back to user limit
        pr=np.clip(pr,60.0,np.maximum(wl,60.0))
        p=os.path.join(OUT,f"gbm_q{int(round(q*100))}.npz"); np.savez(p,est=pr)
        cov=(pr>=wd).mean(); print(f"  gbm-q{int(round(q*100))}: median est/dur={np.median(pr/np.maximum(wd,1)):.2f} coverage={cov:.3f} miss={int(miss.sum())} -> {p}",flush=True)
    print("BACKFILL_BASELINES_DONE",flush=True)
