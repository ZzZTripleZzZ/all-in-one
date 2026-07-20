"""Diagnostic for the backfill MODEL arm: is there headroom for a learned runtime estimator, and
how big is the cold-start opportunity (users with little/no per-user history, where hist-qXX falls
back to the raw user limit elpl)? Pure pandas, no GPU. Mirrors run_backfill_replay.load() exactly."""
import os, glob, numpy as np, pandas as pd
NFILE=int(os.environ.get('NFILE',0)); DAYS=float(os.environ.get('DAYS',14)); SKIPD=float(os.environ.get('SKIPD',7))
MAXJ=int(os.environ.get('MAXJ',300000))

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
pre=d[d['t']<t0]
span_all=(d['t'].iloc[-1]-d['t'].iloc[0])/86400
print(f"file={os.path.basename(f)} total_jobs={len(d):,} file_span={span_all:.1f}d")
print(f"pre-window(train) jobs={len(pre):,} users={pre['usr'].nunique()} | window(eval) jobs={len(w):,} users={w['usr'].nunique()}")

# cold-start: for each window job, how many prior same-user jobs exist (in full d, submit<t_i)?
pre_counts={}
for u,g in pre.groupby('usr'): pre_counts[u]=len(g)
seen=dict(pre_counts)  # running count as we walk the window in submit order
nprior=np.empty(len(w),int)
for i,u in enumerate(w['usr'].values):
    nprior[i]=seen.get(u,0); seen[u]=seen.get(u,0)+1
cold=(nprior<3)  # hist-qXX falls back to elpl when <3 obs
print(f"\nwindow jobs with <3 prior same-user obs (hist->elpl fallback): {cold.sum():,} ({cold.mean()*100:.1f}%)")
print(f"  of those, users entirely NEW (0 prior): {(nprior==0).sum():,} ({(nprior==0).mean()*100:.1f}%)")

# how loose is elpl vs true dur (the status-quo estimate the model must beat, esp. on cold-start)?
r=w['lim'].values/np.maximum(w['dur'].values,1.0)
print(f"\nelpl/dur ratio (status-quo looseness): median={np.median(r):.2f} p90={np.quantile(r,0.9):.1f} mean={r.mean():.1f}")
print(f"  cold-start subset elpl/dur: median={np.median(r[cold]):.2f} p90={np.quantile(r[cold],0.9):.1f}")

# headroom: an *oracle-ish* per-user q90 of PAST dur vs true dur (how tight can history alone get?)
print(f"\ndur seconds: median={np.median(w['dur']):.0f} p90={np.quantile(w['dur'],0.9):.0f} max={w['dur'].max():.0f}")
print(f"nn nodes:    median={np.median(w['nn']):.0f} p90={np.quantile(w['nn'],0.9):.0f} max={w['nn'].max():.0f}")
# does elpl alone predict dur? corr of logs
lg=lambda a: np.log1p(np.asarray(a,float))
print(f"corr(log elpl, log dur)={np.corrcoef(lg(w['lim']),lg(w['dur']))[0,1]:.3f} | corr(log nn, log dur)={np.corrcoef(lg(w['nn']),lg(w['dur']))[0,1]:.3f}")
print("PROBE_DONE")
