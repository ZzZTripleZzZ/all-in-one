"""EASY-backfilling REPLAY on F-DATA (Fugaku job logs): the decision-level headline for the HPC leg
(playbook P1; precedent Gaussier SC'15 — better runtime estimates -> better backfilling -> lower slowdown).
Replay a contiguous window of real jobs (submit=adt, nodes=nnumr, true runtime=duration, user limit=elpl)
through an EASY scheduler; the ONLY thing that varies between arms is the WALLTIME ESTIMATE used for
scheduling decisions (kill limit stays the user's elpl in all arms, as in real systems):
  user-elpl    operator status quo (user-declared limit)
  hist-q90/99  per-user rolling quantile of PAST durations (causal, cheap predictor)
  oracle       true duration (upper bound)
  model-qXX    (plug-in later: per-job quantile from the generative model at submit time)
Metrics: mean & P95 bounded slowdown (bs=(wait+run)/max(run,60s)), utilization, #backfilled, #killed.
Env: NFILE(0) DAYS(14) SKIPD(7) NODES(0=auto P95 concurrency) HISTQ(0.9,0.99) MAXJ(300000).
"""
import os, sys, glob, heapq, numpy as np, pandas as pd

NFILE=int(os.environ.get('NFILE',0)); DAYS=float(os.environ.get('DAYS',14)); SKIPD=float(os.environ.get('SKIPD',7))
NODES=int(os.environ.get('NODES',0)); MAXJ=int(os.environ.get('MAXJ',300000))
HISTQ=[float(x) for x in os.environ.get('HISTQ','0.9,0.99').split(',')]

def load():
    f=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))[NFILE]
    d=pd.read_parquet(f,columns=['usr','adt','nnumr','elpl','duration'])
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce')
    d=d.dropna(subset=['adt']).sort_values('adt')
    d['t']=d['adt'].astype('datetime64[ns, UTC]').astype(np.int64)//10**9
    d['dur']=pd.to_numeric(d['duration'],errors='coerce').clip(lower=1)
    d['lim']=pd.to_numeric(d['elpl'],errors='coerce')
    d['nn']=pd.to_numeric(d['nnumr'],errors='coerce').fillna(1).clip(lower=1).astype(int)
    d=d.dropna(subset=['dur','lim'])
    d['lim']=np.maximum(d['lim'],d['dur'])                                     # trace consistency: no job outlives its limit
    t0=d['t'].iloc[0]+SKIPD*86400
    w=d[(d['t']>=t0)&(d['t']<t0+DAYS*86400)].head(MAXJ).reset_index(drop=True)
    print(f"file={os.path.basename(f)} window jobs={len(w):,} users={w['usr'].nunique()} span={DAYS}d",flush=True)
    return d,w,t0

def hist_quantile_estimates(d,w,q):
    """causal per-user rolling quantile of past durations (min 3 obs, else fall back to elpl)."""
    est=np.empty(len(w)); hist={}
    past=d[d['t']<w['t'].iloc[0]]
    for u,g in past.groupby('usr'): hist[u]=list(g['dur'].values[-50:])
    for i,(u,dur,lim) in enumerate(zip(w['usr'],w['dur'],w['lim'])):
        h=hist.get(u,[])
        est[i]=min(np.quantile(h,q)*1.1,lim) if len(h)>=3 else lim
        hist.setdefault(u,[]).append(dur)
        if len(hist[u])>50: hist[u]=hist[u][-50:]
    return np.maximum(est,60.0)

def easy_sim(sub,nn,dur,lim,est,cap):
    """EASY backfilling; returns (start[n], killed[n], nbf). Estimates drive decisions; kill at lim."""
    n=len(sub); order=np.argsort(sub,kind='stable')
    start=np.full(n,-1.0); killed=np.zeros(n,bool); nbf=0
    run=[]                                                                     # heap (finish_actual, j)
    runest={}                                                                  # j -> estimated finish
    free=cap; queue=[]; i=0; t=sub[order[0]]
    def finish_actual(j,st): return st+min(dur[j],lim[j])
    while i<n or queue or run:
        while run and run[0][0]<=t+1e-9:
            _,j=heapq.heappop(run); free+=nn[j]; runest.pop(j,None)
        while i<n and sub[order[i]]<=t+1e-9: queue.append(order[i]); i+=1
        progressed=True
        while progressed and queue:
            progressed=False
            h=queue[0]
            if nn[h]<=free:                                                    # start head
                st=t; start[h]=st; killed[h]=dur[h]>lim[h]
                heapq.heappush(run,(finish_actual(h,st),h)); runest[h]=st+est[h]
                free-=nn[h]; queue.pop(0); progressed=True; continue
            # reservation for head: earliest time enough nodes free per ESTIMATED finishes.
            # Expired estimates (job ran past its estimate) are extended to now+10min -- the standard
            # correction (Tsafrir TPDS'07); without it, underestimates poison every reservation.
            ends=sorted((max(runest[j],t+600.0),nn[j]) for j in runest)
            acc=free; shadow=None; spare=0
            for fe,nd in ends:
                acc+=nd
                if acc>=nn[h]: shadow=fe; spare=acc-nn[h]; break
            if shadow is None: break
            # backfill: any queued job that fits now and (est-end<=shadow or uses<=spare)
            for k in list(queue[1:]):
                if nn[k]<=free and (t+est[k]<=shadow+1e-9 or nn[k]<=spare):
                    st=t; start[k]=st; killed[k]=dur[k]>lim[k]
                    heapq.heappush(run,(finish_actual(k,st),k)); runest[k]=st+est[k]
                    free-=nn[k]; queue.remove(k); nbf+=1; progressed=True
                    if nn[k]<=spare: spare-=nn[k]
            if not progressed: break
        # advance time
        nt=[]
        if run: nt.append(run[0][0])
        if i<n: nt.append(sub[order[i]])
        if not nt: break
        t=max(t,min(nt))
    return start,killed,nbf

if __name__=='__main__':
    d,w,t0=load()
    sub=w['t'].values.astype(float); nn=w['nn'].values; dur=w['dur'].values.astype(float); lim=w['lim'].values.astype(float)
    FRAC=float(os.environ.get('FRACCAP',0))                                    # 0<FRAC<1: cap = FRAC * auto-P95 concurrency (DEPRECATED for cross-month use: P95 is burst-dominated and leaves bursty months unsaturated)
    RHO=float(os.environ.get('RHOCAP',0))                                      # 0<RHO<=1.2: cap = mean offered load / RHO (total node-seconds over submit span). RHO~0.95 => heavily saturated, ~0.80 => loaded; matches the 21_04 win band 18k-23k.
    if NODES==0:
        if RHO>0:
            span=float(sub.max()-sub.min())+1.0
            cap=int(float((nn.astype(np.float64)*dur).sum())/span/RHO)
            for _ in range(8):                                                 # fixed point: wide jobs dropped by `fit` carry big node-seconds, so offered must be computed on the FILTERED set
                fitc=nn<=cap
                off=float((nn[fitc].astype(np.float64)*dur[fitc]).sum())/span
                ncap=max(int(off/RHO),int(nn.min()))
                if ncap==cap: break
                cap=ncap
            print(f"fixed-point fit-offered={off:,.0f} nodes avg over {span/86400:.1f}d, RHO={RHO} -> cap={cap}",flush=True)
        else:
            ev=[]
            for s,du,k in zip(sub,dur,nn): ev.append((s,k)); ev.append((s+du,-k))
            ev.sort(); cur=0; peak=[]
            for _,dk in ev: cur+=dk; peak.append(cur)
            cap=int(np.quantile(peak,0.95))
            if FRAC>0: cap=int(cap*FRAC); print(f"auto-P95 concurrency scaled by {FRAC} -> cap={cap}",flush=True)
            else: cap=max(cap,int(nn.max()))
    else: cap=NODES
    arms={'user-elpl':np.maximum(lim,60.0),'oracle':np.maximum(dur,60.0)}
    for q in HISTQ: arms[f'hist-q{int(q*100)}']=hist_quantile_estimates(d,w,q)
    mfile=os.environ.get('MODEL_EST','')
    if mfile:                                                                  # plug-in: npz {est:[n]} aligned to window jobs
        arms['model']=np.maximum(np.load(os.path.expanduser(mfile))['est'],60.0)
    fit=nn<=cap                                                                # jobs wider than the machine can never start: drop (standard when down-scaling capacity) else they deadlock FCFS
    print(f"cluster capacity={cap} nodes | dropped too-wide jobs: {(~fit).sum()} ({(~fit).mean()*100:.2f}%)",flush=True)
    sub,nn,dur,lim=sub[fit],nn[fit],dur[fit],lim[fit]
    arms={nm:np.asarray(est,float)[fit] for nm,est in arms.items()}
    print(f"\n{'arm':<12} {'meanBS':>8} {'P95BS':>8} {'util%':>7} {'#backfill':>10} {'medWait(s)':>11}",flush=True)
    for nm,est in arms.items():
        start,killed,nbf=easy_sim(sub,nn,dur,lim,est,cap)
        ok=start>=0
        wait=start[ok]-sub[ok]; run=np.minimum(dur[ok],lim[ok])
        bs=np.maximum((wait+run)/np.maximum(run,60.0),1.0)
        mk=(start[ok]+run).max()-sub.min()                                     # makespan of the replay
        util=float((nn[ok]*run).sum()/(cap*mk))*100.0
        print(f"{nm:<12} {bs.mean():>8.2f} {np.quantile(bs,0.95):>8.2f} {util:>7.1f} {nbf:>10,} {np.median(wait):>11.0f}  (started {ok.mean()*100:.1f}%)",flush=True)
    print("BACKFILL_REPLAY_DONE",flush=True)
