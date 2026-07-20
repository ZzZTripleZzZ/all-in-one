"""BACKBONE BAKE-OFF v2: fix the two structural handicaps found in the any-io run (462476), then re-fight.
Diagnosis from 462476 + depth run: (i) autoregressive ROLLOUT error accumulation is why one-model loses to
per-cell GBM at long horizons (GBM predicts t+h DIRECTLY); (ii) plain GRU Pareto-dominated HSTU -> recurrence,
not recsys attention, is the right mixer here; (iii) time-awareness is our one consistently-winning lever.
So: keep the HSTU CONCEPTS (K-bin tokenization + entity-event generative transduction + time-indexed memory),
swap the mixer (TGRv2 time-gated recurrence), and predict every horizon DIRECTLY (HorizonHead, no rollout).

Arms (all same data/split/epochs, 3 seeds):
  TGR+direct        ours: time-gated recurrence + direct multi-horizon mixture head
  TGR-notime+direct ablation: same arch, no time terms  -> isolates the dt lever
  GRU+direct        ablation: cuDNN GRU + direct head   -> isolates the recurrence arch
  HSTU+direct       fairness: was rollout HSTU's real problem?
  TGR+rollout       same TRAINED weights as TGR+direct, h=1 chained -> isolates direct-vs-rollout
  GBM-specialist / persistence / EWMA(0.3)  (operator baselines; GBM = one model PER cell)
Rollout-trained baselines (HSTU/TFM/LSTM/GRU+rollout) already measured in job 462476, same protocol/seeds.

Metrics per cell: per-series MASE (median point). Distributional at Lin=64: pinball@0.9/0.99 + CRPS from a
9-level quantile grid (the old 3-level "CRPS" under-covered; S=512 sampling). Serving latency measured inline.
Env: KC(256) D(512) H(8) L(8) EP(10) MIX(4) NSEED(3) SPLIT(96) LOUTMAX(64) EVALMAX(3000) BS(256) NFILES(0) ARMS().
"""
import os, sys, glob, time, pickle, numpy as np, pandas as pd, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, GRUv2, TGRv2, ContinuousInput, HorizonHead, dev
import metrics_v2 as MET
from sklearn.ensemble import HistGradientBoostingRegressor

KC=int(os.environ.get('KC',256)); D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8))
EP=int(os.environ.get('EP',10)); MIX=int(os.environ.get('MIX',4)); NSEED=int(os.environ.get('NSEED',3))
SPLIT=int(os.environ.get('SPLIT',96)); LOUTMAX=int(os.environ.get('LOUTMAX',64)); N=SPLIT+LOUTMAX
EVALMAX=int(os.environ.get('EVALMAX',3000)); BS=int(os.environ.get('BS',256)); NFILES=int(os.environ.get('NFILES',0))
SMOKE=int(os.environ.get('SMOKE',0)); SEED0=int(os.environ.get('SEED0',0))    # seed range [SEED0,NSEED): split seeds across parallel jobs
LINS=[8,16,32,64,96]; HS=[1,4,16,32,64]; QS9=tuple(np.linspace(0.1,0.9,9)); DLIN=64
if SMOKE: LINS=[8,32]; HS=[1,16]; DLIN=32; EP=2; NSEED=1; EVALMAX=800

def load():
    files=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))
    if SMOKE and NFILES==0: files=files[:1]
    elif NFILES>0: files=files[:NFILES]
    d=pd.concat([pd.read_parquet(f,columns=['usr','adt','duration']) for f in files],ignore_index=True)
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce'); d=d.dropna(subset=['adt']).sort_values(['usr','adt'])
    d['t']=d['adt'].astype('datetime64[ns, UTC]').astype(np.int64)//10**9      # force ns: parquet is us -> //1e9 was 1000x off, flattening the time signal
    d['r']=np.log1p(pd.to_numeric(d['duration'],errors='coerce').clip(lower=0)); d=d.dropna(subset=['r'])
    R,T=[],[]
    for u,g in d.groupby('usr',sort=False):
        if len(g)<N//2: continue
        R.append(g['r'].values.astype(np.float64)); T.append(g['t'].values.astype(np.int64))
    return R,T

def train_windows(R,T,idxs,mu,sd,edges):
    Z,C,TS,V=[],[],[],[]
    for i in idxs:
        r=R[i]; z=(r-mu)/sd; c=np.clip(np.digitize(r,edges),0,KC-1)
        for s in range(0,len(r),N):
            zz=z[s:s+N]
            if len(zz)<8: continue
            pad=N-len(zz)
            Z.append(np.concatenate([zz,np.full(pad,0.0)])); C.append(np.concatenate([c[s:s+N],np.full(pad,KC)]))
            TS.append(np.concatenate([T[i][s:s+N],np.full(pad,T[i][s:s+N][-1])])); V.append(np.concatenate([np.ones(len(zz)),np.zeros(pad)]))
    return np.array(Z,np.float32),np.array(C,np.int64),np.array(TS,np.int64),np.array(V,np.float32)

def eval_windows(R,T,idxs,mu,sd,edges):
    Z,C,TS,SID=[],[],[],[]
    for i in idxs:
        r=R[i]; z=(r-mu)/sd; c=np.clip(np.digitize(r,edges),0,KC-1)
        for s in range(0,len(r)-N+1,N):
            Z.append(z[s:s+N]); C.append(c[s:s+N]); TS.append(T[i][s:s+N]); SID.append(i)
    Z=np.array(Z,np.float32); C=np.array(C,np.int64); TS=np.array(TS,np.int64); SID=np.array(SID)
    if len(Z)>EVALMAX:
        k=np.random.default_rng(0).choice(len(Z),EVALMAX,replace=False); Z,C,TS,SID=Z[k],C[k],TS[k],SID[k]
    return Z,C,TS,SID

def train_direct(model, Ztr,Ctr,TStr,Vtr, epochs=EP, lr=1e-3):
    m=model.to(dev); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr,weight_decay=0.01)
    nw=len(Ztr); to=lambda a: torch.tensor(a).to(dev)
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(nw)
        for i in range(0,nw,BS):
            j=idx[i:i+BS]; z=to(Ztr[j]); c=to(Ctr[j]); t=to(TStr[j]); v=to(Vtr[j])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m((z,c),t)
            hid=hid.float(); loss=0.0
            for h in HS:                                                       # direct target at every horizon
                loss=loss+m.head.nll(hid[:,:N-h], h, z[:,h:], v[:,:N-h]*v[:,h:])
            loss=loss/len(HS)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m

def ctx_buffer(Z,C,TS,Lin):
    B=len(Z); bz=np.zeros((B,N),np.float32); bc=np.full((B,N),KC,np.int64); bt=np.zeros((B,N),np.int64)
    bz[:,:Lin]=Z[:,SPLIT-Lin:SPLIT]; bc[:,:Lin]=C[:,SPLIT-Lin:SPLIT]
    bt[:,:Lin]=TS[:,SPLIT-Lin:SPLIT]; bt[:,Lin:]=bt[:,Lin-1:Lin]               # no FUTURE timestamps needed (unlike rollout)
    return bz,bc,bt

@torch.no_grad()
def eval_direct(m, Ze,Ce,Te, mu,sd, bs=512):
    """returns {(Lin,H): median_r [n]} + distributional quantiles at Lin=DLIN: {(H): [n,9] r-space}."""
    med={}; qgrid={}
    for Lin in LINS:
        bz,bc,bt=ctx_buffer(Ze,Ce,Te,Lin); hids=[]
        for i in range(0,len(bz),bs):
            z=torch.tensor(bz[i:i+bs]).to(dev); c=torch.tensor(bc[i:i+bs]).to(dev); t=torch.tensor(bt[i:i+bs]).to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hids.append(m((z,c),t)[:,Lin-1].float())
        hid=torch.cat(hids)                                                     # [n,D] one forward per Lin serves ALL horizons
        for Hh in HS:
            q=m.head.quantiles(hid.unsqueeze(1),Hh,(0.5,))[:,0,0].cpu().numpy(); med[(Lin,Hh)]=mu+sd*q
            if Lin==DLIN:
                q=m.head.quantiles(hid.unsqueeze(1),Hh,QS9+(0.99,),S=4096)[:,0,:].cpu().numpy(); qgrid[Hh]=mu+sd*q   # [n,10]: 9-grid + P99; S=4096 (P99 from 512 samples too noisy, review)
    return med,qgrid

@torch.no_grad()
def eval_rollout(m, Ze,Ce,Te, Lin, mu,sd, edges, bs=256):
    """h=1 chained rollout with the SAME direct-trained weights (ablation: direct-vs-rollout, same model)."""
    edg=torch.tensor(edges,device=dev); out=[]
    for i in range(0,len(Ze),bs):
        z=torch.tensor(Ze[i:i+bs]).to(dev); c=torch.tensor(Ce[i:i+bs]).to(dev); tsw=torch.tensor(Te[i:i+bs]).to(dev); B=z.shape[0]
        bz=torch.zeros((B,N),device=dev); bc=torch.full((B,N),KC,device=dev,dtype=torch.long)
        bt=torch.zeros((B,N),device=dev,dtype=torch.long); bt[:,:Lin]=tsw[:,SPLIT-Lin:SPLIT]; bt[:,Lin:]=bt[:,Lin-1:Lin]   # review fix: NO ground-truth future arrival times (was a covariate leak favoring time-aware arms)
        bz[:,:Lin]=z[:,SPLIT-Lin:SPLIT]; bc[:,:Lin]=c[:,SPLIT-Lin:SPLIT]; meds=[]
        for p in range(Lin,Lin+LOUTMAX):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m((bz,bc),bt)[:,p-1].float()
            mz=m.head.quantiles(hid.unsqueeze(1),1,(0.5,))[:,0,0]
            bz[:,p]=mz; rr=mu+sd*mz; bc[:,p]=torch.clamp(torch.bucketize(rr,edg),0,KC-1); meds.append(rr.cpu().numpy())
        out.append(np.stack(meds,1))
    return np.concatenate(out)                                                 # [n,LOUTMAX] r-space medians

if __name__=='__main__':
    print(f"device={dev} D={D} L={L} EP={EP} NSEED={NSEED} N={N} LINS={LINS} HS={HS} SMOKE={SMOKE}",flush=True)
    R,T=load(); n=len(R); print(f"users={n} events={sum(len(r) for r in R):,}",flush=True)
    mk=lambda tg: (lambda: TGRv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N,time_gated=tg))
    ARMS={'TGR+direct':mk(True), 'TGR-notime+direct':mk(False),
          'GRU+direct':      lambda: GRUv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N),
          'HSTU+direct':     lambda: HSTUv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N)}
    _sel=os.environ.get('ARMS','')
    if _sel: ARMS={k:v for k,v in ARMS.items() if k in _sel.split(',')}
    surf={a:{} for a in list(ARMS)+['TGR+rollout(sameWts)','GBM-specialist','persistence','EWMA']}
    dist={a:{} for a in list(ARMS)}                                            # arm -> H -> [ (pin90,pin99,crps9) per seed ]
    lat={}                                                                     # serving-latency measurements
    for seed in range(SEED0,NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        rng=np.random.default_rng(seed); e=rng.permutation(n); tec=set(e[:max(1,int(n*0.2))].tolist())
        tr=[i for i in range(n) if i not in tec]; te=[i for i in range(n) if i in tec]
        rall=np.concatenate([R[i] for i in tr]); mu=float(rall.mean()); sd=float(rall.std()+1e-6)
        edges=np.quantile(rall,np.linspace(0,1,KC+1)[1:-1])
        Ztr,Ctr,TStr,Vtr=train_windows(R,T,tr,mu,sd,edges); Ze,Ce,Te,SIDe=eval_windows(R,T,te,mu,sd,edges)
        scale={i:float(np.mean(np.abs(np.diff(R[i])))) for i in tr if len(R[i])>1}   # train users: their data IS training data
        for s_ in np.unique(SIDe):                                                   # review fix: held-out users' MASE scale from CONTEXT regions only (never the scored target segment)
            scale[int(s_)]=float(sd*np.mean(np.abs(np.diff(Ze[SIDe==s_][:,:SPLIT],axis=1))))
        print(f"[seed {seed}] train-win={len(Ztr):,} eval-win={len(Ze):,}",flush=True)
        true={Hh: mu+sd*Ze[:,SPLIT+Hh-1] for Hh in HS}
        for arm,ctor in ARMS.items():
            m=train_direct(ctor(),Ztr,Ctr,TStr,Vtr)
            t0=time.time(); med,qg=eval_direct(m,Ze,Ce,Te,mu,sd); tdir=time.time()-t0
            lat.setdefault(arm,[]).append(tdir)
            for (Lin,Hh),p in med.items():
                mm,_=MET.mase_per_series(p,true[Hh],SIDe,scale); surf[arm].setdefault((Lin,Hh),[]).append(mm)
            for Hh,q in qg.items():                                            # q: [n,10] = 9-grid + P99
                p90=MET.pinball(true[Hh],q[:,7],0.9); p99=MET.pinball(true[Hh],q[:,9],0.99)
                crps=MET.crps_from_quantiles(true[Hh],q[:,:9],QS9)
                dist[arm].setdefault(Hh,[]).append((p90,p99,crps))
            if arm=='TGR+direct':                                              # rollout ablation with the SAME weights
                t0=time.time(); roll=eval_rollout(m,Ze,Ce,Te,DLIN,mu,sd,edges); troll=time.time()-t0
                lat.setdefault('TGR+rollout(sameWts)',[]).append(troll)
                for Hh in HS:
                    mm,_=MET.mase_per_series(roll[:,Hh-1],true[Hh],SIDe,scale)
                    surf['TGR+rollout(sameWts)'].setdefault((DLIN,Hh),[]).append(mm)
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {arm} done ({tdir:.0f}s direct-eval)",flush=True)
        # GBM per cell + persistence + EWMA
        for Lin in LINS:
            Xtr=Ztr[:,SPLIT-Lin:SPLIT]; Xte=Ze[:,SPLIT-Lin:SPLIT]
            ew=np.zeros(len(Ze))
            alpha=0.3; s=Xte[:,0].copy()
            for kcol in range(1,Lin): s=alpha*Xte[:,kcol]+(1-alpha)*s
            ew_r=mu+sd*s
            for Hh in HS:
                ok=(Vtr[:,SPLIT+Hh-1]>0)&(Vtr[:,SPLIT-Lin:SPLIT].min(1)>0)     # review fix: drop pad-fabricated rows (they dragged GBM toward the global mean)
                gm=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,max_iter=80 if SMOKE else 200,learning_rate=0.08,random_state=seed).fit(Xtr[ok],(mu+sd*Ztr[:,SPLIT+Hh-1])[ok])
                mm,_=MET.mase_per_series(gm.predict(Xte),true[Hh],SIDe,scale); surf['GBM-specialist'].setdefault((Lin,Hh),[]).append(mm)
                mp,_=MET.mase_per_series(mu+sd*Ze[:,SPLIT-1],true[Hh],SIDe,scale); surf['persistence'].setdefault((Lin,Hh),[]).append(mp)
                me,_=MET.mase_per_series(ew_r,true[Hh],SIDe,scale); surf['EWMA'].setdefault((Lin,Hh),[]).append(me)
        ck=os.path.expanduser(f"~/nfm/logs/bakeoff_ckpt_s{seed}_{os.environ.get('SLURM_JOB_ID','x')}.pkl")
        with open(ck,'wb') as fh: pickle.dump({'surf':{k:dict(v) if isinstance(v,dict) else v for k,v in surf.items()},'dist':dist,'lat':lat,'seed':seed},fh)
        print(f"  [seed {seed}] checkpoint -> {ck}",flush=True)                # timeout-proof: per-seed partials survive

    print(f"\n==== BAKE-OFF: per-series MASE surface, mean over {NSEED-SEED0} seed(s) ====",flush=True)
    for meth in surf:
        if not surf[meth]: continue
        print(f"\n-- {meth} --  (rows=input len, cols=horizon)")
        print("  Lin\\H "+" ".join(f"{h:>7}" for h in HS))
        for Lin in LINS:
            row=" ".join(f"{np.mean(surf[meth][(Lin,h)]):>7.3f}" if (Lin,h) in surf[meth] else f"{'-':>7}" for h in HS)
            print(f"  {Lin:>4}  {row}")
    print(f"\n==== distributional at Lin={DLIN} (pin@0.9 pin@0.99 CRPS-9grid), mean over seeds ====")
    for arm in dist:
        if not dist[arm]: continue
        cells=" | ".join(f"h{Hh}: "+" ".join(f"{np.mean([v[k] for v in dist[arm][Hh]]):.4f}" for k in range(3)) for Hh in HS if Hh in dist[arm])
        print(f"  {arm:<20} {cells}")
    print(f"\n==== serving latency (full-grid eval wall, {EVALMAX} windows) ====")
    for k,v in lat.items(): print(f"  {k:<22} {np.mean(v):7.1f}s   ({'ALL '+str(len(LINS)*len(HS))+' cells, no rollout' if 'direct' in k else str(LOUTMAX)+'-step rollout, ONE Lin'})")
    print("BAKEOFF_DONE",flush=True)
