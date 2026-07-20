"""HEADLINE thesis experiment: ANY input length x ANY output horizon, ONE model.
Train ONE generative model per backbone (HSTU / Transformer / LSTM / GRU, all with the SAME heavy-tail
MixtureHead), then evaluate it across a grid (input-length L_in x output-horizon H) by autoregressive
rollout. Compare to (a) a quantile-GBM SPECIALIST trained per (L_in, H) cell -> |grid| models, and
(b) persistence. The point: one generative model covers the whole grid; specialists need one per cell.
Metric per cell = per-series MASE (point=mixture median) on log-duration; also CRPS for the generative models.
Env: KC(256) D H L EP MIX NSEED SPLIT(96) LOUTMAX(64) EVALMAX(3000) NFILES(0).
"""
import os, sys, glob, numpy as np, pandas as pd, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, TFMv2, LSTMv2, GRUv2, RetNetV2, MambaV2, ContinuousInput, MixtureHead, train_v2, dev
import metrics_v2 as MET
from sklearn.ensemble import HistGradientBoostingRegressor

KC=int(os.environ.get('KC',256)); D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8))
EP=int(os.environ.get('EP',10)); MIX=int(os.environ.get('MIX',4)); NSEED=int(os.environ.get('NSEED',2))
SPLIT=int(os.environ.get('SPLIT',96)); LOUTMAX=int(os.environ.get('LOUTMAX',64)); N=SPLIT+LOUTMAX
EVALMAX=int(os.environ.get('EVALMAX',3000)); NFILES=int(os.environ.get('NFILES',0)); SMOKE=int(os.environ.get('SMOKE',0))
BS=int(os.environ.get('BS',256))   # training batch; lower for memory-heavy sequential-scan backbones (Mamba)
LINS=[8,16,32,64,96] if not SMOKE else [8,32]; HS=[1,4,16,32,64] if not SMOKE else [1,16]; QS=(0.5,0.9,0.99)
if SMOKE: SPLIT=48; LOUTMAX=32; N=SPLIT+LOUTMAX; LINS=[8,32]; HS=[1,16]

def load():
    files=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))
    if SMOKE and NFILES==0: files=files[:1]
    elif NFILES>0: files=files[:NFILES]
    d=pd.concat([pd.read_parquet(f,columns=['usr','adt','duration']) for f in files],ignore_index=True)
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce'); d=d.dropna(subset=['adt']).sort_values(['usr','adt'])
    d['t']=d['adt'].astype('datetime64[ns, UTC]').astype(np.int64)//10**9      # force ns: parquet is us -> //1e9 was 1000x off
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
    if not Z: return None
    Z=np.array(Z,np.float32); C=np.array(C,np.int64); TS=np.array(TS,np.int64); SID=np.array(SID)
    if len(Z)>EVALMAX:
        k=np.random.default_rng(0).choice(len(Z),EVALMAX,replace=False); Z,C,TS,SID=Z[k],C[k],TS[k],SID[k]
    return Z,C,TS,SID

@torch.no_grad()
def rollout_grid(m, Z, C, TS, Lin, mu, sd, edges, bs=256):
    """context = z[SPLIT-Lin:SPLIT] placed at buffer [0:Lin]; roll out LOUTMAX steps -> median r per horizon.
    Returns med_r [n, LOUTMAX]."""
    m.eval(); edg=torch.tensor(edges,device=dev); out=[]
    for i in range(0,len(Z),bs):
        z=torch.tensor(Z[i:i+bs]).to(dev); c=torch.tensor(C[i:i+bs]).to(dev); tsw=torch.tensor(TS[i:i+bs]).to(dev); B=z.shape[0]
        bz=torch.zeros((B,N),device=dev); bc=torch.full((B,N),KC,device=dev,dtype=torch.long)
        bt=torch.zeros((B,N),device=dev,dtype=torch.long); seg=tsw[:,SPLIT-Lin:SPLIT+LOUTMAX]; bt[:,:seg.shape[1]]=seg
        bz[:,:Lin]=z[:,SPLIT-Lin:SPLIT]; bc[:,:Lin]=c[:,SPLIT-Lin:SPLIT]; meds=[]
        for p in range(Lin,Lin+LOUTMAX):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                h=m((bz,bc),bt)[:,p-1]
            medz=m.head.quantiles(h.unsqueeze(1),(0.5,))[:,0,0].float()
            bz[:,p]=medz; rr=mu+sd*medz; bc[:,p]=torch.clamp(torch.bucketize(rr,edg),0,KC-1); meds.append((mu+sd*medz).cpu().numpy())
        out.append(np.stack(meds,1))                                   # [B,LOUTMAX]
    return np.concatenate(out)

def per_series_mase_cell(pred, Z, SID, mu, sd, scale, H):
    true=mu+sd*Z[:,SPLIT+H-1]                                          # true r at horizon H
    return MET.mase_per_series(pred[:,H-1], true, SID, scale)

if __name__=='__main__':
    print(f"device={dev} D={D} L={L} EP={EP} MIX={MIX} NSEED={NSEED} SPLIT={SPLIT} LOUTMAX={LOUTMAX} N={N} LINS={LINS} HS={HS} SMOKE={SMOKE}",flush=True)
    R,T=load(); n=len(R); print(f"users={n} events={sum(len(r) for r in R):,}",flush=True)
    BB={'HSTU':HSTUv2,'HSTU++':lambda i,h,**kw: HSTUv2(i,h,ffn=True,**kw),
        'RetNet':RetNetV2,'Mamba':MambaV2,'TFM':TFMv2,'LSTM':LSTMv2,'GRU':GRUv2}
    _sel=os.environ.get('BACKBONES','')
    if _sel: BB={k:BB[k] for k in _sel.split(',') if k in BB}
    surf={bb:{} for bb in list(BB)+['GBM-specialist','persistence']}     # surf[method][(Lin,H)] = [mase per seed]
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        rng=np.random.default_rng(seed); e=rng.permutation(n); tec=set(e[:max(1,int(n*0.2))].tolist())
        tr=[i for i in range(n) if i not in tec]; te=[i for i in range(n) if i in tec]
        rall=np.concatenate([R[i] for i in tr]); mu=float(rall.mean()); sd=float(rall.std()+1e-6)
        edges=np.quantile(rall,np.linspace(0,1,KC+1)[1:-1]); scale={i:float(np.mean(np.abs(np.diff(R[i])))) for i in range(n) if len(R[i])>1}
        Ztr,Ctr,TStr,Vtr=train_windows(R,T,tr,mu,sd,edges); ev=eval_windows(R,T,te,mu,sd,edges)
        if ev is None: print("no eval windows"); break
        Ze,Ce,Te,SIDe=ev; print(f"[seed {seed}] train-win={len(Ztr):,} eval-win={len(Ze):,}",flush=True)
        def batches():
            idx=np.random.permutation(len(Ztr))
            for i in range(0,len(idx),BS):
                j=idx[i:i+BS]
                yield (torch.tensor(Ztr[j]).to(dev),torch.tensor(Ctr[j]).to(dev)),torch.tensor(TStr[j]).to(dev),torch.tensor(Ztr[j]).to(dev),torch.tensor(Vtr[j]).to(dev)
        for bb,cls in BB.items():
            m=train_v2(cls(ContinuousInput(D,KC),MixtureHead(D,MIX),D=D,H=H,L=L,N=N),lambda: batches(),epochs=EP)
            for Lin in LINS:
                med=rollout_grid(m,Ze,Ce,Te,Lin,mu,sd,edges)
                for Hh in HS:
                    mm,_=per_series_mase_cell(med,Ze,SIDe,mu,sd,scale,Hh); surf[bb].setdefault((Lin,Hh),[]).append(mm)
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {bb} done",flush=True)
        # GBM specialist per (Lin,H) + persistence
        for Lin in LINS:
            Xtr=Ztr[:,SPLIT-Lin:SPLIT]; Xte=Ze[:,SPLIT-Lin:SPLIT]
            for Hh in HS:
                ytr=mu+sd*Ztr[:,SPLIT+Hh-1]
                gm=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,max_iter=80 if SMOKE else 200,learning_rate=0.08).fit(Xtr,ytr)
                pred=gm.predict(Xte); true=mu+sd*Ze[:,SPLIT+Hh-1]; mm,_=MET.mase_per_series(pred,true,SIDe,scale)
                surf['GBM-specialist'].setdefault((Lin,Hh),[]).append(mm)
                pers=mu+sd*Ze[:,SPLIT-1]; mp,_=MET.mase_per_series(pers,true,SIDe,scale); surf['persistence'].setdefault((Lin,Hh),[]).append(mp)
    # report MASE surface
    print(f"\n==== ANY-INPUT x ANY-OUTPUT: per-series MASE surface, mean over {NSEED} seed(s) ====",flush=True)
    print("(ONE generative model per backbone covers ALL cells; GBM-specialist = one model PER cell)")
    for meth in surf:
        if not surf[meth]: continue
        print(f"\n-- {meth} --  (rows=input len, cols=horizon)")
        print("  Lin\\H "+" ".join(f"{h:>7}" for h in HS))
        for Lin in LINS:
            row=" ".join(f"{np.mean(surf[meth][(Lin,h)]):>7.3f}" if (Lin,h) in surf[meth] else f"{'-':>7}" for h in HS)
            print(f"  {Lin:>4}  {row}")
    nc=len(LINS)*len(HS)
    print(f"\n#models to cover the {nc}-cell grid:  each generative backbone = 1  |  GBM-specialist = {nc}",flush=True)
    print("ANYIO_V2_DONE",flush=True)
