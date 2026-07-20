"""v2 forecasting: HPC job-duration on F-DATA with a HEAVY-TAIL mixture-density head, evaluated on the
UNSATURATED metrics (tail-MASE / quantile pinball / CRPS / coverage) where a calibrated distribution wins,
plus plain MASE (where we honestly tie). Models: HSTU-v2(+MixtureHead), TFM-v2(+SAME head, isolates the
backbone), quantile-GBM specialist (per horizon+quantile), persistence. Target = standardized log-duration.
Env: KC(256) CTX(32) HMAX(16) D H L EP MIX(4) NSEED(3) SMOKE(0) FRAC(1.0).
"""
import os, sys, glob, numpy as np, pandas as pd, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, TFMv2, ContinuousInput, MixtureHead, train_v2, dev
import metrics_v2 as MET
from sklearn.ensemble import HistGradientBoostingRegressor

KC=int(os.environ.get('KC',256)); CTX=int(os.environ.get('CTX',32)); HMAX=int(os.environ.get('HMAX',16)); N=CTX+HMAX
D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8)); EP=int(os.environ.get('EP',8))
MIX=int(os.environ.get('MIX',4)); NSEED=int(os.environ.get('NSEED',3)); SMOKE=int(os.environ.get('SMOKE',0)); FRAC=float(os.environ.get('FRAC',1.0))
QS=(0.5,0.9,0.99); HZ=[h for h in [1,2,4,8,16] if h<=HMAX]; PADz=0.0; PADc=KC

NFILES=int(os.environ.get('NFILES',0))   # 0 = all; else first NFILES months
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
        if len(g)<8: continue
        R.append(g['r'].values.astype(np.float64)); T.append(g['t'].values.astype(np.int64))
    return R,T

def win_train(R,T,idxs,mu,sd,edges):
    Z,C,TS,V=[],[],[],[]
    for i in idxs:
        r=R[i]; z=(r-mu)/sd; c=np.clip(np.digitize(r,edges),0,KC-1)
        for s in range(0,len(r),N):
            zz=z[s:s+N]
            if len(zz)<8: continue
            pad=N-len(zz); m=np.concatenate([np.ones(len(zz)),np.zeros(pad)])
            Z.append(np.concatenate([zz,np.full(pad,PADz)])); C.append(np.concatenate([c[s:s+N],np.full(pad,PADc)]))
            TS.append(np.concatenate([T[i][s:s+N],np.full(pad,T[i][s:s+N][-1])])); V.append(m)
    return np.array(Z,np.float32),np.array(C,np.int64),np.array(TS,np.int64),np.array(V,np.float32)

def win_eval(R,T,idxs,mu,sd,edges):
    Zc,Cc,TSf,Rf=[],[],[],[]
    for i in idxs:
        r=R[i]; z=(r-mu)/sd; c=np.clip(np.digitize(r,edges),0,KC-1)
        for s in range(0,len(r)-N+1,N):
            Zc.append(z[s:s+CTX]); Cc.append(c[s:s+CTX]); TSf.append(T[i][s:s+N]); Rf.append(r[s+CTX:s+CTX+HMAX])
    if not Zc: return None
    return np.array(Zc,np.float32),np.array(Cc,np.int64),np.array(TSf,np.int64),np.array(Rf,np.float64)

@torch.no_grad()
def rollout_mix(m,Zc,Cc,TSf,mu,sd,edges,bs=256):
    m.eval(); QP=[]                                    # per-horizon predicted r-quantiles [n,HMAX,len(QS)]
    for i in range(0,len(Zc),bs):
        z=torch.tensor(Zc[i:i+bs]).to(dev); c=torch.tensor(Cc[i:i+bs]).to(dev); ts=torch.tensor(TSf[i:i+bs]).to(dev); B=z.shape[0]
        bz=torch.full((B,N),PADz,device=dev); bc=torch.full((B,N),PADc,device=dev,dtype=torch.long)
        bz[:,:CTX]=z; bc[:,:CTX]=c; qh=[]
        edg=torch.tensor(edges,device=dev)
        for p in range(CTX,N):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                h=m((bz,bc),ts)[:,p-1]
            qz=m.head.quantiles(h.unsqueeze(1),QS)[:,0].float()          # [B,len(QS)] z-space
            medz=qz[:,QS.index(0.5)]; bz[:,p]=medz
            rr=mu+sd*medz; bc[:,p]=torch.clamp(torch.bucketize(rr,edg),0,KC-1)
            qh.append((mu+sd*qz).cpu().numpy())                          # r-space quantiles
        QP.append(np.stack(qh,1))                                       # [B,HMAX,len(QS)]
    return np.concatenate(QP)

def gbm_quant(Zc,Rf,tr_ctx,tr_fut):
    """quantile-GBM specialist: features=context z; one model per (horizon,quantile). Returns [n,HMAX,len(QS)] r-quantiles."""
    Xtr=tr_ctx; out=np.zeros((len(Zc),HMAX,len(QS)))
    for hi,h in enumerate(range(1,HMAX+1)):
        for qi,q in enumerate(QS):
            gm=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=80 if SMOKE else 200,learning_rate=0.08)
            gm.fit(Xtr,tr_fut[:,h-1]); out[:,hi,qi]=gm.predict(Zc)
    return out

def evalall(name, qpred_r, Rf, sid, scale):                            # qpred_r [n,HMAX,len(QS)] r-space
    med=qpred_r[:,:,QS.index(0.5)]                                      # point = median
    res={}
    for hi,h in enumerate(HZ):
        j=h-1; yt=Rf[:,j]; ptm=med[:,j]
        mm,mmed=MET.mase_per_series(ptm,yt,sid,scale)
        tail=MET.tail_mask(yt,0.10)
        pb={q:MET.pinball(yt,qpred_r[:,j,qi],q) for qi,q in enumerate(QS)}
        pbt={q:MET.pinball(yt[tail],qpred_r[tail,j,qi],q) for qi,q in enumerate(QS)}
        crps=MET.crps_from_quantiles(yt,qpred_r[:,j,:],QS); crps_t=MET.crps_from_quantiles(yt[tail],qpred_r[tail,j,:],QS)
        cov=MET.coverage(yt,qpred_r[:,j,0]-1e9 if len(QS)<2 else qpred_r[:,j,0], qpred_r[:,j,-1])   # [q0.5,q0.99] as ~ lower..upper
        tm,_=MET.mase_per_series(ptm[tail],yt[tail],sid[tail],scale)
        res[h]=dict(MASE=mm,MASEmed=mmed,tailMASE=tm,pin90=pb[0.9],pin99=pb[0.99],tpin90=pbt[0.9],tpin99=pbt[0.99],CRPS=crps,tCRPS=crps_t)
    return res

if __name__=='__main__':
    print(f"device={dev} KC={KC} CTX={CTX} HMAX={HMAX} D={D} H={H} L={L} EP={EP} MIX={MIX} NSEED={NSEED} SMOKE={SMOKE} FRAC={FRAC}",flush=True)
    R,T=load(); n=len(R); print(f"users={n} events={sum(len(r) for r in R):,}",flush=True)
    agg={}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        rng=np.random.default_rng(seed); e=rng.permutation(n); tec=set(e[:max(1,int(n*0.2))].tolist())
        tr=[i for i in range(n) if i not in tec]; te=[i for i in range(n) if i in tec]
        if FRAC<1.0: tr=tr[:max(1,int(len(tr)*FRAC))]
        rall=np.concatenate([R[i] for i in tr]); mu=float(rall.mean()); sd=float(rall.std()+1e-6)
        edges=np.quantile(rall,np.linspace(0,1,KC+1)[1:-1])
        scale={i: float(np.mean(np.abs(np.diff(R[i])))) for i in range(n) if len(R[i])>1}   # per-series naive-MAE (r)
        Ztr,Ctr,TStr,Vtr=win_train(R,T,tr,mu,sd,edges); ev=win_eval(R,T,te,mu,sd,edges)
        if ev is None: print("no eval windows"); break
        Zc,Cc,TSf,Rf=ev
        sid=np.concatenate([[i]*((len(R[i])-N)//N+1) for i in te if len(R[i])>=N])   # series id per eval window (approx)
        sid=sid[:len(Zc)] if len(sid)>=len(Zc) else np.resize(sid,len(Zc))
        print(f"[seed {seed}] train-win={len(Ztr):,} eval-win={len(Zc):,} mu={mu:.2f} sd={sd:.2f}",flush=True)
        def batches(Z,C,TS,Vv,bs=256):
            def it():
                idx=np.random.permutation(len(Z))
                for i in range(0,len(idx),bs):
                    j=idx[i:i+bs]
                    yield (torch.tensor(Z[j]).to(dev),torch.tensor(C[j]).to(dev)),torch.tensor(TS[j]).to(dev),torch.tensor(Z[j]).to(dev),torch.tensor(Vv[j]).to(dev)
            return it
        # models with the SAME MixtureHead (isolates backbone)
        hstu=train_v2(HSTUv2(ContinuousInput(D,KC),MixtureHead(D,MIX),D=D,H=H,L=L,N=N),batches(Ztr,Ctr,TStr,Vtr),epochs=EP)
        tfm =train_v2(TFMv2 (ContinuousInput(D,KC),MixtureHead(D,MIX),D=D,H=H,L=L,N=N),batches(Ztr,Ctr,TStr,Vtr),epochs=EP)
        qh=rollout_mix(hstu,Zc,Cc,TSf,mu,sd,edges); qt=rollout_mix(tfm,Zc,Cc,TSf,mu,sd,edges)
        # baselines
        per=np.repeat((mu+sd*Zc[:,-1:]),HMAX,axis=1)[:,:,None].repeat(len(QS),2)     # persistence (point -> flat quantiles)
        tr_ctx=Zc*0; # build GBM train features from TRAIN windows' contexts
        Gz,Gc,Gt,Gv=win_train(R,T,tr,mu,sd,edges); gtr_ctx=Gz[:,:CTX]; gtr_fut=(mu+sd*Gz[:,CTX:CTX+HMAX])
        gbm=gbm_quant(Zc,Rf,gtr_ctx,gtr_fut)
        for nm,qp in [('HSTU-mix',qh),('TFM-mix',qt),('GBM-quant',gbm),('persistence',per)]:
            res=evalall(nm,qp,Rf,sid,scale)
            agg.setdefault(nm,[]).append(res)
        del hstu,tfm; torch.cuda.empty_cache() if dev=='cuda' else None
    # report mean over seeds
    print(f"\n==== v2 forecasting, mean over {NSEED} seed(s) ====",flush=True)
    for nm in agg:
        print(f"\n-- {nm} --")
        print(f"  {'h':>3} {'MASE':>7} {'MASEmed':>8} {'tailMASE':>9} {'pin90':>7} {'pin99':>7} {'tpin90':>7} {'tpin99':>7} {'CRPS':>7} {'tCRPS':>7}")
        for h in HZ:
            v={k:np.mean([r[h][k] for r in agg[nm]]) for k in agg[nm][0][h]}
            print(f"  {h:>3} {v['MASE']:>7.3f} {v['MASEmed']:>8.3f} {v['tailMASE']:>9.3f} {v['pin90']:>7.3f} {v['pin99']:>7.3f} {v['tpin90']:>7.3f} {v['tpin99']:>7.3f} {v['CRPS']:>7.3f} {v['tCRPS']:>7.3f}")
    print("\nFORECAST_V2_DONE",flush=True)
