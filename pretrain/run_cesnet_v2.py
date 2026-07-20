"""CESNET network-traffic leg: hourly per-institution volume forecasting + CAPACITY-PROVISIONING Pareto.
The networking-native evaluation (playbook P1/P3): sweep the predicted quantile tau -> set next-hour capacity
= Q_tau -> report (overprovision%, violation%) tradeoff curves; operator status-quo rules (last-week-same-hour
peak x1.2, EWMA) are DOTS that our curve should dominate. Also plain forecasting metrics (per-series MASE vs
train-region naive scale, pinball, CRPS-9) for the support table. Chronological 80/20 per series (clean).
Regular hourly grid = the uniform-sampling case (time gating expected neutral here; that is an honest finding).
Env: D(512) H(8) L(8) EP(8) MIX(4) NSEED(2) SPLIT(168) HMAX(24) STRIDE(24) EVALMAX(3000) BS(128) ARMS() SMOKE(0).
"""
import os, sys, time, pickle, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, GRUv2, TGRv2, ContinuousInput, HorizonHead, dev
import metrics_v2 as MET
from sklearn.ensemble import HistGradientBoostingRegressor

D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8))
EP=int(os.environ.get('EP',8)); MIX=int(os.environ.get('MIX',4)); NSEED=int(os.environ.get('NSEED',2))
SPLIT=int(os.environ.get('SPLIT',168)); HMAX=int(os.environ.get('HMAX',24)); N=SPLIT+HMAX
STRIDE=int(os.environ.get('STRIDE',24)); EVALMAX=int(os.environ.get('EVALMAX',3000)); BS=int(os.environ.get('BS',128))
KC=int(os.environ.get('KC',256)); SMOKE=int(os.environ.get('SMOKE',0))
HS=[1,4,24]; QS9=tuple(np.linspace(0.1,0.9,9)); TAUS=(0.5,0.8,0.9,0.95,0.99)
if SMOKE: EP=2; NSEED=1; EVALMAX=400
NPZ=os.path.expanduser("~/nfm/data/cesnet/cesnet_hourly.npz")

def load():
    z=np.load(NPZ); r=z['r']; v=z['valid']; ts=z['ts']
    if SMOKE: r=r[:40]; v=v[:40]
    return r,v,ts

def windows(r,v,ts,lo,hi,stride):
    """fully-valid length-N windows with origin in [lo,hi); returns Z,TS,SID (z-space filled later)."""
    W=[]; S=[]; Tw=[]
    for i in range(len(r)):
        for s in range(lo,hi-N+1,stride):
            if v[i,s:s+N].min()>0: W.append(r[i,s:s+N]); Tw.append(ts[s:s+N]); S.append(i)
    return np.array(W,np.float32),np.array(Tw,np.int64),np.array(S)

def train_direct(model,Zt,Tt,epochs=EP,lr=1e-3):
    m=model.to(dev); opt=torch.optim.AdamW(m.parameters(),lr,weight_decay=0.01); nw=len(Zt)
    to=lambda a: torch.tensor(a).to(dev)
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(nw)
        for i in range(0,nw,BS):
            j=idx[i:i+BS]; z=to(Zt[j]); t=to(Tt[j])
            zz=(z-MU)/SD; cb=torch.clamp(torch.bucketize(z,EDG_T),0,KC-1)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m((zz,cb),t)
            hid=hid.float(); loss=0.0
            for h in HS: loss=loss+m.head.nll(hid[:,:N-h],h,zz[:,h:],torch.ones_like(zz[:,h:]))
            loss=loss/len(HS)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m

@torch.no_grad()
def eval_direct(m,Ze,Te,bs=256):
    """origin at SPLIT-1; returns {H: quantile matrix [n, len(QS9)+2]} in r-space (9-grid + tau95 + tau99)."""
    out={h:[] for h in HS}; qs=QS9+(0.95,0.99)
    for i in range(0,len(Ze),bs):
        z=torch.tensor(Ze[i:i+bs]).to(dev); t=torch.tensor(Te[i:i+bs]).to(dev)
        zz=(z-MU)/SD; cb=torch.clamp(torch.bucketize(z,EDG_T),0,KC-1)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            hid=m((zz,cb),t)[:,SPLIT-1].float()
        for h in HS:
            q=m.head.quantiles(hid.unsqueeze(1),h,qs,S=4096)[:,0,:].cpu().numpy()
            out[h].append(MU+SD*q)
    return {h:np.concatenate(v) for h,v in out.items()}

def pareto(cap_r,act_r):
    cap=np.expm1(cap_r); act=np.expm1(act_r)
    over=np.maximum(cap-act,0).sum()/max(act.sum(),1e-9)*100.0
    viol=float((act>cap).mean()*100.0)
    return over,viol

if __name__=='__main__':
    print(f"device={dev} D={D} L={L} EP={EP} NSEED={NSEED} SPLIT={SPLIT} HMAX={HMAX} STRIDE={STRIDE} SMOKE={SMOKE}",flush=True)
    r,v,ts=load(); S,T=r.shape; cut=int(T*0.8)
    print(f"series={S} bins={T} train<{cut} test>={cut}",flush=True)
    global MU,SD,EDG_T
    tr_mask=v[:,:cut]>0; MU=float(r[:,:cut][tr_mask].mean()); SD=float(r[:,:cut][tr_mask].std()+1e-6)
    EDG_T=torch.tensor(np.quantile(r[:,:cut][tr_mask],np.linspace(0,1,KC+1)[1:-1]),dtype=torch.float32).to(dev)
    scale={i:float(np.mean(np.abs(np.diff(r[i,:cut][v[i,:cut]>0])))) for i in range(S)}   # train-region naive scale (clean)
    Zt,Tt,_=windows(r,v,ts,0,cut,STRIDE); Ze,Te,SIDe=windows(r,v,ts,cut,T,STRIDE)
    if len(Ze)>EVALMAX:
        k=np.random.default_rng(0).choice(len(Ze),EVALMAX,replace=False); Ze,Te,SIDe=Ze[k],Te[k],SIDe[k]
    print(f"train-win={len(Zt):,} eval-win={len(Ze):,}",flush=True)
    true={h: Ze[:,SPLIT+h-1] for h in HS}
    mk=lambda tg: (lambda: TGRv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N,time_gated=tg))
    ARMS={'TGR+direct':mk(True),'GRU+direct':lambda: GRUv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N),
          'HSTU+direct':lambda: HSTUv2(ContinuousInput(D,KC),HorizonHead(D,MIX),D=D,H=H,L=L,N=N)}
    _sel=os.environ.get('ARMS','')
    if _sel: ARMS={k:v for k,v in ARMS.items() if k in _sel.split(',')}
    res={}; par={}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        for arm,ctor in ARMS.items():
            m=train_direct(ctor(),Zt,Tt); Q=eval_direct(m,Ze,Te)
            for h in HS:
                q=Q[h]; med=q[:,4]                                             # cols = (.1..“.9”)+(.95,.99): 0.5->4, 0.8->7, 0.9->8, 0.95->9, 0.99->10
                mase,_=MET.mase_per_series(med,true[h],SIDe,scale)
                p90=MET.pinball(true[h],q[:,8],0.9); p99=MET.pinball(true[h],q[:,10],0.99)
                crps=MET.crps_from_quantiles(true[h],q[:,:9],QS9)
                res.setdefault((arm,h),[]).append((mase,p90,p99,crps))
            capcols={0.5:4,0.8:7,0.9:8,0.95:9,0.99:10}
            for tau in TAUS:
                ov,vi=pareto(Q[1][:,capcols[tau]],true[1]); par.setdefault((arm,tau),[]).append((ov,vi))
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {arm} done",flush=True)
        # baselines (seed 0 only: deterministic)
        if seed==0:
            last=Ze[:,SPLIT-1]; week=Ze[:,SPLIT-168] if SPLIT>=168 else Ze[:,0]
            al=0.3; ew=Ze[:,0].copy()
            for kcol in range(1,SPLIT): ew=al*Ze[:,kcol]+(1-al)*ew
            for h in HS:
                for nm,pred in [('persistence',last),('seasonal-naive',Ze[:,SPLIT+h-1-168] if SPLIT+h-1-168>=0 else week),('EWMA',ew)]:
                    mase,_=MET.mase_per_series(pred,true[h],SIDe,scale); res.setdefault((nm,h),[]).append((mase,np.nan,np.nan,np.nan))
                Xtr=Zt[:,:SPLIT]; ytr=Zt[:,SPLIT+h-1]
                gm=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,max_iter=80 if SMOKE else 200,learning_rate=0.08,random_state=0).fit(Xtr,ytr)
                mase,_=MET.mase_per_series(gm.predict(Ze[:,:SPLIT]),true[h],SIDe,scale); res.setdefault(('GBM-specialist',h),[]).append((mase,np.nan,np.nan,np.nan))
            # operator provisioning dots: last-week-same-hour x1.0/x1.2 (in bytes), EWMA x1.2
            wk=Ze[:,SPLIT+1-1-168] if SPLIT-167>=0 else week
            for nm,cap_r in [('rule:lastweek x1.0',wk),('rule:lastweek x1.2',np.log1p(1.2*np.expm1(wk))),('rule:EWMA x1.2',np.log1p(1.2*np.expm1(ew)))]:
                ov,vi=pareto(cap_r,true[1]); par.setdefault((nm,None),[]).append((ov,vi))
    print(f"\n==== CESNET hourly volume: MASE | pin90 | pin99 | CRPS9, mean over seeds ====",flush=True)
    meths=sorted(set(k[0] for k in res))
    for nm in meths:
        row=" | ".join(f"h{h}: "+" ".join(f"{np.nanmean([v[i] for v in res[(nm,h)]]):.3f}" for i in range(4)) for h in HS if (nm,h) in res)
        print(f"  {nm:<18} {row}")
    print(f"\n==== next-hour capacity-provisioning Pareto (overprovision% , violation%) ====",flush=True)
    for (nm,tau),vals in sorted(par.items(),key=lambda x:(x[0][0],x[0][1] or 0)):
        ov=np.mean([v[0] for v in vals]); vi=np.mean([v[1] for v in vals])
        print(f"  {nm:<18} tau={tau}: over={ov:6.1f}%  viol={vi:5.2f}%")
    with open(os.path.expanduser(f"~/nfm/logs/cesnet_res_{os.environ.get('SLURM_JOB_ID','x')}.pkl"),'wb') as fh:
        pickle.dump({'res':res,'par':par},fh)
    print("CESNET_DONE",flush=True)
