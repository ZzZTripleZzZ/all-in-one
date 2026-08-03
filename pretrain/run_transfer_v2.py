"""Cross-domain transfer experiment for the FM claim: pretrain the v2 backfill
architecture on NON-HPC telemetry (CESNET per-institution hourly traffic + Azure2017
per-VM CPU), then adapt it to the Fugaku walltime-estimator task under a limited
target-history budget, against a from-scratch arm at the same budget.

All target-side arms reuse run_backfill_model_v2's data pipeline, training loop,
inference-history contract (per-user history seeded from the FULL eval-month
pre-window, identical across arms), and npz interface, so replay numbers are
directly comparable to the paper's Table III pipeline. Only the TRAINING corpus is
budget-limited (last PREDAYS days of the pre-window).

MODE=pretrain : SRC domains -> CKPT (state_dict; same D/H/L/N/MIX as the estimator)
MODE=scratch  : budget-limited from-scratch estimator (PREDAYS in {0.25,1,2,7})
MODE=transfer : same budget, weights initialized from CKPT, full finetune
MODE=zeroshot : CKPT weights untouched; only normalization stats fitted on the budget
                window (no gradient step); est npz produced the same way

Source streams are cast into the estimator's 7-channel format: [lagged value(z),
0, 0, log1p(gap_s), sin(hod), cos(hod), dow/6] with per-domain standardization of
the value channel/target, so mu=0, sd=1 during pretraining.
"""
import os, sys, glob, pickle, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(_ROOT,"common")); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

MODE=os.environ.get('MODE','scratch')                                          # pretrain | scratch | transfer | zeroshot
CKPT=os.path.expanduser(os.environ.get('CKPT','~/nfm/data/transfer/src_ca.pt'))
PREDAYS=float(os.environ.get('PREDAYS',7))                                     # target-history budget in days (7 = full pre-window)
SRC=os.environ.get('SRC','cesnet,azure').split(',')
MAXW=int(os.environ.get('MAXW',250000))                                        # pretrain window cap
AZMAX=int(os.environ.get('AZMAX',6000))                                        # azure series cap (pt pool)
PSTRIDE=int(os.environ.get('PSTRIDE',48))                                      # pretrain window stride

import run_backfill_model_v2 as bf                                             # env-configured at import; heavy work is __main__-guarded
from nfm_v2 import dev

lg=bf.lg; NCH=bf.NCH

# ---------- source domains -> (z_seq, valid_seq, ts_seq) per entity ----------
def _ffill(a):
    a=np.asarray(a,np.float32).copy(); ok=np.isfinite(a)
    if not ok.any(): return np.zeros_like(a)
    idx=np.where(ok,np.arange(len(a)),0); np.maximum.accumulate(idx,out=idx); a=a[idx]
    m=np.isfinite(a)
    if not m[0]: a[:m.argmax()]=a[m.argmax()]
    return a

def src_streams():
    out=[]
    if 'cesnet' in SRC:
        z=np.load(os.path.expanduser('~/nfm/data/cesnet/cesnet_hourly.npz'))
        print(f"cesnet npz keys={z.files}",flush=True)
        r=z['r'].astype(np.float32)                                            # [S,T] log1p bytes
        val=(z['valid'].astype(np.float32) if 'valid' in z.files else np.isfinite(r).astype(np.float32))
        ts=z['ts'].astype(np.int64)                                            # [T] hourly epochs
        ok=val>0; mu=float(r[ok].mean()); sd=float(r[ok].std()+1e-6)
        for i in range(r.shape[0]):
            zz=(np.where(ok[i],r[i],0.0)-mu)/sd
            out.append((zz.astype(np.float32),val[i],ts.copy()))
        print(f"cesnet: {r.shape[0]} entities x {r.shape[1]} hours mu={mu:.3f} sd={sd:.3f}",flush=True)
    if 'azure' in SRC:
        with open(os.path.expanduser('~/nfm/data/cloudops/azure_vm_traces_2017.pkl'),'rb') as fh:
            zc=pickle.load(fh)
        series=zc['pt']
        if len(series)>AZMAX:
            series=[series[k] for k in np.random.default_rng(0).choice(len(series),AZMAX,replace=False)]
        pool=np.concatenate([s[d][np.isfinite(s[d])] for s in series[:500] for d in range(s.shape[0])])
        pool=np.log1p(np.maximum(pool,0)); mu=float(pool.mean()); sd=float(pool.std()+1e-6)
        n=0
        for s in series:
            for dvar in range(s.shape[0]):
                raw=s[dvar]
                if len(raw)<8: continue
                valid=np.isfinite(raw).astype(np.float32)
                zz=(np.log1p(np.maximum(_ffill(raw),0))-mu)/sd
                ts=(np.arange(len(raw),dtype=np.int64)*300)                    # 5-min grid, synthetic clock
                out.append((zz.astype(np.float32),valid,ts)); n+=1
        print(f"azure: {n} streams mu={mu:.3f} sd={sd:.3f}",flush=True)
    return out

def build_src_windows(streams):
    LMAX=bf.LMAX
    X=[];Y=[];V=[];TS=[]
    for zz,valid,ts in streams:
        m=len(zz)
        if m<4: continue
        x=np.zeros((m,NCH),np.float32)
        x[1:,0]=zz[:-1]
        x[1:,3]=np.log1p(np.maximum(np.diff(ts),0))
        hod=(ts%86400)/86400.0
        x[:,4]=np.sin(2*np.pi*hod); x[:,5]=np.cos(2*np.pi*hod)
        x[:,6]=((ts//86400)%7)/6.0
        for s in range(0,max(1,m-1),PSTRIDE):
            e=min(s+LMAX,m); T=e-s
            xin=np.zeros((LMAX,NCH),np.float32); xin[:T]=x[s:e]
            if s>0: xin[0,0]=zz[s-1]; xin[0,3]=np.log1p(max(ts[s]-ts[s-1],0))
            y=np.zeros(LMAX,np.float32); y[:T]=zz[s:e]
            v=np.zeros(LMAX,np.float32); v[:T]=valid[s:e]
            t_=np.zeros(LMAX,np.int64); t_[:T]=ts[s:e]-ts[s]
            X.append(xin);Y.append(y);V.append(v);TS.append(t_)
            if e>=m: break
    X=np.stack(X);Y=np.stack(Y);V=np.stack(V);TS=np.stack(TS)
    if len(X)>MAXW:
        k=np.random.default_rng(0).choice(len(X),MAXW,replace=False)
        X,Y,V,TS=X[k],Y[k],V[k],TS[k]
    return X,Y,V,TS

# ---------- bf.train clone: identical math + per-epoch print + optional init ----------
def train_loud(kind,X,Y,V,TS,mu,sd,N,init=None):
    m=bf.make(kind,N)
    if init is not None:
        m.load_state_dict(init,strict=True); print("init: loaded pretrained state_dict (strict)",flush=True)
    m=m.to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=0.01); nb=len(X)
    Yz=(Y-mu)/sd
    Xt=torch.tensor(X); Yt=torch.tensor(Yz); Vt=torch.tensor(V); Tt=torch.tensor(TS)
    vm=Vt.reshape(-1).bool(); flat=Xt.reshape(-1,NCH)[vm]
    cmu=flat.mean(0); csd=flat.std(0)+1e-6
    for ep in range(bf.EP):
        m.train(); idx=np.random.permutation(nb); tot=0.0; cnt=0
        for i in range(0,nb,bf.BS):
            j=idx[i:i+bf.BS]; x=((Xt[j]-cmu)/csd).to(dev); y=Yt[j].to(dev); v=Vt[j].to(dev); t=Tt[j].to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m(x,t)
            loss=m.head.nll(hid.float(),y,v)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tot+=float(loss)*len(j); cnt+=len(j)
        print(f"  ep{ep} nll={tot/max(cnt,1):.4f}",flush=True)
    m.eval(); return m,cmu.to(dev),csd.to(dev)

def feat_stats(X,V):
    Xt=torch.tensor(X); Vt=torch.tensor(V)
    vm=Vt.reshape(-1).bool(); flat=Xt.reshape(-1,NCH)[vm]
    return (flat.mean(0)).to(dev),(flat.std(0)+1e-6).to(dev)

if __name__=='__main__':
    np.random.seed(bf.SEED); torch.manual_seed(bf.SEED)
    if MODE=='pretrain':
        streams=src_streams()
        X,Y,V,TS=build_src_windows(streams)
        print(f"pretrain windows={len(X):,} SRC={SRC} EP={bf.EP} D={bf.D} H={bf.H} L={bf.L} LMAX={bf.LMAX}",flush=True)
        m,cmu,csd=train_loud(bf.ARM,X,Y,V,TS,0.0,1.0,bf.LMAX)
        os.makedirs(os.path.dirname(CKPT),exist_ok=True)
        torch.save({'sd':m.state_dict(),
                    'cfg':dict(D=bf.D,H=bf.H,L=bf.L,LMAX=bf.LMAX,MIX=bf.MIX,ARM=bf.ARM,SRC=SRC,
                               MAXW=MAXW,EP=bf.EP,USE_TIME=os.environ.get('USE_TIME','1'))},CKPT)
        print(f"TRANSFER_PRETRAIN_DONE -> {CKPT}",flush=True)
    else:
        d,w,pre,t0,fn=bf.load(bf.NFILE)
        if PREDAYS>=6.99: tr=pre
        else: tr=pre[pre['t']>=t0-PREDAYS*86400].reset_index(drop=True)
        print(f"eval file={fn} window={len(w):,} | MODE={MODE} PREDAYS={PREDAYS} train jobs={len(tr):,} "
              f"(full pre={len(pre):,}) SEED={bf.SEED} D={bf.D} L={bf.L} EP={bf.EP}",flush=True)
        alld=lg(tr['dur'].values); mu=float(alld.mean()); sd=float(alld.std()+1e-6)
        print(f"log1p(dur) budget-train stats mu={mu:.3f} sd={sd:.3f}",flush=True)
        X,Y,V,TS=bf.build_train([tr]); print(f"train windows={len(X):,}",flush=True)
        init=None
        if MODE in ('transfer','zeroshot'):
            ck=torch.load(CKPT,map_location='cpu')
            print(f"ckpt cfg={ck['cfg']}",flush=True)
            init=ck['sd']
        if MODE=='zeroshot':
            m=bf.make(bf.ARM,bf.LMAX); m.load_state_dict(init,strict=True); m=m.to(dev); m.eval()
            cmu,csd=feat_stats(X,V)
        else:
            m,cmu,csd=train_loud(bf.ARM,X,Y,V,TS,mu,sd,bf.LMAX,init=init)
        est=bf.infer(m,cmu,csd,w,pre,mu,sd,bf.LMAX)                            # history contract: FULL pre-window, unchanged
        tru=w['dur'].values
        for qi,q in enumerate(bf.QS):
            p=os.path.join(bf.OUT,f"{bf.ARM.lower()}_q{int(round(q*100))}.npz")
            np.savez(p,est=est[:,qi])
            e=est[:,qi]; cov=(e>=tru).mean(); rat=np.median(e/np.maximum(tru,1))
            le=np.log1p(e); lt=np.log1p(tru); r_=lt-le
            pin=float(np.mean(np.maximum(q*r_,(q-1)*r_)))
            print(f"  q{int(round(q*100))}: median est={np.median(e):.0f}s coverage={cov:.3f} "
                  f"median est/dur={rat:.2f} pinball(log)={pin:.4f} -> {p}",flush=True)
        print(f"TRANSFER_{MODE.upper()}_DONE",flush=True)
