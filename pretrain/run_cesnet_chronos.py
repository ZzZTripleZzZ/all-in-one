"""Chronos-Bolt ZERO-SHOT baseline on the CESNET leg, under the EXACT protocol of run_cesnet_v2.py:
same npz, same 80/20 chronological cut, same fully-valid windows, same rng(0) EVALMAX subsample,
same per-series train-region MASE scale, same metrics_v2 calls, same provisioning pareto.
Context = the window's first SPLIT hours in r-space (log1p bytes, as stored); Chronos returns
quantiles in the same space. Defensive baseline vs arXiv:2511.17529 (TTM on CESNET).
Env: SPLIT(168) HMAX(24) STRIDE(24) EVALMAX(3000) BS(256) MODEL(amazon/chronos-bolt-small)."""
import os, sys, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
import metrics_v2 as MET

SPLIT=int(os.environ.get('SPLIT',168)); HMAX=int(os.environ.get('HMAX',24)); N=SPLIT+HMAX
STRIDE=int(os.environ.get('STRIDE',24)); EVALMAX=int(os.environ.get('EVALMAX',3000)); BS=int(os.environ.get('BS',256))
MODEL=os.environ.get('MODEL','amazon/chronos-bolt-small')
HS=[1,4,24]; QS9=tuple(np.linspace(0.1,0.9,9)); TAUS=(0.5,0.8,0.9,0.95,0.99)
QLEV=list(QS9)+[0.95,0.99]
NPZ=os.path.expanduser("~/nfm/data/cesnet/cesnet_hourly.npz")

def windows(r,v,ts,lo,hi,stride):
    W=[]; S=[]
    for i in range(len(r)):
        for s in range(lo,hi-N+1,stride):
            if v[i,s:s+N].min()>0: W.append(r[i,s:s+N]); S.append(i)
    return np.array(W,np.float32),np.array(S)

def pareto(cap_r,act_r):
    cap=np.expm1(cap_r); act=np.expm1(act_r)
    over=np.maximum(cap-act,0).sum()/max(act.sum(),1e-9)*100.0
    viol=float((act>cap).mean()*100.0)
    return over,viol

if __name__=='__main__':
    from chronos import BaseChronosPipeline
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    z=np.load(NPZ); r=z['r']; v=z['valid']; ts=z['ts']
    S,T=r.shape; cut=int(T*0.8)
    scale={i:float(np.mean(np.abs(np.diff(r[i,:cut][v[i,:cut]>0])))) for i in range(S)}
    Ze,SIDe=windows(r,v,ts,cut,T,STRIDE)
    if len(Ze)>EVALMAX:
        k=np.random.default_rng(0).choice(len(Ze),EVALMAX,replace=False); Ze,SIDe=Ze[k],SIDe[k]
    print(f"model={MODEL} device={dev} eval-win={len(Ze):,} (protocol-matched to run_cesnet_v2)",flush=True)
    true={h: Ze[:,SPLIT+h-1] for h in HS}
    pipe=BaseChronosPipeline.from_pretrained(MODEL,device_map=dev,torch_dtype=torch.bfloat16)
    Qrows=[]
    for i in range(0,len(Ze),BS):
        ctx=torch.tensor(Ze[i:i+BS,:SPLIT])
        q,_=pipe.predict_quantiles(ctx,prediction_length=HMAX,quantile_levels=QLEV)
        Qrows.append(q.float().cpu().numpy())                                  # [B, HMAX, len(QLEV)]
    Q=np.concatenate(Qrows,0)
    res={}
    for h in HS:
        qh=Q[:,h-1,:]                                                          # [n, 11] cols=9-grid+0.95+0.99
        med=qh[:,4]
        mase,_=MET.mase_per_series(med,true[h],SIDe,scale)
        p90=MET.pinball(true[h],qh[:,8],0.9); p99=MET.pinball(true[h],qh[:,10],0.99)
        crps=MET.crps_from_quantiles(true[h],qh[:,:9],QS9)
        res[h]=(mase,p90,p99,crps)
    print("\n==== Chronos-Bolt zero-shot: MASE | pin90 | pin99 | CRPS9 ====",flush=True)
    print("  chronos-0shot      "+" | ".join(f"h{h}: "+" ".join(f"{x:.3f}" for x in res[h]) for h in HS),flush=True)
    print("\n==== next-hour provisioning Pareto (overprovision% , violation%) ====",flush=True)
    capcols={0.5:4,0.8:7,0.9:8,0.95:9,0.99:10}
    for tau in TAUS:
        ov,vi=pareto(Q[:,0,capcols[tau]],true[1])
        print(f"  chronos-0shot      tau={tau}: over={ov:6.1f}%  viol={vi:5.2f}%",flush=True)
    print("CESNET_CHRONOS_DONE",flush=True)
