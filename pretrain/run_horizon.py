"""Multi-horizon forecasting runner: ONE generative model (any input/output length, HSTU/TFM)
vs per-horizon specialists (GBM) vs persistence/seasonal. Per-horizon MASE. Our-vocab tokenizer.
Env: K vocab (1024), CTX (32), HMAX (16), D/H/L/EP model. Usage: python run_horizon.py [burst|fdata|cesnet|all]"""
import os, sys, numpy as np, torch
# --- repo layout: shared libs live in common/ and data/ (see repo root README) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("common", "data"):
    sys.path.insert(0, os.path.join(_ROOT, _p))
from nfm_core import Quantizer, HSTU, TFM, LSTMGen, GRUGen, train_gen, rollout, mase, rank_metrics, dev
import nfm_data, baselines_sota as bs
import os as _os

K=int(os.environ.get('K',1024)); CTX=int(os.environ.get('CTX',32)); HMAX=int(os.environ.get('HMAX',16))
N=CTX+HMAX; HZ=[h for h in [1,2,4,8,16,24,32] if h<=HMAX]
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4)); EP=int(os.environ.get('EP',4))
np.random.seed(0); torch.manual_seed(0)

def gen_windows(V,T,idxs,q,frac=None):                 # length-N (tok,ts) windows for the generative model
    Tok,Ts=[],[]
    for i in idxs:
        tk=q.to_tok(V[i]); ts=T[i]; hi=int(len(tk)*frac) if frac else len(tk)
        for s in range(0,hi,N):
            w=tk[s:s+N]; wt=ts[s:s+N]
            if len(w)>=8:
                pad=N-len(w); last=wt[-1] if len(wt) else 0
                Tok.append(np.concatenate([w,np.full(pad,K)])); Ts.append(np.concatenate([wt,np.full(pad,last)]))
    return np.array(Tok),np.array(Ts,np.int64)

def eval_windows(V,T,idxs,q,lo=None):                  # (ctx tok, ctx raw val, full ts, future raw val)
    Cx,Cxv,TsF,Fu=[],[],[],[]
    for i in idxs:
        tk=q.to_tok(V[i]); ts=T[i]; v=V[i]; start=int(len(tk)*lo) if lo else 0
        for s in range(start,len(tk)-N+1,N):
            Cx.append(tk[s:s+CTX]); Cxv.append(v[s:s+CTX]); TsF.append(ts[s:s+N]); Fu.append(v[s+CTX:s+CTX+HMAX])
    if not Cx: return None
    return np.array(Cx),np.array(Cxv,float),np.array(TsF,np.int64),np.array(Fu,float)

def run(name, loader):
    V,T=loader(); n=len(V)
    if n>=10:
        rng=np.random.default_rng(0); e=rng.permutation(n); tec=set(e[:max(1,int(n*0.2))].tolist())
        tr=[i for i in range(n) if i not in tec]; te=[i for i in range(n) if i in tec]; split=f'entity-holdout({n})'
        q=Quantizer(K).fit(np.concatenate([V[i] for i in tr]))
        scale=float(np.mean([np.mean(np.abs(np.diff(V[i]))) for i in tr if len(V[i])>1]))
        Tok,Ts=gen_windows(V,T,tr,q); ev=eval_windows(V,T,te,q)
    else:                                              # single stream -> temporal 80/20
        split='temporal(single-stream)'; cut=int(len(V[0])*0.8); q=Quantizer(K).fit(V[0][:cut])
        scale=float(np.mean(np.abs(np.diff(V[0][:cut]))))
        Tok,Ts=gen_windows(V,T,[0],q,frac=0.8); ev=eval_windows(V,T,[0],q,lo=0.8)
    if ev is None or len(Tok)<10: print(f"### {name}: too few windows"); return
    Cx,Cxv,TsF,Fu=ev
    print(f"\n### {name}  split={split}  vocab K={K}  gen-win={len(Tok):,}  eval-win={len(Cx):,}  scale(naive-MAE)={scale:.3g}  CTX={CTX} HMAX={HMAX}")
    res={}
    # naive baselines on RAW values (one rule, all horizons)
    res['persist(1)']=mase(np.repeat(Cxv[:,-1:],HMAX,axis=1),Fu,scale)
    res['ctx-mean(1)']=mase(np.repeat(Cxv.mean(1,keepdims=True),HMAX,axis=1),Fu,scale)
    P=int(_os.environ.get('SEASON',0))     # seasonal-naive: y[t+h]=y[t+h-P] (strong baseline for seasonal data)
    if P>0:
        sp=np.zeros((len(Cxv),HMAX))
        for h in range(1,HMAX+1):
            idx=CTX-1+h-P; sp[:,h-1]=Cxv[:,idx] if 0<=idx<CTX else Cxv[:,-1]
        res[f'seasonal{P}(1)']=mase(sp,Fu,scale)
    # FIXED-horizon GBM REGRESSOR specialists on raw values (fair strong specialist; ONE per horizon)
    Xtr,Ytr=[],[]
    for i in ([0] if n<10 else [j for j in range(n) if j not in tec]):
        v=V[i]; hi=int(len(v)*0.8) if n<10 else len(v)
        for s in range(0,hi-N+1,max(1,N//3)):
            Xtr.append(v[s:s+CTX]); Ytr.append(v[s+CTX:s+CTX+HMAX])
    res[f'GBMreg*({len(HZ)})']=mase(bs.gbm_reg(np.array(Xtr,float),np.array(Ytr,float),Cxv,HZ,HMAX),Fu,scale)
    # ours + deep generative baselines: ONE model each -> ALL horizons via autoregressive rollout (run FIRST, clean cuda)
    rankres={}; nxt=q.to_tok(Fu[:,0])       # true 1-step next token (bin) for HSTU-faithful ranking
    for tag,Model in [('LSTM(1)',LSTMGen),('GRU(1)',GRUGen),('TFM(1)',TFM),('HSTU(1)',HSTU)]:
        m=train_gen(Model(K,D=D,H=H,L=L,N=N),Tok,Ts,K,epochs=EP)
        res[tag]=mase(rollout(m,Cx,TsF,K,CTX,HMAX,q.centers),Fu,scale)
        if tag in ('TFM(1)','HSTU(1)'): rankres[tag]=rank_metrics(m,Cx,TsF,nxt,K,CTX)
        del m; torch.cuda.empty_cache()
    # SOTA TSFM zero-shot: Chronos (ONE model, any horizon -> direct rival to our flexibility claim); LAST to avoid cuda-state clobber
    if _os.environ.get('CHRONOS','1')=='1':
        try: res['Chronos(1)']=mase(bs.chronos_zeroshot(Cxv,HMAX),Fu,scale)
        except Exception as e: print("  [chronos skipped]",str(e)[:120])
    # table
    names=list(res); print("  MASE h | "+" | ".join(f"{k:>12}" for k in names))
    for h in HZ:
        print(f"      {h:>3} | "+" | ".join(f"{res[k][h-1]:>12.3f}" for k in names))
    print(f"  #models for all {len(HZ)} horizons:  GBMreg*={len(HZ)} (fixed-horizon)  |  HSTU/TFM/LSTM/GRU/Chronos = 1 each (any-horizon)")
    for tag,rm in rankres.items():
        print(f"  [{tag} HSTU-faithful 1-step next-token]  "+"  ".join(f"{k}={v:.3f}" for k,v in rm.items()))

if __name__=='__main__':
    print("device:",dev, torch.cuda.get_device_name(0) if dev=='cuda' else '', f"| K={K} CTX={CTX} HMAX={HMAX} D={D} L={L} EP={EP} horizons={HZ}")
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    order=[which] if which in nfm_data.LOADERS else list(nfm_data.LOADERS)
    labels={'burst':'BurstGPT/DC response-len','fdata':'F-DATA/HPC job-duration','cesnet':'CESNET/NET n_bytes','beam':'ITU-Beam/wireless DL-throughput','azure':'Azure-VM/DC cpu-util','m100':'M100/HPC node-power','borg':'Borg/DC task-duration','alibaba':'Alibaba/DC task-duration'}
    for d in order: run(labels[d], nfm_data.LOADERS[d])
