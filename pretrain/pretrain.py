"""CROSS-DOMAIN PRETRAINED HSTU (the real thesis): ONE HSTU pretrained on pooled
DC+HPC+NET+wireless telemetry, evaluated per-domain ZERO-SHOT vs Chronos (both pretrained
=> FAIR comparison, unlike from-scratch). Chronos-style per-series scaling + shared vocab so
one model spans domains of different magnitude. Env: K,CTX,HMAX,D/H/L,PRE_EP, DOMAINS(csv)."""
import os, sys, numpy as np, torch
# --- repo layout: shared libs live in common/ and data/ (see repo root README) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("common", "data"):
    sys.path.insert(0, os.path.join(_ROOT, _p))
from nfm_core import HSTU, dev, mase, train_gen, rollout, rank_metrics
import nfm_data, baselines_sota as bs

K=int(os.environ.get('K',1024)); CTX=int(os.environ.get('CTX',64)); HMAX=int(os.environ.get('HMAX',16))
N=CTX+HMAX; HZ=[h for h in [1,2,4,8,16,24,32] if h<=HMAX]
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4)); PRE_EP=int(os.environ.get('PRE_EP',8))
DOMAINS=os.environ.get('DOMAINS','fdata,beam,cesnet').split(',')
np.random.seed(0); torch.manual_seed(0)

class ScaleTok:                                   # per-series scale + shared vocab (Chronos-style)
    def __init__(self,K): self.K=K
    def fit(self,series):
        norm=np.concatenate([np.asarray(v)/(np.mean(np.abs(v))+1e-8) for v in series if len(v)>1])
        norm=norm[np.isfinite(norm)]
        self.edges=np.quantile(norm,np.linspace(0,1,self.K+1)[1:-1])
        tok=np.clip(np.digitize(norm,self.edges),0,self.K-1)
        self.centers=np.array([np.median(norm[tok==k]) if (tok==k).any() else 0.0 for k in range(self.K)])
        return self
    def tok(self,v,s): return np.clip(np.digitize(np.asarray(v)/s,self.edges),0,self.K-1).astype(np.int64)
    def val(self,t,s): return self.centers[np.clip(np.asarray(t),0,self.K-1)]*s

def split(n): rng=np.random.default_rng(0); e=rng.permutation(n); k=max(1,int(n*0.2)); return e[k:],e[:k]

# load domains -> (train V/T, test V/T)
DAT={}
for d in DOMAINS:
    V,T=nfm_data.LOADERS[d]()
    if len(V)>=10:
        tr,te=split(len(V)); DAT[d]=([V[i] for i in tr],[T[i] for i in tr],[V[i] for i in te],[T[i] for i in te])
    else:
        c=int(len(V[0])*0.8); DAT[d]=([V[0][:c]],[T[0][:c]],[V[0][c:]],[T[0][c:]])
    print(f"loaded {d}: train-series={len(DAT[d][0])} test-series={len(DAT[d][2])}")

tokz=ScaleTok(K).fit([v for d in DAT for v in DAT[d][0]])       # shared vocab from ALL train series

def gen_windows(Vs,Ts):
    Tok,TS=[],[]
    for v,ts in zip(Vs,Ts):
        for s0 in range(0,len(v),N):
            w=np.asarray(v[s0:s0+N]); wt=np.asarray(ts[s0:s0+N])
            if len(w)<8: continue
            sc=np.mean(np.abs(w[:CTX])) if len(w)>=CTX else np.mean(np.abs(w)); sc+=1e-8
            tk=tokz.tok(w,sc); pad=N-len(tk)
            Tok.append(np.concatenate([tk,np.full(pad,K)])); TS.append(np.concatenate([wt,np.full(pad,wt[-1] if len(wt) else 0)]))
    return (np.array(Tok),np.array(TS,np.int64)) if Tok else (np.zeros((0,N),np.int64),np.zeros((0,N),np.int64))

PTok,PTs=[],[]
for d in DAT:
    tk,ts=gen_windows(DAT[d][0],DAT[d][1]); PTok.append(tk); PTs.append(ts); print(f"  {d}: {len(tk)} pretrain windows")
PTok=np.concatenate(PTok); PTs=np.concatenate(PTs)
print(f"TOTAL pretrain windows (pooled {list(DAT)}): {len(PTok):,}  vocab K={K} CTX={CTX} HMAX={HMAX} D={D} L={L} PRE_EP={PRE_EP}")

m=train_gen(HSTU(K,D=D,H=H,L=L,N=N),PTok,PTs,K,epochs=PRE_EP); print("PRETRAINED one HSTU across domains.")

def eval_domain(name,teV,teT):
    scale=np.mean([np.mean(np.abs(np.diff(v))) for v in teV if len(v)>1])
    Cx,Cxv,TsF,Fu,Sc=[],[],[],[],[]
    for v,ts in zip(teV,teT):
        v=np.asarray(v); ts=np.asarray(ts)
        for s0 in range(0,len(v)-N+1,N):
            ctx=v[s0:s0+CTX]; sc=np.mean(np.abs(ctx))+1e-8
            Cx.append(tokz.tok(ctx,sc)); Cxv.append(ctx); TsF.append(ts[s0:s0+N]); Fu.append(v[s0+CTX:s0+CTX+HMAX]); Sc.append(sc)
    if not Cx: print(f"### {name}: no eval windows"); return
    Cx=np.array(Cx); Cxv=np.array(Cxv,float); TsF=np.array(TsF,np.int64); Fu=np.array(Fu,float); Sc=np.array(Sc)
    pv=rollout(m,Cx,TsF,K,CTX,HMAX,tokz.centers,scale=Sc)   # expected-value forecast, un-normalized per window
    res={'persist(1)':mase(np.repeat(Cxv[:,-1:],HMAX,1),Fu,scale),'HSTU-pretrained(1)':mase(pv,Fu,scale)}
    try: res['Chronos(1)']=mase(bs.chronos_zeroshot(Cxv,HMAX),Fu,scale)
    except Exception as e: print("  chronos skip",str(e)[:60])
    print(f"\n### {name}  (HSTU pretrained on {list(DAT)}, ZERO-SHOT)  eval-win={len(Cx)}  scale={scale:.3g}")
    print("  MASE h | "+" | ".join(f"{k:>18}" for k in res))
    for h in HZ: print(f"       {h:>3} | "+" | ".join(f"{res[k][h-1]:>18.3f}" for k in res))
    nxt=np.array([tokz.tok(np.array([Fu[i,0]]),Sc[i])[0] for i in range(len(Fu))])   # HSTU-faithful 1-step next-token ranking
    rm=rank_metrics(m,Cx,TsF,nxt,K,CTX)
    print("  HSTU-faithful next-token: "+"  ".join(f"{k}={v:.3f}" for k,v in rm.items()))

for d in DAT: eval_domain(d,DAT[d][2],DAT[d][3])
