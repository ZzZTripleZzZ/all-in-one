#!/usr/bin/env python
"""ALL-IN-ONE multi-horizon forecasting: ONE generative model (any input/output length)
vs a SUITE of FIXED-horizon specialist models (one trained per horizon).

Thesis (user's framing): classical ML forecasting fixes (input_len, output_len) -> you must
train a separate model for every horizon/task. Our generative model is autoregressive: ONE
model, arbitrary input length, arbitrary output length, rolled out to cover ALL horizons.

Setup: univariate on each dataset's target token sequence (quantized). Entity-holdout split.
Per horizon h: exact-bin accuracy + MAE(|delta bin|). Baseline 'GBM*' = one HistGBM PER horizon
(so it needs len(HORIZONS) models); our HSTU/TFM = ONE model for all horizons.
Env: CTX (context len, default 32), HMAX (max horizon, default 16), D/H/L/EP model size."""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from bench_single import SeqModel, burst, fdata, cesnet, nids_cic, dev

CTX=int(os.environ.get('CTX',32)); HMAX=int(os.environ.get('HMAX',16))
HORIZONS=[h for h in [1,2,4,8,16,32] if h<=HMAX]
N=CTX+HMAX
torch.manual_seed(0); np.random.seed(0)

def split_ent(n):
    rng=np.random.default_rng(0); e=rng.permutation(n); k=max(1,int(n*0.2))
    return e[k:], e[:k]                                   # train_ents, test_ents

def gen_windows(seqs, V):                                 # length-N next-token windows for the generative model
    W=[]
    for s in seqs:
        for i in range(0,len(s),N):
            w=s[i:i+N]
            if len(w)>=8: W.append(np.concatenate([w,np.full(N-len(w),V)]))
    return np.array(W,np.int64) if W else np.zeros((0,N),np.int64)

def cf_windows(seqs):                                     # (context CTX, future HMAX) pairs
    C,Fu=[],[]
    for s in seqs:
        for i in range(0,len(s)-CTX-HMAX+1, N):
            C.append(s[i:i+CTX]); Fu.append(s[i+CTX:i+CTX+HMAX])
    return (np.array(C,np.int64),np.array(Fu,np.int64)) if C else (np.zeros((0,CTX),np.int64),np.zeros((0,HMAX),np.int64))

def train_gen(kind, W, V, D, H, L, epochs, bs=256):
    m=SeqModel(kind,V,0,D,H,L,N).to(dev); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=0.01)
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(len(W))
        for i in range(0,len(idx),bs):
            tok=torch.tensor(W[idx[i:i+bs]]).to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lg=m(tok)[:,:-1]; loss=F.cross_entropy(lg.reshape(-1,V),tok[:,1:].reshape(-1),ignore_index=V)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def rollout(m, C, V, bs=512):                             # autoregressive greedy rollout, ONE model -> all horizons
    m.eval(); out=[]
    for i in range(0,len(C),bs):
        c=torch.tensor(C[i:i+bs]).to(dev); B=c.shape[0]
        buf=torch.full((B,N),V,device=dev,dtype=torch.long); buf[:,:CTX]=c
        for p in range(CTX,N):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lg=m(buf)[:,p-1].float()
            buf[:,p]=lg.argmax(-1)
        out.append(buf[:,CTX:].cpu().numpy())
    return np.concatenate(out)                            # [n, HMAX]

def per_h(pred, Fu, name):                                # pred[n,HMAX], Fu[n,HMAX]
    return {h:( (pred[:,h-1]==Fu[:,h-1]).mean(), np.abs(pred[:,h-1].astype(int)-Fu[:,h-1].astype(int)).mean() ) for h in HORIZONS}

def run(name, loader, D, H, L, epochs):
    st,_,V=loader(); n=len(st)
    tr_e,te_e=split_ent(n)
    tr_seqs=[st[i] for i in tr_e]; te_seqs=[st[i] for i in te_e]
    Wg=gen_windows(tr_seqs,V); Cte,Fte=cf_windows(te_seqs); Ctr,Ftr=cf_windows(tr_seqs)
    print(f"\n### {name}  entities tr/te={len(tr_e)}/{len(te_e)}  gen-windows={len(Wg):,}  eval-windows={len(Cte):,}  V={V}  CTX={CTX} HMAX={HMAX}")
    if len(Wg)<10 or len(Cte)<10: print("  too few windows, skip"); return
    results={}
    # baseline 1: persistence (repeat last context token) -- one rule, all horizons
    per=np.repeat(Cte[:,-1:],HMAX,axis=1); results['persist(1)']=per_h(per,Fte,'persist')
    # baseline 2: FIXED-HORIZON specialists -- ONE HistGBM trained PER horizon (needs len(HORIZONS) models)
    gbm_pred=np.zeros_like(Fte)
    Xtr=Ctr.astype(np.float32); Xte=Cte.astype(np.float32)
    if len(Xtr)>200000:
        s=np.random.choice(len(Xtr),200000,replace=False); Xtr=Xtr[s]; Ftr=Ftr[s]
    for h in HORIZONS:
        g=HistGradientBoostingClassifier(max_iter=150,max_depth=8); g.fit(Xtr,Ftr[:,h-1])
        gbm_pred[:,h-1]=g.predict(Xte)
    results[f'GBM*({len(HORIZONS)})']=per_h(gbm_pred,Fte,'gbm')
    # ours: ONE generative model per arch, rolled out to all horizons
    for kind in ['tfm','hstu']:
        m=train_gen(kind,Wg,V,D,H,L,epochs); pr=rollout(m,Cte,V)
        results[f'{kind.upper()}(1)']=per_h(pr,Fte,kind); del m; torch.cuda.empty_cache()
    # table
    names=list(results); print("horizon | "+" | ".join(f"{k:>14}" for k in names)+"    [acc | MAE]")
    for h in HORIZONS:
        print(f"  h={h:<4}| "+" | ".join(f"{results[k][h][0]:.3f}/{results[k][h][1]:4.2f}".rjust(14) for k in names))
    print(f"  #models to cover all {len(HORIZONS)} horizons:  persist=0  GBM*={len(HORIZONS)}  HSTU=1  TFM=1")

if __name__=='__main__':
    print("device:",dev, torch.cuda.get_device_name(0) if dev=='cuda' else '', f"| CTX={CTX} HMAX={HMAX} horizons={HORIZONS}")
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4)); EP=int(os.environ.get('EP',4))
    print(f"model: D={D} H={H} L={L} epochs={EP}")
    if which in ('all','burst'):  run("BurstGPT/DC  response-len", burst, D,H,L,EP)
    if which in ('all','fdata'):  run("F-DATA/HPC   job-duration", fdata, D,H,L,EP)
    if which in ('all','cesnet'): run("CESNET/NET   nbytes",       cesnet, D,H,L,EP)
    if which in ('all','cic'):    run("CIC-IDS2017/SEC attack",    nids_cic, D,H,L,EP)
