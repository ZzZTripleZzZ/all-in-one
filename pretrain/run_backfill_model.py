"""MODEL runtime-estimator arm for the EASY-backfill replay (run_backfill_replay.py plug-in).
The NFM (HSTU) predicts, at each job's submit time, the distribution of its log-runtime from the
user's PAST job history plus the current job's submit-time covariates (nodes, user limit), and we
read a quantile as the walltime estimate. This is the generative sequence model in "any-history ->
next-value distribution" mode -- the SAME backbone/head as the F-DATA/CloudOps legs, no rollout.

Why it can beat hist-qXX: hist-qXX is the per-user *marginal* empirical quantile (one number per
user, ignoring the current job); the model gives a *conditional* quantile p(dur | nodes, limit,
full past jobs). Causality is enforced two ways: (1) model WEIGHTS are trained only on pre-window
jobs (t < t0); (2) at inference the per-user history is grown in submit order, exactly mirroring
hist_quantile_estimates() -- the only thing that changes vs the baseline is empirical->learned.

Framing (teacher-forced, causal): position t input = [ log1p(dur_{t-1}) (0 at t=0), log1p(nn_t),
log1p(lim_t) ]; target = log1p(dur_t). Hidden_t sees covariates of job t + full jobs 0..t-1, so it
predicts dur_t from what is known at t's submit. Read the last position for the current job.

Emits one npz per quantile: {est:[len(w)]} in seconds, clipped to [60, lim], aligned to the FULL
window (pre-fit, same order as run_backfill_replay builds dur/nn/lim). Env: NFILE DAYS SKIPD MAXJ
LMAX(96) D(256) H(4) L(4) EP(25) MIX(4) BS(256) ARM(HSTU) QS(0.5,0.7,0.8,0.9,0.95) OUT(dir).
"""
import os, sys, glob, numpy as np, pandas as pd, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, RNNv2, MixtureHead, dev
import torch.nn as nn

NFILE=int(os.environ.get('NFILE',0)); DAYS=float(os.environ.get('DAYS',14)); SKIPD=float(os.environ.get('SKIPD',7))
MAXJ=int(os.environ.get('MAXJ',300000)); LMAX=int(os.environ.get('LMAX',96))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
EP=int(os.environ.get('EP',25)); MIX=int(os.environ.get('MIX',4)); BS=int(os.environ.get('BS',256))
ARM=os.environ.get('ARM','HSTU'); STRIDE=int(os.environ.get('STRIDE',LMAX//2))
QS=[float(x) for x in os.environ.get('QS','0.5,0.7,0.8,0.9,0.95').split(',')]
OUT=os.path.expanduser(os.environ.get('OUT','~/nfm/data/backfill_est')); os.makedirs(OUT,exist_ok=True)
SMOKE=int(os.environ.get('SMOKE',0))
SEED=int(os.environ.get('SEED',0))
TRAINF=int(os.environ.get('TRAINF',-1))                                        # >=0: train weights+stats on THAT file's pre-window, apply zero-shot to NFILE's window (per-user history still comes from NFILE)
if SMOKE: EP=3; D=64; L=2; MAXJ=20000

def load(idx=None):
    f=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))[NFILE if idx is None else idx]
    d=pd.read_parquet(f,columns=['usr','adt','nnumr','elpl','duration'])
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce')
    d=d.dropna(subset=['adt']).sort_values('adt')
    d['t']=d['adt'].astype('datetime64[ns, UTC]').astype(np.int64)//10**9
    d['dur']=pd.to_numeric(d['duration'],errors='coerce').clip(lower=1)
    d['lim']=pd.to_numeric(d['elpl'],errors='coerce')
    d['nn']=pd.to_numeric(d['nnumr'],errors='coerce').fillna(1).clip(lower=1).astype(int)
    d=d.dropna(subset=['dur','lim']); d['lim']=np.maximum(d['lim'],d['dur'])
    t0=d['t'].iloc[0]+SKIPD*86400
    w=d[(d['t']>=t0)&(d['t']<t0+DAYS*86400)].head(MAXJ).reset_index(drop=True)
    pre=d[d['t']<t0].reset_index(drop=True)
    return d,w,pre,t0,os.path.basename(f)

lg=lambda a: np.log1p(np.asarray(a,float))

class TriInput(nn.Module):                                                     # [B,N,3] -> [B,N,D]
    def __init__(self,D): super().__init__(); self.lin=nn.Linear(3,D)
    def forward(self,inp): return self.lin(inp.float())

def make(kind,N):
    inp=TriInput(D); head=MixtureHead(D,MIX)
    if kind=='HSTU': return HSTUv2(inp,head,D=D,H=H,L=L,N=N)
    if kind=='GRU':  return RNNv2(inp,head,D=D,H=H,L=L,N=N,cell='gru')

def build_train(pre,st):
    """per-user sliding windows over pre-window jobs; every position is a target. Returns X[B,N,3], Y[B,N], V[B,N]."""
    X=[];Y=[];V=[]
    for u,g in pre.groupby('usr'):
        g=g.sort_values('t'); dur=lg(g['dur'].values); nnf=lg(g['nn'].values); limf=lg(g['lim'].values)
        m=len(g)
        for s in range(0,max(1,m-1),STRIDE):
            e=min(s+LMAX,m); seq=slice(s,e); T=e-s
            xd=dur[seq]; xn=nnf[seq]; xl=limf[seq]
            lag=np.concatenate([[0.0],xd[:-1]])                                # dur_{t-1}, 0 at window start
            xin=np.zeros((LMAX,3),np.float32); xin[:T,0]=lag; xin[:T,1]=xn; xin[:T,2]=xl
            y=np.zeros(LMAX,np.float32); y[:T]=xd; v=np.zeros(LMAX,np.float32); v[:T]=1.0
            X.append(xin);Y.append(y);V.append(v)
            if e>=m: break
    return np.stack(X),np.stack(Y),np.stack(V)

def standardize(pre):
    a=lg(pre['dur'].values); return float(a.mean()),float(a.std()+1e-6)

def train(kind,X,Y,V,mu,sd,N):
    m=make(kind,N).to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=0.01); nb=len(X)
    Yz=(Y-mu)/sd; ts=torch.zeros(BS,N,dtype=torch.long,device=dev)
    Xt=torch.tensor(X); Yt=torch.tensor(Yz); Vt=torch.tensor(V)
    vm=Vt.reshape(-1).bool(); flat=Xt.reshape(-1,3)[vm]                         # channel stats over VALID positions only (exclude pad/BOS zeros)
    cmu=flat.mean(0); csd=flat.std(0)+1e-6
    for ep in range(EP):
        m.train(); idx=np.random.permutation(nb)
        for i in range(0,nb,BS):
            j=idx[i:i+BS]; x=((Xt[j]-cmu)/csd).to(dev); y=Yt[j].to(dev); v=Vt[j].to(dev); t=ts[:len(j)]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m(x,t)
            loss=m.head.nll(hid.float(),y,v)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m,cmu.to(dev),csd.to(dev)

@torch.no_grad()
def infer(m,cmu,csd,d,w,pre,mu,sd,N):
    """grow per-user history in submit order (mirrors hist_quantile_estimates), read last position."""
    histd={};histn={};histl={}
    for u,g in pre.groupby('usr'):
        g=g.sort_values('t'); histd[u]=list(lg(g['dur'].values)); histn[u]=list(lg(g['nn'].values)); histl[u]=list(lg(g['lim'].values))
    est=np.zeros((len(w),len(QS)),np.float32)
    Xbuf=[];ridx=[];rows=[]
    wu=w['usr'].values; wd=lg(w['dur'].values); wn=lg(w['nn'].values); wl=lg(w['lim'].values); wlim=w['lim'].values
    def flush():
        if not Xbuf: return
        x=((torch.tensor(np.stack(Xbuf))-cmu.cpu())/csd.cpu()).to(dev); t=torch.zeros(len(Xbuf),N,dtype=torch.long,device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            hid=m(x,t).float()
        hr=hid[torch.arange(len(Xbuf)),torch.tensor(ridx,device=dev)]           # [B,D] at read index
        qz=m.head.quantiles(hr.unsqueeze(1),tuple(QS),S=2048)[:,0,:].cpu().numpy()   # [B,len(QS)] z-space
        sec=np.expm1(qz*sd+mu)
        for b,r in enumerate(rows):
            est[r]=np.clip(sec[b],60.0,max(60.0,wlim[r]))
        Xbuf.clear();ridx.clear();rows.clear()
    for i in range(len(w)):
        u=wu[i]; hd=histd.get(u,[]); hn=histn.get(u,[]); hl=histl.get(u,[])
        pd_=hd[-(LMAX-1):]; pn=hn[-(LMAX-1):]; pl=hl[-(LMAX-1):]; k=len(pd_)     # past jobs in-window
        xin=np.zeros((N,3),np.float32)
        for tpos in range(k):                                                  # past positions: cov of job tpos + lagged dur
            xin[tpos,0]=pd_[tpos-1] if tpos>0 else 0.0; xin[tpos,1]=pn[tpos]; xin[tpos,2]=pl[tpos]
        xin[k,0]=pd_[-1] if k>0 else 0.0; xin[k,1]=wn[i]; xin[k,2]=wl[i]         # current job at position k
        Xbuf.append(xin); ridx.append(k); rows.append(i)
        histd.setdefault(u,[]).append(wd[i]); histn.setdefault(u,[]).append(wn[i]); histl.setdefault(u,[]).append(wl[i])
        if len(Xbuf)>=BS: flush()
    flush()
    return est

if __name__=='__main__':
    np.random.seed(SEED); torch.manual_seed(SEED)
    d,w,pre,t0,fn=load()
    if TRAINF>=0 and TRAINF!=NFILE:
        _,_,pretr,_,fntr=load(TRAINF)                                          # zero-shot: training corpus from another month
    else:
        pretr,fntr=pre,fn
    print(f"eval file={fn} train file={fntr} pre(train)={len(pretr):,} window(eval)={len(w):,} LMAX={LMAX} arm={ARM} D={D} L={L} EP={EP} SEED={SEED} QS={QS}",flush=True)
    mu,sd=standardize(pretr); print(f"log1p(dur) train stats mu={mu:.3f} sd={sd:.3f}",flush=True)
    N=LMAX
    X,Y,V=build_train(pretr,STRIDE); print(f"train windows={len(X):,} (positions/win<= {LMAX})",flush=True)
    m,cmu,csd=train(ARM,X,Y,V,mu,sd,N)
    est=infer(m,cmu,csd,d,w,pre,mu,sd,N)
    for qi,q in enumerate(QS):
        p=os.path.join(OUT,f"{ARM.lower()}_q{int(round(q*100))}.npz")
        np.savez(p,est=est[:,qi]);
        e=est[:,qi]; print(f"  q{int(round(q*100))}: est sec median={np.median(e):.0f} p90={np.quantile(e,0.9):.0f} -> {p}",flush=True)
    # quick honesty check: model estimate accuracy vs true dur (not the replay, just calibration)
    tru=w['dur'].values
    for qi,q in enumerate(QS):
        e=est[:,qi]; cov=(e>=tru).mean(); rat=np.median(e/np.maximum(tru,1))
        print(f"  q{int(round(q*100))} coverage(P[est>=dur])={cov:.3f} median est/dur={rat:.2f}",flush=True)
    print("BACKFILL_MODEL_DONE",flush=True)
