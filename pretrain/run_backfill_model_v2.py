"""v2 backfill runtime estimator — fixes the two self-diagnosed gaps of v1:
(1) v1 ran the time-aware backbone TIME-BLIND (ts=zeros: the continuous relative-time bias never
    fired) and gave the model FEWER covariates than the GBM baseline (no clock features). v2 feeds
    real submit timestamps into the attention bias and adds inter-arrival + clock covariates. Three
    arms isolate WHERE timing helps on a decision task: ARM=HSTU USE_TIME=1 (timing in attention +
    covariates) / ARM=HSTU USE_TIME=0 (covariates only, same backbone) / ARM=GRU (covariates only,
    recurrent). USE_TIME is read by nfm_v2 at import, so it is set per-job in the sub script.
(2) v1 trained on one month's 7-day pre-window (~27k jobs). v2 optionally pretrains on FULL prior
    months (TRAINMONTHS=comma-separated file indices, all strictly earlier than the eval month =>
    causal), giving the pretraining-scale axis of the FM claim.
Interface (npz per quantile in OUT), inference-history contract (per-user history seeded from the
eval month's pre-window only, grown in submit order), and clipping match v1 exactly, so replay
numbers are directly comparable to v1 and to hist/tsafrir/GBM.
Features per position (7ch): [lag log1p(dur), log1p(nn), log1p(lim), log1p(gap_s), sin(2*pi*hod),
cos(2*pi*hod), dow/6]; ts = submit seconds relative to the sequence start (int64, model log1p-
compresses gaps internally).
"""
import os, sys, glob, numpy as np, pandas as pd, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, RNNv2, MixtureHead, dev
import torch.nn as nn

NFILE=int(os.environ.get('NFILE',0)); DAYS=float(os.environ.get('DAYS',14)); SKIPD=float(os.environ.get('SKIPD',7))
MAXJ=int(os.environ.get('MAXJ',300000)); LMAX=int(os.environ.get('LMAX',96))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
EP=int(os.environ.get('EP',30)); MIX=int(os.environ.get('MIX',6)); BS=int(os.environ.get('BS',256))
ARM=os.environ.get('ARM','HSTU'); STRIDE=int(os.environ.get('STRIDE',24))
QS=[float(x) for x in os.environ.get('QS','0.5,0.6').split(',')]
OUT=os.path.expanduser(os.environ.get('OUT','~/nfm/data/backfill_est_v2')); os.makedirs(OUT,exist_ok=True)
SEED=int(os.environ.get('SEED',0))
ABL=os.environ.get('ABL','full')                                               # full | nohist (covariates-only: zero lag+gap, use LMAX=1) | nocov (history-only: zero nn+lim)
TRAINMONTHS=[int(x) for x in os.environ.get('TRAINMONTHS','').split(',') if x!='']
NCH=7

def load(idx):
    f=sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))[idx]
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

class CovInput(nn.Module):                                                     # [B,N,NCH] -> [B,N,D]
    def __init__(self,D): super().__init__(); self.lin=nn.Linear(NCH,D)
    def forward(self,inp): return self.lin(inp.float())

def make(kind,N):
    inp=CovInput(D); head=MixtureHead(D,MIX)
    if kind=='HSTU': return HSTUv2(inp,head,D=D,H=H,L=L,N=N)
    if kind=='GRU':  return RNNv2(inp,head,D=D,H=H,L=L,N=N,cell='gru')

def feats(durL,nnL,limL,tS,k0):
    """features + relative ts for positions k0..len-1 of one user stream (arrays already log where noted)."""
    T=len(durL)
    x=np.zeros((T,NCH),np.float32)
    x[1:,0]=durL[:-1]                                                          # lagged dur (0 at stream start)
    x[:,1]=nnL; x[:,2]=limL
    x[1:,3]=lg(np.maximum(np.diff(tS),0)); x[0,3]=0.0                          # inter-arrival gap
    hod=(tS%86400)/86400.0
    x[:,4]=np.sin(2*np.pi*hod); x[:,5]=np.cos(2*np.pi*hod)
    x[:,6]=((tS//86400)%7)/6.0
    if ABL=='nohist': x[:,0]=0.0; x[:,3]=0.0
    if ABL=='nocov':  x[:,1]=0.0; x[:,2]=0.0
    return x

def build_train(dfs):
    X=[];Y=[];V=[];TS=[]
    for df in dfs:
        for u,g in df.groupby('usr'):
            g=g.sort_values('t')
            durL=lg(g['dur'].values); nnL=lg(g['nn'].values); limL=lg(g['lim'].values); tS=g['t'].values.astype(np.int64)
            m=len(g); xf=feats(durL,nnL,limL,tS,0)
            for s in range(0,max(1,m-1),STRIDE):
                e=min(s+LMAX,m); T=e-s
                xin=np.zeros((LMAX,NCH),np.float32); xin[:T]=xf[s:e]
                if s>0 and ABL!='nohist': xin[0,0]=durL[s-1]; xin[0,3]=lg([max(tS[s]-tS[s-1],0)])[0]   # window start keeps true lag/gap
                y=np.zeros(LMAX,np.float32); y[:T]=durL[s:e]
                v=np.zeros(LMAX,np.float32); v[:T]=1.0
                ts=np.zeros(LMAX,np.int64); ts[:T]=tS[s:e]-tS[s]
                X.append(xin);Y.append(y);V.append(v);TS.append(ts)
                if e>=m: break
    return np.stack(X),np.stack(Y),np.stack(V),np.stack(TS)

def train(kind,X,Y,V,TS,mu,sd,N):
    m=make(kind,N).to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=0.01); nb=len(X)
    Yz=(Y-mu)/sd
    Xt=torch.tensor(X); Yt=torch.tensor(Yz); Vt=torch.tensor(V); Tt=torch.tensor(TS)
    vm=Vt.reshape(-1).bool(); flat=Xt.reshape(-1,NCH)[vm]
    cmu=flat.mean(0); csd=flat.std(0)+1e-6
    for ep in range(EP):
        m.train(); idx=np.random.permutation(nb)
        for i in range(0,nb,BS):
            j=idx[i:i+BS]; x=((Xt[j]-cmu)/csd).to(dev); y=Yt[j].to(dev); v=Vt[j].to(dev); t=Tt[j].to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m(x,t)
            loss=m.head.nll(hid.float(),y,v)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m,cmu.to(dev),csd.to(dev)

@torch.no_grad()
def infer(m,cmu,csd,w,pre,mu,sd,N):
    """per-user history seeded from eval-month pre-window ONLY (v1/hist contract), grown in submit order."""
    hist={}
    for u,g in pre.groupby('usr'):
        g=g.sort_values('t')
        hist[u]={'d':list(lg(g['dur'].values)),'n':list(lg(g['nn'].values)),
                 'l':list(lg(g['lim'].values)),'t':list(g['t'].values.astype(np.int64))}
    est=np.zeros((len(w),len(QS)),np.float32)
    Xbuf=[];Tbuf=[];ridx=[];rows=[]
    wu=w['usr'].values; wd=lg(w['dur'].values); wn=lg(w['nn'].values); wl=lg(w['lim'].values)
    wt=w['t'].values.astype(np.int64); wlim=w['lim'].values
    def flush():
        if not Xbuf: return
        x=((torch.tensor(np.stack(Xbuf))-cmu.cpu())/csd.cpu()).to(dev)
        t=torch.tensor(np.stack(Tbuf)).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            hid=m(x,t).float()
        hr=hid[torch.arange(len(Xbuf)),torch.tensor(ridx,device=dev)]
        qz=m.head.quantiles(hr.unsqueeze(1),tuple(QS),S=2048)[:,0,:].cpu().numpy()
        sec=np.expm1(qz*sd+mu)
        for b,r in enumerate(rows):
            est[r]=np.clip(sec[b],60.0,max(60.0,wlim[r]))
        Xbuf.clear();Tbuf.clear();ridx.clear();rows.clear()
    for i in range(len(w)):
        u=wu[i]; h=hist.setdefault(u,{'d':[],'n':[],'l':[],'t':[]})
        K=LMAX-1
        past=lambda lst: lst[-K:] if K>0 else []
        durL=np.array(past(h['d'])+[0.0]); nnL=np.array(past(h['n'])+[wn[i]])
        limL=np.array(past(h['l'])+[wl[i]]); tS=np.array(past(h['t'])+[int(wt[i])],dtype=np.int64)
        k=len(durL)-1                                                          # current job position
        xf=feats(durL,nnL,limL,tS,0)
        if k>0 and ABL!='nohist': xf[k,0]=durL[k-1]                            # lag = last finished dur (feats already does this)
        xin=np.zeros((N,NCH),np.float32); xin[:k+1]=xf
        ts=np.zeros(N,np.int64); ts[:k+1]=tS-tS[0]
        Xbuf.append(xin); Tbuf.append(ts); ridx.append(k); rows.append(i)
        h['d'].append(wd[i]); h['n'].append(wn[i]); h['l'].append(wl[i]); h['t'].append(int(wt[i]))
        if len(Xbuf)>=BS: flush()
    flush()
    return est

if __name__=='__main__':
    np.random.seed(SEED); torch.manual_seed(SEED)
    d,w,pre,t0,fn=load(NFILE)
    corp=[pre]; names=[f"{fn}:pre"]
    for mi in TRAINMONTHS:
        assert mi<NFILE, f"TRAINMONTHS {mi} not strictly earlier than eval file {NFILE}"
        dm,_,_,_,fm=load(mi); corp.append(dm); names.append(f"{fm}:full")
    njobs=sum(len(c) for c in corp)
    print(f"eval file={fn} window={len(w):,} | train corpus={names} jobs={njobs:,} | arm={ARM} ABL={ABL} USE_TIME={os.environ.get('USE_TIME','1')} SEED={SEED} D={D} L={L} EP={EP}",flush=True)
    alld=np.concatenate([lg(c['dur'].values) for c in corp]); mu=float(alld.mean()); sd=float(alld.std()+1e-6)
    print(f"log1p(dur) train stats mu={mu:.3f} sd={sd:.3f}",flush=True)
    N=LMAX
    X,Y,V,TS=build_train(corp); print(f"train windows={len(X):,}",flush=True)
    m,cmu,csd=train(ARM,X,Y,V,TS,mu,sd,N)
    est=infer(m,cmu,csd,w,pre,mu,sd,N)
    tru=w['dur'].values
    for qi,q in enumerate(QS):
        p=os.path.join(OUT,f"{ARM.lower()}_q{int(round(q*100))}.npz")
        np.savez(p,est=est[:,qi])
        e=est[:,qi]; cov=(e>=tru).mean(); rat=np.median(e/np.maximum(tru,1))
        print(f"  q{int(round(q*100))}: median est={np.median(e):.0f}s coverage={cov:.3f} median est/dur={rat:.2f} -> {p}",flush=True)
    print("BACKFILL_MODEL_V2_DONE",flush=True)
