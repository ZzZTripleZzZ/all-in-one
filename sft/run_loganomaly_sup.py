"""SUPERVISED past-label log-anomaly on BGL: predict per-event anomaly y_t from events_{<=t} + labels_{<t}
(recsys-style event+outcome interleaving = HSTU's item+action mechanism, arXiv:2402.17152). This is the user's
original "多变量 + 把过去的label加进seq" idea, done NON-LEAKY per the 2026-07-16 design (see memory
project_nfm_pastlabel_design). BGL has per-LINE alert labels, so a per-event target y_t genuinely exists.

Input at position t = (EventId_t, prev_label) with prev_label = y_{t-1} (pos 0 = <START>=2); target = y_t.
Strictly causal, so position t sees e_{<=t}, y_{<t}, NEVER y_t. We report BOTH regimes and the controls that
decide whether the label channel is real signal or leakage:
  A) teacher-forced: feed TRUE y_{t-1}  -> upper bound (valid only under an analyst/revealed-label assumption)
  B) free-running:   feed the model's OWN yhat_{t-1}  -> pure detection = THE HEADLINE
  persistence: label-only Markov P(y_t | y_{t-1}), NO events = decisive control (beat it big or logs add nothing)
  zero-label: label channel forced to <START> everywhere = supervised-without-past-labels control
  event-shuffle self-test: permute the EventId channel, keep labels -> F1 must COLLAPSE toward persistence
Chronological split; per-event Precision/Recall/F1 + PR-AUC (never accuracy; ~7% positives).
Env: DATASET(BGL) W(120) D(256) H(4) L(4) EP(10) NSEED(3) TRAIN_FRAC(0.8) BS(256) TRAIN_STRIDE(W) EVALWIN(1500)
     METHODS(HSTU-dt,LSTM(DeepLog),TFM(GPT)).
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, LSTMv2, TFMv2, MultiFieldInput, CategoricalHead, dev
from sklearn.metrics import average_precision_score, precision_recall_curve

DATASET=os.environ.get('DATASET','BGL'); W=int(os.environ.get('W',120))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
EP=int(os.environ.get('EP',10)); NSEED=int(os.environ.get('NSEED',3)); BS=int(os.environ.get('BS',256))
TRAIN_FRAC=float(os.environ.get('TRAIN_FRAC',0.8)); TRAIN_STRIDE=int(os.environ.get('TRAIN_STRIDE',W))
EVALWIN=int(os.environ.get('EVALWIN',1500)); SMOKE=int(os.environ.get('SMOKE',0))
if SMOKE: EP=2; NSEED=1; EVALWIN=200
NPZ=os.path.expanduser(f"~/nfm/data/loghub/{DATASET}_events.npz")
METH=os.environ.get('METHODS','HSTU-dt,LSTM(DeepLog),TFM(GPT)').split(',')

def load_raw():
    z=np.load(NPZ); return z['eid'].astype(np.int64), z['ts'].astype(np.int64), z['lab'].astype(np.int64), int(z['vocab'])
def build(eid,ts,lab,lo,hi,stride):
    E,Y,T=[],[],[]
    for s in range(lo,hi-W+1,stride): E.append(eid[s:s+W]); Y.append(lab[s:s+W]); T.append(ts[s:s+W])
    return np.asarray(E,np.int64),np.asarray(Y,np.int64),np.asarray(T,np.int64)

def make(kind,V):
    inp=MultiFieldInput([V,3],D); head=CategoricalHead(D,2)                     # fields: EventId(V) + prev-label(3: 0/1/START=2)
    if kind=='HSTU-dt':         return HSTUv2(inp,head,D=D,H=H,L=L,N=W,decoupled_time=True)
    if kind.startswith('HSTU'): return HSTUv2(inp,head,D=D,H=H,L=L,N=W)
    if kind=='TFM(GPT)':        return TFMv2(inp,head,D=D,H=H,L=L,N=W)
    if kind=='LSTM(DeepLog)':   return LSTMv2(inp,head,D=D,H=H,L=L,N=W)

def prevlab(y):                                                                # [B,W] -> y_{t-1}, pos0=<START>=2 (NO leakage)
    p=torch.full_like(y,2); p[:,1:]=y[:,:-1]; return p
def uses_time(kind): return kind in ('HSTU-time','HSTU-dt')

def train_sup(kind,E,Y,T,V,zero_label=False):
    m=make(kind,V).to(dev); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],1e-3,weight_decay=0.01)
    ut=uses_time(kind); nw=len(E); to=lambda a: torch.tensor(a).to(dev)
    for ep in range(EP):
        m.train()
        for i in np.array_split(np.random.permutation(nw),max(1,nw//BS)):
            e=to(E[i]); y=to(Y[i]); t=to(T[i]) if ut else to(np.tile(np.arange(W),(len(i),1)))
            pl=torch.full_like(y,2) if zero_label else prevlab(y)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                h=m((e,pl),t)
            logits=m.head.proj(h).float()                                      # [B,W,2] per-position P(y_t)
            loss=F.cross_entropy(logits.reshape(-1,2), y.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m,ut

@torch.no_grad()
def eval_tf(m,ut,E,Y,T,zero_label=False,shuffle_events=False):                 # regime A / controls (teacher-forced)
    sc,tr=[],[]; nw=len(E); to=lambda a: torch.tensor(a).to(dev)
    for i in range(0,nw,BS):
        e=E[i:i+BS].copy()
        if shuffle_events:
            for r in range(len(e)): e[r]=e[r][np.random.permutation(e.shape[1])]   # break event->label link, keep labels
        e=to(e); y=to(Y[i:i+BS]); t=to(T[i:i+BS]) if ut else to(np.tile(np.arange(W),(y.shape[0],1)))
        pl=torch.full_like(y,2) if zero_label else prevlab(y)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            p=torch.softmax(m.head.proj(m((e,pl),t)).float(),-1)[...,1]         # [B,W] P(anom)
        sc.append(p.reshape(-1).cpu().numpy()); tr.append(Y[i:i+BS].reshape(-1))
    return np.concatenate(sc), np.concatenate(tr)

@torch.no_grad()
def eval_fr(m,ut,E,Y,T):                                                       # regime B: FREE-RUNNING, feed model's own yhat
    sc,tr=[],[]; nw=len(E); to=lambda a: torch.tensor(a).to(dev)
    for i in range(0,nw,BS):
        e=to(E[i:i+BS]); y=to(Y[i:i+BS]); B_=e.shape[0]
        t=to(T[i:i+BS]) if ut else to(np.tile(np.arange(W),(B_,1)))
        pl=torch.full((B_,W),2,device=dev,dtype=torch.long); out=torch.zeros((B_,W),device=dev)
        for pos in range(W):                                                   # causal: hidden@pos uses pl[<=pos] only
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                p=torch.softmax(m.head.proj(m((e,pl),t)[:,pos]).float(),-1)[:,1]
            out[:,pos]=p
            if pos+1<W: pl[:,pos+1]=(p>0.5).long()                             # feed OWN prediction as next prev-label
        sc.append(out.reshape(-1).cpu().numpy()); tr.append(Y[i:i+BS].reshape(-1))
    return np.concatenate(sc), np.concatenate(tr)

def persistence(Etr,Ytr,Ete,Yte):                                             # label-only Markov P(y_t|y_{t-1}), NO events
    a=np.zeros((3,2))                                                          # prev in {0,1,2} -> count of y_t in {0,1}
    for y in Ytr:
        pv=2
        for v in y: a[pv,v]+=1; pv=v
    pr=(a[:,1]+1)/(a.sum(1)+2)                                                  # P(y_t=1 | prev)
    sc=[]
    for y in Yte:
        pv=2
        for v in y: sc.append(pr[pv]); pv=v
    return np.array(sc), Yte.reshape(-1)

def M(score,y):
    y=np.asarray(y,int); score=np.asarray(score,float); ok=np.isfinite(score); y=y[ok]; score=score[ok]
    if len(np.unique(y))<2: return dict(prauc=float('nan'),f1=float('nan'),P=float('nan'),R=float('nan'))
    pa=average_precision_score(y,score); pr,rc,_=precision_recall_curve(y,score); f1=2*pr*rc/(pr+rc+1e-12)
    k=int(np.nanargmax(f1)); return dict(prauc=pa,f1=float(f1[k]),P=float(pr[k]),R=float(rc[k]))

if __name__=='__main__':
    print(f"device={dev} DATASET={DATASET} W={W} D={D} L={L} EP={EP} NSEED={NSEED} STRIDE={TRAIN_STRIDE} EVALWIN={EVALWIN} METH={METH}",flush=True)
    eid,ts,lab,V=load_raw(); n=len(eid); cut=int(n*TRAIN_FRAC)
    Etr,Ytr,Ttr=build(eid,ts,lab,0,cut,TRAIN_STRIDE); Ete,Yte,Tte=build(eid,ts,lab,cut,n,W)
    if len(Ete)>EVALWIN:
        k=np.random.default_rng(0).choice(len(Ete),EVALWIN,replace=False); Ete,Yte,Tte=Ete[k],Yte[k],Tte[k]
    print(f"events={n:,} vocab={V} | train-win={len(Etr):,} eval-win={len(Ete):,} "
          f"train-event-anom={Ytr.mean():.3f} eval-event-anom={Yte.mean():.3f}",flush=True)
    # non-negotiable causal self-check: prev-label input never equals current target
    _p=prevlab(torch.tensor(Yte[:64])); assert (_p[:,0]==2).all() and (_p[:,1:]==torch.tensor(Yte[:64])[:,:-1]).all(), "LEAK: prevlab wrong"
    print("causal-mask check: prev-label = y_{t-1}, pos0=START, no y_t in input. OK",flush=True)

    rows={}                                                                    # (method, regime) -> [metric dict per seed]
    def add(key,d): rows.setdefault(key,[]).append(d)
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        add(('persistence','label-only'), M(*persistence(Etr,Ytr,Ete,Yte)))
        for kind in METH:
            m,ut=train_sup(kind,Etr,Ytr,Ttr,V)
            add((kind,'A:teacher-forced'), M(*eval_tf(m,ut,Ete,Yte,Tte)))
            add((kind,'B:free-running'),   M(*eval_fr(m,ut,Ete,Yte,Tte)))
            add((kind,'shuffle-events'),   M(*eval_tf(m,ut,Ete,Yte,Tte,shuffle_events=True)))
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            mz,_=train_sup(kind,Etr,Ytr,Ttr,V,zero_label=True)                 # supervised WITHOUT past labels
            add((kind,'zero-label'), M(*eval_tf(mz,ut,Ete,Yte,Tte,zero_label=True)))
            del mz; torch.cuda.empty_cache() if dev=='cuda' else None
            r=rows[(kind,'B:free-running')][-1]; print(f"  [seed {seed}] {kind:<14} B-free F1={r['f1']:.3f} PRAUC={r['prauc']:.3f}",flush=True)

    print(f"\n==== {DATASET} SUPERVISED past-label, per-event, chronological, mean+/-std over {NSEED} seed(s) ====",flush=True)
    print("HEADLINE = B:free-running (pure detection). A = feedback upper bound. Beat persistence + shuffle must collapse.")
    print(f"  {'method / regime':<32} {'PR-AUC':<13} {'best-F1':<13} {'P':<6} {'R':<6}")
    order=[('persistence','label-only')]+[(k,r) for k in METH for r in ('zero-label','B:free-running','A:teacher-forced','shuffle-events')]
    for key in order:
        if key not in rows: continue
        A=rows[key]; g=lambda f:(np.nanmean([a[f] for a in A]),np.nanstd([a[f] for a in A]))
        (pa,ps),(fa,fs),(Pa,_),(Ra,_)=g('prauc'),g('f1'),g('P'),g('R')
        print(f"  {key[0]+' ['+key[1]+']':<32} {pa:.3f}+-{ps:<5.3f} {fa:.3f}+-{fs:<5.3f} {Pa:.3f}  {Ra:.3f}")
    print("\nREAD: B:free-running >> persistence AND >> zero-label AND shuffle-events collapses  => label+event both real, non-leaky.")
    print("If B ~= persistence or shuffle-events stays high => it's just label autocorrelation (leaky). ")
    print("SUP_DONE",flush=True)
