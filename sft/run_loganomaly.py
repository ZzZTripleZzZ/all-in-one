"""HPC log-anomaly detection (BGL / Thunderbird), the SOTA-bet security line.

Same generative next-event mechanism as our forecasting leg, applied to system-log EVENT-ID streams: train a
generative model on NORMAL log windows only, score a test window by how surprising its events are (mean
-log p, and the DeepLog top-g miss rate). SOTA to beat is the GPT-class self-supervised regime (LogGPT
BGL F1 0.958 / TB 0.986, arXiv:2309.14482), NOT the supervised upper bound.

Our differentiation: HSTU + TIME-AWARE attention over the real inter-arrival timestamps, which supercomputer
logs have in abundance (bursty, heavy-tailed) and which DeepLog/LogGPT ignore (position-only). The controlled
ablation HSTU-time vs HSTU-notime isolates that lever; HSTU vs TFM(GPT-proxy) vs LSTM(DeepLog) isolates
architecture. All baselines are RE-RUN under one protocol (paper numbers are not comparable -- same method
swings F1 0.43<->0.96 by protocol, ICSE'22 arXiv:2202.04301).

Protocol (reviewer-proof): CHRONOLOGICAL split (never random); non-overlapping fixed windows of W log lines;
window label = 1 if any line in it is an alert; train the generative models on NORMAL windows from the first
TRAIN_FRAC of the stream only; test on all windows in the last (1-TRAIN_FRAC). Report PR-AUC (threshold-free)
and best-F1 (precision/recall at the F1-optimal threshold), mean +/- std over seeds.

Input: ~/nfm/data/loghub/<DATASET>_events.npz (ts,eid,lab,vocab) from prep_loghub.py.
Env: DATASET(BGL) W(100) D(256) H(4) L(4) EP(10) NSEED(3) TRAIN_FRAC(0.8) TOPG(10) BS(256) MAXTRAIN(0=all).
"""
import os, sys, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, TFMv2, LSTMv2, CategoricalInput, CategoricalHead, train_v2, dev
from sklearn.metrics import average_precision_score, precision_recall_curve
from collections import Counter

DATASET=os.environ.get('DATASET','BGL'); W=int(os.environ.get('W',100))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
EP=int(os.environ.get('EP',10)); NSEED=int(os.environ.get('NSEED',3)); BS=int(os.environ.get('BS',256))
TRAIN_FRAC=float(os.environ.get('TRAIN_FRAC',0.8)); TOPG=int(os.environ.get('TOPG',10))
MAXTRAIN=int(os.environ.get('MAXTRAIN',0)); SMOKE=int(os.environ.get('SMOKE',0))
TRAIN_STRIDE=int(os.environ.get('TRAIN_STRIDE',W))    # <W => OVERLAPPING train windows (more data, less overfit); eval stays non-overlapping
if SMOKE: EP=2; NSEED=1
NPZ=os.path.expanduser(f"~/nfm/data/loghub/{DATASET}_events.npz")

def load_raw():
    z=np.load(NPZ); return z['eid'].astype(np.int64), z['ts'].astype(np.int64), z['lab'].astype(np.int8), int(z['vocab'])

def build_windows(eid, ts, lab, lo, hi, stride):
    """length-W windows in [lo,hi) stride apart; window label=1 iff it contains any alert line."""
    E,T,Y=[],[],[]
    for s in range(lo, hi-W+1, stride):
        E.append(eid[s:s+W]); T.append(ts[s:s+W]); Y.append(1 if lab[s:s+W].sum()>0 else 0)
    return np.asarray(E,np.int64), np.asarray(T,np.int64), np.asarray(Y,np.int8)

def metrics(score, y):
    y=np.asarray(y,int); score=np.asarray(score,float)
    if len(np.unique(y))<2: return dict(prauc=float('nan'),f1=float('nan'),P=float('nan'),R=float('nan'))
    prauc=average_precision_score(y,score)
    prec,rec,_=precision_recall_curve(y,score); f1=2*prec*rec/(prec+rec+1e-12)
    k=int(np.nanargmax(f1)); return dict(prauc=prauc,f1=float(f1[k]),P=float(prec[k]),R=float(rec[k]))

def make(kind,V):
    if kind=='HSTU-dt':         return HSTUv2(CategoricalInput(V,D),CategoricalHead(D,V),D=D,H=H,L=L,N=W,decoupled_time=True)
    if kind.startswith('HSTU'): return HSTUv2(CategoricalInput(V,D),CategoricalHead(D,V),D=D,H=H,L=L,N=W)
    if kind=='TFM(GPT)':        return TFMv2(CategoricalInput(V,D),CategoricalHead(D,V),D=D,H=H,L=L,N=W)
    if kind=='LSTM(DeepLog)':   return LSTMv2(CategoricalInput(V,D),CategoricalHead(D,V),D=D,H=H,L=L,N=W)

def train_model(kind,Etr,Ttr,V):
    use_time = kind in ('HSTU-time','HSTU-dt'); nw=len(Etr); to=lambda a: torch.tensor(a).to(dev)
    def batches():
        idx=np.random.permutation(nw)
        for i in range(0,nw,BS):
            j=idx[i:i+BS]; e=to(Etr[j]); t=to(Ttr[j]) if use_time else to(np.tile(np.arange(W),(len(j),1)))
            yield e,t,e,torch.ones_like(e,dtype=torch.float32)
    return train_v2(make(kind,V),batches,epochs=EP), use_time

@torch.no_grad()
def score(m,use_time,Ete,Tte):
    """window scores by MAX / top-5-mean surprise and top-g miss COUNT. A window is anomalous if it CONTAINS a
    surprising event (DeepLog 'any position anomalous'); mean-over-window dilutes a 1-in-100 alert to nothing."""
    nw=len(Ete); smax=np.empty(nw); stop=np.empty(nw); mcnt=np.empty(nw); to=lambda a: torch.tensor(a).to(dev)
    for i in range(0,nw,BS):
        sl=slice(i,i+BS); e=to(Ete[sl]); t=to(Tte[sl]) if use_time else to(np.tile(np.arange(W),(e.shape[0],1)))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            h=m(e,t)[:,:-1]
        logits=m.head.proj(h).float(); tgt=e[:,1:]
        lp=torch.log_softmax(logits,-1); s=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)     # [B,W-1]
        topg=logits.topk(min(TOPG,logits.shape[-1]),-1).indices
        m_=(tgt.unsqueeze(-1)!=topg).all(-1).float()
        smax[sl]=s.max(1).values.cpu().numpy()
        stop[sl]=s.topk(min(5,s.shape[1]),1).values.mean(1).cpu().numpy()
        mcnt[sl]=m_.sum(1).cpu().numpy()                                                    # count of top-g misses
    return smax,stop,mcnt

def count_scores(Etr,Ete,V):
    uni=Counter(); big=Counter(); prev=Counter()
    for w in Etr:
        uni.update(w.tolist())
        for a,b in zip(w[:-1],w[1:]): big[(int(a),int(b))]+=1; prev[int(a)]+=1
    tot=sum(uni.values())
    lu=lambda x: np.log((uni.get(int(x),0)+1)/(tot+V))
    def lb(a,b):
        c=big.get((int(a),int(b)),0); pc=prev.get(int(a),0); return np.log((c+1)/(pc+V)) if pc>0 else lu(b)
    fs=np.empty(len(Ete)); ms=np.empty(len(Ete))
    for i,w in enumerate(Ete):
        fs[i]=max(-lu(x) for x in w[1:]); ms[i]=max(-lb(w[k-1],w[k]) for k in range(1,len(w)))
    return fs,ms                                                                 # frequency, Markov (per-window MAX surprise)

if __name__=='__main__':
    print(f"device={dev} DATASET={DATASET} W={W} D={D} L={L} EP={EP} NSEED={NSEED} TRAIN_FRAC={TRAIN_FRAC} TOPG={TOPG} STRIDE={TRAIN_STRIDE}",flush=True)
    eid,ts,lab,V=load_raw(); n=len(eid); cut=int(n*TRAIN_FRAC)
    Etr_all,Ttr_all,Ytr_all=build_windows(eid,ts,lab,0,cut,TRAIN_STRIDE)        # OVERLAPPING train windows (more data)
    Ete,Tte,Yte=build_windows(eid,ts,lab,cut,n,W)                              # non-overlapping eval (no double-count)
    normal_tr=np.where(Ytr_all==0)[0]                                          # train on NORMAL windows only
    print(f"events={n:,} vocab={V} | train-win={len(Etr_all):,} (normal={len(normal_tr):,}, stride={TRAIN_STRIDE}) "
          f"eval-win={len(Ete):,} eval-anomaly-rate={Yte.mean():.3f}",flush=True)

    METH=['HSTU-time','HSTU-dt','HSTU-notime','TFM(GPT)','LSTM(DeepLog)']     # HSTU-dt = decoupled-time-channel improvement
    _sel=os.environ.get('METHODS','')
    if _sel: METH=[m for m in METH if m in _sel.split(',')]                   # subset filter (e.g. METHODS=HSTU-dt)
    acc={m:{'max':[], 'top5':[], 'cnt':[]} for m in METH}; acc['frequency']={'max':[]}; acc['Markov']={'max':[]}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        idx=normal_tr.copy(); np.random.shuffle(idx)
        if MAXTRAIN>0: idx=idx[:MAXTRAIN]
        Etr,Ttr=Etr_all[idx],Ttr_all[idx]
        for kind in METH:
            m,ut=train_model(kind,Etr,Ttr,V); sm,st,mc=score(m,ut,Ete,Tte)
            acc[kind]['max'].append(metrics(sm,Yte)); acc[kind]['top5'].append(metrics(st,Yte)); acc[kind]['cnt'].append(metrics(mc,Yte))
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {kind:<14} PRAUC(max)={acc[kind]['max'][-1]['prauc']:.3f} F1(max)={acc[kind]['max'][-1]['f1']:.3f} | top5={acc[kind]['top5'][-1]['prauc']:.3f} cnt={acc[kind]['cnt'][-1]['prauc']:.3f}",flush=True)
        fs,mk=count_scores(Etr,Ete,V)
        acc['frequency']['max'].append(metrics(fs,Yte)); acc['Markov']['max'].append(metrics(mk,Yte))
        print(f"  [seed {seed}] frequency PRAUC(max)={acc['frequency']['max'][-1]['prauc']:.3f}  Markov PRAUC(max)={acc['Markov']['max'][-1]['prauc']:.3f}",flush=True)

    print(f"\n==== {DATASET} log-anomaly, train-on-normal, mean +/- std over {NSEED} seed(s) ====",flush=True)
    print(f"SOTA to match/beat (LogGPT, same regime): BGL F1 0.958 / Thunderbird F1 0.986. Metrics: PR-AUC | best-F1 | P | R.")
    def rep(mth,sck,tag):
        A=acc[mth][sck]
        g=lambda k: (np.nanmean([a[k] for a in A]),np.nanstd([a[k] for a in A]))
        (pa,ps),(fa,fs2),(Pa,Ps),(Ra,Rs)=g('prauc'),g('f1'),g('P'),g('R')
        print(f"  {mth+' '+tag:<24} {pa:.3f}+-{ps:<4.3f}  {fa:.3f}+-{fs2:<4.3f}  {Pa:.3f}  {Ra:.3f}")
    print(f"  {'method':<24} {'PR-AUC':<11} {'best-F1':<11} {'P':<6} {'R':<6}")
    for m in METH: rep(m,'max','[max]')             # MAX surprise = DeepLog-faithful window score (headline)
    for m in METH: rep(m,'top5','[top5]')           # top-5-mean surprise (robust variant)
    for m in METH: rep(m,'cnt','[cnt]')             # top-g miss COUNT
    for m in ['frequency','Markov']: rep(m,'max','[max]')
    print("\nKEY ablation: HSTU-time vs HSTU-notime = value of real inter-arrival timing (our differentiation);")
    print("HSTU vs TFM(GPT) vs LSTM(DeepLog) = architecture at matched protocol/epochs.")
    print("LOGANOMALY_DONE",flush=True)
