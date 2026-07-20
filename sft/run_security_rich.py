"""RICH-event LANL security: unsupervised compromised-user detection by generative next-event surprise.

Entity = user. Event = the FULL authentication tuple (dst-computer, src-computer, auth-type, logon-type, success).
A generative multi-field model learns P(next event | history) on BENIGN users only; each user is scored by the
model's JOINT surprise -log P(actual next event) over their stream. Red-team lateral movement authenticates to
unusual destinations from unusual sources with unusual (auth-type, logon-type, outcome) combinations, so those
events are less predictable and score higher.

WHAT THIS EXPERIMENT IS (and is NOT). It is NOT a bid for lateral-movement SOTA -- specialized authentication-
graph models (Argus S&P'24 AP 0.23; CyberGFM'26 AP 0.76) win that. It demonstrates two things: (1) CROSS-DOMAIN
GENERALITY -- the SAME generative telemetry model used for HPC forecasting, with no security-specific engineering,
gives a competitive detector; (2) a CONTROLLED representation-richness ablation nobody has published on LANL:
HSTU-rich vs HSTU-dstonly (same backbone, same epochs, richer event) isolates whether enriching the event -- not
the sequence model -- is the lever the gate identified.

METRICS. AUC saturates near 1.0 on LANL for every method family, so it is reported only as context. The HEADLINE
metrics are AP (AUPRC), recall@alert-budget (R@50 / R@100), and detection@low-FPR (TPR@1% / TPR@5% FPR) -- the
precision-oriented metrics the LANL literature (Argus'24, Larroche'25, CyberGFM'26) shows actually separate methods.

BASELINES. Sequence: LSTM-rich, Markov(bigram-dst), frequency(field unigrams), all epoch-matched, at rich AND
dst-only representations. Cheap-but-strong: new-edge heuristic (fraction of previously-unseen (src,dst) edges --
the direct signature of lateral movement, Turcotte/Heard line). Non-sequence: Isolation Forest on per-user
aggregated features (Tuor's own baseline). Auth-package filtering: NONE excluded (all auth types kept and
normalized) -- reported explicitly because selective exclusion inflates measured performance (Larroche'25).

Protocol: train on a fraction of benign users only; eval = held-out benign + ALL compromised users, so no
compromised user is ever seen in training (one-class novelty). Multi-seed mean +/- std.
Input: ~/nfm/data/lanl/lanl_rich_s*.csv (cols time,user,dst,src,auth,logon,success,label) via LANL_CSV.
Env: DSTV(2000) SRCV(2000) N(64) D(256) H(4) L(4) EP(10) NSEED(3) TRAIN_FRAC(0.5) BS(256).
"""
import os, sys, math, numpy as np, pandas as pd, torch
from collections import Counter, defaultdict
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import (HSTUv2, LSTMv2, MultiFieldInput, CategoricalInput, MultiFieldHead, CategoricalHead,
                    train_v2, dev)
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.ensemble import IsolationForest

DSTV=int(os.environ.get('DSTV',2000)); SRCV=int(os.environ.get('SRCV',2000)); N=int(os.environ.get('N',64))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
EP=int(os.environ.get('EP',10)); NSEED=int(os.environ.get('NSEED',3)); BS=int(os.environ.get('BS',256))
TRAIN_FRAC=float(os.environ.get('TRAIN_FRAC',0.5)); SMOKE=int(os.environ.get('SMOKE',0))
NUSERS=int(os.environ.get('NUSERS',0))          # cap users (keeps ALL compromised) for fast validation smokes
CSV=os.path.expanduser(os.environ.get('LANL_CSV',"~/nfm/data/lanl/lanl_rich_s5.csv"))
if SMOKE: EP=2; NSEED=1; DSTV=500; SRCV=500

# --------------------------------------------------------------------------- data
def load():
    d=pd.read_csv(CSV,header=None,names=['t','user','dst','src','auth','logon','succ','lab'],
                  dtype={'dst':str,'src':str,'auth':str,'logon':str}).sort_values(['user','t'])
    U={}
    for u,g in d.groupby('user',sort=False):
        if len(g)<8: continue
        U[u]=dict(dst=g['dst'].values, src=g['src'].values, auth=g['auth'].values, logon=g['logon'].values,
                  succ=g['succ'].values.astype(np.int64), t=g['t'].values.astype(np.int64), lab=int(g['lab'].max()))
    return U

def build_vocabs(U, tr):
    cd=Counter(); cs=Counter(); ca=Counter(); cl=Counter()
    for u in tr:
        cd.update(U[u]['dst']); cs.update(U[u]['src']); ca.update(U[u]['auth']); cl.update(U[u]['logon'])
    dst={k:i for i,(k,_) in enumerate(cd.most_common(DSTV))}
    src={k:i for i,(k,_) in enumerate(cs.most_common(SRCV))}
    auth={k:i for i,(k,_) in enumerate(ca.most_common())}
    logon={k:i for i,(k,_) in enumerate(cl.most_common())}
    sizes=dict(dst=len(dst)+1,src=len(src)+1,auth=len(auth)+1,logon=len(logon)+1,succ=2)   # +1 OOV/pad bucket
    return dict(dst=dst,src=src,auth=auth,logon=logon), sizes

FIELDS=['dst','src','auth','logon']                                            # succ is already 0/1
def enc_user(U, u, voc, sizes):
    d=U[u]; oov={f:sizes[f]-1 for f in FIELDS}
    di=np.array([voc['dst'].get(x,oov['dst']) for x in d['dst']],np.int64)
    si=np.array([voc['src'].get(x,oov['src']) for x in d['src']],np.int64)
    ai=np.array([voc['auth'].get(x,oov['auth']) for x in d['auth']],np.int64)
    li=np.array([voc['logon'].get(x,oov['logon']) for x in d['logon']],np.int64)
    return di,si,ai,li,d['succ'],d['t']

def windows(U, users, voc, sizes):
    cols={k:[] for k in ['dst','src','auth','logon','succ','ts','val']}; UID=[]
    pad={f:sizes[f] for f in FIELDS}
    for u in users:
        di,si,ai,li,su,ts=enc_user(U,u,voc,sizes)
        for s in range(0,len(di),N):
            k=len(di[s:s+N]); p=N-k
            cols['dst'].append(np.concatenate([di[s:s+N],np.full(p,pad['dst'])]))
            cols['src'].append(np.concatenate([si[s:s+N],np.full(p,pad['src'])]))
            cols['auth'].append(np.concatenate([ai[s:s+N],np.full(p,pad['auth'])]))
            cols['logon'].append(np.concatenate([li[s:s+N],np.full(p,pad['logon'])]))
            cols['succ'].append(np.concatenate([su[s:s+N],np.full(p,0)]))
            cols['ts'].append(np.concatenate([ts[s:s+N],np.full(p,ts[s:s+N][-1] if k else 0)]))
            cols['val'].append(np.concatenate([np.ones(k),np.zeros(p)])); UID.append(u)
    A=lambda k,dt: np.array(cols[k],dt)
    return dict(dst=A('dst',np.int64),src=A('src',np.int64),auth=A('auth',np.int64),logon=A('logon',np.int64),
                succ=A('succ',np.int64),ts=A('ts',np.int64),val=A('val',np.float32),uid=np.array(UID))

# --------------------------------------------------------------------------- generative models (rich / dstonly)
def make_model(kind, sizes):
    vl=[sizes['dst'],sizes['src'],sizes['auth'],sizes['logon'],sizes['succ']]
    if kind=='rich':      return HSTUv2(MultiFieldInput(vl,D),MultiFieldHead(D,vl),D=D,H=H,L=L,N=N)
    if kind=='rich-lstm': return LSTMv2(MultiFieldInput(vl,D),MultiFieldHead(D,vl),D=D,H=H,L=L,N=N)
    if kind=='dstonly':   return HSTUv2(CategoricalInput(sizes['dst'],D),CategoricalHead(D,sizes['dst']),D=D,H=H,L=L,N=N)

def _rich_inp(W, sl, to): return (to(W['dst'][sl]),to(W['src'][sl]),to(W['auth'][sl]),to(W['logon'][sl]),to(W['succ'][sl]))
def train_neural(kind, W, sizes):
    nw=len(W['dst']); rich=kind!='dstonly'; to=lambda a: torch.tensor(a).to(dev)
    def batches():
        idx=np.random.permutation(nw)
        for i in range(0,nw,BS):
            j=idx[i:i+BS]; ts=to(W['ts'][j]); val=to(W['val'][j])
            inp=_rich_inp(W,j,to) if rich else to(W['dst'][j])
            yield inp,ts,inp,val
    m=train_v2(make_model(kind,sizes),batches,epochs=EP); m.eval(); return m,rich

@torch.no_grad()
def neural_event_surprise(m, rich, W):
    nw=len(W['dst']); per=defaultdict(list); to=lambda a: torch.tensor(a).to(dev)
    for i in range(0,nw,BS):
        sl=slice(i,i+BS); ts=to(W['ts'][sl])
        inp=_rich_inp(W,sl,to) if rich else to(W['dst'][sl])
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            h=m(inp,ts)[:,:-1]
        tgt=tuple(t[:,1:] for t in inp) if rich else inp[:,1:]
        s=m.head.surprise(h.float(),tgt).float().cpu().numpy(); v=W['val'][sl][:,1:]; uid=W['uid'][sl]
        for r in range(s.shape[0]):
            mk=v[r]>0
            if mk.any(): per[uid[r]].extend(s[r][mk].tolist())
    return per

# --------------------------------------------------------------------------- count baselines (rich / dstonly)
def count_scores(U, tr, ev, voc, sizes):
    uni=dict(dst=Counter(),auth=Counter(),logon=Counter(),succ=Counter()); big=Counter(); prev=Counter()
    for u in tr:
        di,si,ai,li,su,_=enc_user(U,u,voc,sizes)
        uni['dst'].update(di.tolist()); uni['auth'].update(ai.tolist()); uni['logon'].update(li.tolist()); uni['succ'].update(su.tolist())
        for a,b in zip(di[:-1],di[1:]): big[(int(a),int(b))]+=1; prev[int(a)]+=1
    Vd,Va,Vl=sizes['dst'],sizes['auth'],sizes['logon']
    lu=lambda f,x,V: math.log((uni[f].get(int(x),0)+1)/(sum(uni[f].values())+V))
    def lb(a,b):
        c=big.get((int(a),int(b)),0); pc=prev.get(int(a),0)
        return math.log((c+1)/(pc+Vd)) if pc>0 else lu('dst',b,Vd)
    pf_r=defaultdict(list); pm_r=defaultdict(list); pf_d=defaultdict(list); pm_d=defaultdict(list)
    for u in ev:
        di,si,ai,li,su,_=enc_user(U,u,voc,sizes)
        for k in range(1,len(di)):
            base=lu('auth',ai[k],Va)+lu('logon',li[k],Vl)+lu('succ',su[k],2)
            pf_r[u].append(-(lu('dst',di[k],Vd)+base)); pm_r[u].append(-(lb(di[k-1],di[k])+base))
            pf_d[u].append(-lu('dst',di[k],Vd));        pm_d[u].append(-lb(di[k-1],di[k]))
    return {'freq-rich':pf_r,'Markov-rich':pm_r,'freq-dstonly':pf_d,'Markov-dstonly':pm_d}

# --------------------------------------------------------------------------- cheap/non-sequence baselines
def new_edge_scores(U, tr, ev):
    """per-user fraction of (src,dst) edges never seen in the benign-train corpus (Turcotte/Heard new-edge)."""
    E=set()
    for u in tr: E.update(zip(U[u]['src'],U[u]['dst']))
    sc={}
    for u in ev:
        e=list(zip(U[u]['src'],U[u]['dst'])); sc[u]=float(np.mean([1.0 if x not in E else 0.0 for x in e])) if e else 0.0
    return sc

def user_features(U, u):
    d=U[u]; n=len(d['dst']); edges=list(zip(d['src'],d['dst']))
    def ent(xs):
        c=Counter(xs); tot=sum(c.values()); return -sum((v/tot)*math.log(v/tot) for v in c.values()) if tot else 0.0
    fr=lambda arr,val: float(np.mean(arr==val)) if n else 0.0
    return np.array([n, len(set(d['dst'])), len(set(d['src'])), len(set(edges)), float(np.mean(d['succ']==0)),
                     fr(d['auth'],'NTLM'), fr(d['auth'],'Kerberos'), fr(d['auth'],'Negotiate'),
                     fr(d['logon'],'Network'), float(np.mean(np.isin(d['logon'],['RemoteInteractive','NetworkCleartext']))),
                     ent(d['dst']), ent(edges), len(set(edges))/max(n,1)],dtype=np.float64)

def iso_forest_scores(U, tr, ev, seed):
    Xtr=np.stack([user_features(U,u) for u in tr]); Xev=np.stack([user_features(U,u) for u in ev])
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9; Xtr=(Xtr-mu)/sd; Xev=(Xev-mu)/sd
    clf=IsolationForest(n_estimators=300,random_state=seed).fit(Xtr)
    s=-clf.score_samples(Xev)                                                   # higher = more anomalous
    return {u:float(s[i]) for i,u in enumerate(ev)}

# --------------------------------------------------------------------------- metrics (AP headline; AUC context)
def metrics(s, y):
    s=np.asarray(s,float); y=np.asarray(y,int)
    if len(np.unique(y))<2: return [float('nan')]*6
    ap=average_precision_score(y,s); auc=roc_auc_score(y,s)
    order=np.argsort(-s); P=int(y.sum())
    r50=y[order[:50]].sum()/max(P,1); r100=y[order[:100]].sum()/max(P,1)
    fpr,tpr,_=roc_curve(y,s); d1=float(tpr[fpr<=0.01].max()) if (fpr<=0.01).any() else 0.0
    d5=float(tpr[fpr<=0.05].max()) if (fpr<=0.05).any() else 0.0
    return [ap,r50,r100,d1,d5,auc]                                              # AP, R@50, R@100, det@1%, det@5%, AUC

def agg_scores(per, users, lab, agg):
    y=[lab[u] for u in users]
    if agg=='mean': s=[float(np.mean(per.get(u,[0.0]))) for u in users]
    elif agg=='max': s=[float(np.max(per.get(u,[0.0]))) for u in users]
    else:            s=[float(np.sort(per.get(u,[0.0]))[-5:].mean()) for u in users]   # top5
    return metrics(s,y)

# --------------------------------------------------------------------------- main
SURPRISE=['HSTU-rich','LSTM-rich','freq-rich','Markov-rich','HSTU-dstonly','freq-dstonly','Markov-dstonly']
SINGLE=['new-edge','IsoForest']
MNAMES=['AP','R@50','R@100','det@1%','det@5%','AUC']
if __name__=='__main__':
    print(f"device={dev} DSTV={DSTV} SRCV={SRCV} N={N} D={D} H={H} L={L} EP={EP} NSEED={NSEED} TRAIN_FRAC={TRAIN_FRAC} CSV={os.path.basename(CSV)}",flush=True)
    U=load(); users=list(U); lab={u:U[u]['lab'] for u in users}
    if NUSERS>0:                                                              # keep ALL compromised + a benign subsample
        comp=[u for u in users if lab[u]==1]; ben=[u for u in users if lab[u]==0]
        np.random.default_rng(0).shuffle(ben); keep=set(comp+ben[:max(NUSERS-len(comp),0)])
        U={u:v for u,v in U.items() if u in keep}; users=list(U); lab={u:U[u]['lab'] for u in users}
    nben=sum(1 for u in users if lab[u]==0); ncomp=len(users)-nben
    print(f"users={len(users)} benign={nben} compromised={ncomp} events={sum(len(U[u]['dst']) for u in users):,}",flush=True)
    if ncomp<5: print("too few compromised -- check prep"); sys.exit(1)

    acc=defaultdict(lambda: defaultdict(list))                                  # method -> agg -> [metric-vec per seed]
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        ben=[u for u in users if lab[u]==0]; comp=[u for u in users if lab[u]==1]
        rng=np.random.default_rng(seed); rng.shuffle(ben)
        ntr=int(len(ben)*TRAIN_FRAC); tr=ben[:ntr]; ev=ben[ntr:]+comp
        voc,sizes=build_vocabs(U,tr); Wtr=windows(U,tr,voc,sizes); Wev=windows(U,ev,voc,sizes)
        prev=np.mean([lab[u] for u in ev])
        print(f"[seed {seed}] train-users={len(tr)} eval-users={len(ev)} comp={sum(lab[u] for u in ev)} prev={prev:.3f} "
              f"vocab dst={sizes['dst']} src={sizes['src']} auth={sizes['auth']} logon={sizes['logon']} train-win={len(Wtr['dst'])}",flush=True)
        # generative
        for kind,mth in [('rich','HSTU-rich'),('rich-lstm','LSTM-rich'),('dstonly','HSTU-dstonly')]:
            m,rich=train_neural(kind,Wtr,sizes); per=neural_event_surprise(m,rich,Wev)
            for agg in ['mean','max','top5']: acc[mth][agg].append(agg_scores(per,ev,lab,agg))
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {mth} AP(top5)={acc[mth]['top5'][-1][0]:.3f} AUC={acc[mth]['top5'][-1][5]:.3f}",flush=True)
        # count baselines
        for mth,per in count_scores(U,tr,ev,voc,sizes).items():
            for agg in ['mean','max','top5']: acc[mth][agg].append(agg_scores(per,ev,lab,agg))
        # cheap / non-sequence (single score per user)
        y=[lab[u] for u in ev]
        ne=new_edge_scores(U,tr,ev); acc['new-edge']['native'].append(metrics([ne[u] for u in ev],y))
        iso=iso_forest_scores(U,tr,ev,seed); acc['IsoForest']['native'].append(metrics([iso[u] for u in ev],y))
        print(f"  [seed {seed}] new-edge AP={acc['new-edge']['native'][-1][0]:.3f}  IsoForest AP={acc['IsoForest']['native'][-1][0]:.3f}",flush=True)

    # ---- headline: surprise methods at top5 agg, single methods native. AP is the headline; AUC last (context).
    def line(mth,agg):
        A=np.array(acc[mth][agg]); mu=np.nanmean(A,0); sd=np.nanstd(A,0)
        cells=" ".join(f"{mu[i]:>5.3f}+-{sd[i]:<4.3f}" for i in range(6))
        print(f"  {mth:<15} {cells}")
    print(f"\n==== RICH-EVENT LANL security, mean +/- std over {NSEED} seed(s) ====",flush=True)
    print("HEADLINE metrics (AP first, AUC last = context only; AUC saturates on LANL). Surprise methods @ top5-agg.")
    print(f"  {'method':<15} "+" ".join(f"{n:>10}" for n in MNAMES))
    for mth in SURPRISE: line(mth,'top5')
    for mth in SINGLE:   line(mth,'native')
    print("\n-- aggregation sensitivity (HSTU-rich): mean / max / top5 --")
    for agg in ['mean','max','top5']: line('HSTU-rich',agg)
    print("\nKEY: HSTU-rich vs HSTU-dstonly = event-representation lever (controlled, novel on LANL);")
    print("HSTU-rich vs freq-rich/Markov-rich = sequence-model lever; vs new-edge/IsoForest = cheap/non-seq baselines.")
    print("SECURITY_RICH_DONE",flush=True)
