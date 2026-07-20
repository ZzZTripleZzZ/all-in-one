"""HPC/enterprise-security line: UNSUPERVISED compromised-user detection on LANL auth streams.
Entity = source user; token = DESTINATION COMPUTER (recsys "item"); timestamp = auth time (irregular).
A next-destination density model is trained on BENIGN users only; each user is scored by the model's
SURPRISE (-log P(actual next dest)) aggregated per user. Compromised users (red-team lateral movement
to unusual destinations) score higher. Per-user labels => tractable base rate.

Rebuilt 2026-07-16 after the methodology review (top-venue rigor):
 - AUC is the HEADLINE (prevalence-independent); AUPRC reported WITH its eval prevalence; plus precision@k
   (operationally meaningful for rare-event detection). (was: AUPRC headline at a subsampling-inflated rate)
 - Baselines are EPOCH-MATCHED next-token-surprise models — HSTU vs a plain LSTM vs Markov(bigram) vs
   frequency(unigram) — so "the time-aware FM helps" is a fair architecture comparison. (was: a misleading
   under-trained 1-epoch HSTU "from-scratch").
 - Multiple per-user AGGREGATIONS reported (max / mean / top-k) to show the result is not an artifact of max.
 - Symmetric event filtering upstream (prep_lanl.sh) removes the benign-vs-compromised preprocessing confound.
 - Multi-seed mean±std. Split: train on part of benign only (learn "normal"), eval = held-out benign + ALL
   compromised, so no compromised user is ever seen in training (clean one-class novelty setup).
Input: ~/nfm/data/lanl/lanl_events*.csv (via LANL_CSV). Env: DVOCAB,N,D,H,L,PRE_EP,NSEED,TRAIN_FRAC(0.5).
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from collections import Counter
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from nfm_core import HSTU, LSTMGen, dev, train_gen
from sklearn.metrics import roc_auc_score, average_precision_score

KV=int(os.environ.get('DVOCAB',2048)); N=int(os.environ.get('N',64))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
PRE_EP=int(os.environ.get('PRE_EP',10)); NSEED=int(os.environ.get('NSEED',3)); PAD=KV
TRAIN_FRAC=float(os.environ.get('TRAIN_FRAC',0.5))   # fraction of benign users used to learn "normal"; rest -> eval
CSV=os.path.expanduser(os.environ.get('LANL_CSV',"~/nfm/data/lanl/lanl_events.csv"))

def load_raw():
    d=pd.read_csv(CSV,header=None,names=['t','user','src','dst','lab']).sort_values(['user','t'])
    Dst,Ts,Comp=[],[],[]
    for u,g in d.groupby('user',sort=False):
        if len(g)<8: continue
        Dst.append(g['dst'].values); Ts.append(g['t'].values.astype(np.int64)); Comp.append(int(g['lab'].max()))
    return Dst,Ts,np.array(Comp)

def build_vocab_freq(Dst,tr):                              # TRAIN-ONLY vocab + unigram + bigram
    c=Counter(); big=Counter()
    for i in tr:
        toks=Dst[i].tolist(); c.update(toks)
        for a,b in zip(toks[:-1],toks[1:]): big[(a,b)]+=1
    tot=sum(c.values())
    comp2id={w:i for i,(w,_) in enumerate(c.most_common(KV-1))}; UNK=KV-1
    uni=np.full(KV,1.0/max(tot,1))                         # unigram prob over token ids (train)
    for w,cnt in c.items():
        if w in comp2id: uni[comp2id[w]]=cnt/tot
    return comp2id, UNK, uni, big, c, tot

def tokenize(Dst,comp2id,UNK):
    return [np.array([comp2id.get(x,UNK) for x in d],dtype=np.int64) for d in Dst]

def windows(tok,ts,idxs):
    Tk,Ts,who=[],[],[]
    for i in idxs:
        tk,tt=tok[i],ts[i]
        for s in range(0,len(tk),N):
            w=tk[s:s+N]
            if len(w)<8: continue
            pad=N-len(w); Tk.append(np.concatenate([w,np.full(pad,PAD)]))
            Ts.append(np.concatenate([tt[s:s+N],np.full(pad,tt[s:s+N][-1])])); who.append(i)
    return np.array(Tk),np.array(Ts,np.int64),np.array(who)

@torch.no_grad()
def model_surprise(m,Tk,Ts):                              # per-window per-position surprise -log P(next)
    m.eval(); out=[]
    for i in range(0,len(Tk),256):
        tk=torch.tensor(Tk[i:i+256]).to(dev); ts=torch.tensor(Ts[i:i+256]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            lg=m(tk,ts)[:,:-1].float()
        logp=F.log_softmax(lg,-1); nxt=tk[:,1:]; valid=(nxt!=PAD)
        s=-logp.gather(2,nxt.clamp(max=KV-1).unsqueeze(-1)).squeeze(-1)
        s=torch.where(valid,s,torch.full_like(s,float('nan'))).cpu().numpy()
        out.append(s)
    return np.concatenate(out) if out else np.zeros((0,N-1))    # [win, N-1] with nan on pad

def per_user_from_windows(surp,who,nusers):               # aggregate window surprises -> per-user max/mean/top3
    from collections import defaultdict
    vals=defaultdict(list)
    for row,w in zip(surp,who):
        v=row[~np.isnan(row)]
        if len(v): vals[w].append(v)
    mx=np.full(nusers,np.nan); mn=np.full(nusers,np.nan); t3=np.full(nusers,np.nan)
    for w,chunks in vals.items():
        allv=np.concatenate(chunks)
        mx[w]=allv.max(); mn[w]=allv.mean(); t3[w]=np.sort(allv)[-3:].mean()
    return mx,mn,t3

def markov_freq_scores(tok,uni,big,tr_tot,idxs,nusers):   # bigram + unigram per-user max surprise (numpy, train-fit)
    mk=np.full(nusers,np.nan); fq=np.full(nusers,np.nan)
    # bigram prob with add-1 smoothing over the seen-from-a transitions
    from collections import defaultdict
    outcount=defaultdict(int)
    for (a,b),c in big.items(): outcount[a]+=c
    for i in idxs:
        tk=tok[i]
        # unigram surprise
        us=-np.log(uni[np.clip(tk,0,KV-1)]+1e-12)
        fq[i]=us.max() if len(us) else np.nan
        # bigram surprise
        ss=[]
        for a,b in zip(tk[:-1],tk[1:]):
            num=big.get((a,b),0)+1; den=outcount.get(a,0)+KV
            ss.append(-np.log(num/den))
        mk[i]=max(ss) if ss else us.max()
    return mk,fq

def metrics(score,Y):
    ok=~np.isnan(score); s=score[ok]; y=Y[ok]
    if len(np.unique(y))<2: return None
    order=np.argsort(-s)
    p_at={k:float(y[order[:k]].mean()) for k in (10,20)}
    return dict(auc=roc_auc_score(y,s),auprc=average_precision_score(y,s),p10=p_at[10],p20=p_at[20])

if __name__=='__main__':
    print("device:",dev,f"| DVOCAB={KV} N={N} D={D} L={L} PRE_EP={PRE_EP} NSEED={NSEED} TRAIN_FRAC={TRAIN_FRAC}")
    Dst,Ts,Comp=load_raw(); n=len(Dst)
    print(f"users(seq>=8)={n}  compromised={int(Comp.sum())}  full-population base rate={Comp.mean():.4f}  (CSV={os.path.basename(CSV)})")
    R={k:{'auc':[],'auprc':[],'p10':[],'p20':[]} for k in ['HSTU-surprise','LSTM-surprise','Markov-bigram','frequency']}
    AGG={a:{'auc':[]} for a in ['HSTU-max','HSTU-mean','HSTU-top3']}
    prev=[]
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        rng=np.random.default_rng(seed); benign=np.where(Comp==0)[0].copy(); comp=np.where(Comp==1)[0]
        rng.shuffle(benign); ntr=int(round(len(benign)*TRAIN_FRAC))
        tr=benign[:ntr]; te=np.concatenate([benign[ntr:],comp])           # eval = held-out benign + ALL compromised
        comp2id,UNK,uni,big,ctr,tr_tot=build_vocab_freq(Dst,tr); tok=tokenize(Dst,comp2id,UNK)
        Tk_tr,Ts_tr,_=windows(tok,Ts,tr); Tk_te,Ts_te,who_te=windows(tok,Ts,te)
        Yte=Comp[te]; prev.append(Yte.mean())
        # epoch-matched next-token models (both trained on benign "normal")
        hstu=train_gen(HSTU(KV,D=D,H=H,L=L,N=N),Tk_tr,Ts_tr,KV,epochs=PRE_EP)
        lstm=train_gen(LSTMGen(KV,D=D,H=H,L=L,N=N),Tk_tr,Ts_tr,KV,epochs=PRE_EP)
        hmx,hmn,ht3=per_user_from_windows(model_surprise(hstu,Tk_te,Ts_te),who_te,n)
        lmx,_,_     =per_user_from_windows(model_surprise(lstm,Tk_te,Ts_te),who_te,n)
        mk,fq=markov_freq_scores(tok,uni,big,tr_tot,te,n)
        for nm,sc in [('HSTU-surprise',hmx),('LSTM-surprise',lmx),('Markov-bigram',mk),('frequency',fq)]:
            m=metrics(sc[te],Yte)
            if m:
                for k in R[nm]: R[nm][k].append(m[k])
        for a,sc in [('HSTU-max',hmx),('HSTU-mean',hmn),('HSTU-top3',ht3)]:
            mm=metrics(sc[te],Yte)
            if mm: AGG[a]['auc'].append(mm['auc'])
        print(f"[seed {seed}] train(normal)={len(tr)} eval={len(te)} (compromised={int(Yte.sum())}, eval base rate={Yte.mean():.3f})")
        del hstu,lstm; torch.cuda.empty_cache() if dev=='cuda' else None
    ep=np.mean(prev)
    print(f"\n=== PER-USER compromise detection, mean+/-std over {NSEED} seeds (eval base rate ~{ep:.1%}) ===")
    print(f"  {'scorer (max-agg)':<22}{'AUC':>15}{'AUPRC':>15}{'P@10':>9}{'P@20':>9}")
    for nm in R:
        a=np.array(R[nm]['auc']); p=np.array(R[nm]['auprc']); q10=np.array(R[nm]['p10']); q20=np.array(R[nm]['p20'])
        if len(a): print(f"  {nm:<22}{a.mean():>7.3f}+/-{a.std():<5.3f}{p.mean():>7.3f}+/-{p.std():<5.3f}{q10.mean():>9.3f}{q20.mean():>9.3f}")
    print(f"\n=== aggregation robustness (HSTU AUC) ===")
    for a in AGG:
        v=np.array(AGG[a]['auc'])
        if len(v): print(f"  {a:<12} AUC {v.mean():.3f}+/-{v.std():.3f}")
    print("SEC_ANOMALY_DONE")
