"""HPC-security line, UNSUPERVISED anomaly detection (the reframe that fixes the imbalance problem).
For each auth event, surprise = -log P(actual dest-computer | history) from a model pretrained on NORMAL
(benign) users only. Red-team lateral movement authenticates to unusual destinations -> high surprise.
Aggregate surprise PER USER (max) and ask whether COMPROMISED users score higher than benign users.
Per-user labels make the imbalance ~1-2% instead of ~1e-6.

FIXED after 2026-07-15 code review (top-venue rigor):
 - TRAIN-ONLY vocabulary AND frequency baseline (was transductive over train+test).
 - MULTI-SEED: mean +/- std over NSEED seeds (the benign train/eval split varies per seed).
Compares: pretrained-HSTU surprise vs from-scratch(1ep) surprise vs frequency (rarity) baseline.
Input: ~/nfm/data/lanl/lanl_events.csv. Env: DVOCAB,N,D,H,L,PRE_EP,NSEED(3).
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from collections import Counter
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from nfm_core import HSTU, dev, train_gen
from sklearn.metrics import roc_auc_score, average_precision_score

KV=int(os.environ.get('DVOCAB',2048)); N=int(os.environ.get('N',64))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
PRE_EP=int(os.environ.get('PRE_EP',10)); NSEED=int(os.environ.get('NSEED',3)); PAD=KV
CSV=os.path.expanduser(os.environ.get('LANL_CSV',"~/nfm/data/lanl/lanl_events.csv"))

def load_raw():
    d=pd.read_csv(CSV,header=None,names=['t','user','src','dst','lab']).sort_values(['user','t'])
    Dst,Ts,Comp=[],[],[]
    for u,g in d.groupby('user',sort=False):
        if len(g)<8: continue
        Dst.append(g['dst'].values); Ts.append(g['t'].values.astype(np.int64)); Comp.append(int(g['lab'].max()))
    return Dst,Ts,np.array(Comp)

def build_vocab_freq(Dst,tr):                              # TRAIN-ONLY vocab + dst frequency
    c=Counter()
    for i in tr: c.update(Dst[i].tolist())
    tot=sum(c.values())
    comp2id={w:i for i,(w,_) in enumerate(c.most_common(KV-1))}
    freq={w:cnt/tot for w,cnt in c.items()}
    return comp2id, KV-1, freq

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
def per_user_surprise(m,Tk,Ts,who,nusers,bs=256):
    m.eval(); score=np.full(nusers,-1e9)
    for i in range(0,len(Tk),bs):
        tk=torch.tensor(Tk[i:i+bs]).to(dev); ts=torch.tensor(Ts[i:i+bs]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            lg=m(tk,ts)[:,:-1].float()
        logp=F.log_softmax(lg,-1); nxt=tk[:,1:]; valid=(nxt!=PAD)
        s=-logp.gather(2,nxt.clamp(max=KV-1).unsqueeze(-1)).squeeze(-1)
        s=torch.where(valid,s,torch.full_like(s,-1e9)).amax(1).cpu().numpy()
        for b,wi in enumerate(who[i:i+bs]): score[wi]=max(score[wi],float(s[b]))
    return score

if __name__=='__main__':
    print("device:",dev,f"| DVOCAB={KV} N={N} D={D} L={L} PRE_EP={PRE_EP} NSEED={NSEED}")
    Dst,Ts,Comp=load_raw(); n=len(Dst)
    print(f"users(seq>=8)={n}  compromised users={int(Comp.sum())}  per-user base rate={Comp.mean():.4f}")
    res={'pretrained-HSTU surprise':[], 'from-scratch-HSTU(1ep) surprise':[], 'frequency baseline (rarity)':[]}
    resp={k:[] for k in res}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        rng=np.random.default_rng(seed); benign=np.where(Comp==0)[0].copy(); comp=np.where(Comp==1)[0]
        rng.shuffle(benign); n_te_b=int(round(len(benign)*0.2))
        te=np.concatenate([comp,benign[:n_te_b]]); tr=benign[n_te_b:]        # train = benign only ("normal")
        comp2id,UNK,freq=build_vocab_freq(Dst,tr); tok=tokenize(Dst,comp2id,UNK)
        Tk_tr,Ts_tr,_=windows(tok,Ts,tr); Tk_te,Ts_te,who_te=windows(tok,Ts,te)
        Yte=Comp[te]
        pre=train_gen(HSTU(KV,D=D,H=H,L=L,N=N),Tk_tr,Ts_tr,KV,epochs=PRE_EP)
        scr=train_gen(HSTU(KV,D=D,H=H,L=L,N=N),Tk_tr,Ts_tr,KV,epochs=1)
        sp=per_user_surprise(pre,Tk_te,Ts_te,who_te,n); ss=per_user_surprise(scr,Tk_te,Ts_te,who_te,n)
        fb=np.full(n,-1e9)                                                    # frequency baseline: max TRAIN-rarity per user
        for i in te: fb[i]=float(np.max(-np.log([freq.get(x,1e-9) for x in Dst[i]])))
        print(f"[seed {seed}] train(normal) users={len(tr)} eval users={len(te)} compromised={int(Yte.sum())}")
        for nm,sc in [('pretrained-HSTU surprise',sp),('from-scratch-HSTU(1ep) surprise',ss),('frequency baseline (rarity)',fb)]:
            s=sc[te]; ok=s>-1e8
            a=roc_auc_score(Yte[ok],s[ok]); p=average_precision_score(Yte[ok],s[ok])
            res[nm].append(a); resp[nm].append(p)
        del pre,scr; torch.cuda.empty_cache() if dev=='cuda' else None
    print(f"\n=== PER-USER compromise detection, mean+/-std over {NSEED} seeds ===")
    print(f"  {'scorer':<34}{'AUC':>16}{'AUPRC':>16}")
    for k in res:
        a=np.array(res[k]); p=np.array(resp[k])
        print(f"  {k:<34}{a.mean():>8.3f}+/-{a.std():<6.3f}{p.mean():>8.3f}+/-{p.std():<6.3f}")
    print("SEC_ANOMALY_DONE")
