"""HPC-security line (Frame 2): LANL 2015 auth events as entity-event streams.
Entity = source user; token = DESTINATION COMPUTER a user authenticates to (the recsys "item");
timestamp = auth time (irregular). We pretrain the HSTU backbone autoregressively to predict the
next destination computer per user, then FREEZE the backbone and finetune a lightweight binary head
to flag red-team (malicious) auth EVENTS. Compares pretrained-frozen+head vs from-scratch. AUC/AUPRC.

FIXED after 2026-07-15 code review (top-venue rigor):
 - NO label shift: the head predicts the label at the SAME position, so the model sees the event's OWN
   destination token (red-team is discriminated by the anomalous destination). (was next-position shift)
 - STRATIFIED user split: red-team users guaranteed in both train and test (was random -> could yield 0 test pos).
 - CLASS-WEIGHTED loss (capped inverse frequency) for the extreme imbalance.
 - TRAIN-ONLY destination-computer vocabulary (was transductive over train+test).
 - MULTI-SEED: mean +/- std over NSEED seeds.

Input: ~/nfm/data/lanl/lanl_events.csv (time,user,srccomp,dstcomp,label) from data/prep_lanl.sh.
Env: DVOCAB(2048), N(64), D/H/L, PRE_EP, FT_EP, NSEED(3).
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from collections import Counter
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from nfm_core import HSTU, dev, train_gen
from sklearn.metrics import roc_auc_score, average_precision_score

KV=int(os.environ.get('DVOCAB',2048)); N=int(os.environ.get('N',64))
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
PRE_EP=int(os.environ.get('PRE_EP',10)); FT_EP=int(os.environ.get('FT_EP',6)); NSEED=int(os.environ.get('NSEED',3))
PAD=KV; PADLAB=-999.0
CSV=os.path.expanduser(os.environ.get('LANL_CSV',"~/nfm/data/lanl/lanl_events.csv"))

def load_raw():
    d=pd.read_csv(CSV,header=None,names=['t','user','src','dst','lab']).sort_values(['user','t'])
    Dst,Ts,Lab,Comp=[],[],[],[]
    for u,g in d.groupby('user',sort=False):
        if len(g)<8: continue
        Dst.append(g['dst'].values); Ts.append(g['t'].values.astype(np.int64))
        Lab.append(g['lab'].values.astype(np.int64)); Comp.append(int(g['lab'].max()))   # per-user compromised flag
    return Dst,Ts,Lab,np.array(Comp)

def build_vocab(Dst,tr):                                   # TRAIN-ONLY top-(KV-1) dest computers + UNK
    c=Counter()
    for i in tr: c.update(Dst[i].tolist())
    comp2id={w:i for i,(w,_) in enumerate(c.most_common(KV-1))}
    return comp2id, KV-1

def tokenize(Dst,comp2id,UNK):
    return [np.array([comp2id.get(x,UNK) for x in d],dtype=np.int64) for d in Dst]

def windows(tok,ts,lab,idxs):
    Tk,Ts,Lb=[],[],[]
    for i in idxs:
        tk,tt,yy=tok[i],ts[i],lab[i]
        for s in range(0,len(tk),N):
            w=tk[s:s+N]
            if len(w)<8: continue
            pad=N-len(w); wt=tt[s:s+N]
            Tk.append(np.concatenate([w,np.full(pad,PAD)]))
            Ts.append(np.concatenate([wt,np.full(pad,wt[-1])]))
            Lb.append(np.concatenate([yy[s:s+N].astype(float),np.full(pad,PADLAB)]))
    return np.array(Tk),np.array(Ts,np.int64),np.array(Lb)

def ntrain(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6

def train_head(m,Tk,Ts,Y,epochs,w,bs=256,lr=2e-3):        # NO shift: predict label at SAME position p (sees token p)
    m=m.to(dev); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr,weight_decay=0.01)
    wt=torch.tensor(w,dtype=torch.float32,device=dev)
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(len(Tk))
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]; tk=torch.tensor(Tk[j]).to(dev); ts=torch.tensor(Ts[j]).to(dev); y=torch.tensor(Y[j]).to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                out=m(tk,ts)                               # [B,N,2] same-position classification
            valid=(y>PADLAB+1).reshape(-1)
            loss=F.cross_entropy(out.reshape(-1,2).float()[valid], y.reshape(-1)[valid].long(), weight=wt)  # .float(): bf16 logits vs f32 weight
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def eval_cls(m,Tk,Ts,Y,bs=256):
    m.eval(); P,YY=[],[]
    for i in range(0,len(Tk),bs):
        tk=torch.tensor(Tk[i:i+bs]).to(dev); ts=torch.tensor(Ts[i:i+bs]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            pr=m(tk,ts).float().softmax(-1)[...,1]         # same-position P(malicious)
        tgt=Y[i:i+bs]; valid=tgt>PADLAB+1
        P.append(pr.cpu().numpy()[valid]); YY.append(tgt[valid].astype(int))
    P=np.concatenate(P); Y=np.concatenate(YY)
    if len(np.unique(Y))<2: return dict(auc=float('nan'),auprc=float('nan'),pos=int(Y.sum()),n=len(Y))
    return dict(auc=roc_auc_score(Y,P),auprc=average_precision_score(Y,P),pos=int(Y.sum()),n=len(Y))

def stratified_split(comp,seed):                           # red-team users guaranteed in BOTH train and test
    rng=np.random.default_rng(seed); ci=np.where(comp==1)[0].copy(); bi=np.where(comp==0)[0].copy()
    rng.shuffle(ci); rng.shuffle(bi)
    cte=ci[:max(1,int(round(len(ci)*0.2)))]; bte=bi[:int(round(len(bi)*0.2))]
    te=np.concatenate([cte,bte]); tr=np.concatenate([ci[len(cte):],bi[len(bte):]])
    return tr,te

if __name__=='__main__':
    print("device:",dev,f"| DVOCAB={KV} N={N} D={D} L={L} PRE_EP={PRE_EP} FT_EP={FT_EP} NSEED={NSEED}")
    Dst,Ts,Lab,Comp=load_raw(); n=len(Dst)
    nev=sum(len(x) for x in Lab); npos=sum(int(y.sum()) for y in Lab)
    print(f"users(seq>=8)={n}  compromised users={int(Comp.sum())}  events={nev:,}  malicious events={npos}  event base rate={npos/max(nev,1):.2e}")
    res={'frozen+head':{'auc':[],'auprc':[]}, 'from-scratch':{'auc':[],'auprc':[]}}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        tr,te=stratified_split(Comp,seed)
        comp2id,UNK=build_vocab(Dst,tr); tok=tokenize(Dst,comp2id,UNK)
        Tk_tr,Ts_tr,Lb_tr=windows(tok,Ts,Lab,tr); Tk_te,Ts_te,Lb_te=windows(tok,Ts,Lab,te)
        pos=int((Lb_tr==1).sum()); neg=int((Lb_tr==0).sum()); w=[1.0, float(min(neg/max(pos,1),100.0))]  # capped inverse-freq
        print(f"[seed {seed}] train users={len(tr)} test users={len(te)} | train-pos={pos} test-pos={int((Lb_te==1).sum())} | cls-weight={w[1]:.1f}")
        pre=train_gen(HSTU(KV,D=D,H=H,L=L,N=N),Tk_tr,Ts_tr,KV,epochs=PRE_EP)
        def frozen():
            m=HSTU(KV,D=D,H=H,L=L,N=N); m.load_state_dict(pre.state_dict()); m.freeze_backbone().swap_head(2); return m
        rP=eval_cls(train_head(frozen(),Tk_tr,Ts_tr,Lb_tr,FT_EP,w),Tk_te,Ts_te,Lb_te)
        rS=eval_cls(train_head(HSTU(KV,D=D,H=H,L=L,N=N).swap_head(2),Tk_tr,Ts_tr,Lb_tr,PRE_EP+FT_EP,w),Tk_te,Ts_te,Lb_te)
        print(f"   frozen+head AUC={rP['auc']:.3f} AUPRC={rP['auprc']:.3f} | from-scratch AUC={rS['auc']:.3f} AUPRC={rS['auprc']:.3f} (test-pos={rP['pos']})")
        res['frozen+head']['auc'].append(rP['auc']); res['frozen+head']['auprc'].append(rP['auprc'])
        res['from-scratch']['auc'].append(rS['auc']); res['from-scratch']['auprc'].append(rS['auprc'])
        del pre; torch.cuda.empty_cache() if dev=='cuda' else None
    print(f"\n=== per-event red-team detection, mean+/-std over {NSEED} seeds ===")
    print(f"  {'method':<28}{'AUC':>16}{'AUPRC':>16}")
    for k in res:
        a=np.array(res[k]['auc']); p=np.array(res[k]['auprc'])
        print(f"  {k:<28}{np.nanmean(a):>8.3f}+/-{np.nanstd(a):<6.3f}{np.nanmean(p):>8.3f}+/-{np.nanstd(p):<6.3f}")
    print("SECURITY_DONE")
