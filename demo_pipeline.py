"""FM PIPELINE DEMO (single-domain F-DATA, at SCALE): pretrain HSTU BACKBONE (autoregressive)
-> FREEZE backbone -> insert task-specific HEAD -> finetune ONLY the head -> eval.
THREE downstream tasks off ONE frozen backbone (head-swap):
  (A) job-duration FORECASTING (regression rollout, MASE)
  (B) next-job-FAILURE classification (binary head; acc/F1/AUC/AUPRC)
  (C) next-job NODE-COUNT regression (linear head; MAE/Pearson)
Shows pretrained+frozen-backbone+cheap-head-FT vs from-scratch + Chronos. Env: D,L,PRE_EP,FT_EP,CHRONOS."""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from nfm_core import HSTU, dev, mase, train_gen, rollout, Quantizer
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve
import baselines_sota as bs

K=1024; CTX=32; HMAX=16; N=CTX+HMAX
D=int(os.environ.get('D',256)); H=int(os.environ.get('H',4)); L=int(os.environ.get('L',4))
PRE_EP=int(os.environ.get('PRE_EP',8)); FT_EP=int(os.environ.get('FT_EP',6))
np.random.seed(0); torch.manual_seed(0)
PAD=-999.0

def load():                                   # per-user (duration, ts, fail, log-nodecount)
    cols=['usr','adt','duration','exit state','nnumr']
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))],ignore_index=True)
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce'); d=d.dropna(subset=['adt']).sort_values(['usr','adt'])
    d['t']=d['adt'].astype(np.int64)//10**9
    d['fail']=(d['exit state'].astype(str).str.lower()=='failed').astype(np.int64)
    d['ln']=np.log1p(pd.to_numeric(d['nnumr'],errors='coerce').fillna(0).clip(lower=0))
    Dur,Ts,Fail,Ln=[],[],[],[]
    for u,g in d.groupby('usr',sort=False):
        if len(g)<8: continue
        Dur.append(g['duration'].values.astype(float)); Ts.append(g['t'].values.astype(np.int64))
        Fail.append(g['fail'].values.astype(np.int64)); Ln.append(g['ln'].values.astype(np.float32))
    return Dur,Ts,Fail,Ln

def windows(seqs,ts,extras,q):                # extras=list of aligned label-seq-lists -> windowed together
    Tk,Ts=[],[]; Ex=[[] for _ in extras]
    for k,(v,tt) in enumerate(zip(seqs,ts)):
        tkv=q.to_tok(v)
        for s in range(0,len(tkv),N):
            w=tkv[s:s+N]
            if len(w)<8: continue
            pad=N-len(w); wt=tt[s:s+N]
            Tk.append(np.concatenate([w,np.full(pad,K)])); Ts.append(np.concatenate([wt,np.full(pad,wt[-1])]))
            for j,exs in enumerate(extras):
                we=np.asarray(exs[k][s:s+N],float); Ex[j].append(np.concatenate([we,np.full(pad,PAD)]))
    return np.array(Tk),np.array(Ts,np.int64),[np.array(e) for e in Ex]

def ntrain(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6

def _opt(m,lr): return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr,weight_decay=0.01)

def train_head(m, Tok, Ts, Y, epochs, kind, bs=256, lr=2e-3):   # kind: 'cls' (CE, Y int) or 'reg' (MSE, Y float)
    m=m.to(dev); opt=_opt(m,lr)
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(len(Tok))
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]; tk=torch.tensor(Tok[j]).to(dev); ts=torch.tensor(Ts[j]).to(dev); y=torch.tensor(Y[j]).to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                out=m(tk,ts)[:, :-1]                          # predict NEXT position
            tgt=y[:,1:]; valid=tgt>PAD+1
            if kind=='cls':
                loss=F.cross_entropy(out.reshape(-1,2)[valid.reshape(-1)], tgt.reshape(-1)[valid.reshape(-1)].long())
            else:
                loss=F.mse_loss(out[...,0].reshape(-1)[valid.reshape(-1)].float(), tgt.reshape(-1)[valid.reshape(-1)].float())
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def eval_cls(m, Tok, Ts, Y, bs=256):
    m.eval(); P,YY=[],[]
    for i in range(0,len(Tok),bs):
        tk=torch.tensor(Tok[i:i+bs]).to(dev); ts=torch.tensor(Ts[i:i+bs]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            pr=m(tk,ts)[:, :-1].float().softmax(-1)[...,1]
        tgt=Y[i:i+bs][:,1:]; valid=tgt>PAD+1
        P.append(pr.cpu().numpy()[valid]); YY.append(tgt[valid].astype(int))
    P=np.concatenate(P); Y=np.concatenate(YY)
    prec,rec,_=precision_recall_curve(Y,P); f1s=2*prec*rec/(prec+rec+1e-9)
    return dict(acc=((P>0.5)==Y).mean(), f1_50=f1_score(Y,(P>0.5).astype(int),zero_division=0),
                f1_best=float(np.nanmax(f1s)), auc=roc_auc_score(Y,P) if len(np.unique(Y))>1 else float('nan'),
                auprc=average_precision_score(Y,P) if len(np.unique(Y))>1 else float('nan'))

@torch.no_grad()
def eval_reg(m, Tok, Ts, Y, bs=256):
    m.eval(); P,YY=[],[]
    for i in range(0,len(Tok),bs):
        tk=torch.tensor(Tok[i:i+bs]).to(dev); ts=torch.tensor(Ts[i:i+bs]).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            pr=m(tk,ts)[:, :-1, 0].float()
        tgt=Y[i:i+bs][:,1:]; valid=tgt>PAD+1
        P.append(pr.cpu().numpy()[valid]); YY.append(tgt[valid])
    P=np.concatenate(P); Y=np.concatenate(YY)
    mae=np.abs(P-Y).mean(); corr=np.corrcoef(P,Y)[0,1]
    return dict(mae=mae, corr=corr, naive_mae=np.abs(Y-Y.mean()).mean())

if __name__=='__main__':
    print("device:",dev, f"| D={D} L={L} PRE_EP={PRE_EP} FT_EP={FT_EP}")
    Dur,Ts,Fail,Ln=load(); n=len(Dur); print(f"F-DATA users(seq>=8)={n}  jobs={sum(len(x) for x in Dur):,}")
    rng=np.random.default_rng(0); e=rng.permutation(n); te=set(e[:max(1,int(n*0.2))].tolist())
    tr=[i for i in range(n) if i not in te]; teL=[i for i in range(n) if i in te]
    q=Quantizer(K).fit(np.concatenate([Dur[i] for i in tr]))
    scale=float(np.mean([np.mean(np.abs(np.diff(Dur[i]))) for i in tr if len(Dur[i])>1]))
    Tok_tr,Ts_tr,(Fail_tr,Ln_tr)=windows([Dur[i] for i in tr],[Ts[i] for i in tr],[[Fail[i] for i in tr],[Ln[i] for i in tr]],q)
    Tok_te,Ts_te,(Fail_te,Ln_te)=windows([Dur[i] for i in teL],[Ts[i] for i in teL],[[Fail[i] for i in teL],[Ln[i] for i in teL]],q)
    print(f"pretrain windows={len(Tok_tr):,}  test windows={len(Tok_te):,}")

    print("\n===== STAGE 1: PRETRAIN backbone (autoregressive next-token) =====")
    pre=train_gen(HSTU(K,D=D,H=H,L=L,N=N),Tok_tr,Ts_tr,K,epochs=PRE_EP); print(f"pretrained. params={ntrain(pre):.1f}M")
    def frozen_copy(headV):
        m=HSTU(K,D=D,H=H,L=L,N=N); m.load_state_dict(pre.state_dict()); m.freeze_backbone().swap_head(headV); return m

    print("\n===== TASK A: job-duration FORECASTING (MASE) =====")
    Cxv,Fur,Cxt,Tsf=[],[],[],[]
    for i in teL:
        v=Dur[i]; tt=Ts[i]; tk=q.to_tok(v)
        for s in range(0,len(v)-N+1,N):
            Cxt.append(tk[s:s+CTX]); Cxv.append(v[s:s+CTX]); Tsf.append(tt[s:s+N]); Fur.append(v[s+CTX:s+CTX+HMAX])
    Cxt=np.array(Cxt); Cxv=np.array(Cxv,float); Tsf=np.array(Tsf,np.int64); Fur=np.array(Fur,float)
    ftm=train_gen(frozen_copy(K),Tok_tr,Ts_tr,K,epochs=FT_EP)
    scr=train_gen(HSTU(K,D=D,H=H,L=L,N=N),Tok_tr,Ts_tr,K,epochs=PRE_EP)
    resA={'persist':mase(np.repeat(Cxv[:,-1:],HMAX,1),Fur,scale),
          'pretrained-zeroshot':mase(rollout(pre,Cxt,Tsf,K,CTX,HMAX,q.centers),Fur,scale),
          'frozen+headFT':mase(rollout(ftm,Cxt,Tsf,K,CTX,HMAX,q.centers),Fur,scale),
          'from-scratch':mase(rollout(scr,Cxt,Tsf,K,CTX,HMAX,q.centers),Fur,scale)}
    if os.environ.get('CHRONOS','1')=='1':
        try: resA['Chronos']=mase(bs.chronos_zeroshot(Cxv,HMAX),Fur,scale)
        except Exception as ex: print("  chronos skip",str(ex)[:60])
    print("  MASE h | "+" | ".join(f"{k:>20}" for k in resA))
    for h in [1,4,8,16]: print(f"      {h:>3} | "+" | ".join(f"{resA[k][h-1]:>20.3f}" for k in resA))
    del ftm,scr; torch.cuda.empty_cache()

    print("\n===== TASK B: next-job-FAILURE classification (head-swap, freeze backbone) =====")
    print(f"  failure base rate={Fail_tr[Fail_tr>PAD+1].mean():.4f}")
    bP=eval_cls(train_head(frozen_copy(2),Tok_tr,Ts_tr,Fail_tr,FT_EP,'cls'),Tok_te,Ts_te,Fail_te)
    bS=eval_cls(train_head(HSTU(K,D=D,H=H,L=L,N=N).swap_head(2),Tok_tr,Ts_tr,Fail_tr,PRE_EP+FT_EP,'cls'),Tok_te,Ts_te,Fail_te)
    print(f"  {'method':<26}{'acc':>7}{'F1@.5':>8}{'F1best':>8}{'AUC':>7}{'AUPRC':>7}{'train(M)':>10}")
    for nm,r,mm in [('pretrained+frozen+headFT',bP,'P'),('from-scratch(full)',bS,'S')]:
        tp=0.00 if mm=='P' else ntrain(HSTU(K,D=D,H=H,L=L,N=N))
        print(f"  {nm:<26}{r['acc']:>7.3f}{r['f1_50']:>8.3f}{r['f1_best']:>8.3f}{r['auc']:>7.3f}{r['auprc']:>7.3f}{tp:>10.2f}")

    print("\n===== TASK C: next-job NODE-COUNT regression (head-swap, freeze backbone) =====")
    cP=eval_reg(train_head(frozen_copy(1),Tok_tr,Ts_tr,Ln_tr,FT_EP,'reg'),Tok_te,Ts_te,Ln_te)
    cS=eval_reg(train_head(HSTU(K,D=D,H=H,L=L,N=N).swap_head(1),Tok_tr,Ts_tr,Ln_tr,PRE_EP+FT_EP,'reg'),Tok_te,Ts_te,Ln_te)
    print(f"  {'method':<26}{'MAE':>8}{'Pearson':>9}{'naiveMAE':>10}")
    print(f"  {'pretrained+frozen+headFT':<26}{cP['mae']:>8.3f}{cP['corr']:>9.3f}{cP['naive_mae']:>10.3f}")
    print(f"  {'from-scratch(full)':<26}{cS['mae']:>8.3f}{cS['corr']:>9.3f}{cS['naive_mae']:>10.3f}")
    print("\nDEMODONE")
