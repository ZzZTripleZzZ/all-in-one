#!/usr/bin/env python
"""Single-domain strong-baseline benchmark for the network foundation model.
Proposed HSTU  vs  param-matched causal Transformer  vs  LSTM  vs  GBM  vs  naive.
Datasets: BurstGPT (DC) / F-DATA Fugaku (HPC) / CESNET-TimeSeries24 (NET). CUDA+bf16.
Metric: next-token accuracy + macro-F1 (imbalance-aware).  Data lives in ~/nfm/data.

NOTE (methodology): current split is random 80/20 over windows (comparable to the
laptop validation). Final paper numbers need temporal/entity-holdout split -- see TODO."""
import sys, os, time, glob, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
DATA=os.path.expanduser("~/nfm/data")

# =================== models ===================
class RelPosBias(nn.Module):
    def __init__(self,N): super().__init__(); self.N=N; self.w=nn.Parameter(torch.empty(2*N-1).normal_(0,0.02))
    def forward(self):
        N=self.N; t=F.pad(self.w[:2*N-1],[0,N]).repeat(N); t=t[:-N].reshape(1,N,3*N-2); r=(2*N-1)//2
        return t[:,:,r:-r]
class HSTULayer(nn.Module):            # faithful HSTU (arXiv 2402.17152): SiLU-attn, U-gate
    def __init__(self,D,dqk,dv,H,N,p=0.1):
        super().__init__(); self.D,self.dqk,self.dv,self.H,self.p=D,dqk,dv,H,p
        self.W=nn.Parameter(torch.empty(D,(2*dv+2*dqk)*H).normal_(0,0.02))
        self.o=nn.Linear(dv*H,D); nn.init.xavier_uniform_(self.o.weight); self.rel=RelPosBias(N)
    def forward(self,x,mask):
        B,N,_=x.shape; H,dqk,dv=self.H,self.dqk,self.dv
        uvqk=F.silu(F.layer_norm(x,[self.D])@self.W)
        u,v,q,k=torch.split(uvqk,[dv*H,dv*H,dqk*H,dqk*H],-1)
        q=q.view(B,N,H,dqk); k=k.view(B,N,H,dqk); v=v.view(B,N,H,dv)
        qk=torch.einsum('bnhd,bmhd->bhnm',q,k)+self.rel().unsqueeze(1)
        a=(F.silu(qk)/N)*mask.view(1,1,N,N)
        av=torch.einsum('bhnm,bmhd->bnhd',a,v).reshape(B,N,H*dv)
        return self.o(F.dropout(u*F.layer_norm(av,[dv*H]),self.p,self.training))+x
class TfmLayer(nn.Module):             # matched causal softmax transformer (pre-LN)
    def __init__(self,D,H,p=0.1):
        super().__init__(); self.H=H; self.ln1=nn.LayerNorm(D); self.ln2=nn.LayerNorm(D)
        self.qkv=nn.Linear(D,3*D); self.o=nn.Linear(D,D); self.p=p
        self.mlp=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
    def forward(self,x,mask=None):
        B,N,D=x.shape; H=self.H; hd=D//H
        h=self.ln1(x); q,k,v=self.qkv(h).view(B,N,3,H,hd).permute(2,0,3,1,4).unbind(0)
        att=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.p if self.training else 0.0)
        x=x+self.o(att.transpose(1,2).reshape(B,N,D))
        return x+self.mlp(self.ln2(x))
class SeqModel(nn.Module):
    def __init__(self,kind,V,nfeat,D,H,L,N,dqk=None,dv=None):
        super().__init__(); self.kind=kind; self.N=N
        self.tok=nn.Embedding(V+1,D,padding_idx=V); self.feat=nn.Linear(nfeat,D) if nfeat>0 else None
        self.pos=nn.Embedding(N,D) if kind in ('tfm','lstm') else None
        if kind=='hstu':
            dqk=dqk or D//H; dv=dv or D//H
            self.layers=nn.ModuleList([HSTULayer(D,dqk,dv,H,N) for _ in range(L)])
            self.register_buffer('mask',torch.tril(torch.ones(N,N)))
        elif kind=='tfm':
            self.layers=nn.ModuleList([TfmLayer(D,H) for _ in range(L)])
        elif kind=='lstm':
            self.lstm=nn.LSTM(D,D,L,batch_first=True,dropout=0.1 if L>1 else 0.0)
        self.head=nn.Linear(D,V)
    def forward(self,tok,feat=None):
        x=self.tok(tok)
        if self.feat is not None and feat is not None: x=x+self.feat(feat)
        if self.pos is not None:
            x=x+self.pos(torch.arange(tok.shape[1],device=tok.device))[None]
        if self.kind=='hstu':
            for l in self.layers: x=l(x,self.mask)
        elif self.kind=='tfm':
            for l in self.layers: x=l(x)
        else:
            x,_=self.lstm(x)
        return self.head(x)

# =================== windowing + split ===================
def windows(seqs_tok, seqs_feat, N, V):
    T,Ffl,E=[],[],[]
    for ei,(tk,ft) in enumerate(zip(seqs_tok,seqs_feat)):
        for s in range(0,len(tk),N):
            wt=tk[s:s+N]; wf=ft[s:s+N]
            if len(wt)<8: continue
            pad=N-len(wt)
            T.append(np.concatenate([wt,np.full(pad,V)]))
            Ffl.append(np.concatenate([wf,np.zeros((pad,ft.shape[1]))]))
            E.append(ei)
    return np.array(T,np.int64), np.array(Ffl,np.float32), np.array(E,np.int64)

# =================== neural train/eval ===================
def train_eval(kind, Xt, Xf, tr, te, V, N, D=256, H=4, L=4, epochs=4, bs=256, amp=True):
    nfeat=Xf.shape[2]
    m=SeqModel(kind,V,nfeat,D,H,L,N).to(dev)
    nparam=sum(p.numel() for p in m.parameters())/1e6
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=0.01)
    def batches(ids,shuffle):
        ids=np.random.permutation(ids) if shuffle else ids
        for i in range(0,len(ids),bs):
            j=ids[i:i+bs]
            yield torch.tensor(Xt[j]).to(dev), torch.tensor(Xf[j]).to(dev)
    t0=time.time()
    for ep in range(epochs):
        m.train()
        for tok,ft in batches(tr,True):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                logit=m(tok,ft)[:,:-1]; loss=F.cross_entropy(logit.reshape(-1,V),tok[:,1:].reshape(-1),ignore_index=V)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); P,Ttg=[],[]; nll=tot=0.0
    with torch.no_grad():
        for tok,ft in batches(te,False):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                logit=m(tok,ft)[:,:-1].float()
            tgt=tok[:,1:]; valid=tgt!=V
            nll+=F.cross_entropy(logit.reshape(-1,V),tgt.reshape(-1),ignore_index=V,reduction='sum').item()
            tot+=valid.sum().item()
            P.append(logit.argmax(-1)[valid].cpu().numpy()); Ttg.append(tgt[valid].cpu().numpy())
    P=np.concatenate(P); Ttg=np.concatenate(Ttg)
    acc=(P==Ttg).mean(); mf1=f1_score(Ttg,P,average='macro',zero_division=0); ppl=math.exp(nll/tot)
    return dict(name=kind.upper(),acc=acc,mf1=mf1,ppl=ppl,params=nparam,sec=time.time()-t0)

# =================== baselines ===================
def pairs(Xt, ids, N, V):
    cur,prev,nxt=[],[],[]
    for j in ids:
        t=Xt[j]; v=t!=V
        for p in range(N-1):
            if v[p] and v[p+1]:
                cur.append(t[p]); prev.append(t[p-1] if p>0 and v[p-1] else V); nxt.append(t[p+1])
    return np.array(cur),np.array(prev),np.array(nxt)
def naive_and_gbm(Xt, Xf, tr, te, V, N):
    ctr,ptr,ntr=pairs(Xt,tr,N,V); cte,pte,nte=pairs(Xt,te,N,V)
    maj=np.bincount(ntr,minlength=V).argmax()
    res=[]
    res.append(dict(name='marginal',acc=(nte==maj).mean(),mf1=f1_score(nte,np.full_like(nte,maj),average='macro',zero_division=0)))
    res.append(dict(name='persistence',acc=(cte==nte).mean(),mf1=f1_score(nte,cte,average='macro',zero_division=0)))
    M=np.zeros((V,V))
    for c,n in zip(ctr,ntr): M[c,n]+=1
    mk=M.argmax(1); mk[M.sum(1)==0]=maj; pr=mk[cte]
    res.append(dict(name='Markov',acc=(pr==nte).mean(),mf1=f1_score(nte,pr,average='macro',zero_division=0)))
    # GBM on [cur, prev, current feats] -> next token (subsample for speed)
    def feat_at(ids):
        Xr,yr=[],[]
        for j in ids:
            t=Xt[j]; f=Xf[j]; v=t!=V
            for p in range(1,N-1):
                if v[p] and v[p+1]: Xr.append(np.concatenate([[t[p],t[p-1]],f[p]])); yr.append(t[p+1])
        return np.array(Xr,np.float32),np.array(yr)
    Xtr,ytr=feat_at(tr); Xte,yte=feat_at(te)
    if len(Xtr)>300000:
        s=np.random.choice(len(Xtr),300000,replace=False); Xtr,ytr=Xtr[s],ytr[s]
    g=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,max_depth=8)
    g.fit(Xtr,ytr); pg=g.predict(Xte)
    res.append(dict(name='GBM',acc=(pg==yte).mean(),mf1=f1_score(yte,pg,average='macro',zero_division=0)))
    return res

# =================== dataset adapters (schemas from hstu_val.py) ===================
def qbin(x,K):
    x=np.asarray(x,float); q=np.quantile(x[np.isfinite(x)],np.linspace(0,1,K+1)[1:-1])
    return np.clip(np.digitize(x,q),0,K-1).astype(np.int64)
def fdata(target='duration'):
    K=32; cols=['usr','adt','exit state','duration','nnumr','cnumr','freq_req','pri','qdt','sdt']
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(DATA+"/fdata/*.parquet"))],ignore_index=True)
    for c in ['adt','qdt','sdt']: d[c]=pd.to_datetime(d[c],utc=True)
    d=d.sort_values(['usr','adt'])
    d['qw']=np.log1p((d['sdt']-d['qdt']).dt.total_seconds().clip(lower=0)); d['hour']=d['adt'].dt.hour
    for c in ['nnumr','cnumr']: d[c]=np.log1p(d[c])
    if target=='exit-state': d['tok']=(d['exit state'].astype(str).str.lower()=='failed').astype(np.int64); V=2
    else: d['tok']=qbin(d['duration'],K); V=K
    feats=['nnumr','cnumr','freq_req','pri','qw','hour']; st,sf=[],[]
    for u,g in d.groupby('usr',sort=False):
        if len(g)<8: continue
        st.append(g['tok'].values); sf.append(g[feats].values.astype(np.float32))
    return st,sf,V
def burst():
    K=32; files=sorted(glob.glob(DATA+"/burstgpt/*.csv"))
    d=pd.concat([pd.read_csv(f,usecols=['Timestamp','Model','Request tokens','Response tokens','Log Type']) for f in files],
                ignore_index=True).sort_values('Timestamp')
    d['ia']=np.log1p(d['Timestamp'].diff().clip(lower=0).fillna(0)); d['req']=np.log1p(d['Request tokens'])
    d['gpt4']=(d['Model']=='GPT-4').astype(float); d['api']=(d['Log Type']=='API log').astype(float)
    d['tok']=qbin(d['Response tokens'],K); feats=['req','gpt4','ia','api']
    return [d['tok'].values],[d[feats].values.astype(np.float32)],K
def cesnet():
    K=32; files=sorted(glob.glob(DATA+"/cesnet/**/agg_1_hour/*.csv",recursive=True)); frames=[]
    for f in files:
        c=pd.read_csv(f)
        if len(c)>=16: frames.append(c)
    allb=np.concatenate([f['n_bytes'].values for f in frames])
    q=np.quantile(allb[np.isfinite(allb)],np.linspace(0,1,K+1)[1:-1]); st,sf=[],[]
    for c in frames:
        tok=np.clip(np.digitize(c['n_bytes'].values,q),0,K-1).astype(np.int64)
        ft=np.stack([np.log1p(c['n_flows']),np.log1p(c['n_packets']),c['tcp_udp_ratio_bytes'].fillna(0),
                     c['dir_ratio_bytes'].fillna(0),np.log1p(c['avg_duration'].fillna(0)),(c['id_time']%24)],1).astype(np.float32)
        st.append(tok); sf.append(ft)
    return st,sf,K
def nids_cic():                        # CIC-IDS2017 per-source_ip attack-class sequence (imbalanced -> macro-F1 matters)
    f=glob.glob(DATA+"/nids/CICIDS_Flow.parquet")[0]
    fn=['Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets',
        'Total Length of Bwd Packets','Flow Bytes/s','Flow Packets/s','Down/Up Ratio']
    d=pd.read_parquet(f, columns=['source_ip','Timestamp','attack_label']+fn)
    d['t']=pd.to_datetime(d['Timestamp'],errors='coerce').astype('int64')
    d=d.dropna(subset=['t','attack_label'])
    cats=sorted(d['attack_label'].astype(str).unique()); cmap={c:i for i,c in enumerate(cats)}
    d['tok']=d['attack_label'].astype(str).map(cmap).astype(np.int64); V=len(cats)
    print(f"  [cic] classes={V}: {d['attack_label'].astype(str).value_counts().to_dict()}")
    for c in fn:
        d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        d[c]=np.log1p(d[c].clip(lower=0).fillna(0))
    d=d.sort_values(['source_ip','t']); st,sf=[],[]
    for ip,g in d.groupby('source_ip',sort=False):
        if len(g)<8: continue
        st.append(g['tok'].values); sf.append(g[fn].values.astype(np.float32))
    return st,sf,V

# =================== runner ===================
def run(name, loader, N=64, D=256, H=4, L=4, epochs=4):
    st,sf,V=loader()
    Xt,Xf,E=windows(st,sf,N,V)
    n_ent=int(E.max())+1
    if n_ent>=10:                                          # entity holdout: test = unseen entities
        rng=np.random.default_rng(0); ents=rng.permutation(n_ent); test_ents=ents[:max(1,int(n_ent*0.2))]
        mask=np.isin(E,test_ents); te=np.where(mask)[0]; tr=np.where(~mask)[0]; split=f'entity-holdout ({n_ent} entities)'
    else:                                                  # single/few streams: temporal last-20%
        cut=int(len(Xt)*0.8); tr=np.arange(cut); te=np.arange(cut,len(Xt)); split='temporal last-20%'
    F0=Xf.shape[2]; mu=Xf[tr].reshape(-1,F0).mean(0); sd=Xf[tr].reshape(-1,F0).std(0)+1e-6; Xf=(Xf-mu)/sd
    print(f"\n### {name}  windows={len(Xt):,}  (tr={len(tr):,}/te={len(te):,})  V={V}  N={N}  (split: {split})")
    rows=naive_and_gbm(Xt,Xf,tr,te,V,N)
    for kind in ['lstm','tfm','hstu']:
        rows.append(train_eval(kind,Xt,Xf,tr,te,V,N,D=D,H=H,L=L,epochs=epochs))
    print(f"{'model':<12}{'acc':>8}{'macroF1':>9}{'params(M)':>11}{'sec':>7}")
    for r in rows:
        print(f"{r['name']:<12}{r['acc']:>8.3f}{r['mf1']:>9.3f}{r.get('params',0):>11.1f}{r.get('sec',0):>7.0f}")
    best_base=max(r['acc'] for r in rows if r['name'] in ('marginal','persistence','Markov','GBM'))
    hstu=[r for r in rows if r['name']=='HSTU'][0]['acc']
    tfm=[r for r in rows if r['name']=='TFM'][0]['acc']
    print(f"  HSTU vs best-baseline: {hstu-best_base:+.3f} | HSTU vs matched-Transformer: {hstu-tfm:+.3f}")
    return rows

if __name__=='__main__':
    print("device:",dev, torch.cuda.get_device_name(0) if dev=='cuda' else '')
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    cfg=dict(N=int(os.environ.get('N',64)),D=int(os.environ.get('D',256)),H=int(os.environ.get('H',4)),
             L=int(os.environ.get('L',4)),epochs=int(os.environ.get('EP',4)))
    print("config:",cfg)
    if which in ('all','burst'):  run("BurstGPT/DC  next-response-len(32)", burst, **cfg)
    if which in ('all','fdata'):  run("F-DATA/HPC   next-job-duration(32)", fdata, **cfg)
    if which in ('all','cesnet'): run("CESNET/NET   next-tick-nbytes(32)", cesnet, **cfg)
    if which in ('all','cic'):    run("CIC-IDS2017/SEC  next-flow-attack", nids_cic, **cfg)
