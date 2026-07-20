"""Faithful minimal HSTU (Actions Speak Louder than Words, arXiv 2402.17152) — dense/causal, MPS.
Validates the PAPER'S method (generative sequential transduction, autoregressive next-EVENT prediction)
on 3 modern large datasets, vs marginal / Markov / persistence baselines. Token = next event's (quantized) label."""
import sys, time, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
DATA="/private/tmp/claude-501/-Users-zifanzhang-Library-Mobile-Documents-com-apple-CloudDocs-Reseach-Overleaf/60cef121-2a9d-48ea-a7f0-152d82a310c7/scratchpad/data"

# ---------------- faithful HSTU block ----------------
class RelPosBias(nn.Module):
    def __init__(self, N): super().__init__(); self.N=N; self.w=nn.Parameter(torch.empty(2*N-1).normal_(0,0.02))
    def forward(self):
        N=self.N; t=F.pad(self.w[:2*N-1],[0,N]).repeat(N); t=t[:-N].reshape(1,N,3*N-2); r=(2*N-1)//2
        return t[:,:,r:-r]                                   # [1,N,N]
class HSTULayer(nn.Module):
    def __init__(self, D,dqk,dv,H,N,p=0.1):
        super().__init__(); self.D,self.dqk,self.dv,self.H,self.p=D,dqk,dv,H,p
        self.W=nn.Parameter(torch.empty(D,(2*dv+2*dqk)*H).normal_(0,0.02))
        self.o=nn.Linear(dv*H,D); nn.init.xavier_uniform_(self.o.weight); self.rel=RelPosBias(N)
    def forward(self,x,mask):                                # x[B,N,D], mask[N,N]
        B,N,_=x.shape; H,dqk,dv=self.H,self.dqk,self.dv
        uvqk=F.silu(F.layer_norm(x,[self.D])@self.W)         # SiLU pointwise projection
        u,v,q,k=torch.split(uvqk,[dv*H,dv*H,dqk*H,dqk*H],-1)
        q=q.view(B,N,H,dqk); k=k.view(B,N,H,dqk); v=v.view(B,N,H,dv)
        qk=torch.einsum('bnhd,bmhd->bhnm',q,k)+self.rel().unsqueeze(1)
        a=(F.silu(qk)/N)*mask.view(1,1,N,N)                  # SiLU-normalized attention (not softmax)
        av=torch.einsum('bhnm,bmhd->bnhd',a,v).reshape(B,N,H*dv)
        return self.o(F.dropout(u*F.layer_norm(av,[dv*H]),self.p,self.training))+x   # U-gating + residual
class HSTU(nn.Module):
    def __init__(self,V,nfeat,D=64,dqk=32,dv=32,H=2,L=2,N=64):
        super().__init__(); self.N=N
        self.tok=nn.Embedding(V+1,D,padding_idx=V); self.feat=nn.Linear(nfeat,D) if nfeat>0 else None
        self.layers=nn.ModuleList([HSTULayer(D,dqk,dv,H,N) for _ in range(L)]); self.head=nn.Linear(D,V)
        self.register_buffer('mask',torch.tril(torch.ones(N,N)))
    def forward(self,tok,feat=None):
        x=self.tok(tok)
        if self.feat is not None and feat is not None: x=x+self.feat(feat)
        for l in self.layers: x=l(x,self.mask)
        return self.head(x)

# ---------------- windowing + train/eval ----------------
def windows(seqs_tok, seqs_feat, N, V):
    T,Ffl=[],[]
    for tk,ft in zip(seqs_tok,seqs_feat):
        for s in range(0,len(tk),N):
            wt=tk[s:s+N]; wf=ft[s:s+N]
            if len(wt)<8: continue
            pad=N-len(wt)
            T.append(np.concatenate([wt,np.full(pad,V)]))
            Ffl.append(np.concatenate([wf,np.zeros((pad,ft.shape[1]))]))
    return np.array(T,np.int64), np.array(Ffl,np.float32)

def run(name, target, seqs_tok, seqs_feat, V, N=64, epochs=2, bs=256):
    Xt,Xf=windows(seqs_tok,seqs_feat,N,V)
    idx=np.random.permutation(len(Xt)); cut=int(len(Xt)*0.8); tr,te=idx[:cut],idx[cut:]
    nfeat=Xf.shape[2]
    # standardize feats on train
    mu=Xf[tr].reshape(-1,nfeat).mean(0); sd=Xf[tr].reshape(-1,nfeat).std(0)+1e-6
    Xf=(Xf-mu)/sd
    def batch(ids):
        for i in range(0,len(ids),bs):
            j=ids[i:i+bs]
            yield (torch.tensor(Xt[j]).to(dev), torch.tensor(Xf[j]).to(dev))
    m=HSTU(V,nfeat,N=N).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    t0=time.time()
    for ep in range(epochs):
        m.train()
        for tok,ft in batch(np.random.permutation(tr)):
            logit=m(tok,ft)[:, :-1]                          # predict next token at each pos
            tgt=tok[:,1:]
            loss=F.cross_entropy(logit.reshape(-1,V),tgt.reshape(-1),ignore_index=V)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval HSTU
    m.eval(); corr=tot=0; import math; nll=0.0
    with torch.no_grad():
        for tok,ft in batch(te):
            logit=m(tok,ft)[:, :-1]; tgt=tok[:,1:]; valid=tgt!=V
            pred=logit.argmax(-1)
            corr+=((pred==tgt)&valid).sum().item(); tot+=valid.sum().item()
            nll+=F.cross_entropy(logit.reshape(-1,V),tgt.reshape(-1),ignore_index=V,reduction='sum').item()
    acc_h=corr/tot; ppl=math.exp(nll/tot)
    # baselines from (current,next) pairs
    def pairs(ids):
        cur,nxt=[],[]
        for j in ids:
            t=Xt[j]; v=t!=V
            for p in range(N-1):
                if v[p] and v[p+1]: cur.append(t[p]); nxt.append(t[p+1])
        return np.array(cur),np.array(nxt)
    ctr,ntr=pairs(tr); cte,nte=pairs(te)
    maj=np.bincount(ntr,minlength=V).argmax(); acc_marg=(nte==maj).mean()
    # markov
    M=np.zeros((V,V));
    for c,n in zip(ctr,ntr): M[c,n]+=1
    mk=M.argmax(1); mk[M.sum(1)==0]=maj; acc_mk=(mk[cte]==nte).mean()
    acc_pers=(cte==nte).mean()                              # persistence
    print(f"[{name} | {target} | V={V}] windows={len(Xt):,} test_pos={tot:,} "
          f"train_time={time.time()-t0:.0f}s")
    print(f"    ACC  marginal={acc_marg:.3f}  persistence={acc_pers:.3f}  Markov={acc_mk:.3f}  |  "
          f"HSTU={acc_h:.3f}  (ppl={ppl:.2f})")
    best=max(acc_marg,acc_pers,acc_mk)
    print(f"    VERDICT: HSTU {'BEATS' if acc_h>best+0.005 else 'ties/loses to'} best baseline "
          f"({acc_h-best:+.3f}) — {'paradigm shows real signal' if acc_h>best+0.005 else 'no gain here'}")
    return acc_h,best

def qbin(x, K):                                            # global quantile bucketize -> K tokens
    x=np.asarray(x,float); q=np.quantile(x[np.isfinite(x)],np.linspace(0,1,K+1)[1:-1])
    return np.clip(np.digitize(x,q),0,K-1).astype(np.int64)

# ---------------- dataset adapters ----------------
def fdata(target):
    K=32
    d=pd.read_parquet(DATA+"/fdata/23_09.parquet",
        columns=['usr','adt','exit state','duration','nnumr','cnumr','freq_req','pri','qdt','sdt'])
    for c in ['adt','qdt','sdt']: d[c]=pd.to_datetime(d[c], utc=True)
    d=d.sort_values(['usr','adt'])
    d['qw']=np.log1p((d['sdt']-d['qdt']).dt.total_seconds().clip(lower=0)); d['hour']=d['adt'].dt.hour
    for c in ['nnumr','cnumr']: d[c]=np.log1p(d[c])
    if target=='exit-state': d['tok']=(d['exit state'].astype(str).str.lower()=='failed').astype(np.int64); V=2
    else: d['tok']=qbin(d['duration'],K); V=K
    feats=['nnumr','cnumr','freq_req','pri','qw','hour']
    st,sf=[],[]
    for u,g in d.groupby('usr',sort=False):
        if len(g)<8: continue
        st.append(g['tok'].values); sf.append(g[feats].values.astype(np.float32))
    return st,sf,V

def burst():
    K=32; d=pd.read_csv(DATA+"/burstgpt/BurstGPT_3.csv",
        usecols=['Timestamp','Model','Request tokens','Response tokens','Log Type']).sort_values('Timestamp')
    d=d.iloc[:800000]                                      # slice for laptop; full file=5.3M
    d['ia']=np.log1p(d['Timestamp'].diff().clip(lower=0).fillna(0))
    d['req']=np.log1p(d['Request tokens']); d['gpt4']=(d['Model']=='GPT-4').astype(float)
    d['api']=(d['Log Type']=='API log').astype(float)
    d['tok']=qbin(d['Response tokens'],K)
    feats=['req','gpt4','ia','api']
    return [d['tok'].values],[d[feats].values.astype(np.float32)],K   # global stream

def cesnet():
    K=32; files=sorted(glob.glob(DATA+"/cesnet/ip_addresses_sample/agg_1_hour/*.csv"))
    frames=[]
    for f in files:
        c=pd.read_csv(f)
        if len(c)<16: continue
        frames.append(c)
    allb=np.concatenate([f['n_bytes'].values for f in frames])
    q=np.quantile(allb[np.isfinite(allb)],np.linspace(0,1,K+1)[1:-1])   # GLOBAL bins across IPs
    st,sf=[],[]
    for c in frames:
        tok=np.clip(np.digitize(c['n_bytes'].values,q),0,K-1).astype(np.int64)
        ft=np.stack([np.log1p(c['n_flows']),np.log1p(c['n_packets']),
                     c['tcp_udp_ratio_bytes'].fillna(0),c['dir_ratio_bytes'].fillna(0),
                     np.log1p(c['avg_duration'].fillna(0)),(c['id_time']%24)],1).astype(np.float32)
        st.append(tok); sf.append(ft)
    return st,sf,K

def nids(which):
    f=DATA+("/nids/UNSW_Flow.parquet" if which=='unsw' else "/nids/CICIDS_Flow.parquet")
    tcol='stime' if which=='unsw' else 'Timestamp'
    fn=(['dur','sbytes','dbytes','spkts','dpkts','sload','dload','sttl','dttl'] if which=='unsw'
        else ['Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets',
              'Total Length of Bwd Packets','Flow Bytes/s','Flow Packets/s','Down/Up Ratio'])
    d=pd.read_parquet(f, columns=['source_ip',tcol,'attack_label']+fn)
    d['t']=(pd.to_numeric(d[tcol],errors='coerce') if which=='unsw'
            else pd.to_datetime(d[tcol],errors='coerce').astype('int64'))
    d=d.dropna(subset=['t','attack_label'])
    cats=sorted(d['attack_label'].astype(str).unique()); cmap={c:i for i,c in enumerate(cats)}
    d['tok']=d['attack_label'].astype(str).map(cmap).astype(np.int64); V=len(cats)
    print(f"  [{which}] classes={V}: {d['attack_label'].astype(str).value_counts().to_dict()}")
    for c in fn:
        d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        d[c]=np.log1p(d[c].clip(lower=0).fillna(0))
    d=d.sort_values(['source_ip','t'])
    st,sf=[],[]
    for ip,g in d.groupby('source_ip',sort=False):
        if len(g)<8: continue
        st.append(g['tok'].values); sf.append(g[fn].values.astype(np.float32))
    print(f"  [{which}] hosts(seq>=8)={len(st):,} flows={sum(len(s) for s in st):,}")
    return st,sf,V

if __name__=='__main__':
    print("device:",dev)
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    if which in ('all','fdata'):
        st,sf,V=fdata('exit-state'); run("F-DATA/HPC","next-job-fail(2cls)",st,sf,V)
        st,sf,V=fdata('duration');   run("F-DATA/HPC","next-job-duration(32)",st,sf,V)
    if which in ('all','burst'):
        st,sf,V=burst();  run("BurstGPT/DC","next-response-len(32)",st,sf,V)
    if which in ('all','cesnet'):
        st,sf,V=cesnet(); run("CESNET/NET","next-tick-nbytes(32)",st,sf,V)
    if which in ('all','unsw'):
        st,sf,V=nids('unsw'); run("UNSW-NB15/SEC","next-flow-attack(10cls)",st,sf,V)
    if which in ('all','cic'):
        st,sf,V=nids('cic'); run("CIC-IDS2017/SEC","next-flow-attack(15cls)",st,sf,V)
