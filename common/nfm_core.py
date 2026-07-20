"""NFM core: our-vocab tokenizer (bucketize) + time-aware HSTU (rab^{p,t}, two bucketizes)
+ standard causal Transformer + autoregressive multi-horizon rollout + MASE.
Faithful to meta-recsys/generative-recommenders + arXiv 2402.17152 (Eq1-3, SiLU-attn/N, U-gate).
Smoke test (synthetic) in __main__."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled=False    # cudnn ver mismatch after torch reinstall breaks LSTM flatten; RNN uses native path (HSTU/TFM use cuBLAS, unaffected)
# rab^{p,t} ablation switches (P0-2): USE_TIME=0 -> position-only HSTU (isolates the time-gap bias);
# USE_POS=0 -> time-only. Default both on = full HSTU. Set via env BEFORE importing this module.
_USE_POS  = os.environ.get('USE_POS','1')=='1'
_USE_TIME = os.environ.get('USE_TIME','1')=='1'

# ===================== tokenizer: OUR vocab via bucketize =====================
class Quantizer:
    """Bucketize continuous values into K quantile bins = our token vocabulary."""
    def __init__(self,K=1024): self.K=K
    def fit(self,vals):
        v=np.asarray(vals,float); v=v[np.isfinite(v)]
        self.edges=np.quantile(v,np.linspace(0,1,self.K+1)[1:-1])          # K-1 internal edges
        tok=np.clip(np.digitize(v,self.edges),0,self.K-1)
        self.centers=np.array([np.median(v[tok==k]) if (tok==k).any() else
                               (self.edges[min(k,len(self.edges)-1)] if len(self.edges) else 0.0)
                               for k in range(self.K)],dtype=np.float64)
        return self
    def to_tok(self,vals): return np.clip(np.digitize(np.asarray(vals,float),self.edges),0,self.K-1).astype(np.int64)
    def to_val(self,tok):  return self.centers[np.clip(np.asarray(tok),0,self.K-1)]

# ===================== rab^{p,t}: position + TIME-bucket bias =====================
class RelBias(nn.Module):
    def __init__(self,N,nb=128):
        super().__init__(); self.N=N; self.nb=nb
        self.pos=nn.Parameter(torch.empty(2*N-1).normal_(0,0.02))          # positional (p)
        self.tsw=nn.Parameter(torch.empty(nb+1).normal_(0,0.02))           # time buckets (t)
    def forward(self,ts):                                                  # ts [B,N] int64
        N=self.N; B=ts.shape[0]
        t=F.pad(self.pos[:2*N-1],[0,N]).repeat(N); t=t[:-N].reshape(1,N,3*N-2); r=(2*N-1)//2
        pos_bias=t[:,:,r:-r]                                               # [1,N,N]
        ext=torch.cat([ts,ts[:,N-1:N]],dim=1)                             # [B,N+1]
        delta=(ext[:,1:].unsqueeze(2)-ext[:,:-1].unsqueeze(1)).float()     # [B,N,N] pairwise Δt
        buck=torch.clamp((torch.log(delta.abs().clamp(min=1))/0.301).long(),0,self.nb)  # repo's fn, 128 buckets
        ts_bias=torch.index_select(self.tsw,0,buck.view(-1)).view(B,N,N)
        return pos_bias*(1.0 if _USE_POS else 0.0)+ts_bias*(1.0 if _USE_TIME else 0.0)  # ablation-gated [B,N,N]

class HSTULayer(nn.Module):
    def __init__(self,D,H,N,nb=128,p=0.1):
        super().__init__(); self.D,self.H,self.p=D,H,p; self.dqk=self.dv=D//H
        self.W=nn.Parameter(torch.empty(D,(2*self.dv+2*self.dqk)*H).normal_(0,0.02))
        self.o=nn.Linear(self.dv*H,D); nn.init.xavier_uniform_(self.o.weight); self.rel=RelBias(N,nb)
    def forward(self,x,ts,mask):
        B,N,_=x.shape; H,dqk,dv=self.H,self.dqk,self.dv
        uvqk=F.silu(F.layer_norm(x,[self.D])@self.W)                       # Eq1
        u,v,q,k=torch.split(uvqk,[dv*H,dv*H,dqk*H,dqk*H],-1)
        q=q.view(B,N,H,dqk); k=k.view(B,N,H,dqk); v=v.view(B,N,H,dv)
        qk=torch.einsum('bnhd,bmhd->bhnm',q,k)+self.rel(ts).unsqueeze(1)   # + rab^{p,t}
        a=(F.silu(qk)/N)*mask                                             # Eq2: SiLU-normalized (not softmax)
        av=torch.einsum('bhnm,bmhd->bnhd',a,v).reshape(B,N,H*dv)
        return self.o(F.dropout(u*F.layer_norm(av,[dv*H]),self.p,self.training))+x   # Eq3: U-gate

class HSTU(nn.Module):                                     # BACKBONE (task-agnostic) + SEPARATE swappable task HEAD
    def __init__(self,V,D=256,H=4,L=4,N=64,nb=128):
        super().__init__(); self.N=N; self.V=V; self.D=D; self.tok=nn.Embedding(V+1,D,padding_idx=V)
        self.layers=nn.ModuleList([HSTULayer(D,H,N,nb) for _ in range(L)])
        self.register_buffer('mask',torch.tril(torch.ones(N,N)).view(1,1,N,N))
        self.head=nn.Linear(D,V)                           # task head (untied, swappable per downstream task)
    def backbone(self,tok,ts):                             # -> [B,N,D] hidden states
        x=self.tok(tok)
        for l in self.layers: x=l(x,ts,self.mask)
        return x
    def forward(self,tok,ts): return self.head(self.backbone(tok,ts))
    def freeze_backbone(self):                             # for downstream finetune: freeze backbone, train only head/inserted module
        for p in self.tok.parameters(): p.requires_grad_(False)
        for p in self.layers.parameters(): p.requires_grad_(False)
        for l in self.layers: l.p=0.0                      # disable dropout on frozen layers: head is a linear probe on
        return self                                        #   DETERMINISTIC features (train_gen calls .train(), else eval-time mismatch)
    def swap_head(self,Vnew):                              # attach a fresh task-specific head
        self.head=nn.Linear(self.D,Vnew).to(self.tok.weight.device); return self

class TFM(nn.Module):                                                     # standard causal softmax transformer
    def __init__(self,V,D=256,H=4,L=4,N=64,**kw):
        super().__init__(); self.N=N; self.H=H; self.tok=nn.Embedding(V+1,D,padding_idx=V)
        self.pos=nn.Embedding(N,D)
        self.ln1=nn.ModuleList([nn.LayerNorm(D) for _ in range(L)]); self.ln2=nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.qkv=nn.ModuleList([nn.Linear(D,3*D) for _ in range(L)]); self.o=nn.ModuleList([nn.Linear(D,D) for _ in range(L)])
        self.mlp=nn.ModuleList([nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)) for _ in range(L)])
        self.head=nn.Linear(D,V)
    def forward(self,tok,ts=None):
        B,N=tok.shape; H=self.H; D=self.tok.embedding_dim; hd=D//H
        x=self.tok(tok)+self.pos(torch.arange(N,device=tok.device))[None]
        for i in range(len(self.qkv)):
            q,k,v=self.qkv[i](self.ln1[i](x)).view(B,N,3,H,hd).permute(2,0,3,1,4).unbind(0)
            a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+self.o[i](a.transpose(1,2).reshape(B,N,D)); x=x+self.mlp[i](self.ln2[i](x))
        return self.head(x)

class RNNGen(nn.Module):                                                  # LSTM/GRU generative baseline (classic net-traffic predictor)
    def __init__(self,V,D=256,H=4,L=4,N=64,cell='lstm',**kw):
        super().__init__(); self.N=N; L=min(L,2)        # cap RNN depth: deep LSTM/GRU don't train (esp cudnn off)
        self.tok=nn.Embedding(V+1,D,padding_idx=V); self.pos=nn.Embedding(N,D)
        R=nn.LSTM if cell=='lstm' else nn.GRU
        self.rnn=R(D,D,L,batch_first=True,dropout=0.1 if L>1 else 0.0); self.head=nn.Linear(D,V)
    def forward(self,tok,ts=None):
        x=self.tok(tok)+self.pos(torch.arange(tok.shape[1],device=tok.device))[None]
        x,_=self.rnn(x); return self.head(x)
def LSTMGen(V,**kw): return RNNGen(V,cell='lstm',**kw)
def GRUGen(V,**kw):  return RNNGen(V,cell='gru',**kw)

# ===================== train + rollout + MASE =====================
def train_gen(model, Tok, Ts, V, epochs=4, bs=256, lr=2e-3):
    m=model.to(dev); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr,weight_decay=0.01)  # respects freeze_backbone()
    for ep in range(epochs):
        m.train(); idx=np.random.permutation(len(Tok))
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]; tk=torch.tensor(Tok[j]).to(dev); ts=torch.tensor(Ts[j]).to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                lg=m(tk,ts)[:,:-1]; loss=F.cross_entropy(lg.reshape(-1,V),tk[:,1:].reshape(-1),ignore_index=V)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def rollout(m, Ctx, TsFull, V, CTX, HMAX, centers, scale=None, bs=512):   # EXPECTED-VALUE forecast (MAE-fair, cf Chronos median); argmax fed back for continuation
    m.eval(); N=CTX+HMAX; cen=torch.tensor(np.asarray(centers,dtype=np.float32),device=dev); out=[]
    for i in range(0,len(Ctx),bs):
        c=torch.tensor(Ctx[i:i+bs]).to(dev); ts=torch.tensor(TsFull[i:i+bs]).to(dev); B=c.shape[0]
        buf=torch.full((B,N),V,device=dev,dtype=torch.long); buf[:,:CTX]=c; vals=[]
        for p in range(CTX,N):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                lg=m(buf,ts)[:,p-1].float()
            probs=lg.softmax(-1); med=(probs.cumsum(-1)<0.5).sum(-1).clamp(0,cen.numel()-1)
            buf[:,p]=med; vals.append(cen[med])        # MEDIAN forecast (MAE-optimal for MASE; robust to skew, unlike mean)
        out.append(torch.stack(vals,1).cpu().numpy())
    fc=np.concatenate(out)                              # [n,HMAX] median forecast (normalized if scale given)
    return fc*np.asarray(scale)[:,None] if scale is not None else fc

@torch.no_grad()
def rank_metrics(m, Cx, TsF, nxt, K, CTX, ks=(1,5,10), bs=512):   # HSTU-style 1-step next-token ranking (HR@K/NDCG@K/MRR)
    m.eval(); N=TsF.shape[1]; hr={k:0.0 for k in ks}; nd={k:0.0 for k in ks}; mrr=0.0; tot=0
    for i in range(0,len(Cx),bs):
        c=torch.tensor(Cx[i:i+bs]).to(dev); ts=torch.tensor(TsF[i:i+bs]).to(dev); B=c.shape[0]
        buf=torch.full((B,N),K,device=dev,dtype=torch.long); buf[:,:CTX]=c
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            lg=m(buf,ts)[:,CTX-1].float()                # [B,K] next-token logits
        tgt=torch.tensor(nxt[i:i+bs]).to(dev).long()
        rank=(lg>lg.gather(1,tgt[:,None])).sum(1)+1      # 1-indexed rank of the true token
        for k in ks:
            hit=(rank<=k).float(); hr[k]+=hit.sum().item(); nd[k]+=(hit/torch.log2(rank.float()+1)).sum().item()
        mrr+=(1.0/rank.float()).sum().item(); tot+=B
    return {**{f'HR@{k}':hr[k]/tot for k in ks}, **{f'NDCG@{k}':nd[k]/tot for k in ks}, 'MRR':mrr/tot}

def mase(pred_val, true_val, scale):                    # scale = scalar global 1-step-naive MAE (standard MASE denom)
    ae=np.abs(np.asarray(pred_val)-np.asarray(true_val)) # [n,HMAX]
    if np.isscalar(scale): return ae.mean(0)/max(float(scale),1e-8)
    return (ae/np.maximum(np.asarray(scale)[:,None],1e-8)).mean(0)

# ===================== synthetic smoke =====================
if __name__=='__main__':
    print("device:",dev); rng=np.random.default_rng(0)
    K=256; CTX=32; HMAX=16; N=CTX+HMAX
    # synthetic: AR(1)+seasonal random-walk-ish value series, regular timestamps
    seqs=[];
    for _ in range(400):
        L=rng.integers(N, N*4); e=rng.normal(0,1,L); v=np.cumsum(e)+3*np.sin(np.arange(L)/6.0); seqs.append(v)
    q=Quantizer(K).fit(np.concatenate(seqs))
    # gen windows (next-token) with timestamps = position index
    Tok=[]; Ts=[]
    for v in seqs:
        tk=q.to_tok(v)
        for s in range(0,len(tk),N):
            w=tk[s:s+N]
            if len(w)>=8:
                pad=N-len(w); Tok.append(np.concatenate([w,np.full(pad,K)])); Ts.append(np.arange(N))
    Tok=np.array(Tok); Ts=np.array(Ts,np.int64)
    # eval windows
    Cx=[]; Fu=[]; TsF=[]; sc=[]
    for v in seqs[-80:]:
        tk=q.to_tok(v)
        for s in range(0,len(tk)-N+1,N):
            Cx.append(tk[s:s+CTX]); Fu.append(v[s+CTX:s+CTX+HMAX]); TsF.append(np.arange(N))
            sc.append(np.mean(np.abs(np.diff(v[s:s+CTX]))))
    Cx=np.array(Cx); Fu=np.array(Fu); TsF=np.array(TsF,np.int64); sc=np.array(sc)
    print(f"vocab K={K}  gen-windows={len(Tok)}  eval-windows={len(Cx)}")
    for name,Model in [('HSTU',HSTU),('TFM',TFM)]:
        m=train_gen(Model(K,D=128,H=4,L=3,N=N),Tok,Ts,K,epochs=3)
        pt=rollout(m,Cx,TsF,K,CTX,HMAX); pv=q.to_val(pt)
        ms=mase(pv,Fu,sc)
        print(f"  {name}: MASE h1={ms[0]:.3f} h4={ms[3]:.3f} h8={ms[7]:.3f} h16={ms[15]:.3f} | params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
        del m; torch.cuda.empty_cache() if dev=='cuda' else None
    # persistence baseline
    per=q.to_val(np.repeat(Cx[:,-1:],HMAX,axis=1)); print(f"  persistence: MASE h1={mase(per,Fu,sc)[0]:.3f} h16={mase(per,Fu,sc)[15]:.3f}")
