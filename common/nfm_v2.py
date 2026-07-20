"""NFM v2 backbone (redesign after the 2026-07-16 gate/pre-mortem): keeps HSTU's SiLU pointwise
attention + U-gate, but (1) replaces the coarse log-bucket time bias with a CONTINUOUS multi-scale
Fourier bias of the exact gap, (2) makes the output HEAD a pluggable heavy-tail mixture density
(continuous targets) or categorical (discrete), (3) adds LayerScale + grad-clip for depth scaling.
ONE backbone, pluggable input/head => same `-log p(next)` surprise serves forecasting AND detection.
Faithful HSTU core: `SiLU(QK^T + bias)/N` attention, U-gate, causal mask, arXiv 2402.17152.
"""
import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled=False
_USE_POS  = os.environ.get('USE_POS','1')=='1'      # rab ablation gates (P0-2), read at import
_USE_TIME = os.environ.get('USE_TIME','1')=='1'

# ===================== continuous multi-scale time bias (replaces coarse log-buckets) =====================
class ContinuousRelBias(nn.Module):
    """Drop-in for RelBias. Positional Toeplitz component kept; TIME bias = continuous multi-scale
    Fourier features of the exact causal gap Δt, projected PER-HEAD. Returns [B,H,N,N].
    (XTSFormer 2402.02258 / ContiFormer 2402.10635 / Time2Vec-style; the coarse log-bucket can't
    resolve heavy-tailed HPC inter-arrival gaps -- our ablation showed the bucket bias was inconsistent.)"""
    def __init__(self, N, H, Fr=16):
        super().__init__(); self.N=N; self.H=H; self.Fr=Fr
        self.pos = nn.Parameter(torch.empty(2*N-1).normal_(0,0.02))
        w = torch.exp(torch.linspace(math.log(1e-2), math.log(4.0), Fr))     # geometric multi-scale freqs
        self.w   = nn.Parameter(w); self.phi = nn.Parameter(torch.zeros(Fr))
        self.mix = nn.Linear(2*Fr+1, H)
    def forward(self, ts):                                                   # ts [B,N] -> [B,H,N,N]
        B,N = ts.shape
        t=F.pad(self.pos[:2*N-1],[0,N]).repeat(N); t=t[:-N].reshape(1,N,3*N-2); r=(2*N-1)//2
        pos_bias=t[:,:,r:-r].unsqueeze(1)                                    # [1,1,N,N]
        d=(ts[:,:,None]-ts[:,None,:]).float().clamp(min=0)                   # [B,N,N] causal gap
        g=torch.log1p(d)                                                     # log-compress heavy tail
        ang=g.unsqueeze(-1)*self.w+self.phi                                  # [B,N,N,Fr]
        feat=torch.cat([torch.sin(ang),torch.cos(ang),g.unsqueeze(-1)],-1)  # [B,N,N,2Fr+1]
        time_bias=self.mix(feat).permute(0,3,1,2)                           # [B,H,N,N]
        return pos_bias*(1.0 if _USE_POS else 0.0)+time_bias*(1.0 if _USE_TIME else 0.0)

class HSTULayer2(nn.Module):
    """Faithful HSTU layer (SiLU pointwise attn + U-gate) + LayerScale for deep stacks.
    ffn=True adds a SwiGLU MFFN after the U-gate (FuXi-alpha, arXiv 2502.03036, +13% NDCG over HSTU) = HSTU++."""
    def __init__(self, D, H, p=0.1, ls_init=1e-4, ffn=False):
        super().__init__(); self.D,self.H,self.p=D,H,p; self.dqk=self.dv=D//H
        self.W=nn.Parameter(torch.empty(D,(2*self.dv+2*self.dqk)*H).normal_(0,0.02))
        self.o=nn.Linear(self.dv*H,D); nn.init.xavier_uniform_(self.o.weight)
        self.ls=nn.Parameter(torch.full((D,),ls_init))
        self.ffn=ffn
        if ffn:
            self.ln2=nn.LayerNorm(D); self.w1=nn.Linear(D,4*D); self.w2=nn.Linear(D,4*D); self.w3=nn.Linear(4*D,D)
            self.ls2=nn.Parameter(torch.full((D,),ls_init))
    def forward(self, x, bias, mask):
        B,N,_=x.shape; H,dqk,dv=self.H,self.dqk,self.dv
        uvqk=F.silu(F.layer_norm(x,[self.D])@self.W)
        u,v,q,k=torch.split(uvqk,[dv*H,dv*H,dqk*H,dqk*H],-1)
        q=q.view(B,N,H,dqk); k=k.view(B,N,H,dqk); v=v.view(B,N,H,dv)
        qk=torch.einsum('bnhd,bmhd->bhnm',q,k)+bias                          # + rab^{p,t} (per-head)
        a=(F.silu(qk)/N)*mask                                                # SiLU-normalized (not softmax)
        av=torch.einsum('bhnm,bmhd->bnhd',a,v).reshape(B,N,H*dv)
        x=self.o(F.dropout(u*F.layer_norm(av,[dv*H]),self.p,self.training))*self.ls + x
        if self.ffn:                                                          # SwiGLU FFN (HSTU++)
            h=self.ln2(x); x=self.w3(F.silu(self.w1(h))*self.w2(h))*self.ls2 + x
        return x

class HSTULayerDT(nn.Module):
    """HSTU layer with a DECOUPLED time channel. The semantic channel keeps SiLU(q.k + pos_bias) attention;
    a SEPARATE time channel has its own value projection and an attention map built ONLY from the real
    inter-event gaps via per-head power-law retention r_h^{log1p(dt)} * cos(theta_h * log1p(dt)) (long-memory
    decay + log-periodicity). The two channels are concatenated and U-gated, so timing drives its OWN pathway
    instead of being added into the content-matching logits (which contaminates semantic retrieval). Strictly
    more expressive than a scalar additive time bias. Mechanism: FuXi-Linear temporal retention (arXiv 2602.23671)
    in the FuXi-alpha AMS concat+gate structure (arXiv 2502.03036)."""
    def __init__(self, D, H, p=0.1, ls_init=1e-4, ffn=False):
        super().__init__(); self.D,self.H,self.p=D,H,p; self.dqk=self.dv=D//H
        self.W=nn.Parameter(torch.empty(D,(2*self.dv+2*self.dqk)*H).normal_(0,0.02))
        self.Wt=nn.Linear(D,self.dv*H)                                         # time-channel value V_t
        self.log_r=nn.Parameter(torch.linspace(-0.5,-3.0,H))                   # per-head decay (r=exp(log_r) in (0,1))
        self.theta=nn.Parameter(torch.linspace(0.0,2.0,H))                     # per-head log-periodicity
        self.merge=nn.Linear(2*self.dv*H,self.dv*H)                            # concat(semantic,time) -> dv*H
        self.o=nn.Linear(self.dv*H,D); nn.init.xavier_uniform_(self.o.weight)
        self.ls=nn.Parameter(torch.full((D,),ls_init)); self.ffn=ffn
        if ffn:
            self.ln2=nn.LayerNorm(D); self.w1=nn.Linear(D,4*D); self.w2=nn.Linear(D,4*D); self.w3=nn.Linear(4*D,D)
            self.ls2=nn.Parameter(torch.full((D,),ls_init))
    def forward(self, x, ldt, pos_bias, mask):                                # ldt = log1p(causal gap) [B,N,N]
        B,N,_=x.shape; H,dqk,dv=self.H,self.dqk,self.dv
        xn=F.layer_norm(x,[self.D]); uvqk=F.silu(xn@self.W)
        u,v,q,k=torch.split(uvqk,[dv*H,dv*H,dqk*H,dqk*H],-1)
        q=q.view(B,N,H,dqk); k=k.view(B,N,H,dqk); v=v.view(B,N,H,dv)
        qk=torch.einsum('bnhd,bmhd->bhnm',q,k)+pos_bias                        # semantic: POSITION bias only
        a=(F.silu(qk)/N)*mask
        av_sem=torch.einsum('bhnm,bmhd->bnhd',a,v).reshape(B,N,H*dv)
        vt=self.Wt(xn).view(B,N,H,dv)                                          # decoupled time channel
        r=torch.exp(self.log_r).clamp(max=0.999).view(1,H,1,1); th=self.theta.view(1,H,1,1)
        lb=ldt.unsqueeze(1)                                                    # [B,1,N,N]
        a_t=((r**lb)*torch.cos(th*lb)/N)*mask                                  # [B,H,N,N] power-law decay + log-periodic; /N matches the semantic channel's normalization (un-normalized sum grows with N)
        av_t=torch.einsum('bhnm,bmhd->bnhd',a_t,vt).reshape(B,N,H*dv)
        av=self.merge(torch.cat([av_sem,av_t],-1))
        x=self.o(F.dropout(u*F.layer_norm(av,[dv*H]),self.p,self.training))*self.ls + x
        if self.ffn:
            h=self.ln2(x); x=self.w3(F.silu(self.w1(h))*self.w2(h))*self.ls2 + x
        return x

# ===================== pluggable input adapters =====================
class CategoricalInput(nn.Module):                                          # LANL: dst-computer token
    def __init__(self, V, D): super().__init__(); self.emb=nn.Embedding(V+1,D,padding_idx=V)
    def forward(self, inp): return self.emb(inp)                            # inp: tok [B,N]

class MultiFieldInput(nn.Module):                                          # LANL enriched: several categorical fields
    def __init__(self, vocabs, D):                                         # vocabs: list of field sizes
        super().__init__(); self.embs=nn.ModuleList([nn.Embedding(v+1,D,padding_idx=v) for v in vocabs])
    def forward(self, inp): return sum(e(inp[i]) for i,e in enumerate(self.embs))   # inp: tuple of [B,N]

class ContinuousInput(nn.Module):                                          # F-DATA: continuous z + coarse regime bin
    def __init__(self, D, Kc=256):
        super().__init__(); self.val=nn.Linear(1,D); self.bin=nn.Embedding(Kc+1,D,padding_idx=Kc)
    def forward(self, inp):                                                # inp: (z [B,N] float, cbin [B,N] long)
        z,cbin=inp; return self.val(z.unsqueeze(-1).float())+self.bin(cbin)

# ===================== pluggable heads (unified: nll / surprise / point / quantiles) =====================
class CategoricalHead(nn.Module):
    def __init__(self, D, V): super().__init__(); self.V=V; self.proj=nn.Linear(D,V)
    def nll(self, h, tgt, valid):
        t=tgt.clamp(0,self.V-1)                                                # pad id == V would crash CE; padded positions are masked by valid
        ce=F.cross_entropy(self.proj(h).float().reshape(-1,self.V),t.reshape(-1),reduction='none').view_as(t)
        return (ce*valid).sum()/valid.sum().clamp(min=1)
    def surprise(self, h, tgt):
        return -F.log_softmax(self.proj(h).float(),-1).gather(-1,tgt.clamp(0,self.V-1).unsqueeze(-1)).squeeze(-1)

class MultiFieldHead(nn.Module):
    """Factorized categorical head over several fields (LANL rich event: dst / auth-type / logon-type / success).
    Joint log-likelihood factorizes conditionally on the shared hidden state; joint surprise = sum of per-field -log p.
    This lets the generative model score an event by the JOINT plausibility of (where, how, outcome), not dst alone."""
    def __init__(self, D, vocabs):
        super().__init__(); self.vocabs=list(vocabs)
        self.proj=nn.ModuleList([nn.Linear(D,V) for V in self.vocabs])
    def nll(self, h, tgts, valid):
        tot=0.0; den=valid.sum().clamp(min=1)
        for j,V in enumerate(self.vocabs):
            t=tgts[j].clamp(0,V-1)                                             # padded positions (id=V) -> masked by valid
            ce=F.cross_entropy(self.proj[j](h).float().reshape(-1,V),t.reshape(-1),reduction='none').view_as(t)
            tot=tot+(ce*valid).sum()/den
        return tot
    def surprise(self, h, tgts):                                              # [B,N] joint -log p
        s=0.0
        for j,V in enumerate(self.vocabs):
            lp=F.log_softmax(self.proj[j](h).float(),-1).gather(-1,tgts[j].clamp(0,V-1).unsqueeze(-1)).squeeze(-1)
            s=s-lp
        return s

class MixtureHead(nn.Module):
    """Student-t mixture over standardized target z (heavy-tail, ordinal-preserving). Toto recipe
    (arXiv 2505.14766): scale=softplus+eps, dof=2+softplus (>2 => finite mean/var). All math fp32."""
    def __init__(self, D, M=4, min_scale=1e-3, max_dof=100.0):
        super().__init__(); self.M=M; self.min_scale=min_scale; self.max_dof=max_dof
        self.proj=nn.Linear(D,4*M)
    def params(self, h):
        pl,mu,pt,pn=self.proj(h).float().chunk(4,-1)                        # [B,N,M] each, fp32
        nu=2.0+torch.clamp(F.softplus(pn),max=self.max_dof)
        return torch.log_softmax(pl,-1), mu, F.softplus(pt)+self.min_scale, nu
    def _logT(self, z, mu, tau, nu):                                       # z [...,1]
        zt=(z-mu)/tau
        return (torch.lgamma((nu+1)/2)-torch.lgamma(nu/2)-0.5*torch.log(nu*math.pi)
                -torch.log(tau)-((nu+1)/2)*torch.log1p(zt*zt/nu))
    def _logp(self, h, z):                                                 # z [B,N] -> [B,N]
        lpi,mu,tau,nu=self.params(h)
        return torch.logsumexp(lpi+self._logT(z.unsqueeze(-1),mu,tau,nu),-1)
    def nll(self, h, z, valid):
        lp=self._logp(h,z); return (-lp*valid).sum()/valid.sum().clamp(min=1)
    def surprise(self, h, z): return -self._logp(h,z)
    def _dist(self, h):
        lpi,mu,tau,nu=self.params(h)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=lpi), torch.distributions.StudentT(nu,mu,tau))
    @torch.no_grad()
    def quantiles(self, h, qs=(0.5,0.9,0.99), S=512):                          # S=512: P99 from 128 samples was too noisy
        # sample from the mixture and take empirical quantiles (robust: torch StudentT has no cdf)
        d=self._dist(h); samp=d.sample((S,)).float()                       # [S,B,N]
        qz=torch.quantile(samp, torch.tensor(qs,device=samp.device,dtype=samp.dtype), dim=0)  # [len(qs),B,N]
        return qz.permute(1,2,0)                                           # [B,N,len(qs)] in z-space
    @torch.no_grad()
    def point(self, h): return self.quantiles(h,(0.5,))[...,0]
    @torch.no_grad()
    def sample(self, h): return self._dist(h).sample()

class HorizonHead(nn.Module):
    """DIRECT multi-horizon distributional head: p(x_{t+h} | history_t, h). The horizon h enters as Fourier
    features of log1p(h) projected into a conditioning vector added to the hidden state; one shared Student-t
    MixtureHead then emits the distribution. One backbone forward serves EVERY (input-length, horizon) cell:
    no autoregressive rollout (the error-accumulation that loses to direct GBM specialists at long horizons)
    and no future timestamps needed at inference. Rollout remains available through h=1 chaining."""
    def __init__(self, D, M=4, Fh=8):
        super().__init__(); self.mix=MixtureHead(D,M)
        self.register_buffer('w', torch.exp(torch.linspace(math.log(0.25), math.log(4.0), Fh)))
        self.proj=nn.Linear(2*Fh+1, D)
    def cond(self, hid, h):                                                    # hid [B,T,D], h scalar int
        g=math.log1p(float(h)); ang=self.w*g
        f=torch.cat([torch.sin(ang),torch.cos(ang),torch.tensor([g],device=hid.device,dtype=self.w.dtype)])
        return hid+self.proj(f).to(hid.dtype)
    def nll(self, hid, h, z, valid): return self.mix.nll(self.cond(hid,h), z, valid)
    @torch.no_grad()
    def quantiles(self, hid, h, qs=(0.5,0.9,0.99), S=512): return self.mix.quantiles(self.cond(hid,h), qs, S)   # S>=4096 for tail (P99) estimates

# ===================== backbone (pluggable input+head) + training =====================
class HSTUv2(nn.Module):
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64, Fr=16, share_bias=True, ffn=False, decoupled_time=False):
        super().__init__(); self.N=N; self.D=D; self.share_bias=share_bias; self.decoupled_time=decoupled_time
        self.inp=input_adapter; self.head=head
        if decoupled_time:                                                  # separate time pathway (HSTULayerDT)
            self.rel=None; self.pos_par=nn.Parameter(torch.empty(2*N-1).normal_(0,0.02))
            self.layers=nn.ModuleList([HSTULayerDT(D,H,ffn=ffn) for _ in range(L)])
        else:
            self.rel=ContinuousRelBias(N,H,Fr) if share_bias else None
            self.layers=nn.ModuleList([HSTULayer2(D,H,ffn=ffn) for _ in range(L)])
            if not share_bias:
                for l in self.layers: l.rel=ContinuousRelBias(N,H,Fr)
        self.register_buffer('mask',torch.tril(torch.ones(N,N)).view(1,1,N,N))
    def backbone(self, inp, ts):
        x=self.inp(inp)
        if self.decoupled_time:                                             # POSITION bias (Toeplitz) + log1p gap channel
            N=self.N; p=self.pos_par
            t=F.pad(p[:2*N-1],[0,N]).repeat(N); t=t[:-N].reshape(1,N,3*N-2); rr=(2*N-1)//2
            pos_bias=t[:,:,rr:-rr].unsqueeze(1)                             # [1,1,N,N]
            ldt=torch.log1p((ts[:,:,None]-ts[:,None,:]).float().clamp(min=0))  # [B,N,N]
            for l in self.layers: x=l(x, ldt, pos_bias, self.mask)
            return x
        bias=self.rel(ts) if self.share_bias else None
        for l in self.layers: x=l(x, bias if self.share_bias else l.rel(ts), self.mask)
        return x                                                            # hidden states [B,N,D]
    def forward(self, inp, ts): return self.backbone(inp, ts)
    def freeze_backbone(self):
        for p in self.inp.parameters(): p.requires_grad_(False)
        for p in self.layers.parameters(): p.requires_grad_(False)
        if self.rel is not None:
            for p in self.rel.parameters(): p.requires_grad_(False)
        for l in self.layers: l.p=0.0
        return self

class TFMv2(nn.Module):
    """Matched causal Transformer with the SAME pluggable head (isolates backbone vs head)."""
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64):
        super().__init__(); self.N=N; self.H=H; self.inp=input_adapter; self.head=head
        self.pos=nn.Embedding(N,D)
        self.ln1=nn.ModuleList([nn.LayerNorm(D) for _ in range(L)]); self.ln2=nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.qkv=nn.ModuleList([nn.Linear(D,3*D) for _ in range(L)]); self.o=nn.ModuleList([nn.Linear(D,D) for _ in range(L)])
        self.mlp=nn.ModuleList([nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)) for _ in range(L)])
        self.ls1=nn.ParameterList([nn.Parameter(torch.full((D,),1e-4)) for _ in range(L)])   # LayerScale: stabilize deep stacks
        self.ls2=nn.ParameterList([nn.Parameter(torch.full((D,),1e-4)) for _ in range(L)])
    def backbone(self, inp, ts=None):
        x=self.inp(inp); B,N,D=x.shape; H=self.H; hd=D//H
        x=x+self.pos(torch.arange(N,device=x.device))[None]
        for i in range(len(self.qkv)):
            q,k,v=self.qkv[i](self.ln1[i](x)).view(B,N,3,H,hd).permute(2,0,3,1,4).unbind(0)
            a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+self.o[i](a.transpose(1,2).reshape(B,N,D))*self.ls1[i]; x=x+self.mlp[i](self.ln2[i](x))*self.ls2[i]
        return x
    def forward(self, inp, ts=None): return self.backbone(inp, ts)

class RNNv2(nn.Module):
    """LSTM/GRU backbone with pluggable input+head. NO positional embedding -> pure recurrence handles
    ANY sequence length natively (the cleanest any-input/any-output baseline)."""
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64, cell='lstm', **kw):
        super().__init__(); self.N=N; self.inp=input_adapter; self.head=head
        Lr=min(L,3); R=nn.LSTM if cell=='lstm' else nn.GRU
        self.rnn=R(D,D,Lr,batch_first=True,dropout=0.1 if Lr>1 else 0.0)
    def backbone(self, inp, ts=None):
        with torch.autocast('cuda',enabled=False):                          # nn.LSTM/GRU need consistent fp32 dtype
            x=self.inp(inp).float(); x,_=self.rnn(x)
        return x
    def forward(self, inp, ts=None): return self.backbone(inp, ts)
    def freeze_backbone(self):
        for p in self.inp.parameters(): p.requires_grad_(False)
        for p in self.rnn.parameters(): p.requires_grad_(False)
        return self
def LSTMv2(inp,head,**kw): return RNNv2(inp,head,cell='lstm',**kw)
def GRUv2(inp,head,**kw):  return RNNv2(inp,head,cell='gru',**kw)

class _RetBlock(nn.Module):
    """Retention (RetNet 2307.08621) with a CONTINUOUS per-head time-decay of the real gap Δt.
    Un-normalized (heavy-tail-friendly), like HSTU; the cleanest 'explicit time-decay prior' backbone."""
    def __init__(self, D, H):
        super().__init__(); self.H=H; self.hd=D//H
        self.ln=nn.LayerNorm(D); self.qkv=nn.Linear(D,3*D); self.g=nn.Linear(D,D)
        self.gn=nn.GroupNorm(H,D); self.o=nn.Linear(D,D)
        self.rate=nn.Parameter(-2.0-0.5*torch.arange(H).float())            # per-head decay rate (<0)
        self.ln2=nn.LayerNorm(D); self.w1=nn.Linear(D,4*D); self.w2=nn.Linear(D,4*D); self.w3=nn.Linear(4*D,D)
    def forward(self, x, ts):
        B,N,D=x.shape; H=self.H; hd=self.hd
        h=self.ln(x); q,k,v=self.qkv(h).view(B,N,3,H,hd).permute(2,0,3,1,4).unbind(0)   # [B,H,N,hd]
        dt=(ts[:,None,:,None]-ts[:,None,None,:]).float().clamp(min=0)                     # [B,1,N,N] causal gap
        Dmat=torch.exp(self.rate.view(1,H,1,1)*torch.log1p(dt))                          # (1+Δt)^rate heavy-tail decay
        mask=torch.tril(torch.ones(N,N,device=x.device)).view(1,1,N,N)
        A=(torch.einsum('bhnd,bhmd->bhnm',q,k)*Dmat)*mask
        ret=torch.einsum('bhnm,bhmd->bhnd',A,v).reshape(B,N,D)
        ret=self.gn(ret.transpose(1,2)).transpose(1,2)
        x=x+self.o(F.silu(self.g(h))*ret)
        h2=self.ln2(x); return x+self.w3(F.silu(self.w1(h2))*self.w2(h2))
class RetNetV2(nn.Module):
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64, **kw):
        super().__init__(); self.N=N; self.inp=input_adapter; self.head=head
        self.blocks=nn.ModuleList([_RetBlock(D,H) for _ in range(L)])
    def backbone(self, inp, ts):
        x=self.inp(inp)
        for b in self.blocks: x=b(x,ts)
        return x
    def forward(self, inp, ts): return self.backbone(inp, ts)

class _MambaBlock(nn.Module):
    """Minimal selective SSM (Mamba 2312.00752) with the REAL inter-event gap as the ZOH step Δ ->
    native continuous-time model for irregular events. Pure-torch sequential scan (no CUDA kernels)."""
    def __init__(self, D, dstate=16, expand=2, dconv=4):
        super().__init__(); self.din=expand*D; self.dstate=dstate
        self.ln=nn.LayerNorm(D); self.in_proj=nn.Linear(D,2*self.din)
        self.conv=nn.Conv1d(self.din,self.din,dconv,groups=self.din,padding=dconv-1)
        self.x_proj=nn.Linear(self.din,2*dstate); self.dt_proj=nn.Linear(self.din,self.din)
        self.A_log=nn.Parameter(torch.log(torch.arange(1,dstate+1).float()).repeat(self.din,1))
        self.Dp=nn.Parameter(torch.ones(self.din)); self.out=nn.Linear(self.din,D)
    def forward(self, x, ts):
        B,N,D=x.shape
        h=self.ln(x); xin,z=self.in_proj(h).chunk(2,-1)
        xin=F.silu(self.conv(xin.transpose(1,2))[:,:,:N].transpose(1,2))
        Bm,Cm=self.x_proj(xin).chunk(2,-1)                                   # [B,N,dstate]
        gap=(ts[:,1:]-ts[:,:-1]).clamp(min=0); gap=torch.cat([gap[:,:1],gap],1).float()   # [B,N]
        dt=F.softplus(self.dt_proj(xin))*torch.log1p(gap).unsqueeze(-1)      # [B,N,din] real-time step
        A=-torch.exp(self.A_log); st=torch.zeros(B,self.din,self.dstate,device=x.device); ys=[]
        for t in range(N):
            dA=torch.exp(dt[:,t].unsqueeze(-1)*A)                            # [B,din,dstate]
            st=dA*st + (dt[:,t].unsqueeze(-1)*Bm[:,t].unsqueeze(1))*xin[:,t].unsqueeze(-1)
            ys.append((st*Cm[:,t].unsqueeze(1)).sum(-1)+self.Dp*xin[:,t])
        return x+self.out(torch.stack(ys,1)*F.silu(z))
class MambaV2(nn.Module):
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64, **kw):
        super().__init__(); self.N=N; self.inp=input_adapter; self.head=head
        self.blocks=nn.ModuleList([_MambaBlock(D) for _ in range(min(L,6))])
    def backbone(self, inp, ts):
        with torch.autocast('cuda',enabled=False):
            x=self.inp(inp).float()
            for b in self.blocks: x=b(x,ts)
        return x
    def forward(self, inp, ts): return self.backbone(inp, ts)

class TGRv2(nn.Module):
    """Time-Gated Recurrence backbone: minGRU-style affine recurrence (gates from x only, arXiv 2410.01201)
    + GRU-D / T-LSTM-style per-unit decay of the carried state by the REAL inter-event gap:
        h_t = d_t*(1-z_t)*h_{t-1} + z_t*hbar_t,   d_t = exp(-softplus(w_d) * log1p(dt_t))   (per unit)
    Rationale from our own bake-offs: recurrence is the strongest small generative backbone on telemetry
    (GRU Pareto-dominated HSTU on F-DATA), and time-awareness is our one consistently-winning lever -- this
    keeps HSTU's CONCEPTS (entity-event tokenization, time-indexed memory, one generative transducer) while
    replacing its recsys attention block. All matmuls are precomputed over the sequence; the scan is N
    elementwise steps in fp32. time_gated=False disables both time terms (clean ablation of the dt lever)."""
    def __init__(self, input_adapter, head, D=512, H=8, L=8, N=64, time_gated=True, **kw):
        super().__init__(); self.inp=input_adapter; self.head=head; self.N=N; self.tg=time_gated
        self.ln=nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.pz=nn.ModuleList([nn.Linear(D,D) for _ in range(L)])              # update gate
        self.ph=nn.ModuleList([nn.Linear(D,D) for _ in range(L)])              # candidate
        self.po=nn.ModuleList([nn.Linear(D,D) for _ in range(L)])              # out proj (residual)
        self.vz=nn.ParameterList([nn.Parameter(torch.zeros(D)) for _ in range(L)])      # time term in gate
        self.wd=nn.ParameterList([nn.Parameter(torch.full((D,),-2.0)) for _ in range(L)])  # per-unit decay rate
        self.ls=nn.ParameterList([nn.Parameter(torch.full((D,),0.1)) for _ in range(L)])   # LayerScale
    def backbone(self, inp, ts):
        with torch.autocast('cuda',enabled=False):                             # fp32 scan (bf16 drifts over N steps)
            x=self.inp(inp).float(); B,N,D=x.shape
            if self.tg and ts is not None:
                dt=torch.zeros(B,N,device=x.device); dt[:,1:]=(ts[:,1:]-ts[:,:-1]).float().clamp(min=0)
                tau=torch.log1p(dt).unsqueeze(-1)                              # [B,N,1]
            else: tau=x.new_zeros(B,N,1)
            for l in range(len(self.ln)):
                u=self.ln[l](x)
                z=torch.sigmoid(self.pz[l](u)+self.vz[l]*tau)                  # gate opens/closes with elapsed time
                d=torch.exp(-F.softplus(self.wd[l])*tau)                       # multi-scale forgetting of the carry
                a=d*(1-z); b=z*self.ph[l](u)
                hs=[]; Hc=x.new_zeros(B,D)
                for t in range(N): Hc=a[:,t]*Hc+b[:,t]; hs.append(Hc)          # elementwise scan (matmuls precomputed)
                x=x+self.po[l](torch.stack(hs,1))*self.ls[l]
        return x
    def forward(self, inp, ts): return self.backbone(inp, ts)

def train_v2(model, batches, epochs=8, lr=1e-3, clip=1.0):
    """batches(): iterator yielding (inp, ts, target, valid); target,valid [B,N]. NLL in fp32."""
    m=model.to(dev); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr,weight_decay=0.01)
    for ep in range(epochs):
        m.train()
        for inp,ts,target,valid in batches():
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                h=m(inp,ts)[:,:-1]
            tgt=tuple(t[:,1:] for t in target) if isinstance(target,(tuple,list)) else target[:,1:]   # multi-field aware
            loss=m.head.nll(h.float(), tgt, valid[:,1:])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),clip); opt.step()
    return m
