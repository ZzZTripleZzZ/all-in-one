"""v2 causality test: with REAL timestamps feeding the attention time-bias, the read-position
output must still be invariant to garbage in the padded tail — in BOTH the covariates AND the
timestamps (the [N,N] bias matrix has entries for tail pairs; the causal mask must kill them)."""
import sys, torch, numpy as np
sys.path.insert(0,"/share/hpcproject/zzhang66/nfm/code/common")
from nfm_v2 import HSTUv2, RNNv2, MixtureHead
import torch.nn as nn

NCH=7; D,H,L,N,MIX=32,4,2,16,4
class CovInput(nn.Module):
    def __init__(self,D): super().__init__(); self.lin=nn.Linear(NCH,D)
    def forward(self,inp): return self.lin(inp.float())

torch.manual_seed(0)
for kind in ["HSTU","GRU"]:
    m=(HSTUv2(CovInput(D),MixtureHead(D,MIX),D=D,H=H,L=L,N=N) if kind=="HSTU"
       else RNNv2(CovInput(D),MixtureHead(D,MIX),D=D,H=H,L=L,N=N,cell='gru')).eval()
    B=4; k=torch.tensor([3,7,0,10])
    base=torch.randn(B,N,NCH)
    tbase=torch.cumsum(torch.randint(1,5000,(B,N)),dim=1).long()               # realistic increasing gaps
    a=base.clone(); b=base.clone(); ta=tbase.clone(); tb=tbase.clone()
    for i in range(B):
        a[i,k[i]+1:]=0.0;                       ta[i,k[i]+1:]=0
        b[i,k[i]+1:]=torch.randn(N-k[i]-1,NCH)*9.0
        tb[i,k[i]+1:]=torch.randint(0,10**7,(N-k[i]-1,))                       # garbage ts in tail
    with torch.no_grad():
        ha=m(a,ta)[torch.arange(B),k]; hb=m(b,tb)[torch.arange(B),k]
    dmax=(ha-hb).abs().max().item()
    print(f"{kind}: tail-garbage (covariates+ts) invariance max diff {dmax:.2e}")
    assert dmax<1e-4, f"{kind} LEAKS through the time bias"
print("BF_V2_TEST_OK")
