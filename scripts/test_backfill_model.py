"""Shape + read-index causality test for the backfill model arm (CPU, synthetic).
The estimate for a job is read at position k (the current job); positions k+1..N-1 are pad. Verify
that arbitrary garbage in the padded tail does NOT change the read-position output (causal mask), for
both HSTU and GRU, and that quantiles come out monotone and finite."""
import sys, torch, numpy as np
sys.path.insert(0,"/share/hpcproject/zzhang66/nfm/code/common")
from nfm_v2 import HSTUv2, RNNv2, MixtureHead
import torch.nn as nn

class TriInput(nn.Module):
    def __init__(self,D): super().__init__(); self.lin=nn.Linear(3,D)
    def forward(self,inp): return self.lin(inp.float())

D,H,L,N,MIX=32,4,2,16,4
torch.manual_seed(0)
for kind in ["HSTU","GRU"]:
    inp=TriInput(D); head=MixtureHead(D,MIX)
    m=(HSTUv2(inp,head,D=D,H=H,L=L,N=N) if kind=="HSTU" else RNNv2(inp,head,D=D,H=H,L=L,N=N,cell='gru')).eval()
    B=4; k=torch.tensor([3,7,0,10])                     # per-sample read index (current-job position)
    base=torch.randn(B,N,3); t=torch.zeros(B,N,dtype=torch.long)
    a=base.clone(); b=base.clone()
    for i in range(B):
        a[i,k[i]+1:]=0.0                                # pad tail = zeros
        b[i,k[i]+1:]=torch.randn(N-k[i]-1,3)*9.0        # pad tail = garbage
    with torch.no_grad():
        ha=m(a,t)[torch.arange(B),k]; hb=m(b,t)[torch.arange(B),k]
    d=(ha-hb).abs().max().item()
    tag="OK" if d<1e-4 else f"LEAK d={d:.2e}"
    print(f"{kind}: read-index causality {tag}  (max diff {d:.2e})")
    with torch.no_grad():
        qz=head.quantiles(ha.unsqueeze(1),(0.5,0.7,0.9,0.95),S=256)[:,0,:]
    mono=bool((qz[:,1:]>=qz[:,:-1]-1e-3).all()); fin=bool(torch.isfinite(qz).all())
    print(f"{kind}: quantiles finite={fin} monotone={mono}")
    assert d<1e-4 and fin and mono, f"{kind} FAILED"
print("BFMODEL_TEST_OK")
