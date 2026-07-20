"""Serialized model sizes for the storage-cost column of the deployment table. Measures actual
state_dict bytes (fp32 and bf16) for our models, a fitted GBM pickle at the real config, and the
tsafrir per-user state. Also prints the fleet-multiplication arithmetic: specialists scale as
(grid cells x months), the FM as 1 (+optional per-month finetunes)."""
import io, pickle, sys, numpy as np, torch
sys.path.insert(0,"/share/hpcproject/zzhang66/nfm/code/common")
from nfm_v2 import HSTUv2, RNNv2, MixtureHead, HorizonHead, ContinuousInput
import torch.nn as nn

class CovInput(nn.Module):
    def __init__(self,D,C): super().__init__(); self.lin=nn.Linear(C,D)
    def forward(self,inp): return self.lin(inp.float())

def sd_bytes(m,dtype):
    b=io.BytesIO(); torch.save({k:v.to(dtype) for k,v in m.state_dict().items()},b); return b.tell()

models=[("backfill-v2 estimator (D256L4)",HSTUv2(CovInput(256,7),MixtureHead(256,6),D=256,H=8,L=4,N=96)),
        ("forecasting model (D512L8)",HSTUv2(ContinuousInput(512,256),HorizonHead(512,4),D=512,H=8,L=8,N=160)),
        ("backfill-GRU control (D256L4)",RNNv2(CovInput(256,7),MixtureHead(256,6),D=256,H=8,L=4,N=96,cell='gru'))]
for nm,m in models:
    print(f"  {nm:<34} fp32={sd_bytes(m,torch.float32)/2**20:7.1f} MiB   bf16={sd_bytes(m,torch.bfloat16)/2**20:7.1f} MiB")

from sklearn.ensemble import HistGradientBoostingRegressor
rng=np.random.default_rng(0)
F=rng.normal(size=(26728,10)).astype(np.float32); y=rng.normal(size=26728)
g=HistGradientBoostingRegressor(loss='quantile',quantile=0.6,max_iter=300,min_samples_leaf=40,random_state=0).fit(F,y)
gb=len(pickle.dumps(g))
print(f"  GBM quantile (300 iters, fitted)   pickle={gb/2**20:7.2f} MiB per (quantile x domain-month)")
print(f"  tsafrir state                      2 floats/user: 300 users = {300*16/1024:.1f} KiB")
print("\n  fleet arithmetic (F-DATA alone): specialist route = 25 grid cells x 7 months x GBM "
      f"= {25*7*gb/2**20:,.0f} MiB; FM route = ONE model ({sd_bytes(models[1][1],torch.bfloat16)/2**20:.1f} MiB bf16) "
      "+ optional per-month finetunes")
print("STORAGE_DONE")
