"""Deployment-cost measurements for the NFM paper's efficiency table. All numbers measured on the
CPU compute partition (schedulers run on CPU management nodes — no GPU assumed at inference).
  (1) exact parameter counts for the three model sizes in play;
  (2) CPU inference latency of the backfill estimator (batch 1 = interactive submit path, batch 256
      = periodic re-scoring), median over repeats, torch threads pinned to the job's 4 cores;
  (3) the same for the quantile read (mixture sampling S=2048, the expensive part of serving);
  (4) baseline costs on the same node: tsafrir causal walk per job, GBM quantile predict per job.
Weights are random (latency does not depend on training); shapes match the real runs exactly."""
import os, time, sys, numpy as np, torch
sys.path.insert(0,"/share/hpcproject/zzhang66/nfm/code/common")
torch.set_num_threads(int(os.environ.get('THREADS',4)))
from nfm_v2 import HSTUv2, RNNv2, MixtureHead, HorizonHead, ContinuousInput
import torch.nn as nn

def count(m): return sum(p.numel() for p in m.parameters())

class CovInput(nn.Module):
    def __init__(self,D,C): super().__init__(); self.lin=nn.Linear(C,D)
    def forward(self,inp): return self.lin(inp.float())

def bench(f,rep=30,warm=5):
    for _ in range(warm): f()
    ts=[]
    for _ in range(rep):
        t0=time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))

print(f"torch {torch.__version__} threads={torch.get_num_threads()}")
# ---- (1) parameter counts ----
bf1=HSTUv2(CovInput(256,3),MixtureHead(256,6),D=256,H=8,L=4,N=96)
bf2=HSTUv2(CovInput(256,7),MixtureHead(256,6),D=256,H=8,L=4,N=96)
big=HSTUv2(ContinuousInput(512,256),HorizonHead(512,4),D=512,H=8,L=8,N=160)
gru=RNNv2(CovInput(256,7),MixtureHead(256,6),D=256,H=8,L=4,N=96,cell='gru')
for nm,m in [("backfill-v1 (D256L4)",bf1),("backfill-v2 (D256L4+cov)",bf2),
             ("forecasting (D512L8+HorizonHead)",big),("backfill-GRU (D256L4)",gru)]:
    print(f"  params {nm:<36} {count(m):,}")

# ---- (2)+(3) backfill estimator CPU latency ----
bf2.eval()
for B in [1,256]:
    x=torch.randn(B,96,7); t=torch.cumsum(torch.randint(1,5000,(B,96)),1).long()
    with torch.no_grad():
        fwd=bench(lambda: bf2(x,t))
        hid=bf2(x,t)[:, -1]
        qt=bench(lambda: bf2.head.quantiles(hid.unsqueeze(1),(0.5,0.6),S=2048))
    print(f"  CPU backfill-v2 batch={B:<4} forward={fwd*1e3:8.2f} ms  quantiles(S=2048)={qt*1e3:8.2f} ms  per-job={((fwd+qt)/B)*1e3:7.3f} ms")

# ---- (4) baselines on the same node ----
rng=np.random.default_rng(0)
durs=rng.lognormal(6,2,300000); users=rng.integers(0,300,300000)
def tsafrir_walk():
    h={}
    est=np.empty(len(durs))
    for i,(u,d) in enumerate(zip(users,durs)):
        v=h.get(u)
        est[i]=(v[0]+v[1])/2 if v and len(v)==2 else 1e5
        h.setdefault(u,[]).append(d)
        if len(h[u])>2: h[u]=h[u][-2:]
    return est
t0=time.perf_counter(); tsafrir_walk(); tw=time.perf_counter()-t0
print(f"  CPU tsafrir walk 300k jobs = {tw:.2f} s  ({tw/3e5*1e6:.2f} us/job)")
from sklearn.ensemble import HistGradientBoostingRegressor
F=rng.normal(size=(26728,10)).astype(np.float32); y=rng.normal(size=26728)
g=HistGradientBoostingRegressor(loss='quantile',quantile=0.6,max_iter=300,min_samples_leaf=40,random_state=0)
t0=time.perf_counter(); g.fit(F,y); ft=time.perf_counter()-t0
X1=rng.normal(size=(1,10)); X256=rng.normal(size=(256,10))
p1=bench(lambda: g.predict(X1)); p256=bench(lambda: g.predict(X256))
print(f"  CPU GBM fit(26.7k rows)={ft:.2f} s  predict b1={p1*1e3:.3f} ms  b256={p256*1e3:.3f} ms ({p256/256*1e6:.1f} us/job)")
print("BENCH_DONE")
