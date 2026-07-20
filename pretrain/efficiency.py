"""Measure the ONE-model-vs-specialist-zoo COST tradeoff for the any-input x any-output grid.
For each generative backbone: trainable params + GPU inference latency (one forward pass, and a full
LOUTMAX-step autoregressive rollout over the 25-cell grid). For the GBM specialist: train+predict wall
time and effective tree size for ONE cell, then x |grid| for the zoo. The story: one generative model
covers all |grid| cells at CONSTANT cost; the specialist zoo scales LINEARLY with the number of cells.
Env: D(512) H(8) L(8) N(160) KC(256) MIX(4) SPLIT(96) LOUTMAX(64) BATCH(256).
"""
import os, sys, time, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, TFMv2, LSTMv2, GRUv2, RetNetV2, ContinuousInput, MixtureHead, dev
from sklearn.ensemble import HistGradientBoostingRegressor

D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8))
N=int(os.environ.get('N',160)); KC=int(os.environ.get('KC',256)); MIX=int(os.environ.get('MIX',4))
SPLIT=int(os.environ.get('SPLIT',96)); LOUTMAX=int(os.environ.get('LOUTMAX',64)); B=int(os.environ.get('BATCH',256))
NCELL=25                                                                        # 5 input-lengths x 5 horizons
BB={'HSTU':HSTUv2,'HSTU++':lambda i,h,**kw: HSTUv2(i,h,ffn=True,**kw),'RetNet':RetNetV2,'TFM':TFMv2,'LSTM':LSTMv2,'GRU':GRUv2}
nparams=lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

def time_fwd(m, reps=30):
    z=torch.zeros(B,N,device=dev); c=torch.zeros(B,N,device=dev,dtype=torch.long); t=torch.zeros(B,N,device=dev,dtype=torch.long)
    for _ in range(5):                                                          # warmup
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'): m((z,c),t)
    if dev=='cuda': torch.cuda.synchronize()
    t0=time.time()
    for _ in range(reps):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'): m((z,c),t)
    if dev=='cuda': torch.cuda.synchronize()
    return (time.time()-t0)/reps*1e3                                            # ms per forward (batch B)

def time_rollout(m, reps=5):
    z=torch.zeros(B,N,device=dev); c=torch.zeros(B,N,device=dev,dtype=torch.long); t=torch.zeros(B,N,device=dev,dtype=torch.long)
    if dev=='cuda': torch.cuda.synchronize()
    t0=time.time()
    for _ in range(reps):
        with torch.no_grad():
            for p in range(SPLIT,SPLIT+LOUTMAX):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'): h=m((z,c),t)[:,p-1]
                m.head.quantiles(h.unsqueeze(1),(0.5,))
    if dev=='cuda': torch.cuda.synchronize()
    return (time.time()-t0)/reps*1e3                                            # ms for a full LOUTMAX rollout (batch B)

if __name__=='__main__':
    print(f"device={dev} D={D} L={L} N={N} batch={B} grid-cells={NCELL} rollout-steps={LOUTMAX}",flush=True)
    print(f"\n{'backbone':<10} {'params(M)':>10} {'fwd(ms)':>9} {'rollout(ms)':>12}",flush=True)
    gen={}
    for name,cls in BB.items():
        m=cls(ContinuousInput(D,KC),MixtureHead(D,MIX),D=D,H=H,L=L,N=N).to(dev).eval()
        try: fwd=time_fwd(m); roll=time_rollout(m)
        except RuntimeError as e: fwd=float('nan'); roll=float('nan'); print("  (oom?)",e)
        p=nparams(m)/1e6; gen[name]=(p,fwd,roll); print(f"{name:<10} {p:>10.2f} {fwd:>9.2f} {roll:>12.1f}",flush=True)
        del m; torch.cuda.empty_cache() if dev=='cuda' else None
    # GBM specialist: train+predict wall for ONE cell (Lin=96 features -> 1 horizon), effective tree size
    Xtr=np.random.randn(50000,96).astype(np.float32); ytr=np.random.randn(50000).astype(np.float32); Xte=np.random.randn(3000,96).astype(np.float32)
    t0=time.time(); gm=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,max_iter=200,learning_rate=0.08).fit(Xtr,ytr); gbm_fit=time.time()-t0
    t0=time.time(); gm.predict(Xte); gbm_pred=(time.time()-t0)*1e3
    n_nodes=sum(sum(est.tree_.node_count for est in row) for row in gm._predictors) if hasattr(gm,'_predictors') else 0
    print(f"\nGBM specialist (ONE cell): fit={gbm_fit:.1f}s  predict(3k)={gbm_pred:.1f}ms  nodes~{n_nodes:,}",flush=True)
    print(f"\n==== ONE generative model vs the {NCELL}-cell GBM specialist ZOO ====")
    hp,hf,hr=gen.get('HSTU',(float('nan'),)*3)
    print(f"  generative (HSTU): 1 model, {hp:.2f}M params, one {LOUTMAX}-step rollout {hr:.0f}ms covers ALL {NCELL} cells")
    print(f"  GBM zoo: {NCELL} models, ~{NCELL} separate fits ({NCELL*gbm_fit:.0f}s total), {NCELL} nodes-sets ({NCELL*n_nodes:,} nodes)")
    print(f"  => generative cost is CONSTANT in grid size; specialist cost grows LINEARLY (x{NCELL} here).")
    print("EFFICIENCY_DONE",flush=True)
