"""Fair continuous baselines on RAW values (not our token strawman):
- GBM-regressor: HistGradientBoostingRegressor, ONE per horizon (fixed-horizon specialist).
- Chronos: TSFM zero-shot, ONE model all horizons (direct rival to our arbitrary-horizon claim).
- (PatchTST/DLinear/NHITS via neuralforecast added in run_horizon when enabled.)
All predict raw future values -> compared on MASE alongside our generative model."""
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor

def gbm_reg(Xtr, Ytr, Xte, HZ, HMAX):        # Xtr[n,CTX] raw ctx, Ytr[n,HMAX] raw future, Xte[m,CTX]
    pred=np.zeros((len(Xte),HMAX))
    if len(Xtr)>200000:
        s=np.random.choice(len(Xtr),200000,replace=False); Xtr=Xtr[s]; Ytr=Ytr[s]
    for h in HZ:
        g=HistGradientBoostingRegressor(max_iter=150,max_depth=8); g.fit(Xtr,Ytr[:,h-1]); pred[:,h-1]=g.predict(Xte)
    return pred

_CH=None
def chronos_zeroshot(Cxv, HMAX, model="amazon/chronos-bolt-small", bs=256):
    global _CH
    from chronos import BaseChronosPipeline
    if _CH is None:
        _CH=BaseChronosPipeline.from_pretrained(model, device_map="cuda" if torch.cuda.is_available() else "cpu",
                                                torch_dtype=torch.bfloat16)
    out=[]
    for i in range(0,len(Cxv),bs):
        ctx=[torch.tensor(np.asarray(x,dtype=np.float32)) for x in Cxv[i:i+bs]]
        qf,_=_CH.predict_quantiles(ctx, prediction_length=HMAX, quantile_levels=[0.5])
        out.append(qf[:,:,0].float().cpu().numpy())
    return np.concatenate(out)                # [n,HMAX] median forecast
