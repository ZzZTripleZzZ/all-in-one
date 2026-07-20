import time, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
Xtr=np.random.randn(50000,96).astype('float32'); ytr=np.random.randn(50000).astype('float32'); Xte=np.random.randn(3000,96).astype('float32')
t0=time.time(); gm=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,max_iter=200,learning_rate=0.08).fit(Xtr,ytr); fit=time.time()-t0
t0=time.time(); gm.predict(Xte); pred=(time.time()-t0)*1e3
nodes=sum(len(p[0].nodes) for p in gm._predictors)          # HistGBR: _predictors[iter][output].nodes
print(f"GBM one specialist (1 input-len x 1 horizon cell): fit={fit:.2f}s  predict(3k)={pred:.1f}ms  trees={len(gm._predictors)}  nodes={nodes:,}")
print(f"25-cell GBM zoo: ~{25*fit:.0f}s total training, {25} separate models, {25*nodes:,} total nodes")
print("GBM_COST_DONE")
