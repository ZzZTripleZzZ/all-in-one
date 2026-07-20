"""NET / GEANT traffic matrices — entity=OD pair, ONE model across ALL 529 OD series, predict next-step volume (log kbps).
Baseline: persistence (last value). Model = GBM on lag window, entity-agnostic (tests cross-entity transfer)."""
import glob, os, re, numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA = "/private/tmp/claude-501/-Users-zifanzhang-Library-Mobile-Documents-com-apple-CloudDocs-Reseach-Overleaf/60cef121-2a9d-48ea-a7f0-152d82a310c7/scratchpad/data"
files = sorted(glob.glob(DATA+'/geant/unz/**/IntraTM-*.xml', recursive=True))
def ts(f):
    m = re.search(r'IntraTM-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)', os.path.basename(f))
    return tuple(map(int, m.groups()))
files.sort(key=ts)
rows = []
for f in files:                                   # parse all ~10772 matrices
    src_map = {}
    for src in ET.parse(f).getroot().iter('src'):
        s = src.get('id')
        for dst in src.findall('dst'):
            src_map[(s, dst.get('id'))] = float(dst.text)
    rows.append(src_map)
pairs = sorted(set().union(*[set(r) for r in rows[:50]]))     # OD pairs present early
M = np.array([[r.get(p, np.nan) for p in pairs] for r in rows], float)   # T x P
T, P = M.shape
print(f"[GEANT] matrices(T)={T:,}  OD-pairs(P)={P}  (23-node backbone, 15-min, 2005)")
L = np.log1p(np.clip(M, 0, None))
# build lag samples pooled over all OD pairs; temporal split at 80%
k = 6; cut = int(T*0.8); X, y, istest = [], [], []
for j in range(P):
    s = L[:, j]
    for t in range(k, T-1):
        if np.isnan(s[t-k:t+1]).any() or np.isnan(s[t+1]): continue
        X.append(s[t-k:t+1]); y.append(s[t+1]); istest.append(t>=cut)
X = np.array(X); y = np.array(y); istest = np.array(istest)
tr = ~istest
print(f"[GEANT] pooled samples={len(y):,}  train={tr.sum():,} test={istest.sum():,}  (single entity-agnostic model)")
# persistence baseline = last observed value (lag1 = X[:, -1])
b_persist = mean_absolute_error(y[istest], X[istest, -1])
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6, random_state=0)
reg.fit(X[tr], y[tr]); pred = reg.predict(X[istest])
m_gbm = mean_absolute_error(y[istest], pred)
skill = (1 - m_gbm/b_persist)*100
print(f"[GEANT] MAE(log kbps)  persistence={b_persist:.4f}  |  GBM(1 model, all OD)={m_gbm:.4f}   skill=+{skill:.1f}% vs persistence")
print(f"[GEANT] VERDICT: {'BEATS persistence with ONE shared model across all OD pairs — cross-entity transfer works' if m_gbm<b_persist else 'no gain over persistence'}")
