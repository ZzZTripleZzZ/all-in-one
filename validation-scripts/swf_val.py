"""HPC / ANL-Intrepid SWF — entity=user, per-user job sequence, predict job RUN_TIME (log).
Baselines: global mean, user's requested walltime (req_time), per-user causal running mean. Model must beat all three."""
import gzip, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = "/private/tmp/claude-501/-Users-zifanzhang-Library-Mobile-Documents-com-apple-CloudDocs-Reseach-Overleaf/60cef121-2a9d-48ea-a7f0-152d82a310c7/scratchpad/data"
names = ['job','submit','wait','run','nproc','avgcpu','usedmem','reqproc','reqtime','reqmem',
         'status','uid','gid','exe','queue','partition','prevjob','think']
df = pd.read_csv(DATA+'/hpc/anl.swf.gz', comment=';', sep=r'\s+', header=None, names=names)
df = df[(df['run']>0)&(df['reqtime']>0)&(df['reqproc']>0)].copy()
df = df.sort_values('submit').reset_index(drop=True)
df['y'] = np.log1p(df['run'])
df['hour'] = (df['submit']//3600)%24
# per-user CAUSAL features (only past jobs of same user)
gu = df.groupby('uid', sort=False)
df['u_prev'] = gu['y'].shift(1)
df['u_mean'] = gu['y'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
df['u_cnt']  = gu.cumcount()
df['u_prev'] = df['u_prev'].fillna(df['y'].mean())
df['u_mean'] = df['u_mean'].fillna(df['y'].mean())
# temporal split
cut = int(len(df)*0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
print(f"[SWF] jobs(valid)={len(df):,}  users={df['uid'].nunique():,}  train={len(tr):,} test={len(te):,}  span=8 months")

yte = te['y'].values
b_mean = mean_absolute_error(yte, np.full_like(yte, tr['y'].mean()))
b_req  = mean_absolute_error(yte, np.log1p(te['reqtime'].values))          # user's own walltime estimate
b_user = mean_absolute_error(yte, te['u_mean'].values)                     # per-user running mean
feats = ['reqproc','reqtime','reqmem','queue','partition','hour','u_prev','u_mean','u_cnt']
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6, random_state=0)
reg.fit(tr[feats], tr['y'])
pred = reg.predict(te[feats])
m_gbm = mean_absolute_error(yte, pred); r2 = r2_score(yte, pred)
print(f"[SWF] MAE(log runtime)  globalMean={b_mean:.3f}  reqTime={b_req:.3f}  userMean={b_user:.3f}  |  GBM={m_gbm:.3f}  (R2={r2:.3f})")
best_base = min(b_mean,b_req,b_user)
print(f"[SWF] VERDICT: GBM MAE {m_gbm:.3f} vs best baseline {best_base:.3f} "
      f"({'BEATS baseline — sequential/user history helps' if m_gbm<best_base-0.02 else 'no clear gain over baseline'})")
