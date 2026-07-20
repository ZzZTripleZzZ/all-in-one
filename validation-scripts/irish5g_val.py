"""WIRELESS / Irish 5G (Raca) — entity=session, predict NEXT-second DL throughput. ONE model pooled over sessions.
Held-out WHOLE sessions for test (tests generalization to unseen sessions). Baseline: persistence."""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = "/private/tmp/claude-501/-Users-zifanzhang-Library-Mobile-Documents-com-apple-CloudDocs-Reseach-Overleaf/60cef121-2a9d-48ea-a7f0-152d82a310c7/scratchpad/data"
files = sorted(glob.glob(DATA+'/irish5g/unz/**/*.csv', recursive=True))
numc = ['RSRP','RSRQ','SNR','CQI','RSSI','DL_bitrate','UL_bitrate','Speed','NRxRSRP','NRxRSRQ']
k = 5; rows_tot = 0; parts = []
for i, f in enumerate(files):
    d = pd.read_csv(f, na_values=['-',''])
    for c in numc:
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    if 'DL_bitrate' not in d or len(d) < k+5: continue
    d = d.reset_index(drop=True)
    d['is5g'] = (d.get('NetworkMode','').astype(str)=='5G').astype(int)
    d['y'] = d['DL_bitrate'].shift(-1)                       # next-second throughput
    for L in range(1, k+1): d[f'dl_lag{L}'] = d['DL_bitrate'].shift(L-1)  # lag0..lag(k-1)
    d['sess'] = i
    rows_tot += len(d)
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
feat = [f'dl_lag{L}' for L in range(1,k+1)] + ['RSRP','RSRQ','SNR','CQI','RSSI','Speed','is5g']
df = df.dropna(subset=['y','dl_lag1'])
# hold out whole sessions (unseen-session generalization)
sess = df['sess'].unique(); te_sess = set(sess[::4])       # ~25% sessions held out
te = df[df['sess'].isin(te_sess)]; tr = df[~df['sess'].isin(te_sess)]
print(f"[Irish5G] sessions={len(files)}  rows(total)~{rows_tot:,}  usable={len(df):,}  train={len(tr):,} test={len(te):,} (held-out sessions)")
yte = te['y'].values
b_persist = mean_absolute_error(yte, te['dl_lag1'].values)   # persistence = current second's throughput
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6, random_state=0)
reg.fit(tr[feat], tr['y']); pred = reg.predict(te[feat])
m_gbm = mean_absolute_error(yte, pred); r2 = r2_score(yte, pred)
skill = (1 - m_gbm/b_persist)*100
print(f"[Irish5G] MAE(DL kbps)  persistence={b_persist:,.0f}  |  GBM={m_gbm:,.0f}   skill=+{skill:.1f}%  (R2={r2:.3f})")
print(f"[Irish5G] VERDICT: {'BEATS persistence on unseen sessions — radio-context sequence signal is real' if m_gbm<b_persist else 'no gain over persistence'}")
