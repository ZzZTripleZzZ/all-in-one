"""Prep CESNET-TimeSeries24 institution-level HOURLY traffic into per-institution series for the
network-domain leg (forecasting + capacity-provisioning Pareto). Source: Zenodo 13382427, institutions/
agg_1_hour/<id>.csv with columns id_time,n_flows,n_packets,n_bytes,... and times/times_1_hour.csv mapping
id_time -> wall clock. Target = n_bytes (heavy-tail volume). Series are reindexed onto the full hourly grid;
missing bins get value 0 with valid=0 (institutions can have gaps).
Output: ~/nfm/data/cesnet/cesnet_hourly.npz  { r: [S,T] log1p(bytes), valid: [S,T], ts: [T] unix, sid: [S] }
"""
import os, glob, numpy as np, pandas as pd

root=os.path.expanduser("~/nfm/data/cesnet")
tf=pd.read_csv(f"{root}/times/times_1_hour.csv")
tf['t']=pd.to_datetime(tf['time'],utc=True).astype('datetime64[ns, UTC]').astype(np.int64)//10**9   # force ns (pandas2 may parse as us -> //1e9 was 1000x off)
T=len(tf); print(f"hourly grid: {T} bins  span {(tf['t'].iloc[-1]-tf['t'].iloc[0])/86400:.0f} days",flush=True)
grid=np.full(int(tf['id_time'].max())+1,-1,np.int64); grid[tf['id_time'].values]=np.arange(T)

files=sorted(glob.glob(f"{root}/institutions/agg_1_hour/*.csv"), key=lambda p:int(os.path.basename(p)[:-4]))
R=np.zeros((len(files),T),np.float32); V=np.zeros((len(files),T),np.float32); sid=[]
for k,f in enumerate(files):
    d=pd.read_csv(f,usecols=['id_time','n_bytes'])
    idx=grid[d['id_time'].values]; ok=idx>=0
    R[k,idx[ok]]=np.log1p(d['n_bytes'].values[ok].astype(np.float64)); V[k,idx[ok]]=1.0
    sid.append(int(os.path.basename(f)[:-4]))
cov=V.mean(1)
print(f"institutions: {len(files)}  coverage mean={cov.mean():.3f} min={cov.min():.3f}  (drop <50% coverage: {(cov<0.5).sum()})",flush=True)
keep=cov>=0.5
np.savez(f"{root}/cesnet_hourly.npz", r=R[keep], valid=V[keep], ts=tf['t'].values, sid=np.array(sid)[keep])
print(f"wrote cesnet_hourly.npz  series={keep.sum()} x {T} bins",flush=True)
print("CESNET_PREP_DONE",flush=True)
