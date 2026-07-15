"""Continuous multi-horizon adapters: per-entity (value_seq, timestamp_seq). Raw values
(runner tokenizes via Quantizer -> our vocab). Timestamps feed rab^{p,t} time-bucket bias."""
import glob, os, numpy as np, pandas as pd
DATA=os.path.expanduser("~/nfm/data")

def burst():                              # DC: single request stream; target = response tokens
    files=sorted(glob.glob(DATA+"/burstgpt/*.csv"))
    d=pd.concat([pd.read_csv(f,usecols=['Timestamp','Response tokens']) for f in files],ignore_index=True).sort_values('Timestamp')
    return [d['Response tokens'].values.astype(float)],[d['Timestamp'].values.astype(np.int64)]

def fdata():                              # HPC: per-user job stream; target = duration; ts = arrival epoch
    d=pd.concat([pd.read_parquet(f,columns=['usr','adt','duration']) for f in sorted(glob.glob(DATA+"/fdata/*.parquet"))],ignore_index=True)
    d['adt']=pd.to_datetime(d['adt'],utc=True,errors='coerce'); d=d.dropna(subset=['adt']).sort_values(['usr','adt'])
    d['t']=d['adt'].astype(np.int64)//10**9
    V,T=[],[]
    for u,g in d.groupby('usr',sort=False):
        if len(g)<8: continue
        V.append(g['duration'].values.astype(float)); T.append(g['t'].values.astype(np.int64))
    return V,T

def cesnet():                            # NET: per-IP hourly; target = n_bytes; ts = hour index
    V,T=[],[]
    for f in sorted(glob.glob(DATA+"/cesnet/**/agg_1_hour/*.csv",recursive=True)):
        c=pd.read_csv(f)
        if len(c)<16: continue
        V.append(c['n_bytes'].values.astype(float)); T.append(c['id_time'].values.astype(np.int64))
    return V,T

def beam():                              # wireless: per-beam hourly DL throughput vol; STRONG 168h weekly seasonality
    f=sorted(glob.glob(DATA+"/beam/data/train/*ThpVol*"))[0]
    d=pd.read_csv(f)
    if 'Unnamed: 0' in d.columns: d=d.drop(columns=['Unnamed: 0'])
    arr=d.values.astype(float)           # [840 hours x 2880 beams]
    V,T=[],[]
    for b in range(arr.shape[1]):
        col=np.nan_to_num(arr[:,b])
        if np.count_nonzero(col)<50: continue   # skip near-empty beams
        V.append(col); T.append(np.arange(len(col),dtype=np.int64))
    return V,T

def azure():                             # DC: per-VM 5-min avg CPU utilization (diurnal, UNIFORM sampling); large
    d=pd.concat([pd.read_csv(f,header=None,names=['t','vm','mincpu','maxcpu','avgcpu'],usecols=['t','vm','avgcpu'])
                 for f in sorted(glob.glob(DATA+"/azure/*cpu*.csv*"))],ignore_index=True).sort_values(['vm','t'])
    V,T=[],[]
    for vm,g in d.groupby('vm',sort=False):
        if len(g)<8: continue
        V.append(g['avgcpu'].values.astype(float)); T.append(g['t'].values.astype(np.int64))
    return V,T

def m100():                              # HPC: per-node PSU input power, 15-min aggregated (thermal inertia -> autocorrelated)
    V,T=[],[]
    for f in sorted(glob.glob(DATA+"/m100/*.parquet")):
        try: c=pd.read_parquet(f,columns=['timestamp','ps0_input_power_avg'])
        except Exception: continue
        c=c.dropna(subset=['ps0_input_power_avg']).sort_values('timestamp')
        if len(c)<8 or c['ps0_input_power_avg'].std()<1e-3: continue   # skip idle/flat nodes
        V.append(c['ps0_input_power_avg'].values.astype(float))
        T.append((pd.to_datetime(c['timestamp']).astype(np.int64)//10**9).values.astype(np.int64))
    return V,T

def borg():                              # DC: per-user task-DURATION sequences from Borg lifecycle events (irregular us timestamps)
    cols=['time','minfo','job','task','machine','etype','user','sclass','prio','cpu','mem','disk','constr']
    fr=[]
    for f in sorted(glob.glob(DATA+"/borg/*.csv.gz")):
        c=pd.read_csv(f,header=None,names=cols,usecols=['time','job','task','etype','user'],
                      dtype={'time':np.int64,'job':np.int64,'task':np.int32,'etype':np.int8,'user':'category'})
        fr.append(c[c['etype'].isin([1,4])])          # SCHEDULE, FINISH
    d=pd.concat(fr,ignore_index=True)
    sc=d[d['etype']==1].groupby(['job','task'],sort=False).agg(st=('time','min'),user=('user','first'))
    ft=d[d['etype']==4].groupby(['job','task'],sort=False)['time'].max().rename('ft')
    t=sc.join(ft,how='inner').reset_index(); t['dur']=t['ft']-t['st']
    t=t[t['dur']>0].sort_values(['user','st'])
    V,T=[],[]
    for u,g in t.groupby('user',sort=False,observed=True):
        if len(g)<8: continue
        V.append(np.log1p(g['dur'].values.astype(float))); T.append((g['st'].values//10**6).astype(np.int64))  # us->s
    return V,T

def alibaba():                           # DC: per-job batch-task DURATION sequences (irregular start_time)
    f=sorted(glob.glob(DATA+"/alibaba/batch_task*.csv"))[0]
    cols=['task_name','instance_num','job_name','task_type','status','start_time','end_time','plan_cpu','plan_mem']
    d=pd.read_csv(f,header=None,names=cols,usecols=['job_name','status','start_time','end_time'])
    d=d[(d['status']=='Terminated')&(d['end_time']>d['start_time'])&(d['start_time']>0)].copy()
    d['dur']=d['end_time']-d['start_time']; d=d.sort_values(['job_name','start_time'])
    V,T=[],[]
    for j,g in d.groupby('job_name',sort=False):
        if len(g)<8: continue
        V.append(np.log1p(g['dur'].values.astype(float))); T.append(g['start_time'].values.astype(np.int64))
    return V,T

LOADERS={'burst':burst,'fdata':fdata,'cesnet':cesnet,'beam':beam,'azure':azure,'m100':m100,'borg':borg,'alibaba':alibaba}
