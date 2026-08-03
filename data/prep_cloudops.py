"""Prep CloudOps-TSF (Woo et al. arXiv:2310.05063) directly from the HF repo files, bypassing the broken
dataset script (datasets>=2.x rejects the script + IsADirectoryError). We fetch <config>/train_test.zip and
<config>/pretrain.zip from Salesforce/cloudops_tsf, read the snappy-parquet series, and save a clean npz.
Protocol (from the script): prediction_length=48, freq=5T, stride=48, rolling_evaluations=12; azure2017
target_dim=1 (CRPS), borg2011/alibaba2018 target_dim=2 (CRPS-sum). The last stride*rolling = 48*12 = 576
steps of each train_test series are the 12 non-overlapping 48-step evaluation windows; everything before is
context/history for that series. pretrain.zip = extra unlabelled series for representation pretraining.
Usage: python prep_cloudops.py azure_vm_traces_2017     (or borg_cluster_data_2011 / alibaba_cluster_trace_2018)
"""
import os, sys, glob, zipfile, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

cfg=sys.argv[1] if len(sys.argv)>1 else 'azure_vm_traces_2017'
os.environ.setdefault("HF_HOME","/share/hpcproject/zzhang66/nfm/hf_cache")
root=os.path.expanduser("~/nfm/data/cloudops"); os.makedirs(root,exist_ok=True)
PL,STRIDE,ROLL=48,48,12

def fetch(split):
    d=os.path.join(root,cfg,split); os.makedirs(d,exist_ok=True)
    if not glob.glob(d+"/*.parquet"):
        z=hf_hub_download("Salesforce/cloudops_tsf",f"{cfg}/{split}.zip",repo_type="dataset")
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
    return sorted(glob.glob(d+"/**/*.parquet",recursive=True))

def read_series(parts):
    S=[]
    for p in parts:
        t=pq.read_table(p,columns=['target','item_id'])
        for tg in t.column('target').to_pylist():
            a=np.asarray(tg,dtype=np.float32)
            if a.ndim==1: a=a[None,:]                                         # [1,T] univariate -> [D,T]
            S.append(a)
    return S

print(f"[{cfg}] fetching train_test ...",flush=True); tt=read_series(fetch("train_test"))
print(f"  train_test series={len(tt)}  dims={tt[0].shape[0]}  len range=[{min(s.shape[1] for s in tt)},{max(s.shape[1] for s in tt)}]",flush=True)
try:
    print(f"[{cfg}] fetching pretrain ...",flush=True); pt=read_series(fetch("pretrain"))
    print(f"  pretrain series={len(pt)}",flush=True)
except Exception as e:
    print(f"  pretrain unavailable ({e}); using train_test history only"); pt=[]
D=tt[0].shape[0]; EVAL=PL*ROLL
usable=[s for s in tt if s.shape[1]>=EVAL+PL]                                  # need >=1 context window before the 12 eval windows
print(f"[{cfg}] usable eval series (len>= {EVAL+PL})={len(usable)}",flush=True)
def to_obj(lst):
    a=np.empty(len(lst),dtype=object)
    for i,s in enumerate(lst): a[i]=s.astype(np.float32)
    return a
import pickle
with open(os.path.join(root,f"{cfg}.pkl"),'wb') as fh:
    pickle.dump({'tt':tt,'pt':pt,'D':D,'PL':PL,'STRIDE':STRIDE,'ROLL':ROLL},fh,protocol=4)
print(f"[{cfg}] wrote {cfg}.pkl  D={D} PL={PL} eval_windows_per_series={ROLL}  (tt={len(tt)} pt={len(pt)})",flush=True)
print("CLOUDOPS_PREP_DONE",flush=True)
