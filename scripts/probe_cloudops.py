import os
os.environ.setdefault("HF_HOME","/share/hpcproject/zzhang66/nfm/hf_cache")
try:
    import datasets; print("datasets", datasets.__version__)
except Exception as e:
    print("NO datasets lib:", e); raise SystemExit
from datasets import load_dataset
d=load_dataset("Salesforce/cloudops_tsf","borg_cluster_data_2011")
print("splits:", list(d.keys()))
for sp in d:
    print(f"  {sp}: n={len(d[sp])}")
tr=d[list(d.keys())[0]]; print("cols:", tr.column_names)
r=tr[0]
for k,v in r.items():
    import numpy as np
    ln=(len(v) if hasattr(v,"__len__") and not isinstance(v,str) else v)
    print("  field", k, "->", type(v).__name__, ln)
# target field is usually 'target' (multivariate array) + 'start' + 'feat_static_cat'
print("PROBE_DONE")
