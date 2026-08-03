"""Prep a Loghub-2.0 HPC log dataset (BGL / Thunderbird) into a compact per-line event table for
generative log-anomaly detection. Loghub-2.0's *_structured.csv gives the parsed EventId per line but
DROPPED the alert label and timestamp; the RAW *_full.log keeps both as its first two whitespace tokens
(col1 = label, '-'=normal else an alert category; col2 = unix timestamp). The two files are line-aligned
(LineId 1..N), so we zip them: EventId from the structured CSV, (label, timestamp) from the raw log.

Output: ~/nfm/data/loghub/<name>_events.npz  with ts[int64], eid[int32], lab[int8], and vocab size.
Usage: python prep_loghub.py <BGL|Thunderbird>
"""
import os, sys, numpy as np, pandas as pd

name=sys.argv[1] if len(sys.argv)>1 else 'BGL'
root=os.path.expanduser(f"~/nfm/data/loghub/{name}")
raw=f"{root}/{name}_full.log"; struct=f"{root}/{name}_full.log_structured.csv"
out=os.path.expanduser(f"~/nfm/data/loghub/{name}_events.npz")

print(f"[{name}] reading EventId column from {os.path.basename(struct)} ...",flush=True)
eids=pd.read_csv(struct, usecols=['EventId'], dtype=str)['EventId'].values      # robust CSV parse (Content has commas)
n=len(eids); print(f"  structured rows: {n:,}",flush=True)

print(f"[{name}] streaming raw label+timestamp from {os.path.basename(raw)} ...",flush=True)
lab=np.empty(n,np.int8); ts=np.empty(n,np.int64); bad=0
with open(raw,'rb') as f:
    for i,line in enumerate(f):
        if i>=n: break
        p=line.split(b' ',2)
        if len(p)<2: lab[i]=0; ts[i]=ts[i-1] if i else 0; bad+=1; continue
        lab[i]=0 if p[0]==b'-' else 1
        try: ts[i]=int(p[1])
        except ValueError: ts[i]=ts[i-1] if i else 0; bad+=1
nraw=i+1
if nraw!=n: print(f"  WARNING raw lines {nraw:,} != structured {n:,} (using min)"); m=min(nraw,n); eids=eids[:m]; lab=lab[:m]; ts=ts[:m]; n=m
print(f"  raw lines: {nraw:,}  malformed: {bad}",flush=True)

vocab={e:i for i,e in enumerate(pd.unique(eids))}                              # EventId -> int, in first-seen order
eid=np.fromiter((vocab[e] for e in eids), np.int32, count=n)
print(f"[{name}] vocab={len(vocab)} events  alert-rate={lab.mean():.4f}  span={(ts.max()-ts.min())/86400:.1f} days",flush=True)
np.savez(out, ts=ts, eid=eid, lab=lab, vocab=len(vocab))
print(f"[{name}] wrote {out}  ({n:,} events)",flush=True)
print("LOGHUB_PREP_DONE",flush=True)
