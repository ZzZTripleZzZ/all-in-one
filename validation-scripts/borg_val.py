"""DC / Google Borg 2011 — entity=(job,task), predict NEXT event type from history.
Compares: marginal baseline < bigram-Markov < GBM(context). If GBM>Markov>marginal => learnable sequential structure."""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA = "/private/tmp/claude-501/-Users-zifanzhang-Library-Mobile-Documents-com-apple-CloudDocs-Reseach-Overleaf/60cef121-2a9d-48ea-a7f0-152d82a310c7/scratchpad/data"
cols = ['time','missing','job_id','task_index','machine_id','event_type','user',
        'sched_class','priority','cpu_req','mem_req','disk_req','constraint']
files = sorted(glob.glob(DATA+'/borg/task_events_*.csv.gz'))
df = pd.concat([pd.read_csv(f, header=None, names=cols) for f in files], ignore_index=True)
df = df[['time','job_id','task_index','event_type','sched_class','priority','cpu_req','mem_req']].dropna(
    subset=['event_type','job_id','task_index','time'])
df['event_type'] = df['event_type'].astype(int)
df = df.sort_values(['job_id','task_index','time'])
g = df.groupby(['job_id','task_index'], sort=False)
df['next_event'] = g['event_type'].shift(-1)
df['prev_event'] = g['event_type'].shift(1).fillna(-1).astype(int)   # order-2 context
df['pos'] = g.cumcount()
df['dt'] = df['time'] - g['time'].shift(1)
seq = df.dropna(subset=['next_event']).copy()
seq['next_event'] = seq['next_event'].astype(int)

# split by entity (no same-task leakage): hash job_id
h = (seq['job_id'] % 10)
tr, te = seq[h < 7], seq[h >= 7]
print(f"[Borg] transitions={len(seq):,}  train={len(tr):,} test={len(te):,}  "
      f"entities={df.groupby(['job_id','task_index']).ngroups:,}")
vc = seq['event_type'].value_counts(normalize=True).round(3).to_dict()
print(f"[Borg] event-type mix (0=sub,1=sched,2=evict,3=fail,4=finish,5=kill,...): {vc}")

y_tr, y_te = tr['next_event'].values, te['next_event'].values
# baseline 1: marginal majority
maj = np.bincount(y_tr).argmax()
acc_marg = accuracy_score(y_te, np.full_like(y_te, maj))
# baseline 2: bigram Markov P(next|current event_type)
mk = tr.groupby('event_type')['next_event'].agg(lambda s: s.value_counts().index[0])
pred_mk = te['event_type'].map(mk).fillna(maj).astype(int).values
acc_mk = accuracy_score(y_te, pred_mk)
# model: GBM with context (event types as CATEGORICAL, + order-2 prev_event)
feats = ['event_type','prev_event','sched_class','priority','cpu_req','mem_req','pos','dt']
Xtr, Xte = tr[feats].copy(), te[feats].copy()
for c in ['event_type','prev_event','sched_class']:
    Xtr[c] = Xtr[c].astype('category'); Xte[c] = Xte[c].astype('category')
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6,
                                     categorical_features='from_dtype', random_state=0)
clf.fit(Xtr, y_tr)
pred = clf.predict(Xte)
acc_gbm = accuracy_score(y_te, pred)
f1_gbm = f1_score(y_te, pred, average='macro')
print(f"[Borg] ACC  marginal={acc_marg:.3f}  |  Markov(bigram)={acc_mk:.3f}  |  GBM(context)={acc_gbm:.3f}  (macroF1={f1_gbm:.3f})")
lift = (acc_gbm-acc_marg)/(1-acc_marg)*100
print(f"[Borg] VERDICT: GBM closes {lift:.0f}% of the gap to perfect over marginal; "
      f"{'SIGNAL — sequential context is learnable' if acc_gbm>acc_mk+0.01 else 'weak — context adds little beyond Markov'}")
