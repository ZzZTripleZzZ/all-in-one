"""Metric suite for the v2 experiments (from the 2026-07-16 metrics-design agent).
Forecasting: per-series MASE (+median agg), tail-MASE by magnitude decile, quantile PINBALL,
CRPS (from a quantile grid; CRPS=MAE for point forecasts => fair vs point baselines), interval COVERAGE.
Security: AUC (prevalence-independent headline), AUPRC + prevalence + normalized lift,
detection@fixed-FPR (TPR@alpha, operational), precision@k / recall@k.
All numpy; report the FULL suite (incl. plain MASE where we tie) to pre-empt metric-cherry-picking.
"""
import numpy as np

# ----------------- forecasting -----------------
def pinball(y, qpred, q):
    y=np.asarray(y,float); qpred=np.asarray(qpred,float); e=y-qpred
    return float(np.mean(np.maximum(q*e,(q-1)*e)))

def crps_from_quantiles(y, qpreds, qs):
    """qpreds: [n, Q] predicted values at quantile levels qs. CRPS ~= 2 * mean_q pinball_q (Gneiting&Ranjan 2011)."""
    y=np.asarray(y,float); qpreds=np.asarray(qpreds,float); qs=np.asarray(qs,float)
    tot=0.0
    for j,q in enumerate(qs): tot+=pinball(y,qpreds[:,j],q)
    return 2.0*tot/len(qs)

def mase_per_series(pred, true, series_id, scale):
    """pred,true [n]; series_id [n]; scale: dict sid->per-series train naive-MAE. Returns (mean, median) over series."""
    pred=np.asarray(pred,float); true=np.asarray(true,float); series_id=np.asarray(series_id)
    ae=np.abs(pred-true); per={}
    med=np.nanmedian([v for v in scale.values() if np.isfinite(v) and v>0]) if scale else 1.0
    for s in np.unique(series_id):
        m=series_id==s; sc=scale.get(s,med); sc=med if (not np.isfinite(sc) or sc<1e-3*med) else sc
        per[s]=ae[m].mean()/max(sc,1e-8)
    v=np.array(list(per.values())); return float(np.mean(v)), float(np.median(v))

def mase_global(pred, true, scale):
    return float(np.mean(np.abs(np.asarray(pred,float)-np.asarray(true,float)))/max(scale,1e-8))

def tail_mask(true, top_frac=0.10):
    true=np.asarray(true,float); thr=np.quantile(true,1-top_frac); return true>=thr

def coverage(true, lo, hi):
    true=np.asarray(true,float); return float(np.mean((true>=np.asarray(lo,float))&(true<=np.asarray(hi,float))))

# ----------------- security -----------------
def _roc_auc(y,s):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y); s=np.asarray(s,float); ok=np.isfinite(s)
    return float(roc_auc_score(y[ok],s[ok])) if len(np.unique(y[ok]))>1 else float('nan')

def auprc_norm(y,s):
    from sklearn.metrics import average_precision_score
    y=np.asarray(y); s=np.asarray(s,float); ok=np.isfinite(s); y=y[ok]; s=s[ok]
    if len(np.unique(y))<2: return float('nan'),float('nan'),float('nan')
    ap=float(average_precision_score(y,s)); pi=float(y.mean())
    return ap, pi, (ap-pi)/max(1-pi,1e-9)

def detection_at_fpr(y,s,alpha=0.01):
    """TPR at the largest FPR <= alpha (operational: catch X% of attacks at alpha false-alarm budget)."""
    from sklearn.metrics import roc_curve
    y=np.asarray(y); s=np.asarray(s,float); ok=np.isfinite(s)
    if len(np.unique(y[ok]))<2: return float('nan')
    fpr,tpr,_=roc_curve(y[ok],s[ok]); m=fpr<=alpha
    return float(tpr[m].max()) if m.any() else 0.0

def precision_recall_at_k(y,s,ks=(10,20,50)):
    y=np.asarray(y); s=np.asarray(s,float); ok=np.isfinite(s); y=y[ok]; s=s[ok]
    order=np.argsort(-s); P=int(y.sum()); out={}
    for k in ks:
        k2=min(k,len(y)); topk=y[order[:k2]]
        out[f'P@{k}']=float(topk.mean()); out[f'R@{k}']=float(topk.sum()/max(P,1))
    return out

def security_report(y, scores_by_method, alphas=(0.01,0.05)):
    """scores_by_method: {name: score_array}. Returns {name: {auc,auprc,pi,auprc_norm,det@a,P@k,R@k}}."""
    rep={}
    for nm,s in scores_by_method.items():
        ap,pi,apn=auprc_norm(y,s); r=dict(auc=_roc_auc(y,s),auprc=ap,prevalence=pi,auprc_norm=apn)
        for a in alphas: r[f'det@{a:.0%}FPR']=detection_at_fpr(y,s,a)
        r.update(precision_recall_at_k(y,s)); rep[nm]=r
    return rep
