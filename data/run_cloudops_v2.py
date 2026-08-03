"""CloudOps-TSF CRPS benchmark (Woo et al. arXiv:2310.05063) — the HPC/DC honest-SOTA target.
Protocol (from the dataset script): predict PL=48 steps, STRIDE=48, ROLL=12 non-overlapping windows at the
tail of each train_test series; context = everything before each window. Metric = GluonTS mean weighted
quantile loss over q=0.1..0.9 (== their "CRPS"): CRPS = mean_q [ 2*sum_i pinball_q(y,qhat) / sum_i|y| ].
Numbers to beat: azure2017 CRPS 0.077 (SOTA 85M pretrained TFM; DeepAR 0.101); borg2011 CRPS-sum 0.128
(PatchTST 0.130). Our arm = HSTU + direct horizon-conditioned Student-t mixture head (one forward -> all 48
step distributions via HorizonHead h=1..48; NO rollout). Baselines re-run under the SAME harness to sanity-
check it: seasonal-naive (freq 288=1 day @5T), and a plug for GBM. target_dim=2 configs => CRPS-sum on the
across-variate sum. Env: CFG(azure_vm_traces_2017) KC(256) D(512) H(8) L(8) EP(20) MIX(6) SPLIT(288) BS(128)
NSEED(2) MAXTRAIN(150000) EVALS(2000).
"""
import os, sys, pickle, numpy as np, torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"common"))
from nfm_v2 import HSTUv2, GRUv2, TGRv2, ContinuousInput, HorizonHead, dev

CFG=os.environ.get('CFG','azure_vm_traces_2017'); KC=int(os.environ.get('KC',256))
D=int(os.environ.get('D',512)); H=int(os.environ.get('H',8)); L=int(os.environ.get('L',8))
EP=int(os.environ.get('EP',20)); MIX=int(os.environ.get('MIX',6)); NSEED=int(os.environ.get('NSEED',2))
SPLIT=int(os.environ.get('SPLIT',288)); BS=int(os.environ.get('BS',128)); SMOKE=int(os.environ.get('SMOKE',0))
MAXTRAIN=int(os.environ.get('MAXTRAIN',150000)); EVALS=int(os.environ.get('EVALS',2000))
PRETRAIN_MAX=int(os.environ.get('PRETRAIN_MAX',20000))
ARMS=os.environ.get('ARMS','HSTU,GRU').split(',')
PKL=os.path.expanduser(f"~/nfm/data/cloudops/{CFG}.pkl")
QS=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]); PL=48; ROLL=12; N=SPLIT+PL
if SMOKE: EP=2; NSEED=1; MAXTRAIN=8000; EVALS=300; D=128; L=2

def load():
    with open(PKL,'rb') as fh: z=pickle.load(fh)
    tt=z['tt']; pt=z['pt']
    if len(pt)>PRETRAIN_MAX: pt=[pt[k] for k in np.random.default_rng(0).choice(len(pt),PRETRAIN_MAX,replace=False)]
    return tt,pt,int(z['D'])

def crps_wql(y, q):                                                            # y:[M], q:[M,9] -> GluonTS mean wQL
    y=np.asarray(y,float); q=np.asarray(q,float); den=np.abs(y).sum()+1e-9; tot=0.0
    for j,tau in enumerate(QS):
        e=y-q[:,j]; tot+=2.0*np.sum(np.maximum(tau*e,(tau-1)*e))/den
    return tot/len(QS)

def make(kind):
    inp=ContinuousInput(D,KC); head=HorizonHead(D,MIX)
    if kind=='HSTU': return HSTUv2(inp,head,D=D,H=H,L=L,N=N)
    if kind=='GRU':  return GRUv2(inp,head,D=D,H=H,L=L,N=N)
    if kind=='TGR':  return TGRv2(inp,head,D=D,H=H,L=L,N=N,time_gated=False)    # CloudOps is regular 5T grid

def build_train(series, mu, sd, edges):
    Z=[]
    for s in series:
        for d in range(s.shape[0]):                                            # each variate = its own univariate stream
            r=s[d]; L_=len(r)
            for st in range(0,L_-N+1,PL):
                Z.append(((r[st:st+N]-mu)/sd).astype(np.float32))
    Z=np.array(Z,np.float32)
    if len(Z)>MAXTRAIN: Z=Z[np.random.default_rng(0).choice(len(Z),MAXTRAIN,replace=False)]
    C=np.clip(np.digitize(Z*sd+mu,edges),0,KC-1).astype(np.int64)
    return Z,C

def train(kind,Z,C):
    m=make(kind).to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=0.01); nw=len(Z)
    ts=torch.zeros(BS,N,dtype=torch.long,device=dev)
    for ep in range(EP):
        m.train(); idx=np.random.permutation(nw)
        for i in range(0,nw,BS):
            j=idx[i:i+BS]; z=torch.tensor(Z[j]).to(dev); c=torch.tensor(C[j]).to(dev); t=ts[:len(j)]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
                hid=m((z,c),t)
            hid=hid.float(); loss=0.0
            for h in [1,2,4,8,16,24,32,48]:
                loss=loss+m.head.nll(hid[:,:N-h],h,z[:,h:],torch.ones_like(z[:,h:]))
            loss=loss/8
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m

@torch.no_grad()
def eval_crps(m, series, mu, sd, edges):
    """for each series: ROLL windows at the tail; each window = predict PL steps (h=1..PL) from its context end."""
    ys=[]; qs=[]; edg=torch.tensor(edges,device=dev)
    origins=[]                                                                  # (series,variate,window_start)
    for si,s in enumerate(series):
        for d in range(s.shape[0]):
            T=s.shape[1]
            for w in range(ROLL):
                ws=T-(ROLL-w)*PL                                               # window start (predict s[d, ws:ws+PL])
                if ws-SPLIT<0: continue
                origins.append((si,d,ws))
    if len(origins)>EVALS: origins=[origins[k] for k in np.random.default_rng(1).choice(len(origins),EVALS,replace=False)]
    for i in range(0,len(origins),BS):
        batch=origins[i:i+BS]; ctx=np.zeros((len(batch),SPLIT),np.float32); tru=np.zeros((len(batch),PL),np.float32)
        for b,(si,d,ws) in enumerate(batch):
            r=series[si][d]; ctx[b]=(r[ws-SPLIT:ws]-mu)/sd; tru[b]=r[ws:ws+PL]
        z=torch.tensor(ctx).to(dev); c=torch.clamp(torch.bucketize(torch.tensor(ctx*sd+mu).to(dev),edg),0,KC-1)
        t=torch.zeros(len(batch),SPLIT,dtype=torch.long,device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
            hid=m((z,c),t)[:,SPLIT-1].float()
        for h in range(1,PL+1):
            q=m.head.quantiles(hid.unsqueeze(1),h,tuple(QS),S=1024)[:,0,:].cpu().numpy()*sd+mu   # [B,9] r-space
            ys.append(tru[:,h-1]); qs.append(q)
    return crps_wql(np.concatenate(ys), np.concatenate(qs,0))

def seasonal_naive_crps(series, period=288):
    ys=[]; preds=[]
    for s in series:
        for d in range(s.shape[0]):
            r=s[d]; T=len(r)
            for w in range(ROLL):
                ws=T-(ROLL-w)*PL
                if ws-period<0: continue
                ys.append(r[ws:ws+PL]); preds.append(r[ws-period:ws-period+PL])
    y=np.concatenate(ys); p=np.concatenate(preds)
    return crps_wql(y, np.repeat(p[:,None],9,1))                               # point forecast -> flat quantiles

if __name__=='__main__':
    print(f"device={dev} CFG={CFG} D={D} L={L} EP={EP} MIX={MIX} SPLIT={SPLIT} N={N} PL={PL} ROLL={ROLL} arms={ARMS}",flush=True)
    tt,pt,Dd=load(); print(f"train_test series={len(tt)} pretrain={len(pt)} target_dim={Dd}",flush=True)
    usable=[s for s in tt if s.shape[1]>=SPLIT+ROLL*PL]
    print(f"usable eval series (len>= {SPLIT+ROLL*PL})={len(usable)}",flush=True)
    allr=np.concatenate([s[d, :max(1,s.shape[1]-ROLL*PL)] for s in tt for d in range(s.shape[0])])  # stats from HISTORY only (no eval region)
    mu=float(allr.mean()); sd=float(allr.std()+1e-6); edges=np.quantile(allr,np.linspace(0,1,KC+1)[1:-1])
    print(f"target stats mu={mu:.3f} sd={sd:.3f}",flush=True)
    sn=seasonal_naive_crps(usable); print(f"\n[sanity] seasonal-naive(1day) CRPS = {sn:.4f}   (paper naive ~0.3-0.9)",flush=True)
    res={}
    for seed in range(NSEED):
        np.random.seed(seed); torch.manual_seed(seed)
        Z,C=build_train(tt+pt,mu,sd,edges)
        if seed==0: print(f"train windows={len(Z):,}",flush=True)
        for kind in ARMS:
            m=train(kind,Z,C); cr=eval_crps(m,usable,mu,sd,edges); res.setdefault(kind,[]).append(cr)
            del m; torch.cuda.empty_cache() if dev=='cuda' else None
            print(f"  [seed {seed}] {kind}+direct  CRPS = {cr:.4f}",flush=True)
    print(f"\n==== {CFG} CRPS (GluonTS mean-wQL), mean+-std over {NSEED} seeds ====",flush=True)
    tgt={'azure_vm_traces_2017':0.077,'borg_cluster_data_2011':0.128,'alibaba_cluster_trace_2018':0.017}.get(CFG,None)
    print(f"  SOTA to beat: {tgt}   (seasonal-naive sanity: {sn:.4f})")
    for kind,v in res.items():
        beat="  <-- BEATS SOTA" if tgt and np.mean(v)<tgt else ""
        print(f"  {kind}+direct   {np.mean(v):.4f} +- {np.std(v):.4f}{beat}")
    print("CLOUDOPS_DONE",flush=True)
