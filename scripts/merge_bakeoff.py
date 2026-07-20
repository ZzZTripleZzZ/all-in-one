import pickle, numpy as np
A="/share/hpcproject/zzhang66/nfm/logs/bakeoff_ckpt_s1_464572.pkl"   # seeds 0+1 accumulated
B="/share/hpcproject/zzhang66/nfm/logs/bakeoff_ckpt_s2_464573.pkl"   # seed 2
a=pickle.load(open(A,'rb')); b=pickle.load(open(B,'rb'))
surf={}
for src in (a['surf'],b['surf']):
    for m,d in src.items():
        for k,v in d.items(): surf.setdefault(m,{}).setdefault(k,[]).extend(v)
dist={}
for src in (a['dist'],b['dist']):
    for m,d in src.items():
        for h,v in d.items(): dist.setdefault(m,{}).setdefault(h,[]).extend(v)
LINS=[8,16,32,64,96]; HS=[1,4,16,32,64]
print("==== MERGED 3-seed per-series MASE (mean+-std) ====")
for m in surf:
    if not surf[m]: continue
    n=len(surf[m].get((64,1),[])); print(f"\n-- {m} --  ({n} seeds)  rows=Lin cols=H")
    for L in LINS:
        row=[]
        for h in HS:
            v=surf[m].get((L,h))
            row.append(f"{np.mean(v):6.3f}±{np.std(v):5.3f}" if v else "   -   ")
        print("  L"+str(L).ljust(3)+" "+" ".join(row))
print("\n==== Lin=64 headline row (mean over 3 seeds) ====")
print(f"{'method':<22}"+" ".join(f"h{h:<7}" for h in HS))
for m in ['TGR+direct','TGR-notime+direct','GRU+direct','HSTU+direct','TGR+rollout(sameWts)','GBM-specialist','persistence','EWMA']:
    if m not in surf: continue
    print(f"{m:<22}"+" ".join(f"{np.mean(surf[m][(64,h)]):7.3f}" if (64,h) in surf[m] else "   -   " for h in HS))
print("\n==== distributional Lin=64 (pin90 pin99 CRPS9), 3-seed mean ====")
for m in dist:
    cells=" | ".join(f"h{h}: "+" ".join(f"{np.mean([x[i] for x in dist[m][h]]):.4f}" for i in range(3)) for h in HS if h in dist[m])
    print(f"  {m:<20} {cells}")
print("\nMERGE_DONE")
