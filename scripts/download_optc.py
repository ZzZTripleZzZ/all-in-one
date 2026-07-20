import gdown, os, time
url="https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa"
D=os.path.expanduser("~/nfm/data/optc"); os.makedirs(D,exist_ok=True)
res=gdown.download_folder(url=url, skip_download=True, quiet=True)
tgt=[f for f in res if f.path.startswith("ecar/evaluation/") or f.path=="OpTCRedTeamGroundTruth.pdf"]
print(f"{len(tgt)} files to fetch",flush=True)
ok=fail=skip=0
for i,f in enumerate(tgt):
    out=os.path.join(D,f.path); os.makedirs(os.path.dirname(out),exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out)>0: skip+=1; ok+=1; continue
    done=False
    for a in range(4):
        try:
            gdown.download(id=f.id, output=out, quiet=True)
            if os.path.exists(out) and os.path.getsize(out)>0: ok+=1; done=True; break
        except Exception as e: print(f"  retry{a} {f.path}: {str(e)[:70]}",flush=True); time.sleep(15)
    if not done: fail+=1; print("GIVEUP",f.path,flush=True)
    if i%10==0: print(f"  {i+1}/{len(tgt)} ok={ok} skip={skip} fail={fail}",flush=True)
print(f"OPTC_DL_DONE ok={ok} fail={fail} total_bytes={sum(os.path.getsize(os.path.join(dp,fn)) for dp,_,fs in os.walk(D) for fn in fs)}")
