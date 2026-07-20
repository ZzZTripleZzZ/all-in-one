import gdown, os
D=os.path.expanduser("~/nfm/data/optc/flash"); os.makedirs(D,exist_ok=True)
ids={"10N9ZPolq_L8HivBqzf_jFKbwjSxddsZp":"AIA-51-75.ecar-last.json.gz",
     "1HFSyvmgH0jvdnnnTdKfWRjZYOrLWoIkv":"AIA-201-225.ecar-last.json.gz",
     "1pJLxJsDV8sngiedbfVajMetczIgM3PQd":"AIA-201-225.ecar-2.json.gz",
     "1fRQqc68r8-z5BL7H_eAKIDOeHp7okDuM":"AIA-501-525.ecar-last.json.gz",
     "1VfyGr8wfSe8LBIHBWuYBlU8c2CyEgO5C":"AIA-501-525.ecar-2.json.gz",
     "1xIr8gw-4zc8ESjUpYtrFsbOwhPGUSd15":"extra1.ecar.json.gz",
     "1PvlCp2oQaxEBEFGSQWfcFVj19zLOe7yH":"extra2.ecar.json.gz"}
ok=fail=0
for fid,name in ids.items():
    out=os.path.join(D,name)
    if os.path.exists(out) and os.path.getsize(out)>1_000_000: print("skip",name,flush=True); ok+=1; continue
    try:
        gdown.download(id=fid, output=out, quiet=False)
        if os.path.exists(out) and os.path.getsize(out)>1_000_000: ok+=1
        else: fail+=1; print("EMPTY",name,flush=True)
    except Exception as e: fail+=1; print("FAIL",name,str(e)[:120],flush=True)
print(f"FLASH_DL_DONE ok={ok} fail={fail}",flush=True)
for dp,_,fs in os.walk(D):
    for f in fs: print("  ",f, round(os.path.getsize(os.path.join(dp,f))/1e9,2),"GB")
