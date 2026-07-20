import sys, json, urllib.request
for rec in ["8275861", "8196385"]:
    print(f"\n=== Zenodo record {rec} ===")
    try:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=30) as r:
            d = json.load(r)
        print("title:", d.get("metadata", {}).get("title", "?"))
        for f in d.get("files", []):
            key = f.get("key", "?"); size = f.get("size", 0) / 1e6
            url = f.get("links", {}).get("self", "")
            print(f"  {key:<32} {size:9.1f} MB   {url}")
    except Exception as e:
        print("FAIL:", e)
