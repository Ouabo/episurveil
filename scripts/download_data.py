"""Download configured open datasets with checksums and retrieval metadata."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, urllib.request, yaml
ROOT=Path(__file__).resolve().parents[1]
def download(name,spec):
    target=ROOT/"data"/"raw"/f"{name}.{spec.get('format','bin')}"; target.parent.mkdir(parents=True,exist_ok=True)
    try:
        urllib.request.urlretrieve(spec["url"],target)
        if target.read_bytes()[:40].startswith(b"version https://git-lfs.github.com/spec"):
            digest=""; status="failed: Git LFS pointer downloaded; retrieve the release asset or Zenodo file"
        else:
            digest=hashlib.sha256(target.read_bytes()).hexdigest(); status="downloaded"
    except Exception as exc: digest=""; status=f"failed: {exc}"
    meta={"name":name,"url":spec["url"],"authority":spec.get("authority"),"disease":spec.get("disease"),"licence":spec.get("licence"),"retrieved_at":datetime.now(timezone.utc).isoformat(),"sha256":digest,"status":status}
    out=ROOT/"data"/"metadata"/f"{name}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(meta,indent=2),encoding="utf-8"); return meta
if __name__=="__main__":
    cfg=yaml.safe_load((ROOT/"config"/"data_sources.yaml").read_text())
    for name,spec in cfg["sources"].items(): print(download(name,spec))
