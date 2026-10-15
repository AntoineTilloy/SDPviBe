"""Download a Docker Hub image and unpack its layers into a directory (no docker needed).
usage: python3 tools/pull_image.py bootstrapcollaboration/sdpb 3.1.0-amd64 tools/sdpb_root"""
import json, os, sys, tarfile, urllib.request, io
repo, tag, out = sys.argv[1:4]
tok = json.load(urllib.request.urlopen(
    f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]
def get(path, accept):
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/{path}",
                                 headers={"Authorization": f"Bearer {tok}", "Accept": accept})
    return urllib.request.urlopen(req)
acc = ",".join(["application/vnd.docker.distribution.manifest.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.index.v1+json"])
man = json.load(get(f"manifests/{tag}", acc))
if "manifests" in man:                                  # multi-arch index
    d = [m for m in man["manifests"] if m.get("platform", {}).get("architecture") == "amd64"][0]
    man = json.load(get(f"manifests/{d['digest']}", acc))
cfg = json.load(get(f"blobs/{man['config']['digest']}", "*/*"))
os.makedirs(out, exist_ok=True)
for L in man["layers"]:
    print("layer", L["digest"][:19], L["size"] // 2**20, "MB", flush=True)
    data = get(f"blobs/{L['digest']}", "*/*").read()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            name = os.path.basename(m.name)
            if name.startswith(".wh."):                 # whiteout: delete from lower layer
                p = os.path.join(out, os.path.dirname(m.name), name[4:])
                if os.path.lexists(p) and not os.path.isdir(p): os.remove(p)
                continue
            if m.isdev(): continue
            try: t.extract(m, out, filter="tar")
            except Exception as e: pass
json.dump(cfg, open(os.path.join(out, "image_config.json"), "w"), indent=1)
print("env", cfg["config"].get("Env"), "entry", cfg["config"].get("Entrypoint"), cfg["config"].get("Cmd"))
