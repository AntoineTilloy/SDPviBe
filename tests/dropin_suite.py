"""Drop-in test suite: SDPviBe vs official SDPB on the same SDPs, with the same command lines.

  python tests/dropin_suite.py --work DIR --sdpb "mpirun -n 32 /path/sdpb" --pmp2sdp "mpirun -n 32 /path/pmp2sdp" \
         [--spectrum "/path/spectrum"] [--sdpvibe "bin/sdpb"] [--cases tests/dropin_cases.json] [--only NAME]

For each case: pmp2sdp (binary SDP) -> sdpb and sdpvibe with identical options -> compare terminate reason,
iteration count, out.txt values, y/z/x/c_minus_By, iterations.json (per-iteration agreement), spectrum.
Feature tests (on the case marked "features"): checkpoint resume, text checkpoints in both directions
(SDPB -> SDPviBe and SDPviBe -> SDPB), zip input, --maxRuntime, SIGTERM, --paramFile.
Writes DIR/report.json and DIR/report.md.
"""
import argparse, json, os, shlex, shutil, signal, subprocess, sys, time
import mpmath as mp

mp.mp.prec = 1200
KEYS = ["mu", "P-obj", "D-obj", "P-err", "D-err", "P-step", "D-step", "beta"]


def run(cmd, log, timeout=None, env=None):
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    return p.returncode, time.time() - t0


def out_txt(d):
    r = {}
    for line in open(os.path.join(d, "out.txt")):
        k, _, v = line.partition("=")
        r[k.strip()] = v.strip().rstrip(";").strip('"')
    return r


def digits(a, b):
    """-log10 relative difference (99 if equal)"""
    a, b = mp.mpf(a), mp.mpf(b)
    if a == b: return 99.0
    s = max(abs(a), abs(b))
    return float(-mp.log10(abs(a - b) / s)) if s else 99.0


def vec(path):
    t = open(path).read().split()
    return [mp.mpf(x) for x in t[2:]]


def vec_digits(pa, pb):
    if not (os.path.exists(pa) and os.path.exists(pb)): return None
    a, b = vec(pa), vec(pb)
    if len(a) != len(b): return -1.0
    m = max((abs(x) for x in a), default=mp.mpf(0))
    if m == 0: return 99.0
    d = max(abs(x - y) for x, y in zip(a, b)) / m                  # normwise
    return 99.0 if d == 0 else float(-mp.log10(d))


def iter_digits(pa, pb):
    a = json.load(open(pa)); b = json.load(open(pb))
    worst = 99.0
    for ra, rb in zip(a, b):
        for k in KEYS:
            worst = min(worst, digits(ra[k], rb[k]))
    return len(a), len(b), worst, a, b


def median_iter_time(rows):
    t = sorted(r["iter_time"] for r in rows[2:]) or [r["iter_time"] for r in rows]
    return t[len(t) // 2] if t else None


def compare(ref, gpu):
    oa, ob = out_txt(ref), out_txt(gpu)
    na, nb, wd, ia, ib = iter_digits(os.path.join(ref, "iterations.json"), os.path.join(gpu, "iterations.json"))
    res = dict(reason_sdpb=oa["terminateReason"], reason_sdpvibe=ob["terminateReason"],
               same_reason=oa["terminateReason"] == ob["terminateReason"], iterations_sdpb=na, iterations_sdpvibe=nb,
               iterations_digits=round(wd, 1),
               out_digits={k: round(digits(oa[k], ob[k]), 1) for k in ("primalObjective", "dualObjective", "dualityGap")},
               y_digits=vec_digits(os.path.join(ref, "y.txt"), os.path.join(gpu, "y.txt")),
               z_digits=vec_digits(os.path.join(ref, "z.txt"), os.path.join(gpu, "z.txt")),
               x0_digits=vec_digits(os.path.join(ref, "x_0.txt"), os.path.join(gpu, "x_0.txt")),
               iter_time_sdpb=median_iter_time(ia), iter_time_sdpvibe=median_iter_time(ib))
    ca, cb = (os.path.join(d, "c_minus_By", "c_minus_By.json") for d in (ref, gpu))
    if os.path.exists(ca) and os.path.exists(cb):
        A = json.load(open(ca))["c_minus_By"]; B = json.load(open(cb))["c_minus_By"]
        if len(A) != len(B):
            res["c_minus_By_digits"] = -1.0
        else:
            m = max(abs(mp.mpf(v)) for blk in A for v in blk)
            d = max(abs(mp.mpf(x) - mp.mpf(y)) for ba, bb in zip(A, B) for x, y in zip(ba, bb))
            res["c_minus_By_digits"] = 99.0 if d == 0 else round(float(-mp.log10(d / m)), 1)
    return res


def spectrum_compare(a, b):
    try:
        A = json.load(open(a)); B = json.load(open(b))
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    za = [[z["zero"] for z in blk["zeros"]] for blk in A]; zb = [[z["zero"] for z in blk["zeros"]] for blk in B]
    same_count = [len(x) == len(y) for x, y in zip(za, zb)]
    worst = 99.0
    for x, y in zip(za, zb):
        for u, v in zip(sorted(map(mp.mpf, x)), sorted(map(mp.mpf, y))):
            worst = min(worst, digits(u, v))
    return {"blocks": len(A), "blocks_same_zero_count": sum(same_count), "zeros_digits": round(worst, 1),
            "zeros": sum(map(len, za))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--sdpb", required=True); ap.add_argument("--pmp2sdp", required=True)
    ap.add_argument("--spectrum"); ap.add_argument("--sdpvibe", default="bin/sdpb")
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dropin_cases.json"))
    ap.add_argument("--only"); ap.add_argument("--no-features", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    cases = json.load(open(a.cases))
    report = {"cases": {}, "features": {}}
    rp = os.path.join(a.work, "report.json")
    for c in cases:
        if a.only and c["name"] != a.only: continue
        name = c["name"]; W = os.path.join(a.work, name); os.makedirs(W, exist_ok=True)
        prec = c["precision"]; opts = c.get("options", "")
        common = f"--precision {prec} {opts}"
        sdp = os.path.join(W, "sdp")
        print(f"== {name}", flush=True)
        r = {}
        if not os.path.exists(os.path.join(sdp, "control.json")):
            rc, t = run(f"{a.pmp2sdp} --precision {prec} -i {c['pmp']} -o {sdp}", os.path.join(W, "pmp2sdp.log"))
            r["pmp2sdp_s"] = round(t, 1)
            if rc: r["error"] = "pmp2sdp failed"; report["cases"][name] = r; continue
        ws = "--writeSolution=x,y,z" if os.path.exists(os.path.join(sdp, "normalization.json")) else "--writeSolution=x,y"
        for tag, exe in (("sdpb", a.sdpb), ("sdpvibe", a.sdpvibe)):
            o = os.path.join(W, f"out_{tag}")
            shutil.rmtree(o, ignore_errors=True)
            rc, t = run(f"{exe} -s {sdp} -o {o} -c {W}/ck_{tag} {common} {ws} --noFinalCheckpoint", os.path.join(W, f"{tag}.log"),
                        timeout=c.get("timeout", 36000))
            r[f"{tag}_rc"] = rc; r[f"{tag}_s"] = round(t, 1)
        if r["sdpb_rc"] == 0 and r["sdpvibe_rc"] == 0:
            r.update(compare(os.path.join(W, "out_sdpb"), os.path.join(W, "out_sdpvibe")))
            if a.spectrum and os.path.exists(os.path.join(sdp, "pmp_info.json")):
                for tag in ("sdpb", "sdpvibe"):
                    run(f"{a.spectrum} --pmpInfo={sdp}/pmp_info.json --solution={W}/out_{tag} --precision={prec} "
                        f"--output={W}/spectrum_{tag}.json", os.path.join(W, f"spectrum_{tag}.log"))
                r["spectrum"] = spectrum_compare(os.path.join(W, "spectrum_sdpb.json"), os.path.join(W, "spectrum_sdpvibe.json"))
        report["cases"][name] = r
        print(json.dumps(r), flush=True)
        json.dump(report, open(rp, "w"), indent=1)

    feat = next((c for c in cases if c.get("features")), None)
    if feat and not a.no_features and not a.only:
        try:
            features(a, feat, report, rp)
        except Exception as e:                                       # keep the case results
            import traceback; traceback.print_exc()
            report["features"]["error"] = repr(e)
            json.dump(report, open(rp, "w"), indent=1)
    write_md(a, report)


def features(a, feat, report, rp):
    if True:
        W = os.path.join(a.work, "features"); shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
        sdp = os.path.join(a.work, feat["name"], "sdp")
        common = f"--precision {feat['precision']} {feat.get('options', '')}"
        F = report["features"]
        v = a.sdpvibe
        # 1. resume from a binary checkpoint == uninterrupted run (bitwise)
        run(f"{v} -s {sdp} -o {W}/full -c {W}/full.ck {common} --noFinalCheckpoint", f"{W}/full.log")
        run(f"{v} -s {sdp} -o {W}/part -c {W}/res.ck {common} --maxIterations 20", f"{W}/part.log")
        run(f"{v} -s {sdp} -o {W}/rest -c {W}/res.ck {common}", f"{W}/rest.log")
        fa = json.load(open(f"{W}/full/iterations.json")); fb = json.load(open(f"{W}/part/iterations.json")) + json.load(open(f"{W}/rest/iterations.json"))
        F["resume_bitwise"] = len(fa) == len(fb) and all(x[k] == y[k] for x, y in zip(fa, fb) for k in KEYS)
        F["resume_same_out"] = out_txt(f"{W}/full")["terminateReason"] == out_txt(f"{W}/rest")["terminateReason"]
        # 2. text checkpoint SDPB -> SDPviBe and SDPviBe -> SDPB
        run(f"{a.sdpb} -s {sdp} -o {W}/sref -c {W}/sref.ck {common} --noFinalCheckpoint", f"{W}/sref.log")
        run(f"{a.sdpb} -s {sdp} -o {W}/s20 -c {W}/s20.ck {common} --maxIterations 20 --writeSolution=x,y,X,Y --noFinalCheckpoint", f"{W}/s20.log")
        rc, _ = run(f"{v} -s {sdp} -o {W}/s2v -c {W}/s2v.ck -i {W}/s20 {common}", f"{W}/s2v.log")
        F["text_ck_sdpb_to_sdpvibe"] = dict(rc=rc, reason=out_txt(f"{W}/s2v")["terminateReason"] if rc == 0 else None,
                                           ref_reason=out_txt(f"{W}/sref")["terminateReason"],
                                           iterations=len(json.load(open(f"{W}/s2v/iterations.json"))) + 20 if rc == 0 else None,
                                           ref_iterations=len(json.load(open(f"{W}/sref/iterations.json"))),
                                           dualObjective_digits=round(digits(out_txt(f"{W}/s2v")["dualObjective"], out_txt(f"{W}/sref")["dualObjective"]), 1) if rc == 0 else None)
        run(f"{v} -s {sdp} -o {W}/v20 -c {W}/v20.ck {common} --maxIterations 20 --writeSolution=x,y,X,Y --noFinalCheckpoint", f"{W}/v20.log")
        rc, _ = run(f"{a.sdpb} -s {sdp} -o {W}/v2s -c {W}/v2s.ck -i {W}/v20 {common}", f"{W}/v2s.log")
        F["text_ck_sdpvibe_to_sdpb"] = dict(rc=rc, reason=out_txt(f"{W}/v2s")["terminateReason"] if rc == 0 else None,
                                           iterations=len(json.load(open(f"{W}/v2s/iterations.json"))) + 20 if rc == 0 else None,
                                           dualObjective_digits=round(digits(out_txt(f"{W}/v2s")["dualObjective"], out_txt(f"{W}/sref")["dualObjective"]), 1) if rc == 0 else None)
        # 3. zip input == directory input (bitwise)
        run(f"{a.pmp2sdp} --precision {feat['precision']} -i {feat['pmp']} -o {W}/sdp.zip --zip", f"{W}/zip.log")
        run(f"{v} -s {W}/sdp.zip -o {W}/zipout -c {W}/zip.ck {common} --noFinalCheckpoint", f"{W}/zipout.log")
        za = json.load(open(f"{W}/zipout/iterations.json"))
        F["zip_bitwise"] = len(za) == len(fa) and all(x[k] == y[k] for x, y in zip(fa, za) for k in KEYS)
        # 4. maxRuntime
        run(f"{v} -s {sdp} -o {W}/rt -c {W}/rt.ck {common} --maxRuntime 1 --noFinalCheckpoint", f"{W}/rt.log")
        F["maxRuntime_reason"] = out_txt(f"{W}/rt")["terminateReason"]
        # 5. SIGTERM: stop, checkpoint, resume
        p = subprocess.Popen(f"exec {v} -s {sdp} -o {W}/sig -c {W}/sig.ck {common}", shell=True,
                             stdout=open(f"{W}/sig.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(600):
            time.sleep(0.5)
            if os.path.exists(f"{W}/sig/iterations.json") and open(f"{W}/sig/iterations.json").read().count('"iteration"') >= 5: break
        p.send_signal(signal.SIGTERM); rc = p.wait(timeout=3600)
        F["sigterm"] = dict(rc=rc, reason=out_txt(f"{W}/sig")["terminateReason"], checkpoint=os.path.exists(f"{W}/sig.ck/checkpoint.json"))
        run(f"{v} -s {sdp} -o {W}/sig2 -c {W}/sig.ck {common}", f"{W}/sig2.log")
        F["sigterm"]["resumed_reason"] = out_txt(f"{W}/sig2")["terminateReason"]
        # 6. paramFile
        with open(f"{W}/params.txt", "w") as f:
            f.write(f"precision = {feat['precision']}\nmaxIterations = 7\n# comment\nverbosity = 0\n")
        run(f"{v} -s {sdp} -o {W}/pf -c {W}/pf.ck -p {W}/params.txt {feat.get('options', '')} --noFinalCheckpoint", f"{W}/pf.log")
        F["paramFile"] = dict(reason=out_txt(f"{W}/pf")["terminateReason"], iterations=len(json.load(open(f"{W}/pf/iterations.json"))))
        print(json.dumps(F), flush=True)
        json.dump(report, open(rp, "w"), indent=1)

def write_md(a, report):
    with open(os.path.join(a.work, "report.md"), "w") as f:
        f.write("| case | verdict SDPB / SDPviBe | iterations | per-iteration agreement (digits) | y | s/it SDPB | s/it SDPviBe |\n|---|---|---|---|---|---|---|\n")
        for n, r in report["cases"].items():
            if "iterations_sdpb" not in r: f.write(f"| {n} | {r} |\n"); continue
            f.write(f"| {n} | {r['reason_sdpb']} / {r['reason_sdpvibe']} | {r['iterations_sdpb']} / {r['iterations_sdpvibe']} | "
                    f"{r['iterations_digits']} | {r['y_digits']} | {r['iter_time_sdpb']} | {r['iter_time_sdpvibe']} |\n")
        f.write("\nfeatures: " + json.dumps(report["features"], indent=1) + "\n")


if __name__ == "__main__":
    main()
