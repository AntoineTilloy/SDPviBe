"""Grid scan of the mixed sigma/eps Ising feasibility problem with the GPU solver.
Rows of constant delta = Ds - De share block tables (as in KPSV fig. 5).
usage: python tools/scan_mixed.py --nmax 6 --deltas -0.885,-0.895 --ds 0.516:0.523:0.001 --out results/scan.csv"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--nmax", type=int, default=6)
ap.add_argument("--lmax", type=int, default=30)
ap.add_argument("--deltas", required=True)
ap.add_argument("--ds", required=True, help="start:stop:step")
ap.add_argument("--precision", type=int, default=768)
ap.add_argument("--out", required=True)
ap.add_argument("--extra", default="", help="extra args for ising_mixed_pmp.py")
a = ap.parse_args()
lo, hi, st = map(float, a.ds.split(":"))
dss = [round(lo + i * st, 6) for i in range(int(round((hi - lo) / st)) + 1)]
env = dict(os.environ, OPAL_PREFIX=os.path.join(HERE, "sdpb_root", "usr"), OMPI_MCA_ess_singleton_isolated="1")
done = set()
if os.path.exists(a.out):
    done = {tuple(l.split(",")[:2]) for l in open(a.out).read().split("\n")[1:] if l}
else:
    open(a.out, "w").write("ds,de,verdict,iterations,seconds\n")
py = sys.executable
for d in [float(x) for x in a.deltas.split(",")]:
    for ds in dss:
        de = round(ds - d, 6)
        if (f"{ds}", f"{de}") in done: continue
        r = os.path.join(ROOT, "runs", f"scan_n{a.nmax}", f"{ds}_{de}")
        os.makedirs(r, exist_ok=True)
        t0 = time.time()
        subprocess.run([py, os.path.join(HERE, "ising_mixed_pmp.py"), "--nmax", str(a.nmax), "--lmax", str(a.lmax),
                        "--ds", str(ds), "--de", str(de), "--out", os.path.join(r, "pmp.json")] + a.extra.split(),
                       check=True, capture_output=True)
        subprocess.run([os.path.join(HERE, "run_in_image.sh"), os.path.join(HERE, "sdpb_root"), "pmp2sdp", "--precision",
                        str(a.precision), "-i", os.path.join(r, "pmp.json"), "-o", os.path.join(r, "sdp"),
                        "--outputFormat", "json"], check=True, capture_output=True, env=env)
        out = subprocess.run([py, "-m", "sdpvibe", "-s", os.path.join(r, "sdp"), "--precision",
                              str(a.precision), "--findPrimalFeasible", "--findDualFeasible", "-o", os.path.join(r, "gpu_out")],
                             capture_output=True, text=True).stdout
        verdict = "excluded" if "dual feasible" in out else ("allowed" if "primal feasible" in out else "unknown")
        its = next((l.split("=")[1].strip() for l in out.splitlines() if l.startswith("iterations")), "?")
        with open(a.out, "a") as f:
            f.write(f"{ds},{de},{verdict},{its},{time.time() - t0:.0f}\n")
        print(f"({ds}, {de}) {verdict} [{its} it, {time.time() - t0:.0f}s]", flush=True)
        subprocess.run(["rm", "-rf", os.path.join(r, "sdp"), os.path.join(r, "pmp.json")])
