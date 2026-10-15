"""Single-correlator <ssss> 3d bootstrap PMP (SDPB JSON format), e.g. the 3d Ising gap problem:
is there a functional excluding (Delta_sigma, Delta_epsilon_gap)?

Blocks from scalar_blocks (bootstrapcollaboration docker image, run through tools/run_in_image.sh).
Conventions (checked against exact 4d blocks): abDeriv[m,n] = d_a^m d_b^n g at a=b=0, with
z = (1 + a + sqrt b)/2, polynomial variable x = Delta - (l + d - 2); the full block is a positive
prefactor (damped rational with base 4 r_*, poles = shiftedPoles) times these polynomials.

usage: python3 tools/ising_pmp.py --nmax 6 --ds 0.518 --de 1.41 --out pmp.json [--lmax 30]
"""
import argparse, glob, json, os, re, subprocess
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def blocks_family(nmax, lmax, order, poles, prec, cache, d12="0", d34="0", spins=None):
    """{l: ({(m,n): [coeffs in x]}, [poles in x])} for the given spins (default: even 0..lmax)"""
    d = os.path.join(cache, f"d3-nmax{nmax}-o{order}-p{poles}-prec{prec}-d12_{d12}-d34_{d34}")
    if not glob.glob(os.path.join(d, "*-L0-*")):
        os.makedirs(d, exist_ok=True)
        subprocess.run([os.path.join(HERE, "run_in_image.sh"), os.path.join(HERE, "sb_root"), "scalar_blocks",
                        "--dim", "3", "--order", str(order), "--max-derivs", str(nmax),
                        "--spin-ranges", f"0-{lmax}", "--poles", str(poles), "--delta-12", str(d12), "--delta-34", str(d34),
                        "--output-ab", "--output-poles", "--precision", str(prec), "-o", d,
                        "--num-threads", str(os.cpu_count())], check=True, capture_output=True)
    out = {}
    for l in (spins if spins is not None else range(0, lmax + 1, 2)):
        s = open(glob.glob(os.path.join(d, f"*-L{l}-*"))[0]).read()
        body, _, ptxt = s.partition("shiftedPoles")
        tab = {}
        for m, n, expr in re.findall(r"abDeriv\[(\d+),(\d+)\]\s*->\s*(.*?)(?=,\s*abDeriv|\}|,?\s*$)", body, re.S):
            c = {}
            for coef, p in re.findall(r"([-+]?\s*-?[0-9.]+(?:[eE][-+]?\d+)?)(?:\*x\^(\d+))?", expr):
                coef = coef.replace(" ", "").replace("+-", "-").lstrip("+")
                if coef in ("", "-"): continue
                c[int(p) if p else 0] = mp.mpf(coef)
            tab[(int(m), int(n))] = [c.get(k, mp.mpf(0)) for k in range(max(c) + 1)]
        out[l] = (tab, [mp.mpf(v) for v in re.findall(r"-?[0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?", ptxt)])
    return out


def blocks(nmax, lmax, order, poles, prec, cache):
    return blocks_family(nmax, lmax, order, poles, prec, cache)


def v_derivs(ds, M, Nn):
    """d_a^i d_b^j v^ds at a=b=0, v = ((1-a)^2 - b)/4"""
    D = {}
    for i in range(M + 1):
        for j in range(Nn + 1):
            t = mp.mpf(4) ** (-ds) * mp.binomial(ds, j) * (-1) ** j * mp.binomial(2 * ds - 2 * j, i) * (-1) ** i
            D[(i, j)] = t * mp.factorial(i) * mp.factorial(j)
    return D


def poly_add(p, q, s=1):
    n = max(len(p), len(q)); p = p + [mp.mpf(0)] * (n - len(p)); q = q + [mp.mpf(0)] * (n - len(q))
    return [a + s * b for a, b in zip(p, q)]


def taylor_shift(c, h):
    """coefficients of p(y + h) in y"""
    n = len(c)
    return [sum(c[k] * mp.binomial(k, j) * h ** (k - j) for k in range(j, n)) for j in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--lmax", type=int, default=30)
    ap.add_argument("--order", type=int, default=60)
    ap.add_argument("--poles", type=int, default=14)
    ap.add_argument("--prec", type=int, default=665, help="bits for scalar_blocks and output digits")
    ap.add_argument("--ds", type=str, default="0.518")
    ap.add_argument("--de", type=str, default="1.41")
    ap.add_argument("--out", default="pmp.json")
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data", "blocks"))
    a = ap.parse_args()
    mp.mp.prec = a.prec
    ds, de = mp.mpf(a.ds), mp.mpf(a.de)
    Lam = 2 * a.nmax - 1
    comps = [(m, n) for n in range(a.nmax) for m in range(1, Lam + 1, 2) if m + 2 * n <= Lam]
    B = blocks(a.nmax, a.lmax, a.order, a.poles, a.prec, a.cache)
    V = v_derivs(ds, Lam, a.nmax)

    def F(tab):
        """crossing vector components as polynomials in x: 2 sum C C d(v^ds) g"""
        out = []
        for m, n in comps:
            acc = [mp.mpf(0)]
            for i in range(m + 1):
                for j in range(n + 1):
                    w = 2 * mp.binomial(m, i) * mp.binomial(n, j) * V[(i, j)]
                    acc = poly_add(acc, [w * c for c in tab[(m - i, n - j)]])
            out.append(acc)
        return out

    digits = int(a.prec * 0.30103) + 5
    s = lambda v: mp.nstr(v, digits, strip_zeros=False, min_fixed=-mp.inf, max_fixed=mp.inf) if v else "0"
    norm = [2 * V[(m, n)] for m, n in comps]                 # identity block g = 1
    blocks_json = []
    for l, (tab, poles) in B.items():
        gap = de if l == 0 else mp.mpf(l + 1)
        h = gap - (l + 1)                                     # x = x' + h, x' >= 0
        polys = [taylor_shift(p, h) for p in F(tab)]
        deg = max(len(p) for p in polys)
        polys = [p + [mp.mpf(0)] * (deg - len(p)) for p in polys]
        blocks_json.append({
            "DampedRational": {"base": s(4 * (3 - 2 * mp.sqrt(2))), "constant": s(mp.mpf(4 * (3 - 2 * mp.sqrt(2))) ** gap),
                               "poles": [s(p - h) for p in poles]},
            "polynomials": [[[[s(c) for c in p] for p in polys]]]})
    pmp = {"objective": ["0"] * len(comps), "normalization": [s(v) for v in norm],
           "PositiveMatrixWithPrefactorArray": blocks_json}
    json.dump(pmp, open(a.out, "w"), indent=0)
    print(f"Lambda={Lam}: {len(comps)} components, {len(blocks_json)} blocks (spins 0..{a.lmax}), "
          f"degrees {min(len(b['polynomials'][0][0][0]) for b in blocks_json) - 1}..{max(len(b['polynomials'][0][0][0]) for b in blocks_json) - 1} -> {a.out}")


if __name__ == "__main__":
    main()
