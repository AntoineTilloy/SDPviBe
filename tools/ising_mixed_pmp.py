"""Mixed-correlator 3d Ising bootstrap PMP <ssss>, <ssee>, <eeee> (Kos, Poland, Simmons-Duffin
arXiv:1406.4858, eqs. 3.7, 3.12, 3.20) in SDPB JSON format.  Feasibility of (Delta_sigma, Delta_eps):
  (1 1) a.V_{+,0,0} (1 1)^T = 1                        normalisation (unit operator)
  a.V_{+,De,0} >= 0 (2x2)            isolated eps      a.V_{-,Ds,0} >= 0            isolated sigma
  a.V_{+,D,0}  >= 0 (2x2), D >= gap_even (eps')        a.V_{-,D,0} >= 0, D >= gap_odd (sigma')
  a.V_{+,D,l}  >= 0, l even >= 2, D >= l+1             a.V_{-,D,l} >= 0, l >= 1, D >= l+1
with F^{ij,kl}_{-+} = v^{(Dk+Dj)/2} g^{Dij,Dkl}(u,v) -+ u^{(Dk+Dj)/2} g^{Dij,Dkl}(v,u) and
  V_- = (0, 0, F^{se,se}_-, (-1)^l F^{es,se}_-, -(-1)^l F^{es,se}_+)
  V_+ = ([[F^{ss,ss}_-,0],[0,0]], [[0,0],[0,F^{ee,ee}_-]], 0, [[0,F^{ss,ee}_-/2],[.,0]], [[0,F^{ss,ee}_+/2],[.,0]])
Functional components: per equation, d_a^m d_b^n at a=b=0 with m+2n <= 2 nmax - 1, m odd for F_-
equations (1-4), m even for F_+ (5): 5 nmax (nmax+1)/2 components.
Block conventions (scalar_blocks): checked in tools/ising_pmp.py and against exact 4d blocks.
Sign convention: scalar_blocks' blocks are positive at the crossing point for all spins (Dolan-Osborn
k-functions without the (-1/2)^l factor).  In that normalisation the s-channel coefficients of
<s e s e> are (-1)^l lambda_{seO}^2 (check: leading-twist GFF expansion of u^{(Ds+De)/2} gives a
negative spin-1 coefficient), whereas KPSV's normalisation has them positive.  Hence the vector
used here is (-1)^l times KPSV's V_-.  Verified: reproduces the nmax=6 island of KPSV fig. 4.

usage: python3 tools/ising_mixed_pmp.py --nmax 6 --ds 0.5181 --de 1.4126 --out pmp.json
"""
import argparse, json, os, sys
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ising_pmp import blocks_family, poly_add, taylor_shift


_JOB = None


def _run_job(t): return _JOB(t)


def uv_derivs(p, M, Nn):
    """d_a^i d_b^j v^p at a=b=0, v = ((1-a)^2 - b)/4"""
    D = {}
    for i in range(M + 1):
        for j in range(Nn + 1):
            t = mp.mpf(4) ** (-p) * mp.binomial(p, j) * (-1) ** j * mp.binomial(2 * p - 2 * j, i) * (-1) ** i
            D[(i, j)] = t * mp.factorial(i) * mp.factorial(j)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--lmax", type=int, default=30)
    ap.add_argument("--order", type=int, default=60)
    ap.add_argument("--poles", type=int, default=14)
    ap.add_argument("--prec", type=int, default=665)
    ap.add_argument("--ds", type=str, required=True)
    ap.add_argument("--de", type=str, required=True)
    ap.add_argument("--gap_even", type=str, default="3")
    ap.add_argument("--gap_odd", type=str, default="3")
    ap.add_argument("--out", default="pmp.json")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "blocks"))
    a = ap.parse_args()
    mp.mp.prec = a.prec
    ds, de = mp.mpf(a.ds), mp.mpf(a.de)
    Lam = 2 * a.nmax - 1
    odd = [(m, n) for n in range(a.nmax) for m in range(1, Lam + 1, 2) if m + 2 * n <= Lam]
    even = [(m, n) for n in range(a.nmax) for m in range(0, Lam + 1, 2) if m + 2 * n <= Lam]
    comps = [(e, mn) for e in range(4) for mn in odd] + [(4, mn) for mn in even]
    NC = len(comps)
    dlt = ds - de
    fam0 = blocks_family(a.nmax, a.lmax, a.order, a.poles, a.prec, a.cache, "0", "0", spins=range(0, a.lmax + 1, 2))
    famA = blocks_family(a.nmax, a.lmax, a.order, a.poles, a.prec, a.cache, mp.nstr(dlt, 20), mp.nstr(dlt, 20), spins=range(a.lmax + 1))
    famB = blocks_family(a.nmax, a.lmax, a.order, a.poles, a.prec, a.cache, mp.nstr(-dlt, 20), mp.nstr(dlt, 20), spins=range(a.lmax + 1))
    Vd = {p: uv_derivs(p, Lam, a.nmax) for p in (ds, de, (ds + de) / 2)}

    def Fvec(tab, p, parity):
        """{(m,n): polynomial} of F_{-} (parity 'odd') or F_{+} ('even') derivatives: 2 h_mn, h = v^p g"""
        out = {}
        for m, n in (odd if parity == "odd" else even):
            acc = [mp.mpf(0)]
            for i in range(m + 1):
                for j in range(n + 1):
                    w = 2 * mp.binomial(m, i) * mp.binomial(n, j) * Vd[p][(i, j)]
                    acc = poly_add(acc, [w * c for c in tab[(m - i, n - j)]])
            out[(m, n)] = acc
        return out

    def ident(p, parity):
        """F_{-+} of the unit operator (g = 1)"""
        return {mn: [2 * Vd[p][mn]] for mn in (odd if parity == "odd" else even)}

    Z = lambda: [mp.mpf(0)]

    def Vplus(t_ss, t_ee, t_se_m, t_se_p):
        """2x2 matrix of component lists (polynomials) for V_+"""
        M = [[[Z() for _ in range(NC)] for _ in range(2)] for _ in range(2)]
        for k, (e, mn) in enumerate(comps):
            if e == 0: M[0][0][k] = t_ss[mn]
            elif e == 1: M[1][1][k] = t_ee[mn]
            elif e == 3: M[0][1][k] = M[1][0][k] = [c / 2 for c in t_se_m[mn]]
            elif e == 4: M[0][1][k] = M[1][0][k] = [c / 2 for c in t_se_p[mn]]
        return M

    def Vminus(t_sese, t_esse_m, t_esse_p, l):
        v = [Z() for _ in range(NC)]
        sg = (-1) ** l
        fl = sg      # block convention: see module docstring
        for k, (e, mn) in enumerate(comps):
            if e == 2: v[k] = [fl * c for c in t_sese[mn]]
            elif e == 3: v[k] = [fl * sg * c for c in t_esse_m[mn]]
            elif e == 4: v[k] = [-fl * sg * c for c in t_esse_p[mn]]
        return [[v]]

    digits = int(a.prec * 0.30103) + 5
    s = lambda v: mp.nstr(v, digits, strip_zeros=False, min_fixed=-mp.inf, max_fixed=mp.inf) if v else "0"
    base = 4 * (3 - 2 * mp.sqrt(2))

    def entry(M, D0, l, poles, iso=False):
        """PositiveMatrixWithPrefactor for x' = D - D0 >= 0 (or the single point D0 if iso)"""
        h = D0 - (l + 1)
        dim = len(M)
        if iso:
            chi = base ** D0 / mp.fprod([h - pl for pl in poles])
            P = [[[[s(chi * mp.polyval(list(reversed(p)), h))] for p in M[i][j]] for j in range(dim)] for i in range(dim)]
            return {"DampedRational": {"base": s(base), "constant": "1", "poles": []}, "polynomials": P}
        polys = [[[taylor_shift(p, h) for p in M[i][j]] for j in range(dim)] for i in range(dim)]
        deg = max(len(p) for row in polys for col in row for p in col)
        P = [[[[s(c) for c in p + [mp.mpf(0)] * (deg - len(p))] for p in polys[i][j]] for j in range(dim)] for i in range(dim)]
        return {"DampedRational": {"base": s(base), "constant": s(base ** D0), "poles": [s(pl - h) for pl in poles]},
                "polynomials": P}

    pm = (ds + de) / 2

    def job(task):
        kind, l = task
        if kind == "even":                                   # Z2-even, even spins (2x2)
            tab, poles = fam0[l]
            M = Vplus(Fvec(tab, ds, "odd"), Fvec(tab, de, "odd"), Fvec(tab, pm, "odd"), Fvec(tab, pm, "even"))
            if l == 0:
                return [entry(M, de, 0, poles, iso=True), entry(M, mp.mpf(a.gap_even), 0, poles)]
            return [entry(M, mp.mpf(l + 1), l, poles)]
        tA, poles = famA[l]; tB, polesB = famB[l]            # Z2-odd, all spins (1x1)
        assert [mp.nstr(x, 20) for x in poles] == [mp.nstr(x, 20) for x in polesB]
        M = Vminus(Fvec(tA, pm, "odd"), Fvec(tB, ds, "odd"), Fvec(tB, ds, "even"), l)
        if l == 0:
            return [entry(M, ds, 0, poles, iso=True), entry(M, mp.mpf(a.gap_odd), 0, poles)]
        return [entry(M, mp.mpf(l + 1), l, poles)]

    tasks = [("even", l) for l in range(0, a.lmax + 1, 2)] + [("odd", l) for l in range(0, a.lmax + 1)]
    global _JOB
    _JOB = job
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(min(len(tasks), os.cpu_count())) as pool:
        out = [e for r in pool.map(_run_job, tasks) for e in r]
    # normalisation: (1 1) a.V_{+,0,0} (1 1)^T
    I = Vplus(ident(ds, "odd"), ident(de, "odd"), ident(pm, "odd"), ident(pm, "even"))
    norm = [I[0][0][k][0] + I[1][1][k][0] + 2 * I[0][1][k][0] for k in range(NC)]
    pmp = {"objective": ["0"] * NC, "normalization": [s(v) for v in norm], "PositiveMatrixWithPrefactorArray": out}
    json.dump(pmp, open(a.out, "w"), indent=0)
    print(f"nmax={a.nmax} (Lambda={Lam}): {NC} components, {len(out)} positivity blocks "
          f"(spins 0..{a.lmax}) at (Ds, De) = ({a.ds}, {a.de}), gaps even {a.gap_even} odd {a.gap_odd} -> {a.out}")


if __name__ == "__main__":
    main()
