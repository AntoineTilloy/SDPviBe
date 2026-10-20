"""SDPB's output files, in SDPB 3.1's formats (sdpb/save_solution.cxx, SDP_Solver/run/print_iteration.cxx):

  out.txt                     terminateReason, objectives, gap, errors, "Solver runtime" (s)
  iterations.json             one record per iteration (streamed; an older file is moved to iterations.<k>.json)
  c_minus_By/c_minus_By.json  c - B y per block (used by SDPB's `spectrum`)
  y.txt, z.txt, x_<j>.txt     vectors: "H W" then one value per line
  X_matrix_<2j+p>.txt, Y_...  matrices, same layout (row-major); together with x, y a text checkpoint
Numbers are printed like SDPB prints BigFloats (C++ %g with ceil(precision log10 2) + 1 digits).
"""
import json, os
import mpmath as mp
from .solver import fmt_big


def _vals(A):
    """MPF -> flat list of mpmath numbers (row-major)"""
    return list(A.to_mp().reshape(-1))


def write_text_matrix(path, A, digits, shape=None):
    h, w = shape or (A.shape if len(A.shape) == 2 else (A.shape[0], 1))
    with open(path, "w") as f:
        f.write(f"{h} {w}\n")
        for v in _vals(A):
            f.write(fmt_big(v, digits) + "\n")


def read_text_matrix(path, L):
    """'H W' then H*W values -> MPF (H, W) on the GPU"""
    from . import mpf as F
    with open(path) as f:
        h, w = map(int, f.readline().split())
        vals = f.read().split()
    if len(vals) != h * w:
        raise ValueError(f"{path}: expected {h * w} values, found {len(vals)}")
    if h * w == 0:
        return F.zeros((h, w), L)
    return F.from_strings(vals, L, (h, w))


def write_out_txt(out_dir, res, digits, runtime_s):
    f = lambda v: fmt_big(v, digits)
    with open(os.path.join(out_dir, "out.txt"), "w") as o:
        o.write(f'terminateReason = "{res["terminateReason"]}";\n'
                f'primalObjective = {f(res["primalObjective"])};\n'
                f'dualObjective   = {f(res["dualObjective"])};\n'
                f'dualityGap      = {f(res["dualityGap"])};\n'
                f'primalError     = {f(res["primalError"])};\n'
                f'dualError       = {f(res["dualError"])};\n'
                f'Solver runtime  = {int(runtime_s)};\n')


def z_from_y(y, normalization):
    """SDPB: insert into y, at the position of the largest |normalization| entry, the value making n.z = 1"""
    n = [mp.mpf(v) for v in normalization]
    if len(n) != len(y) + 1:
        raise ValueError(f"normalization has {len(n)} entries, y has {len(y)}")
    k = max(range(len(n)), key=lambda i: abs(n[i]))                  # first maximal |n_i| (max_normalization_index)
    z = list(y[:k]) + [mp.mpf(0)] + list(y[k:])
    nz = mp.fsum(a * b for a, b in zip(n, z))
    z[k] = (1 - nz) / n[k]
    return z


def write_solution(out_dir, state, parts, digits, normalization=None):
    """parts: subset of {'x', 'y', 'z', 'X', 'Y'} (SDPB's --writeSolution)"""
    os.makedirs(out_dir, exist_ok=True)
    if "y" in parts or "z" in parts:
        y = _vals(state["y"])
        if "y" in parts:
            write_vec(os.path.join(out_dir, "y.txt"), y, digits)
        if "z" in parts:
            if normalization is None:
                raise ValueError("Please provide SDP with valid normalization.json or exclude z from --writeSolution arguments.")
            write_vec(os.path.join(out_dir, "z.txt"), z_from_y(y, normalization), digits)
    for j in range(len(state["x"])):
        if "x" in parts:
            write_text_matrix(os.path.join(out_dir, f"x_{j}.txt"), state["x"][j], digits, (state["x"][j].shape[0], 1))
        for p in (0, 1):
            for tag in ("X", "Y"):
                M = state[tag][j][p]
                if tag in parts and M.shape[0] != 0:
                    write_text_matrix(os.path.join(out_dir, f"{tag}_matrix_{2 * j + p}.txt"), M, digits)


def write_vec(path, vals, digits):
    with open(path, "w") as f:
        f.write(f"{len(vals)} 1\n")
        for v in vals:
            f.write(fmt_big(v, digits) + "\n")


def write_c_minus_By(path, blocks, digits):
    """blocks: list (SDPB block order) of MPF vectors"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"c_minus_By": [[fmt_big(v, digits) for v in _vals(b)] for b in blocks]}
    with open(path, "w") as f:
        json.dump(data, f)


class IterationsJson:
    """iterations.json written as SDPB does: '[' first, one record per iteration, ']' at the end"""
    KEYS = ["mu", "P-obj", "D-obj", "gap", "P-err", "p-err", "D-err", "R-err", "P-step", "D-step", "beta",
            "Q_cond_number", "max_block_cond_number", "block_name"]

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "iterations.json")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(self.path):                             # keep the previous run, as SDPB does
            k = 0
            while os.path.exists(os.path.join(out_dir, f"iterations.{k}.json")): k += 1
            os.rename(self.path, os.path.join(out_dir, f"iterations.{k}.json"))
            cm = os.path.join(out_dir, "c_minus_By")
            if os.path.exists(cm):
                os.rename(cm, cm + f".{k}")
        with open(self.path, "w") as f:
            f.write("[")
        self.n = 0

    def append(self, row):
        with open(self.path, "a") as f:
            if self.n: f.write(",")
            f.write(f'\n{{ "iteration":{row["iteration"]}, "total_time": {row["total_time"]:.3f}, "iter_time": {row["iter_time"]:.3f}')
            for k in self.KEYS:
                f.write(f', "{k}": "{row[k]}"')
            f.write(" }")
        self.n += 1

    def close(self):
        with open(self.path, "a") as f:
            f.write("\n]")
