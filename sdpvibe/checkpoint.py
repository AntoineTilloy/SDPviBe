"""Checkpoints, with SDPB's conventions (SDP_Solver/save_checkpoint.cxx, load_checkpoint/):

  * binary: <dir>/sdpvibe_checkpoint_<k>.pt (x, X, y, Y in SDPB block order, exact limbs) and
    <dir>/checkpoint.json {"current": k, "backup": k-1, "format": "sdpvibe", ...}; the previous generation is
    kept as a backup and older ones removed, as SDPB does;
  * text: a directory with x_<j>.txt, y.txt, X_matrix_<k>.txt, Y_matrix_<k>.txt (sdpb/sdpvibe
    --writeSolution=x,y,X,Y output) -- interchangeable with SDPB's text checkpoints.
Loading tries binary, then text.  SDPB's own binary checkpoints (checkpoint_<k>_<rank>) are not readable.
"""
import json, os
import torch
from .mpf import MPF

FORMAT = "sdpvibe"


def _pack(A): return {"m": A.m.contiguous(), "e": A.e.contiguous()}
def _unpack(d): return MPF(d["m"], d["e"])


def structure(sdp):
    return {"N": sdp.N, "blocks": [[b["dim"], b["K"], b["he"], b["ho"]] for b in sdp.layout.blocks]}


def save(ck_dir, state, sdp, L, precision, options):
    if not ck_dir:
        return
    os.makedirs(ck_dir, exist_ok=True)
    meta_path = os.path.join(ck_dir, "checkpoint.json")
    cur = backup = None
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            if meta.get("format") == FORMAT:
                cur, backup = meta.get("current"), meta.get("backup")
        except (OSError, ValueError):
            pass
    k = (cur or 0) + 1
    data = {"format": FORMAT, "version": 1, "precision": precision, "L": L, "structure": structure(sdp),
            "y": _pack(state["y"]), "x": [_pack(v) for v in state["x"]],
            "X": [[_pack(m) for m in ms] for ms in state["X"]], "Y": [[_pack(m) for m in ms] for ms in state["Y"]]}
    path = os.path.join(ck_dir, f"sdpvibe_checkpoint_{k}.pt")
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)
    with open(meta_path + ".tmp", "w") as f:
        json.dump({"current": k, "backup": cur, "version": "sdpvibe", "format": FORMAT, "options": options}, f, indent=4)
    os.replace(meta_path + ".tmp", meta_path)
    if backup is not None:                                          # keep current and one backup generation
        old = os.path.join(ck_dir, f"sdpvibe_checkpoint_{backup}.pt")
        if os.path.exists(old):
            os.remove(old)


def _load_binary(ck_dir, sdp):
    meta_path = os.path.join(ck_dir, "checkpoint.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    if meta.get("format") != FORMAT:
        return None
    for k in (meta.get("current"), meta.get("backup")):
        if k is None:
            continue
        path = os.path.join(ck_dir, f"sdpvibe_checkpoint_{k}.pt")
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d["structure"] != structure(sdp):
            raise ValueError(f"checkpoint {path} was written for a different SDP block structure")
        return {"y": _unpack(d["y"]), "x": [_unpack(v) for v in d["x"]],
                "X": [[_unpack(m) for m in ms] for ms in d["X"]], "Y": [[_unpack(m) for m in ms] for ms in d["Y"]],
                "source": path}
    return None


def _load_text(ck_dir, sdp, L):
    from .sdpb_output import read_text_matrix
    from . import mpf as F
    if not (os.path.exists(os.path.join(ck_dir, "y.txt")) and os.path.exists(os.path.join(ck_dir, "x_0.txt"))):
        return None
    st = {"y": read_text_matrix(os.path.join(ck_dir, "y.txt"), L), "x": [], "X": [], "Y": [], "source": ck_dir + " (text)"}
    for j, b in enumerate(sdp.layout.blocks):
        st["x"].append(read_text_matrix(os.path.join(ck_dir, f"x_{j}.txt"), L))
        for tag in ("X", "Y"):
            ms = []
            for p, h in enumerate((b["he"], b["ho"])):
                n = b["dim"] * h
                f = os.path.join(ck_dir, f"{tag}_matrix_{2 * j + p}.txt")
                if n == 0:
                    ms.append(F.zeros((0, 0), L)); continue
                if not os.path.exists(f):
                    return None
                ms.append(read_text_matrix(f, L))
            st[tag].append(ms)
    return st


def load(ck_dir, sdp, L):
    """state dict (see state.py) or None if the directory holds no usable checkpoint"""
    if not ck_dir or not os.path.isdir(ck_dir):
        return None
    return _load_binary(ck_dir, sdp) or _load_text(ck_dir, sdp, L)
