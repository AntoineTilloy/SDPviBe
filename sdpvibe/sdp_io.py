"""Reading SDPB's SDP input (the output of pmp2sdp), as SDPB 3.x does:

  * a directory, or an archive (zip, tar, tar.gz, ...) containing control.json, objectives.json,
    [normalization.json], block_info_<j>.json and block_data_<j>.bin or block_data_<j>.json;
  * block data in JSON (decimal strings) or in SDPB's binary format (boost::archive::binary_oarchive of
    El::BigFloat = GMP mpf_t records; see pmp2sdp/write_block_data.cxx and sdpb_util/boost_serialization.hxx).

Blocks are converted to sdpvibe's host format (L 32-bit two's-complement limbs per entry, entry-major,
plus binary exponents) by `read_block(src, j, L)`; the binary path uses a small C decoder (_bigfloat.c,
compiled on first use with the system C compiler and cached), with a pure-Python fallback.
"""
import ctypes, hashlib, io, json, os, struct, subprocess, sys, tarfile, tempfile, zipfile
import numpy as np

EZ = -(1 << 40)


# ------------------------------------------------------------------ sources
class SdpSource:
    """A pmp2sdp output: directory or archive"""

    def __init__(self, path):
        self.path = os.fspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"SDP not found: {self.path}")
        self.kind = "dir" if os.path.isdir(self.path) else None
        self._zip = self._tar = None
        if self.kind is None:
            if zipfile.is_zipfile(self.path):
                self.kind = "zip"
            elif tarfile.is_tarfile(self.path):
                self.kind = "tar"
            else:
                raise FileNotFoundError(f"SDP not found or not a directory/zip/tar archive: {self.path}")
        self._names = None

    def _open(self):
        if self.kind == "zip" and self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
            self._prefix = self._common_prefix(self._zip.namelist())
        if self.kind == "tar" and self._tar is None:
            self._tar = tarfile.open(self.path)
            self._prefix = self._common_prefix(self._tar.getnames())

    @staticmethod
    def _common_prefix(names):
        """archives may hold the files inside one top-level directory"""
        for n in names:
            if n.endswith("control.json"):
                return n[: -len("control.json")]
        return ""

    def names(self):
        if self._names is None:
            if self.kind == "dir":
                self._names = set(os.listdir(self.path))
            else:
                self._open()
                allnames = self._zip.namelist() if self.kind == "zip" else self._tar.getnames()
                p = self._prefix
                self._names = {n[len(p):] for n in allnames if n.startswith(p) and "/" not in n[len(p):]}
        return self._names

    def exists(self, name):
        return name in self.names()

    def read(self, name):
        if self.kind == "dir":
            with open(os.path.join(self.path, name), "rb") as f:
                return f.read()
        self._open()
        if self.kind == "zip":
            return self._zip.read(self._prefix + name)
        return self._tar.extractfile(self._prefix + name).read()

    def json(self, name):
        return json.loads(self.read(name))

    def mtime(self, name="control.json"):
        return os.path.getmtime(os.path.join(self.path, name) if self.kind == "dir" else self.path)

    def cache_dir(self):
        """where sdpvibe may cache parsed data: the directory itself, or next to the archive"""
        return self.path if self.kind == "dir" else self.path + ".sdpvibe_cache"


# ------------------------------------------------------------------ binary BigFloat decoder
_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB or None
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bigfloat.c")
    code = open(src, "rb").read()
    tag = hashlib.sha1(code).hexdigest()[:12]
    cache = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "sdpvibe")
    so = os.path.join(cache, f"_bigfloat_{tag}.so")
    try:
        if not os.path.exists(so):
            os.makedirs(cache, exist_ok=True)
            tmp = so + f".{os.getpid()}.tmp"
            cc = os.environ.get("CC", "cc")
            subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", tmp, src], check=True, capture_output=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        f = lib.decode_bigfloats
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        _LIB = lib
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"sdpvibe: C decoder unavailable ({e}); using the slow Python decoder", file=sys.stderr)
        _LIB = False
    return _LIB or None


def _decode_py(buf, pos, n, nl, L):
    """pure-Python version of decode_bigfloats"""
    limbs = np.zeros((n, L), dtype=np.int32); exps = np.empty(n, dtype=np.int64)
    T = 32 * L - 2
    for i in range(n):
        z = buf[pos]; pos += 1
        if z:
            exps[i] = EZ; continue
        size, exp = struct.unpack_from("<iq", buf, pos); pos += 12
        asz = abs(size)
        d = struct.unpack_from(f"<{nl}Q", buf, pos); pos += 8 * nl
        m = sum(d[k] << (64 * k) for k in range(asz))
        if m == 0:
            exps[i] = EZ; continue
        s = max(0, m.bit_length() - T)
        if s:
            m = (m + (1 << (s - 1))) >> s
        if size < 0:
            m = -m
        limbs[i] = np.frombuffer(m.to_bytes(4 * L, "little", signed=True), dtype=np.int32)
        exps[i] = 64 * (exp - asz) + s
    return limbs, exps, pos


class _BinReader:
    def __init__(self, buf, L):
        self.b = buf; self.L = L
        p = 0
        n = struct.unpack_from("<Q", buf, p)[0]; p += 8
        if buf[p:p + n] != b"serialization::archive":
            raise ValueError("not a boost binary archive")
        p += n + 2                                   # signature, library version
        p += 4 + 4                                   # sizeof(int, long, float, double), endianness marker
        self.precision = struct.unpack_from("<Q", buf, p)[0]; p += 8
        self.nl = (self.precision + 127) // 64 + 1   # mpf_t limbs: __GMPF_BITS_TO_PREC(prec) + 1
        self.p = p
        self.first_bigfloat = True; self.first_matrix = True; self.first_vector = True

    def _floats(self, n):
        if n == 0:
            return np.zeros((0, self.L), dtype=np.int32), np.zeros(0, dtype=np.int64)
        if self.first_bigfloat:
            self.p += 5                              # class info: tracking flag + version
            self.first_bigfloat = False
        lib = _lib()
        if lib is None:
            limbs, exps, self.p = _decode_py(self.b, self.p, n, self.nl, self.L)
            return limbs, exps
        limbs = np.empty((n, self.L), dtype=np.int32); exps = np.empty(n, dtype=np.int64)
        arr = np.frombuffer(self.b, dtype=np.uint8)
        q = lib.decode_bigfloats(arr.ctypes.data, len(self.b), self.p, n, self.nl, self.L,
                                 limbs.ctypes.data, exps.ctypes.data, EZ)
        if q < 0:
            raise ValueError("truncated or corrupt binary block data")
        self.p = q
        return limbs, exps

    def matrix(self):
        """El::Matrix<BigFloat> (column-major, leading dimension) -> row-major (h, w) host arrays"""
        if self.first_matrix:
            self.p += 5
            self.first_matrix = False
        h, w, ld = struct.unpack_from("<iii", self.b, self.p); self.p += 12
        limbs, exps = self._floats(ld * w)
        limbs = limbs.reshape(w, ld, self.L)[:, :h].transpose(1, 0, 2).reshape(h * w, self.L)
        exps = exps.reshape(w, ld)[:, :h].T.reshape(-1)
        return h, w, np.ascontiguousarray(limbs), np.ascontiguousarray(exps)

    def vector(self):
        if self.first_vector:
            self.p += 5
            self.first_vector = False
        n = struct.unpack_from("<Q", self.b, self.p)[0]; self.p += 8
        self.p += 4                                  # item version
        return self._floats(n)


# ------------------------------------------------------------------ blocks
def _strings_to_host(strs, L):
    from . import mpf as F
    return F.strings_to_host(strs, L)


def read_block(src, j, L):
    """block j -> {dim, K, bilinear_bases_even: (h, (limbs, exps) | None), bilinear_bases_odd, c, B}
    (B row-major: (K*dim(dim+1)/2) x N)"""
    info = src.json(f"block_info_{j}.json")
    out = {"dim": info["dim"], "K": info["num_points"]}
    if src.exists(f"block_data_{j}.bin"):
        r = _BinReader(src.read(f"block_data_{j}.bin"), L)
        _, _, Bl, Be = r.matrix()
        cl, ce = r.vector()
        bases = [r.matrix(), r.matrix()]
        if r.p != len(r.b):
            raise ValueError(f"block_data_{j}.bin: {len(r.b) - r.p} unexpected trailing bytes")
        for key, (h, w, bl, be) in zip(("bilinear_bases_even", "bilinear_bases_odd"), bases):
            out[key] = (h, (bl, be) if h else None)
        out["c"] = (cl, ce)
        out["B"] = (Bl, Be)
        out["precision"] = r.precision
    else:
        data = src.json(f"block_data_{j}.json")
        for key in ("bilinear_bases_even", "bilinear_bases_odd"):
            rows = data[key]
            out[key] = (len(rows), _strings_to_host([x for row in rows for x in row], L) if rows else None)
        out["c"] = _strings_to_host(data["c"], L)
        out["B"] = _strings_to_host([x for row in data["B"] for x in row], L)
    return out


def read_objectives(src):
    obj = src.json("objectives.json")
    return obj.get("constant", "0"), obj["b"]


def read_normalization(src):
    """normalization vector (decimal strings) or None"""
    if not src.exists("normalization.json"):
        return None
    return src.json("normalization.json")["normalization"]
