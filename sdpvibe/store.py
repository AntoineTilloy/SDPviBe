"""Placement of the large per-chunk arrays: GPU-resident, or streamed from pinned host memory.

Objects are arbitrary trees (MPF, Chol, BMP, lists, ...) whose leaves are CUDA tensors.
  put(key, obj)   store (device: keep the reference; host: async D2H on a copy stream)
  get(key)        device object (waits for a prefetch started earlier, or copies now)
  prefetch(key)   start the H2D copy on a copy stream
  stream(keys)    iterate over device objects, prefetching `ahead` keys in advance
Host buffers are allocated once per key (shapes do not change between iterations) and pinned
with cudaHostRegister (torch's pinned allocator rounds every allocation up to a power of 2).
Copies run on two side streams (H2D, D2H); per-key events order them against each other and
against the compute stream; record_stream keeps the caching allocator from recycling memory
that a copy is still using.
"""
import torch
from .rns import dev


def tmap(f, obj):
    """apply f to every tensor leaf of obj (MPF / Chol / BMP / BTile / list / tuple / dict)"""
    if torch.is_tensor(obj): return f(obj)
    if obj is None or isinstance(obj, (int, float, bool, str)): return obj
    if isinstance(obj, (list, tuple)): return type(obj)(tmap(f, o) for o in obj)
    if isinstance(obj, dict): return {k: tmap(f, v) for k, v in obj.items()}
    new = object.__new__(type(obj))
    names = obj.__dict__.keys() if hasattr(obj, "__dict__") else ()
    for k in list(names) + [s for c in type(obj).__mro__ for s in getattr(c, "__slots__", ())]:
        if hasattr(obj, k): setattr(new, k, tmap(f, getattr(obj, k)))
    return new


def tleaves(obj):
    out = []
    tmap(lambda t: out.append(t) or t, obj)
    return out


def nbytes(obj):
    return sum(t.numel() * t.element_size() for t in tleaves(obj))


class Store:
    def __init__(self, host_keys=(), ahead=1):
        self.host_keys = set(host_keys)          # keys kept in host memory; all others resident
        self.ahead = ahead
        self.dev = {}                            # key -> device object (resident keys)
        self.host = {}                           # key -> host skeleton (pinned leaves)
        self.ev_d2h, self.ev_h2d = {}, {}        # key -> last event of the copy direction
        self.pending = {}                        # key -> (device object, ready event)
        self.h2d = torch.cuda.Stream(device=dev)
        self.d2h = torch.cuda.Stream(device=dev)
        self.bytes_h2d = self.bytes_d2h = 0
        self.host_bytes = 0

    def is_host(self, key): return key in self.host_keys

    def _alloc_host(self, obj):
        """one pinned host buffer per key, carved into views shaped like the leaves of obj"""
        sizes = [(t.numel() * t.element_size() + 255) // 256 * 256 for t in tleaves(obj)]
        buf = torch.empty(max(1, sum(sizes)), dtype=torch.uint8)
        torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
        self.host_bytes += buf.numel()
        off = [0]

        def f(t):
            n = t.numel() * t.element_size()
            v = buf[off[0]:off[0] + n].view(t.dtype).view(t.shape)
            off[0] += (n + 255) // 256 * 256
            return v
        return tmap(f, obj)

    def put(self, key, obj):
        if key not in self.host_keys:
            self.dev[key] = obj
            return
        self.pending.pop(key, None)
        if key not in self.host:
            self.host[key] = self._alloc_host(obj)
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event(); ev.record(cur)
        with torch.cuda.stream(self.d2h):
            self.d2h.wait_event(ev)
            if key in self.ev_h2d: self.d2h.wait_event(self.ev_h2d[key])     # host buffer still being read
            for h, t in zip(tleaves(self.host[key]), tleaves(obj)):
                t = t.contiguous(); t.record_stream(self.d2h)
                h.copy_(t, non_blocking=True)
                self.bytes_d2h += h.numel() * h.element_size()
            e2 = torch.cuda.Event(); e2.record(self.d2h)
        self.ev_d2h[key] = e2

    def prefetch(self, key):
        if key not in self.host_keys or key in self.pending: return
        with torch.cuda.stream(self.h2d):
            if key in self.ev_d2h: self.h2d.wait_event(self.ev_d2h[key])
            def f(h):
                self.bytes_h2d += h.numel() * h.element_size()
                return torch.empty(h.shape, dtype=h.dtype, device=dev).copy_(h, non_blocking=True)
            d = tmap(f, self.host[key])
            ev = torch.cuda.Event(); ev.record(self.h2d)
        self.ev_h2d[key] = ev
        self.pending[key] = (d, ev)

    def get(self, key):
        if key not in self.host_keys: return self.dev[key]
        self.prefetch(key)
        d, ev = self.pending.pop(key)
        cur = torch.cuda.current_stream()
        cur.wait_event(ev)
        for t in tleaves(d): t.record_stream(cur)
        return d

    def stream(self, items, nxt=()):
        """items: sequence of key tuples -> yields tuples of device objects.  nxt: keys needed first by
        the following pass, prefetched while the last item is processed"""
        items = [tuple(it) for it in items]
        for i, it in enumerate(items):
            for j in range(i, min(len(items), i + 1 + self.ahead)):
                for k in items[j]: self.prefetch(k)
            if i == len(items) - 1:
                for k in nxt: self.prefetch(k)
            yield tuple(self.get(k) for k in it)

    def drop(self, key):
        self.dev.pop(key, None); self.pending.pop(key, None)


class MultiStore:
    """one Store per GPU; keys (name, chunk) are routed to the store of the chunk's device
    (dev_of[chunk]).  Each Store is only used from its device's thread."""
    def __init__(self, host_keys, dev_of, devices):
        self.dev_of = dev_of
        self.stores = {}
        for d in devices:
            with torch.cuda.device(d):
                self.stores[d] = Store({k for k in host_keys if dev_of[k[1]] == d})

    def of(self, key): return self.stores[self.dev_of[key[1]]]
    def is_host(self, key): return self.of(key).is_host(key)
    def put(self, key, obj): self.of(key).put(key, obj)
    def get(self, key): return self.of(key).get(key)
    def prefetch(self, key): self.of(key).prefetch(key)
    def drop(self, key): self.of(key).drop(key)

    def stream(self, items, nxt=()):
        """single device only (all keys on one store)"""
        items = [tuple(it) for it in items]
        if not items: return iter(())
        return self.of(items[0][0]).stream(items, nxt)

    @property
    def bytes_h2d(self): return sum(s.bytes_h2d for s in self.stores.values())
    @property
    def bytes_d2h(self): return sum(s.bytes_d2h for s in self.stores.values())
    @property
    def host_bytes(self): return sum(s.host_bytes for s in self.stores.values())
