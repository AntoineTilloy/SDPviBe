"""exactness / rounding test of gemm_nt against Python big integers"""
import random, sys, time, torch
sys.path.insert(0, '.')
from sdpvibe.mpmat import MP, from_ints, to_ints, gemm_nt, add, rescale, headroom_bits

def rint(bits): return random.getrandbits(bits) * random.choice([-1, 1])
def rnd_half_up(v, s):            # round(v / 2^s) half up, s may be <= 0
    if s <= 0: return v << (-s)
    return (v + (1 << (s - 1))) >> s

random.seed(1)
L = 34
for (M, N, K, mode) in [(5, 7, 3, 'col'), (40, 33, 70, 'mat'), (64, 64, 300, -1000), (130, 20, 5000, 'col')]:
    A = [[rint(32 * L - 40) for _ in range(K)] for _ in range(M)]
    B = [[rint(32 * L - 40) for _ in range(K)] for _ in range(N)]
    eA, eB = -1000, -1060
    Am, Bm = from_ints(A, eA, L), from_ints(B, eB, L)
    t = time.time(); C = gemm_nt(Am, Bm, out_exp=mode); torch.cuda.synchronize(); dt = time.time() - t
    Ci, ce = to_ints(C)
    bad = 0
    for i in range(M):
        for j in range(N):
            exact = sum(A[i][k] * B[j][k] for k in range(K))
            s = ce[j] - (eA + eB)
            d = abs(Ci[i][j] - rnd_half_up(exact, s)); bad += d > (1 if mode == 'mat' else 0)
    print(f"gemm M={M} N={N} K={K} mode={mode}: {bad} mismatches, exps {sorted(set(ce))[:3]}, {dt:.2f}s")

# add / rescale
A = [[rint(1000) for _ in range(9)] for _ in range(4)]; B = [[rint(1000) for _ in range(9)] for _ in range(4)]
Am, Bm = from_ints(A, -900, L), from_ints(B, -1000, L)
S = add(Am, Bm, -1); Si, se = to_ints(S)
bad = sum(Si[i][j] != rnd_half_up(A[i][j] * 2**100 - B[i][j], se[j] + 1000) for i in range(4) for j in range(9))
print("add mismatches:", bad, "exp", se[0])
