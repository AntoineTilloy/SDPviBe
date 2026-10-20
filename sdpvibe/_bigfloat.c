/* Decoder for SDPB's binary block data (pmp2sdp --outputFormat bin, boost::archive::binary_oarchive).
 *
 * An El::BigFloat record (boost class version 1) is
 *     uint8 is_zero; if !is_zero: int32 _mp_size, int64 _mp_exp, nl x uint64 limbs (GMP mpf_t),
 * value = sign(_mp_size) * M * 2^(64 (_mp_exp - |_mp_size|)),  M = sum_{i < |size|} d[i] 2^(64 i),
 * nl = GMP precision in limbs + 1 (limbs beyond |size| are not meaningful).
 *
 * decode_bigfloats() converts n consecutive records into sdpvibe's host format: L 32-bit two's-complement
 * limbs per entry (entry-major) holding the mantissa rounded (half up) to at most 32L-2 bits, and a binary
 * exponent; zeros get mantissa 0 and exponent EZ.  Returns the position after the last record, or -1 if
 * the buffer is too short.
 */
#include <stdint.h>
#include <string.h>

int64_t decode_bigfloats(const uint8_t *buf, int64_t len, int64_t pos, int64_t n, int nl, int L,
                         int32_t *out, int64_t *exps, int64_t EZ)
{
    uint32_t w[2 * nl + 2], r[L + 1];
    const int T = 32 * L - 2;
    for (int64_t i = 0; i < n; i++) {
        int32_t *o = out + i * (int64_t)L;
        if (pos >= len) return -1;
        uint8_t is_zero = buf[pos++];
        if (is_zero) { memset(o, 0, 4 * (size_t)L); exps[i] = EZ; continue; }
        if (pos + 12 + 8 * (int64_t)nl > len) return -1;
        int32_t size; int64_t exp;
        memcpy(&size, buf + pos, 4); memcpy(&exp, buf + pos + 4, 8); pos += 12;
        int asz = size < 0 ? -size : size;
        if (asz > nl) return -1;
        memset(w, 0, sizeof(w));
        memcpy(w, buf + pos, 8 * (size_t)asz);            /* little endian: 64-bit limbs = pairs of words */
        pos += 8 * (int64_t)nl;
        int top = 2 * asz - 1;
        while (top >= 0 && w[top] == 0) top--;
        if (top < 0) { memset(o, 0, 4 * (size_t)L); exps[i] = EZ; continue; }
        int bl = 32 * top + 32 - __builtin_clz(w[top]);
        int s = bl > T ? bl - T : 0;                       /* right shift, rounding half up */
        int q = s >> 5, b = s & 31;
        for (int k = 0; k <= L; k++) {
            int i0 = q + k, i1 = q + k + 1;
            uint32_t lo = i0 <= top ? w[i0] : 0, hi = i1 <= top ? w[i1] : 0;
            r[k] = b ? (lo >> b) | (hi << (32 - b)) : lo;
        }
        if (s > 0) {
            int sb = s - 1;
            if ((w[sb >> 5] >> (sb & 31)) & 1u) {
                for (int k = 0; k <= L; k++) { if (++r[k] != 0) break; }
            }
        }
        if (size < 0) {                                    /* two's complement over L words */
            uint64_t c = 1;
            for (int k = 0; k < L; k++) { uint64_t t = (uint64_t)(~r[k]) + c; r[k] = (uint32_t)t; c = t >> 32; }
        }
        memcpy(o, r, 4 * (size_t)L);
        exps[i] = 64 * (exp - asz) + s;
    }
    return pos;
}

/* Inverse of decode_bigfloats: n entries (L 32-bit two's-complement limbs, entry-major, value = M 2^e;
 * zeros: e == EZ or M == 0) -> boost El::BigFloat records with nl GMP limbs.  Writes into out (capacity
 * cap bytes) and returns the number of bytes written, or -1 if out is too small or a mantissa does not fit. */
int64_t encode_bigfloats(const int32_t *in, const int64_t *exps, int64_t n, int L, int nl,
                         uint8_t *out, int64_t cap, int64_t EZ)
{
    uint32_t m[L + 2], d[2 * nl + 2];
    int64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        const int32_t *x = in + i * (int64_t)L;
        memcpy(m, x, 4 * (size_t)L); m[L] = m[L + 1] = 0;
        int neg = (int32_t)m[L - 1] < 0;
        if (neg) {                                         /* magnitude */
            uint64_t c = 1;
            for (int k = 0; k < L; k++) { uint64_t t = (uint64_t)(~m[k]) + c; m[k] = (uint32_t)t; c = t >> 32; }
        }
        int top = L - 1;
        while (top >= 0 && m[top] == 0) top--;
        if (top < 0 || exps[i] == EZ) {
            if (pos + 1 > cap) return -1;
            out[pos++] = 1; continue;                     /* is_zero */
        }
        /* value = M 2^e = D 2^(64 (X - size)) with D an integer of `size` 64-bit limbs:
           shift M left by r = e mod 64 bits, then exponent in limbs */
        int64_t e = exps[i];
        int r = (int)(((e % 64) + 64) % 64);
        int64_t base = (e - r) / 64;                       /* value = (M << r) 2^(64 base) */
        memset(d, 0, sizeof(d));
        int q = r >> 5, b = r & 31;
        for (int k = 0; k <= top + 1 && k + q < 2 * nl + 2; k++) {
            uint32_t lo = m[k], prev = k > 0 ? m[k - 1] : 0;
            d[k + q] = b ? (lo << b) | (prev >> (32 - b)) : lo;
        }
        if (top + 2 + q < 2 * nl + 2 && b) d[top + 1 + q] = (m[top] >> (32 - b));
        int dtop = 2 * nl + 1;
        while (dtop >= 0 && d[dtop] == 0) dtop--;
        int size = dtop / 2 + 1;                           /* 64-bit limbs used */
        if (size > nl) return -1;
        /* drop low zero limbs as GMP does (normalised mpf has no low zero limbs is not required, but keep
           all `size` limbs starting at limb 0) */
        int32_t ssize = neg ? -size : size;
        int64_t X = base + size;
        if (pos + 1 + 12 + 8 * (int64_t)nl > cap) return -1;
        out[pos++] = 0;
        memcpy(out + pos, &ssize, 4); memcpy(out + pos + 4, &X, 8); pos += 12;
        memset(out + pos, 0, 8 * (size_t)nl);
        memcpy(out + pos, d, 8 * (size_t)size);
        pos += 8 * (int64_t)nl;
    }
    return pos;
}
