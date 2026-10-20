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
