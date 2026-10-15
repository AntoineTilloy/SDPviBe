// CPU baseline mimicking SDPB's El::BigFloat path: GMP mpf, naive loops, OpenMP.
// GMP headers are not installed here, so the few mpf entry points are declared by hand
// (layout/ABI from gmp.h, GMP 6.x). Link: gcc -O2 -fopenmp mpf_baseline.c -l:libgmp.so.10 -lm
//
// usage: mpf_baseline <prec_bits> <n> <nrhs> <threads>
//   times: Cholesky (n x n), TRSM P = L^{-1} B (n x nrhs), GEMM (n x n)*(n x nrhs)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef unsigned long mp_limb_t;
typedef struct { int _mp_prec; int _mp_size; long _mp_exp; mp_limb_t *_mp_d; } __mpf_struct;
typedef __mpf_struct mpf_t[1];
void __gmpf_init2(mpf_t, unsigned long);
void __gmpf_clear(mpf_t);
void __gmpf_set_d(mpf_t, double);
void __gmpf_set(mpf_t, const mpf_t);
double __gmpf_get_d(const mpf_t);
void __gmpf_mul(mpf_t, const mpf_t, const mpf_t);
void __gmpf_add(mpf_t, const mpf_t, const mpf_t);
void __gmpf_sub(mpf_t, const mpf_t, const mpf_t);
void __gmpf_div(mpf_t, const mpf_t, const mpf_t);
void __gmpf_sqrt(mpf_t, const mpf_t);
void __gmpf_div_ui(mpf_t, const mpf_t, unsigned long);

static unsigned long PREC;
static __mpf_struct *alloc(size_t n) {
    __mpf_struct *a = malloc(n * sizeof(__mpf_struct));
    for (size_t i = 0; i < n; i++) __gmpf_init2(&a[i], PREC);
    return a;
}
// full-precision pseudo-random value: (integer)/(odd integer) has a non-terminating mantissa
static void rnd(__mpf_struct *x, unsigned *s) {
    *s = *s * 1103515245u + 12345u; double a = (double)(*s >> 8) - 8388608.0;
    *s = *s * 1103515245u + 12345u; double b = 2.0 * (double)(*s >> 9) + 1.0;
    __mpf_struct t; __gmpf_init2(&t, PREC); __gmpf_set_d(&t, b);
    __gmpf_set_d(x, a); __gmpf_div(x, x, &t); __gmpf_clear(&t);
}
static double now() { return omp_get_wtime(); }

int main(int argc, char **argv) {
    PREC = atol(argv[1]); int n = atoi(argv[2]), R = atoi(argv[3]), th = atoi(argv[4]);
    omp_set_num_threads(th);
    // SPD matrix A = G G^T + n I  (G random n x n), built in mpf
    __mpf_struct *G = alloc((size_t)n * n), *A = alloc((size_t)n * n), *B = alloc((size_t)n * R);
    unsigned s = 1;
    for (size_t i = 0; i < (size_t)n * n; i++) rnd(&G[i], &s);
    for (size_t i = 0; i < (size_t)n * R; i++) rnd(&B[i], &s);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        __mpf_struct t; __gmpf_init2(&t, PREC);
        for (int j = 0; j <= i; j++) {
            __gmpf_set_d(&A[(size_t)i*n+j], i == j ? (double)n : 0.0);
            for (int k = 0; k < n; k++) { __gmpf_mul(&t, &G[(size_t)i*n+k], &G[(size_t)j*n+k]); __gmpf_add(&A[(size_t)i*n+j], &A[(size_t)i*n+j], &t); }
            __gmpf_set(&A[(size_t)j*n+i], &A[(size_t)i*n+j]);
        }
        __gmpf_clear(&t);
    }
    // ---- Cholesky (Crout, lower), parallel over rows within each column
    double t0 = now();
    for (int j = 0; j < n; j++) {
        __mpf_struct t; __gmpf_init2(&t, PREC);
        for (int k = 0; k < j; k++) { __gmpf_mul(&t, &A[(size_t)j*n+k], &A[(size_t)j*n+k]); __gmpf_sub(&A[(size_t)j*n+j], &A[(size_t)j*n+j], &t); }
        __gmpf_sqrt(&A[(size_t)j*n+j], &A[(size_t)j*n+j]);
        __gmpf_clear(&t);
        #pragma omp parallel for schedule(static)
        for (int i = j + 1; i < n; i++) {
            __mpf_struct u; __gmpf_init2(&u, PREC);
            for (int k = 0; k < j; k++) { __gmpf_mul(&u, &A[(size_t)i*n+k], &A[(size_t)j*n+k]); __gmpf_sub(&A[(size_t)i*n+j], &A[(size_t)i*n+j], &u); }
            __gmpf_div(&A[(size_t)i*n+j], &A[(size_t)i*n+j], &A[(size_t)j*n+j]);
            __gmpf_clear(&u);
        }
    }
    double tchol = now() - t0;
    // ---- TRSM: B <- L^{-1} B, parallel over right-hand sides
    t0 = now();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int c = 0; c < R; c++) {
        __mpf_struct u; __gmpf_init2(&u, PREC);
        for (int i = 0; i < n; i++) {
            __mpf_struct *bi = &B[(size_t)i*R+c];
            for (int k = 0; k < i; k++) { __gmpf_mul(&u, &A[(size_t)i*n+k], &B[(size_t)k*R+c]); __gmpf_sub(bi, bi, &u); }
            __gmpf_div(bi, bi, &A[(size_t)i*n+i]);
        }
        __gmpf_clear(&u);
    }
    double ttrsm = now() - t0;
    // ---- GEMM C = G * B (n x n)(n x R), naive (Elemental BigFloat Gemm is a naive triple loop)
    __mpf_struct *C = alloc((size_t)n * R);
    t0 = now();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        __mpf_struct u; __gmpf_init2(&u, PREC);
        for (int c = 0; c < R; c++) {
            __gmpf_set_d(&C[(size_t)i*R+c], 0.0);
            for (int k = 0; k < n; k++) { __gmpf_mul(&u, &G[(size_t)i*n+k], &B[(size_t)k*R+c]); __gmpf_add(&C[(size_t)i*R+c], &C[(size_t)i*R+c], &u); }
        }
        __gmpf_clear(&u);
    }
    double tgemm = now() - t0;
    double mac_chol = (double)n*n*n/6, mac_trsm = (double)n*n*R/2, mac_gemm = (double)n*n*R;
    printf("prec=%lu n=%d nrhs=%d threads=%d\n", PREC, n, R, th);
    printf("  cholesky %8.3f s  %7.2f M mp-MAC/s\n", tchol, mac_chol/tchol/1e6);
    printf("  trsm     %8.3f s  %7.2f M mp-MAC/s\n", ttrsm, mac_trsm/ttrsm/1e6);
    printf("  gemm     %8.3f s  %7.2f M mp-MAC/s\n", tgemm, mac_gemm/tgemm/1e6);
    printf("  check L[n-1][n-1]=%.6e  P[n-1][0]=%.6e\n", __gmpf_get_d(&A[(size_t)(n-1)*n+n-1]), __gmpf_get_d(&B[(size_t)(n-1)*R]));
    return 0;
}
