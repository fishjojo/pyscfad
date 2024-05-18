#ifndef HAVE_DEFINED_CCSD_T_H
#define HAVE_DEFINED_CCSD_T_H

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride);

void _make_permute_indices(int *idx, int n);

void add_and_permute(double *out, double *w, double *v, int n, double fac);

void get_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int *idx);
#endif
