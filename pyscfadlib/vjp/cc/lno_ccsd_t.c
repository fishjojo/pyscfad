#include <stdlib.h>
#include "config.h"
#include "pyscf/vhf/fblas.h"
#include "ccsd_t.h"

static double _lno_ccsd_t_get_energy(double *mat, double *w, double *v, double *mo_energy, int nocc,
                                     int a, int b, int c, double fac, double *cache)
{
        const int noo = nocc * nocc;
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';

        dgemm_(&TRANS_N, &TRANS_T, &noo, &nocc, &nocc,
               &D1, w, &noo, mat, &nocc,
               &D0, cache, &noo);

        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                et += fac * cache[n] * v[n] / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
        } } }
        return et;
}


static double contract6(int nocc, int nvir, int a, int b, int c,
                        double *mat, double *mo_energy, double *t1T, double *t2T,
                        double *fvo, double *vooo, double *cache1, void **cache,
                        int *permute_idx, double fac)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
        get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
        get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
        get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
        get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
        get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);

        add_and_permute(z0, w0, v0, nocc, fac);

        double et;
        if (a == c) {
                et = _lno_ccsd_t_get_energy(mat, w0, z0, mo_energy, nocc, a, b, c, 1./6, cache1);
        } else if (a == b || b == c) {
                et = _lno_ccsd_t_get_energy(mat, w0, z0, mo_energy, nocc, a, b, c, .5, cache1);
        } else {
                et = _lno_ccsd_t_get_energy(mat, w0, z0, mo_energy, nocc, a, b, c, 1., cache1);
        }
        return et;
}


void lno_ccsd_t_contract(double *e_tot, double *mat,
                         double *mo_energy, double *t1T, double *t2T,
                         double *vooo, double *fvo,
                         int nocc, int nvir, int a0, int a1, int b0, int b1,
                         void *cache_row_a, void *cache_col_a,
                         void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mat, mo_energy, t1T, t2T,\
               vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
        double *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6(nocc, nvir, a, b, c, mat, mo_energy, t1Thalf, t2T,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                               1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}
