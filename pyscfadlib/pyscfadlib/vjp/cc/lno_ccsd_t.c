#include <stdlib.h>
#include "config.h"
#include "vhf/fblas.h"
#include "vjp/cc/ccsd_t.h"

static double lnoccsdt_get_energy(double *mat, double *w, double *v,
                                  double *mo_energy, int nocc,
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
        if (b == c) {
                et = lnoccsdt_get_energy(mat, w0, z0, mo_energy, nocc, a, b, c, .5, cache1);
        } else {
                et = lnoccsdt_get_energy(mat, w0, z0, mo_energy, nocc, a, b, c, 1., cache1);
        }
        return et;
}

static size_t lnoccsdt_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                                int a0, int a1, void *cache, size_t stride)
{
        size_t nop = nocc * (nocc+nvir) * stride;
        size_t m, a, b, c;

        m = 0;
        for (a = a0; a < a1; a++) {
        for (b = 0; b < nvir; b++) {
        for (c = 0; c <= b; c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
                jobs[m].cache[0] = cache + nop*(a*nvir+b);
                jobs[m].cache[1] = cache + nop*(a*nvir+c);
                jobs[m].cache[2] = cache + nop*(b*nvir+a);
                jobs[m].cache[3] = cache + nop*(b*nvir+c);
                jobs[m].cache[4] = cache + nop*(c*nvir+a);
                jobs[m].cache[5] = cache + nop*(c*nvir+b);
        } } }
        return m;
}

void lnoccsdt_contract(double *e_tot, double *mat,
                       double *mo_energy, double *t1T, double *t2T,
                       double *vooo, double *fvo,
                       int nocc, int nvir, int a0, int a1,
                       void *cache)
{
        int da = a1 - a0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*nvir*(nvir+1)/2);
        size_t njobs = lnoccsdt_gen_jobs(jobs, nocc, nvir, a0, a1,
                                         cache, sizeof(double));
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
