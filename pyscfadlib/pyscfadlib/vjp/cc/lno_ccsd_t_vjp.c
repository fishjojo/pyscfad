#include <stdlib.h>
#include "config.h"
#include "vhf/fblas.h"
#include "vjp/cc/ccsd_t.h"
#include "vjp/util/util.h"

#define MAX_THREADS 128

static void get_wz(double *w0_mat, double *w0, double* z0,
                   int nocc, int nvir, int a, int b, int c,
                   double *mat, double *mo_energy, double *t1T, double *t2T,
                   double *fvo,
                   double *vooo, double *cache1, void **cache,
                   int *permute_idx, double fac)
{
    const int noo = nocc * nocc;
    const int nooo = noo * nocc;
    int *idx0 = permute_idx;
    int *idx1 = idx0 + nooo;
    int *idx2 = idx1 + nooo;
    int *idx3 = idx2 + nooo;
    int *idx4 = idx3 + nooo;
    int *idx5 = idx4 + nooo;
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    double *v0 = cache1;
    double *wtmp = v0 + nooo;
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

    dgemm_(&TRANS_N, &TRANS_T, &noo, &nocc, &nocc,
           &D1, w0, &noo, mat, &nocc,
           &D0, w0_mat, &noo);
}

static void get_d3(double *d3, double *mo_energy, int nocc,
                   int a, int b, int c, double fac)
{
    int i, j, k, n;
    double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];

    for (n = 0, i = 0; i < nocc; i++) {
    for (j = 0; j < nocc; j++) {
    for (k = 0; k < nocc; k++, n++) {
        d3[n] = fac / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
    } } }
}

static void get_v0_bar(double *v0_bar, double *z0_bar, double fac, int nocc, int *idx)
{
    const int nooo = nocc * nocc * nocc;
    int n;
    for (n = 0; n < nooo; n++) {
        v0_bar[n] += z0_bar[idx[n]] * fac;
    }
}

static void get_w0v0d3_bar(double *mat_bar, double *w0_bar, double *v0_bar, double *d3_bar,
                           double *w0_mat, double *mat, double *z0, double *w0, double *d3,
                           double et_bar, int nocc, int *permute_idx,
                           double *cache)
{
    const int noo = nocc * nocc;
    const int nooo = noo * nocc;
    int n;
    int *idx0 = permute_idx;
    int *idx1 = idx0 + nooo;
    int *idx2 = idx1 + nooo;
    int *idx3 = idx2 + nooo;
    int *idx4 = idx3 + nooo;
    int *idx5 = idx4 + nooo;

    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    double *z0_bar = cache;
    double *w0_mat_bar = z0_bar + nooo;

    for (n = 0; n < nooo; n++) {
        z0_bar[n] = et_bar * w0_mat[n] * d3[n];
        d3_bar[n] = et_bar * w0_mat[n] * z0[n];
        w0_mat_bar[n] = et_bar * z0[n] * d3[n];
        v0_bar[n] = 0;
    }

    dgemm_(&TRANS_T, &TRANS_N, &nocc, &nocc, &noo,
           &D1, w0_mat_bar, &noo, w0, &noo,
           &D1, mat_bar, &nocc);

    dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nocc,
           &D1, w0_mat_bar, &noo, mat, &nocc,
           &D0, w0_bar, &noo);

    get_v0_bar(v0_bar, z0_bar,  4., nocc, idx0);
    get_v0_bar(v0_bar, z0_bar, -2., nocc, idx1);
    get_v0_bar(v0_bar, z0_bar, -2., nocc, idx2);
    get_v0_bar(v0_bar, z0_bar,  1., nocc, idx3);
    get_v0_bar(v0_bar, z0_bar,  1., nocc, idx4);
    get_v0_bar(v0_bar, z0_bar, -2., nocc, idx5);

    for (n = 0; n < nooo; n++) {
        w0_bar[n] += v0_bar[n];
    }
}

static void get_mo_energy_bar(double *mo_energy_bar, double *d3_bar, double *d3,
                              int a, int b, int c, int nocc, double fac)
{
    int i, j, k, n;
    double tmp;
    double *pvir = mo_energy_bar + nocc;
    for (n = 0, i = 0; i < nocc; i++) {
    for (j = 0; j < nocc; j++) {
    for (k = 0; k < nocc; k++, n++) {
        tmp = d3_bar[n] * d3[n] * d3[n] / fac;
        mo_energy_bar[i] -= tmp;
        mo_energy_bar[j] -= tmp;
        mo_energy_bar[k] -= tmp;
        pvir[a] += tmp;
        pvir[b] += tmp;
        pvir[c] += tmp;
    } } }


}

static void get_wabc_bar(double *wabc_bar, double *w0_bar,
                         int nocc, int *idx)
{
    const int nooo = nocc * nocc * nocc;
    int n;
    for (n = 0; n < nooo; n++) {
        wabc_bar[n] = w0_bar[idx[n]];
    }
}

static void get_vabc_bar(double *vabc_bar, double *v0_bar,
                         int nocc, int *idx)
{
    const int nooo = nocc * nocc * nocc;
    int n;
    for (n = 0; n < nooo; n++) {
        vabc_bar[n] = v0_bar[idx[n]];
    }
}

static void get_vvov_bar(double *vvop_bar, double *wabc_bar, double *t2T,
                         int nocc, int nvir, int a, int b, int c)
{
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const int nmo = nocc + nvir;
    const int noo = nocc * nocc;
    const size_t nvoo = nvir * noo;

    dgemm_(&TRANS_T, &TRANS_N, &nvir, &nocc, &noo,
           &D1, t2T+c*nvoo, &noo, wabc_bar, &noo,
           &D1, vvop_bar+nocc, &nmo);
}

static void get_vooo_bar(double *vooo_bar, double *wabc_bar, double *t2T,
                         int nocc, int nvir, int a, int b, int c)
{
    const double D1 = 1;
    const double DN1 = -1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const int noo = nocc * nocc;
    const int nooo = noo * nocc;
    const size_t nvoo = nvir * noo;

    dgemm_(&TRANS_T, &TRANS_N, &nocc, &noo, &nocc,
           &DN1, t2T+c*nvoo+b*noo, &nocc, wabc_bar, &nocc,
           &D1, vooo_bar+a*nooo, &nocc);

}

static void get_vvoo_bar(double *vvop_bar, double *vabc_bar, double *t1Thalf,
                         int nocc, int nvir, int a, int b, int c)
{
    const int I1 = 1;
    const double D1 = 1;
    const char TRANS_T = 'T';
    const int nmo = nocc + nvir;
    const int noo = nocc * nocc;
    int i;

    for (i = 0; i < nocc; i++) {
        dgemv_(&TRANS_T, &nocc, &nocc,
               &D1, vabc_bar+i*noo, &nocc, t1Thalf+c*nocc, &I1,
               &D1, vvop_bar+i*nmo, &I1);
    }
}

static void get_t1T_bar(double *t1T_bar, double *vabc_bar, double *vvop,
                        int nocc, int nvir, int a, int b, int c)
{
    const int I1 = 1;
    const double D1 = 1;
    const double Half = .5;
    const char TRANS_N = 'N';
    const int nmo = nocc + nvir;
    const int noo = nocc * nocc;
    int i;

    for (i = 0; i < nocc; i++) {
        dgemv_(&TRANS_N, &nocc, &nocc,
               &Half, vabc_bar+i*noo, &nocc, vvop+i*nmo, &I1,
               &D1, t1T_bar+c*nocc, &I1);
    }
}

static void get_t2T_bar(double *t2T_bar, double *wabc_bar, double *vabc_bar,
                        double *vvop, double *vooo, double *fvohalf,
                        int nocc, int nvir, int a, int b, int c)
{
    const int I1 = 1;
    const double D1 = 1;
    const double DN1 = -1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const int nmo = nocc + nvir;
    const int noo = nocc * nocc;
    const int nooo = noo * nocc;
    const size_t nvoo = nvir * noo;

    dgemm_(&TRANS_N, &TRANS_T, &noo, &nvir, &nocc,
           &D1, wabc_bar, &noo, vvop+nocc, &nmo,
           &D1, t2T_bar+c*nvoo, &noo);

    dgemm_(&TRANS_N, &TRANS_T, &nocc, &nocc, &noo,
           &DN1, wabc_bar, &nocc, vooo+a*nooo, &nocc,
           &D1, t2T_bar+c*nvoo+b*noo, &nocc);

    dgemv_(&TRANS_T, &nocc, &noo,
           &D1, vabc_bar, &nocc, fvohalf+c*nocc, &I1,
           &D1, t2T_bar+b*nvoo+a*noo, &I1);
}

static void get_fvo_bar(double *fvo_bar, double *vabc_bar, double *t2T,
                        int nocc, int nvir, int a, int b, int c)
{
    const int I1 = 1;
    const double D1 = 1;
    const double Half = .5;
    const char TRANS_N = 'N';
    const int noo = nocc * nocc;
    const size_t nvoo = nvir * noo;

    dgemv_(&TRANS_N, &nocc, &noo,
           &Half, vabc_bar, &nocc, t2T+b*nvoo+a*noo, &I1,
           &D1, fvo_bar+c*nocc, &I1);
}

static void get_eris_amps_bar(
        double *t1T_bar, double *t2T_bar, double *fvo_bar,
        double *vooo_bar, double *vvop_bar,
        double *w0_bar, double *v0_bar,
        double *t1Thalf, double *t2T, double *fvohalf,
        double *vooo, double *vvop,
        int nocc, int nvir, int a, int b, int c, int *idx,
        double *cache)
{
    int nooo = nocc * nocc * nocc;
    double *wabc_bar = cache;
    double *vabc_bar = wabc_bar + nooo;

    get_wabc_bar(wabc_bar, w0_bar, nocc, idx);
    get_vabc_bar(vabc_bar, v0_bar, nocc, idx);
    get_vvov_bar(vvop_bar, wabc_bar, t2T, nocc, nvir, a, b, c);
    get_vooo_bar(vooo_bar, wabc_bar, t2T, nocc, nvir, a, b, c);
    get_vvoo_bar(vvop_bar, vabc_bar, t1Thalf, nocc, nvir, a, b, c);
    get_t1T_bar(t1T_bar, vabc_bar, vvop, nocc, nvir, a, b, c);
    get_t2T_bar(t2T_bar, wabc_bar, vabc_bar, vvop, vooo, fvohalf, nocc, nvir, a, b, c);
    get_fvo_bar(fvo_bar, vabc_bar, t2T, nocc, nvir, a, b, c);
}

static void contract6_vjp(double *mat, double *mo_energy, double *t1Thalf, double *t2T,
                          double *fvohalf, double *vooo, void **cache,
                          double *mat_bar, double *mo_energy_bar, double *t1T_bar, double *t2T_bar,
                          double *fvo_bar, double *vooo_bar, void **cache_vjp, double et_bar,
                          int nocc, int nvir, int a, int b, int c,
                          int *permute_idx, double fac, double *cache1)
{
    int nooo = nocc * nocc * nocc;
    int *idx0 = permute_idx;
    int *idx1 = idx0 + nooo;
    int *idx2 = idx1 + nooo;
    int *idx3 = idx2 + nooo;
    int *idx4 = idx3 + nooo;
    int *idx5 = idx4 + nooo;

    double *w0 = cache1;
    double *w0_bar = w0;
    double *z0 = w0 + nooo;
    double *v0_bar = z0;
    double *d3 = z0 + nooo;
    double *d3_bar = d3 + nooo;
    double *w0_mat = d3_bar;

    cache1 += nooo * 4;
    get_wz(w0_mat, w0, z0, nocc, nvir, a, b, c,
           mat, mo_energy, t1Thalf, t2T, fvohalf,
           vooo, cache1, cache, permute_idx, fac);

    if (b == c) {
        get_d3(d3, mo_energy, nocc, a, b, c, .5);
    } else {
        get_d3(d3, mo_energy, nocc, a, b, c, 1.);
    }

    get_w0v0d3_bar(mat_bar, w0_bar, v0_bar, d3_bar,
                   w0_mat, mat, z0, w0, d3,
                   et_bar, nocc, permute_idx, cache1);

    if (b == c) {
        get_mo_energy_bar(mo_energy_bar, d3_bar, d3, a, b, c, nocc, .5);
    } else {
        get_mo_energy_bar(mo_energy_bar, d3_bar, d3, a, b, c, nocc, 1.);
    }

    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[0], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[0], nocc, nvir, a, b, c, idx0, cache1);
    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[1], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[1], nocc, nvir, a, c, b, idx1, cache1);
    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[2], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[2], nocc, nvir, b, a, c, idx2, cache1);
    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[3], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[3], nocc, nvir, b, c, a, idx3, cache1);
    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[4], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[4], nocc, nvir, c, a, b, idx4, cache1);
    get_eris_amps_bar(t1T_bar, t2T_bar, fvo_bar, vooo_bar, cache_vjp[5], w0_bar, v0_bar,
                      t1Thalf, t2T, fvohalf, vooo, cache[5], nocc, nvir, c, b, a, idx5, cache1);
}

static size_t lnoccsdt_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                                int b0, int b1, int c0, int c1,
                                void *cache_row_b, void *cache_col_b,
                                void *cache_row_c, void *cache_col_c,
                                size_t stride)
{
        size_t nov = nocc * (nocc+nvir) * stride;
        int db = b1 - b0;
        int dc = c1 - c0;
        size_t m, a, b, c;

        if (c1 <= b0) {
                m = 0;
                for (a = 0; a < nvir; a++) {
                for (b = b0; b < b1; b++) {
                for (c = c0; c < c1; c++, m++) {
                        jobs[m].a = a;
                        jobs[m].b = b;
                        jobs[m].c = c;
                        jobs[m].cache[0] = cache_col_b + nov*(db*(a)+b-b0);
                        jobs[m].cache[1] = cache_col_c + nov*(dc*(a)+c-c0);
                        jobs[m].cache[2] = cache_row_b + nov*(nvir*(b-b0)+a);
                        jobs[m].cache[3] = cache_row_b + nov*(nvir*(b-b0)+c);
                        jobs[m].cache[4] = cache_row_c + nov*(nvir*(c-c0)+a);
                        jobs[m].cache[5] = cache_row_c + nov*(nvir*(c-c0)+b);
                } } }
        } else {
                m = 0;
                for (a = 0; a < nvir; a++) {
                for (b = b0; b < b1; b++) {
                for (c = c0; c <= b; c++, m++) {
                        jobs[m].a = a;
                        jobs[m].b = b;
                        jobs[m].c = c;
                        jobs[m].cache[0] = cache_col_b + nov*(db*(a)+b-b0);
                        jobs[m].cache[1] = cache_col_b + nov*(db*(a)+c-c0);
                        jobs[m].cache[2] = cache_row_b + nov*(nvir*(b-b0)+a);
                        jobs[m].cache[3] = cache_row_b + nov*(nvir*(b-b0)+c);
                        jobs[m].cache[4] = cache_row_b + nov*(nvir*(c-c0)+a);
                        jobs[m].cache[5] = cache_row_b + nov*(nvir*(c-c0)+b);
                } } }
        }
        return m;
}

void lnoccsdt_energy_vjp(double *mat, double *mo_energy, double *t1T, double *t2T,
                         double *vooo, double *fvo, double et_bar,
                         int nocc, int nvir, int b0, int b1, int c0, int c1,
                         void *cache_row_a, void *cache_col_a,
                         void *cache_row_b, void *cache_col_b,
                         double *mat_bar,
                         double *mo_energy_bar,
                         double *t1T_bar,
                         double *t2T_bar,
                         double *vooo_bar,
                         double *fvo_bar,
                         double *cache_row_a_bar,
                         double *cache_col_a_bar,
                         double *cache_row_b_bar,
                         double *cache_col_b_bar)
{
    int da = nvir;
    int db = b1 - b0;
    int dc = c1 - c0;
    int nmo = nocc + nvir;

    CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*dc);
    size_t njobs = lnoccsdt_gen_jobs(jobs, nocc, nvir, b0, b1, c0, c1,
                                     cache_row_a, cache_col_a,
                                     cache_row_b, cache_col_b, sizeof(double));

    int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
    _make_permute_indices(permute_idx, nocc);

    double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
    double *fvohalf = t1Thalf + nvir*nocc;
    int k;
    for (k = 0; k < nvir*nocc; k++) {
        t1Thalf[k] = t1T[k] * .5;
        fvohalf[k] = fvo[k] * .5;
    }

    double *mat_bar_bufs[MAX_THREADS];
    double *mo_energy_bar_bufs[MAX_THREADS];
    double *t1T_bar_bufs[MAX_THREADS];
    double *t2T_bar_bufs[MAX_THREADS];
    double *vooo_bar_bufs[MAX_THREADS];
    double *fvo_bar_bufs[MAX_THREADS];

    double *cache_row_a_bar_bufs[MAX_THREADS];
    double *cache_col_a_bar_bufs[MAX_THREADS];
    double *cache_row_b_bar_bufs[MAX_THREADS];
    double *cache_col_b_bar_bufs[MAX_THREADS];

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double *mat_bar_priv;
        double *mo_energy_bar_priv;
        double *t1T_bar_priv;
        double *t2T_bar_priv;
        double *vooo_bar_priv;
        double *fvo_bar_priv;
        double *cache_row_a_bar_priv=NULL;
        double *cache_col_a_bar_priv=NULL;
        double *cache_row_b_bar_priv=NULL;
        double *cache_col_b_bar_priv=NULL;
        if (thread_id == 0) {
            mat_bar_priv = mat_bar;
            mo_energy_bar_priv = mo_energy_bar;
            t1T_bar_priv = t1T_bar;
            t2T_bar_priv = t2T_bar;
            vooo_bar_priv = vooo_bar;
            fvo_bar_priv = fvo_bar;
            cache_row_a_bar_priv = cache_row_a_bar;
            cache_col_a_bar_priv = cache_col_a_bar;
            if (c1 <= b0) {
                cache_row_b_bar_priv = cache_row_b_bar;
                cache_col_b_bar_priv = cache_col_b_bar;
            }
        } else {
            mat_bar_priv = calloc(nocc*nocc, sizeof(double));
            mo_energy_bar_priv = calloc(nmo, sizeof(double));
            t1T_bar_priv = calloc(nvir*nocc, sizeof(double));
            t2T_bar_priv = calloc(nvir*nvir*nocc*nocc, sizeof(double));
            vooo_bar_priv = calloc(nvir*nocc*nocc*nocc, sizeof(double));
            fvo_bar_priv = calloc(nvir*nocc, sizeof(double));
            cache_row_a_bar_priv = calloc(da*db*nocc*nmo, sizeof(double));
            if (b0 == 0 && b1 == nvir) {
                cache_col_a_bar_priv = cache_row_a_bar_priv;
            }
            else {
                cache_col_a_bar_priv = calloc(da*db*nocc*nmo, sizeof(double));
            }
            if (c1 <= b0) {
                cache_row_b_bar_priv = calloc(da*dc*nocc*nmo, sizeof(double));
                cache_col_b_bar_priv = calloc(da*dc*nocc*nmo, sizeof(double));
            }
        }

        mat_bar_bufs[thread_id] = mat_bar_priv;
        mo_energy_bar_bufs[thread_id] = mo_energy_bar_priv;
        t1T_bar_bufs[thread_id] = t1T_bar_priv;
        t2T_bar_bufs[thread_id] = t2T_bar_priv;
        vooo_bar_bufs[thread_id] = vooo_bar_priv;
        fvo_bar_bufs[thread_id] = fvo_bar_priv;

        cache_row_a_bar_bufs[thread_id] = cache_row_a_bar_priv;
        cache_col_a_bar_bufs[thread_id] = cache_col_a_bar_priv;
        cache_row_b_bar_bufs[thread_id] = cache_row_b_bar_priv;
        cache_col_b_bar_bufs[thread_id] = cache_col_b_bar_priv;

        CacheJob *jobs_vjp = malloc(sizeof(CacheJob) * da*db*dc);
        lnoccsdt_gen_jobs(jobs_vjp, nocc, nvir, b0, b1, c0, c1,
                          (void*)cache_row_a_bar_priv, (void*)cache_col_a_bar_priv,
                          (void*)cache_row_b_bar_priv, (void*)cache_col_b_bar_priv,
                          sizeof(double));

        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*6 + 2));
        #pragma omp for schedule(dynamic)
        for (k = 0; k < njobs; k++) {
            a = jobs[k].a;
            b = jobs[k].b;
            c = jobs[k].c;
            contract6_vjp(mat, mo_energy, t1Thalf, t2T,
                          fvohalf, vooo, jobs[k].cache,
                          mat_bar_priv, mo_energy_bar_priv, t1T_bar_priv, t2T_bar_priv,
                          fvo_bar_priv, vooo_bar_priv, jobs_vjp[k].cache, et_bar,
                          nocc, nvir, a, b, c,
                          permute_idx, 1., cache1);
        }
        free(jobs_vjp);
        free(cache1);

        omp_dsum_reduce_inplace(mat_bar_bufs, nocc*nocc);
        omp_dsum_reduce_inplace(mo_energy_bar_bufs, nmo);
        omp_dsum_reduce_inplace(t1T_bar_bufs, nvir*nocc);
        omp_dsum_reduce_inplace(t2T_bar_bufs, nvir*nvir*nocc*nocc);
        omp_dsum_reduce_inplace(vooo_bar_bufs, nvir*nocc*nocc*nocc);
        omp_dsum_reduce_inplace(fvo_bar_bufs, nvir*nocc);
        omp_dsum_reduce_inplace(cache_row_a_bar_bufs, da*db*nocc*nmo);
        if (thread_id != 0) {
            free(mat_bar_priv);
            free(mo_energy_bar_priv);
            free(t1T_bar_priv);
            free(t2T_bar_priv);
            free(vooo_bar_priv);
            free(fvo_bar_priv);
            free(cache_row_a_bar_priv);
        }

        if (b0 != 0 || b1 != nvir) {
            omp_dsum_reduce_inplace(cache_col_a_bar_bufs, da*db*nocc*nmo);
            if (thread_id != 0) {
                free(cache_col_a_bar_priv);
            }
        }
        if (c1 <= b0) {
            omp_dsum_reduce_inplace(cache_row_b_bar_bufs, da*dc*nocc*nmo);
            omp_dsum_reduce_inplace(cache_col_b_bar_bufs, da*dc*nocc*nmo);
            if (thread_id != 0) {
                free(cache_row_b_bar_priv);
                free(cache_col_b_bar_priv);
            }
        }
    }
    free(jobs);
    free(permute_idx);
    free(t1Thalf);
}
