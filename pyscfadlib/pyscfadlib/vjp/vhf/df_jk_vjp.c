#include <stdlib.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "vjp/util/util.h"

#define MAX_THREADS     128

// vk_bar in F order
// eri_bar = einsum('pki,ij->pkj', buf1, vk_bar)
// buf1_bar = einsum('ij,pkj->pki', vk_bar, eri)
// eri_bar = einsum('pki,jk->pij', buf1_bar, dm)
// dm_bar = einsum('pij,pki->jk', eri, buf1_bar)
static void _contract_vk(double* eri_tril_bar, double *dm_bar,
                         double *vk_bar, double *buf1,
                         double *eri_tril, double *dm,
                         int nao, double *cache)
{
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const char SIDE_L = 'L';
    const char SIDE_R = 'R';
    const char UPLO_U = 'U';
    const size_t nao2 = nao * nao;
    double *eri = cache;
    double *buf1_bar = eri + nao2;
    double *eri_bar = eri;

    NPdunpack_tril(nao, eri_tril, eri, 0);
    dsymm_(&SIDE_R, &UPLO_U, &nao, &nao,
           &D1, eri, &nao, vk_bar, &nao,
           &D0, buf1_bar, &nao);

    dsymm_(&SIDE_L, &UPLO_U, &nao, &nao,
           &D1, eri, &nao, buf1_bar, &nao,
           &D1, dm_bar, &nao);  //dm_bar in F order

    dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nao,
           &D1, vk_bar, &nao, buf1, &nao,
           &D0, eri_bar, &nao);
    dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &nao,
           &D1, dm, &nao, buf1_bar, &nao,
           &D1, eri_bar, &nao); //dm in F order
    pack_tril(nao, eri_tril_bar, eri_bar);
}


void df_vk_vjp(double *eri_tril_bar, double *dm_bar,
               double *vk_bar, double *buf1,
               double *eri_tril, double *dm,
               int naux, int nao)
{
    const size_t nao2 = (size_t)nao * nao;
    const size_t nao_pair = (size_t)nao * (nao+1) /2;
    double *dm_bar_bufs[MAX_THREADS];
    #pragma omp parallel
    {
        int i;
        int thread_id = omp_get_thread_num();
        double *dm_bar_priv;
        if (thread_id == 0) {
            dm_bar_priv = dm_bar;
        } else {
            dm_bar_priv = calloc(nao2, sizeof(double));
        }
        dm_bar_bufs[thread_id] = dm_bar_priv;
        double *cache = malloc(nao2*2 * sizeof(double));
        #pragma omp for schedule(dynamic)
        for (i = 0; i < naux; i++) {
            _contract_vk(eri_tril_bar+i*nao_pair, dm_bar_priv,
                         vk_bar, buf1+i*nao2, eri_tril+i*nao_pair, dm,
                         nao, cache);
        }
        free(cache);

        omp_dsum_reduce_inplace(dm_bar_bufs, nao2);
        if (thread_id != 0) {
            free(dm_bar_priv);
        }
    }
}
