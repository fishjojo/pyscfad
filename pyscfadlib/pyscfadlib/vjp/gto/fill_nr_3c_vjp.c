#include <stdlib.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#define BLKSIZE 8

static void fill_ij_r0_s2_igtj(double *vjp, double *in, double* ybar, int comp,
                            size_t ip, size_t nij, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double *pin, *ptr_ybar;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        ptr_ybar = ybar + k * nij;
                        pin  = in + k * dij;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        vjp[ic] += pin[j*di+i] * ptr_ybar[j];
                                }
                                ptr_ybar += ip1 + i;
                        }
                }
                in += dij * dk;
        }
}

static void fill_ij_r0_s2_jgti(double *vjp, double *in, double* ybar, int comp,
                            size_t ip, size_t nij, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double *pin, *ptr_ybar;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        ptr_ybar = ybar + k * nij;
                        pin  = in  + k * dij;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        vjp[ic] += pin[i*dj+j] * ptr_ybar[j];
                                }
                                ptr_ybar += ip1 + i;
                        }
                }
                in += dij * dk;
        }
}

static void fill_ij_r0_s2_ieqj(double *vjp, double *in, double* ybar, int comp,
                            size_t ip, size_t nij, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double *pin, *ptr_ybar;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        ptr_ybar = ybar + k * nij;
                        pin  = in  + k * dij;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < i; j++) {
                                        vjp[ic] += pin[j*di+i] * ptr_ybar[j];
                                        vjp[ic] += pin[i*di+j] * ptr_ybar[j];
                                }
                                vjp[ic] += pin[j*di+j] * ptr_ybar[j] * 2.;
                                ptr_ybar += ip1 + i;
                        }
                }
                in += dij * dk;
        }
}

void GTOnr3c_ij_r0_vjp_s2ij(int (*intor)(), double *vjp, double* ybar, double *buf,
                       int comp, int jobid,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nksh = ksh1 - ksh0;

        const int ksh = jobid % nksh + ksh0;
        //const int istart = jobid / nksh * BLKSIZE + ish0;
        //const int iend = MIN(istart + BLKSIZE, ish1);
        //if (istart >= iend) {
        //        return;
        //}

        const int jstart = jobid / nksh * BLKSIZE + jsh0;
        const int jend = MIN(jstart + BLKSIZE, jsh1);
        if (jstart >= jend) {
                return;
        }

        const int i0 = ao_loc[ish0];
        const int i1 = ao_loc[ish1];
        //const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off = i0 * (i0 + 1) / 2;
        const size_t nij = i1 * (i1 + 1) / 2 - off;

        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int k0 = ao_loc[ksh] - ao_loc[ksh0];
        ybar += nij * k0;

        int ish, jsh, ip, jp, di, dj;
        int iatm, jatm;
        int shls[3] = {0, 0, ksh};
        di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        double *cache = buf + di * di * dk * comp;
        double *ptr_ybar;
        double *ptr_vjp_i, *ptr_vjp_j;

        for (ish = ish0; ish < ish1; ish++) {
            iatm = bas[ATOM_OF+ish*BAS_SLOTS];
            ptr_vjp_i = vjp + iatm * comp;
        for (jsh = jstart; jsh < jend; jsh++) {
                if (ish < jsh) {
                        continue;
                }

                ip = ao_loc[ish];
                jp = ao_loc[jsh] - ao_loc[jsh0];
                shls[0] = ish;
                shls[1] = jsh;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];

                (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache);

                ptr_ybar = ybar + (ip * (ip + 1) / 2 - off + jp);
                if (ish != jsh) {
                        fill_ij_r0_s2_igtj(ptr_vjp_i, buf, ptr_ybar, comp, ip, nij, di, dj, dk);
                } else {
                        fill_ij_r0_s2_ieqj(ptr_vjp_i, buf, ptr_ybar, comp, ip, nij, di, dj, dk);
                }

                if (ish != jsh) {
                        shls[0] = jsh;
                        shls[1] = ish;
                        (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache);
                        jatm = bas[ATOM_OF+jsh*BAS_SLOTS];
                        ptr_vjp_j = vjp + jatm * comp;
                        fill_ij_r0_s2_jgti(ptr_vjp_j, buf, ptr_ybar, comp, ip, nij, di, dj, dk);
                }

        } }
}

void GTOnr3c_ij_r0_vjp(int (*intor)(), void (*fill)(), double *vjp, double *ybar, int comp,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int natm_ij, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int nksh = ksh1 - ksh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 3);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);
        const int njobs = (MAX(nish,njsh) / BLKSIZE + 1) * nksh;

#pragma omp parallel
{
        int i, jobid;
        double *buf = malloc(sizeof(double) * (di*di*di*comp + cache_size));
        int thread_id = omp_get_thread_num();
        double *vjp_loc;
        if (thread_id == 0) {
            vjp_loc = vjp;
        } else {
            vjp_loc = calloc(natm_ij*comp, sizeof(double));
        }

        #pragma omp for nowait schedule(dynamic)
        for (jobid = 0; jobid < njobs; jobid++) {
                (*fill)(intor, vjp_loc, ybar, buf, comp, jobid, shls_slice, ao_loc,
                        cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);

        if (thread_id != 0) {
            for (i = 0; i < natm_ij*comp; i++) {
                #pragma omp atomic
                vjp[i] += vjp_loc[i];
            }
            free(vjp_loc);
        }
}
}

