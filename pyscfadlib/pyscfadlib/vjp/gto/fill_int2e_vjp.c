/* Copyright 2021-2024 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */
#include <stdlib.h>
#include "config.h"
#include "cint.h"

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        int di = 0;
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        for (i = i0; i < i1; i++) {
                di = MAX(di, ao_loc[i+1]-ao_loc[i]);
        }
        return di;
}

int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, n;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        int shls[4];
        int cache_size = 0;
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                shls[2] = i;
                shls[3] = i;
                n = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

static int no_prescreen()
{
    return 1;
}


void GTOnr2e_fill_r0_vjp_s4(
        int (*intor)(), int (*fprescreen)(),
        double* vjp, double *ybar, double *buf, int comp, int ishp, int jshp,
        int *shls_slice, int *ao_loc, CINTOpt *cintopt,
        int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int ksh1 = shls_slice[5];
    int lsh0 = shls_slice[6];
    int nk = ao_loc[ksh1] - ao_loc[ksh0];
    size_t nkl = nk * (nk+1) / 2;

    int ish = ishp + ish0;
    int jsh = jshp + jsh0;
    int i0 = ao_loc[ish] - ao_loc[ish0];
    int j0 = ao_loc[jsh] - ao_loc[jsh0];

    if (ish >= jsh) {
        ybar += nkl * (i0*(i0+1)/2 + j0);
    } else {
        ybar += nkl * (j0*(j0+1)/2 + i0);
    }

    int iatm = bas[ATOM_OF+ish*BAS_SLOTS];
    vjp += iatm * comp;

    int di = ao_loc[ish+1] - ao_loc[ish];
    int dj = ao_loc[jsh+1] - ao_loc[jsh];
    int dij = di * dj;
    int k0, l0, dk, dl, dijk, dijkl;
    int i, j, k, l, icomp;
    int ksh, lsh, kshp, lshp;
    int shls[4];
    double *ybar0, *pybar0, *pybar, *buf0, *pbuf, *cache;

    shls[0] = ish;
    shls[1] = jsh;

    for (kshp = 0; kshp < ksh1-ksh0; kshp++) {
    for (lshp = 0; lshp <= kshp; lshp++) {
        ksh = kshp + ksh0;
        lsh = lshp + lsh0;
        shls[2] = ksh;
        shls[3] = lsh;
        k0 = ao_loc[ksh] - ao_loc[ksh0];
        l0 = ao_loc[lsh] - ao_loc[lsh0];
        dk = ao_loc[ksh+1] - ao_loc[ksh];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        dijk = dij * dk;
        dijkl = dijk * dl;
        cache = buf + dijkl * comp;
        if ((*fprescreen)(shls, atm, bas, env) &&
            (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
            ybar0 = ybar + k0*(k0+1)/2+l0;
            buf0 = buf;
            for (icomp = 0; icomp < comp; icomp++) {
                pybar0 = ybar0;
                if (kshp > lshp) {
                    if (ish > jsh) {
                        for (i = 0; i < di; i++, pybar0+=nkl*(i0+i)) {
                        for (j = 0; j < dj; j++) {
                            pybar = pybar0 + nkl*j;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                            for (pbuf = buf0 + k*dij + j*di + i,
                                 l = 0; l < dl; l++) {
                                vjp[icomp] += pybar[l] * pbuf[l*dijk];
                            } }
                        } }
                    } else if (ish < jsh) {
                        for (j = 0; j < dj; j++, pybar0+=nkl*(j0+j)) {
                        for (i = 0; i < di; i++) {
                            pybar = pybar0 + nkl*i;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                            for (pbuf = buf0 + k*dij + j*di + i,
                                 l = 0; l < dl; l++) {
                                vjp[icomp] += pybar[l] * pbuf[l*dijk];
                            } }
                        } }
                    } else {// ish == jsh
                        for (i = 0; i < di; i++, pybar0+=nkl*(i0+i)) {
                        for (j = 0; j <= i; j++) {
                            pybar = pybar0 + nkl*j;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                                for (pbuf = buf0 + k*dij + j*di + i,
                                     l = 0; l < dl; l++) {
                                    vjp[icomp] += pybar[l] * pbuf[l*dijk];
                                }
                                for (pbuf = buf0 + k*dij + i*di + j,
                                     l = 0; l < dl; l++) {
                                    vjp[icomp] += pybar[l] * pbuf[l*dijk];
                                }
                            }
                        } }
                    }
                } else {// ksh == lsh
                    if (ish > jsh) {
                        for (i = 0; i < di; i++, pybar0+=nkl*(i0+i)) {
                        for (j = 0; j < dj; j++) {
                            pybar = pybar0 + nkl*j;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                            for (pbuf = buf0 + k*dij + j*di + i,
                                 l = 0; l <= k; l++) {
                                vjp[icomp] += pybar[l] * pbuf[l*dijk];
                            } }
                        } }
                    } else if (ish < jsh) {
                        for (j = 0; j < dj; j++, pybar0+=nkl*(j0+j)) {
                        for (i = 0; i < di; i++) {
                            pybar = pybar0 + nkl*i;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                            for (pbuf = buf0 + k*dij + j*di + i,
                                 l = 0; l <= k; l++) {
                                vjp[icomp] += pybar[l] * pbuf[l*dijk];
                            } }
                        } }
                    } else {// ish == jsh
                        for (i = 0; i < di; i++, pybar0+=nkl*(i0+i)) {
                        for (j = 0; j <= i; j++) {
                            pybar = pybar0 + nkl*j;
                            for (k = 0; k < dk; k++, pybar+=k0+k) {
                                for (pbuf = buf0 + k*dij + j*di + i,
                                     l = 0; l <= k; l++) {
                                    vjp[icomp] += pybar[l] * pbuf[l*dijk];
                                }
                                for (pbuf = buf0 + k*dij + i*di + j,
                                     l = 0; l <= k; l++) {
                                    vjp[icomp] += pybar[l] * pbuf[l*dijk];
                                }
                            }
                        } }
                    }
                }
                buf0 += dijkl;
            }
        }
    } }
}


void GTOnr2e_fill_r0_vjp(int (*intor)(), void (*fill)(), int (*fprescreen)(),
                         double *vjp, double* ybar, int comp,
                         int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    if (fprescreen == NULL) {
        fprescreen = no_prescreen;
    }

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                             atm, natm, bas, nbas, env);

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    double *vjp_loc;
    if (thread_id == 0) {
        vjp_loc = vjp;
    } else {
        vjp_loc = calloc(natm*comp, sizeof(double));
    }

    int ij, i, j;
    double *buf = malloc(sizeof(double) * (di*di*di*di*comp + cache_size));
    #pragma omp for nowait schedule(dynamic)
    for (ij = 0; ij < nish*njsh; ij++) {
        i = ij / njsh;
        j = ij % njsh;
        (*fill)(intor, fprescreen, vjp_loc, ybar, buf, comp, i, j, shls_slice,
                ao_loc, cintopt, atm, natm, bas, nbas, env);
    }
    free(buf);

    if (thread_id != 0) {
        for (i = 0; i < natm*comp; i++) {
            #pragma omp atomic
            vjp[i] += vjp_loc[i];
        }
        free(vjp_loc);
    }
}

    //minus sign for nuclear derivative
    int i;
    for (i = 0; i < natm*comp; i++) {
        vjp[i] *= -1.;
    }
}
