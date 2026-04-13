/* Copyright 2026 The PySCFAD Authors
 
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
#include <stdbool.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "np_helper/np_helper.h"

#define INTBUFMAX10    8000

static int shloc_partition(int *kshloc, int *ao_loc, int ksh0, int ksh1, int dkmax)
{
    int ksh;
    int nloc = 0;
    int loclast = ao_loc[ksh0];
    kshloc[0] = ksh0;
    for (ksh = ksh0+1; ksh < ksh1; ksh++) {
        assert(ao_loc[ksh+1] - ao_loc[ksh] < dkmax);
        if (ao_loc[ksh+1] - loclast > dkmax) {
            nloc += 1;
            kshloc[nloc] = ksh;
            loclast = ao_loc[ksh];
        }
    }
    nloc += 1;
    kshloc[nloc] = ksh1;
    return nloc;
}


static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
        env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
        env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
        env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}


static void sort2c_s1(double *out, double *buf,
                      int *shls_slice, int *ao_loc, int comp,
                      int jsh, int msh0, int msh1, int jL)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
    const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
    const size_t nij = naoi * naoj;

    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int jp = ao_loc[jsh] - ao_loc[jsh0];
    out += jL * comp * nij + jp;

    int i, j, ish, ic, di, dij;
    size_t off = 0;
    double *pbr, *pout;

    for (ish = msh0; ish < msh1; ish++) {
        di = ao_loc[ish+1] - ao_loc[ish];
        dij = di * dj;
        for (ic = 0; ic < comp; ic++) {
            pout = out + nij*ic + naoj*(ao_loc[ish]-ao_loc[ish0]);
            pbr = buf + off + dij*ic;
            for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                    pout[i*naoj+j] = pbr[j*di+i];
                }
            }
        }
        off += dij * comp;
    }
}


static void sort2c_s2(double *out, double *buf,
                      int *shls_slice, int *ao_loc, int comp,
                      int jsh, int msh0, int msh1, int jL)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
    const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
    const size_t nij = naoi * naoj;

    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int jp = ao_loc[jsh] - ao_loc[jsh0];
    out += jL * comp * nij + jp;

    int i, j, ish, ic, di, dij, i0;
    size_t off = 0;
    double *pbr, *pout;

    for (ish = msh0; ish < msh1; ish++) {
        di = ao_loc[ish+1] - ao_loc[ish];
        dij = di * dj;
        for (ic = 0; ic < comp; ic++) {
            pout = out + nij*ic + naoj*(ao_loc[ish]-ao_loc[ish0]);
            pbr = buf + off + dij*ic;
            for (j = 0; j < dj; j++) {
                i0 = (ish == jsh) ? j : 0;
                for (i = i0; i < di; i++) {
                    pout[i*naoj+j] = pbr[j*di+i];
                }
            }
        }
        off += dij * comp;
    }
}


static int _nr2c_fill(int (*intor)(), void (*sort2c)(), double *out,
                      int comp, int nimgs, int jsh, int ish0,
                      double *buf, double *env_loc,
                      double *Ls, bool *Ls_mask,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];

    ish0 += shls_slice[0];
    jsh += jsh0;
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    int dimax = INTBUFMAX10 / dj;
    int ishloc[ish1-ish0+1];
    int nishloc = shloc_partition(ishloc, ao_loc, ish0, ish1, dimax);

    int m, msh0, msh1, dmjc, ish, di, empty;
    int jL;
    int shls[2];
    double *pbuf, *cache;

    shls[1] = jsh;
    for (m = 0; m < nishloc; m++) {
        msh0 = ishloc[m];
        msh1 = ishloc[m+1];
        dimax = ao_loc[msh1] - ao_loc[msh0];
        dmjc = dj * dimax * comp;
        cache = buf + dmjc;

        for (jL = 0; jL < nimgs; jL++) {
            if (!Ls_mask[jL]) continue;

            pbuf = buf;
            shift_bas(env_loc, env, Ls, jptrxyz, jL);
            for (ish = msh0; ish < msh1; ish++) {
                shls[0] = ish;
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                             env_loc, cintopt, cache)) {
                    empty = 0;
                }
                pbuf += di * dj * comp;
            }
            sort2c(out, buf, shls_slice, ao_loc,
                   comp, jsh, msh0, msh1, jL);
        }
    }
    return !empty;
}


void LATnr2c_fill_s1(int (*intor)(), double *out,
                     int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls, bool *Ls_mask,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr2c_fill(intor, sort2c_s1, out, comp, nimgs, jsh, 0,
               buf, env_loc, Ls, Ls_mask, shls_slice, ao_loc,
               cintopt, atm, natm, bas, nbas, env);
}


void LATnr2c_fill_s2(int (*intor)(), double *out,
                     int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls, bool *Ls_mask,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr2c_fill(intor, sort2c_s2, out, comp, nimgs, jsh, jsh,
               buf, env_loc, Ls, Ls_mask, shls_slice, ao_loc,
               cintopt, atm, natm, bas, nbas, env);
}


void LATnr2c_drv(int (*intor)(), void (*fill)(), double *out,
                 int comp, int nimgs, double *Ls, bool *Ls_mask,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int njsh = jsh1 - jsh0;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);

#pragma omp parallel
{
    int jsh;
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);
    double *buf = malloc(sizeof(double)*(INTBUFMAX10*comp+cache_size));
    #pragma omp for schedule(dynamic)
    for (jsh = 0; jsh < njsh; jsh++) {
    	(*fill)(intor, out, comp, nimgs, jsh,
                buf, env_loc, Ls, Ls_mask,
                shls_slice, ao_loc, cintopt, atm, natm, bas, nbas, env);
    }
    free(buf);
    free(env_loc);
}
}
