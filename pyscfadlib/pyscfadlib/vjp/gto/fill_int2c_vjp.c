#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"

#define IPOW(x, n) ((int) rint(pow(x, n)))

#define NORM_S 0.282094791773878143
#define NORM_P 0.488602511902919921

#define LEN_LABELS 120
#define CACHESIZE 144000

static const int LABELS_INT[] = {// l = 14
    0, 1, 2, 4, 5, 8, 13, 14, 17, 26, 40, 41, 44, 53, 80, 121, 122, 125, 134, 161, 242, 364, 365, 368, 377, 404, 485, 728, 1093, 1094, 1097, 1106, 1133, 1214, 1457, 2186, 3280, 3281, 3284, 3293, 3320, 3401, 3644, 4373, 6560, 9841, 9842, 9845, 9854, 9881, 9962, 10205, 10934, 13121, 19682, 29524, 29525, 29528, 29537, 29564, 29645, 29888, 30617, 32804, 39365, 59048, 88573, 88574, 88577, 88586, 88613, 88694, 88937, 89666, 91853, 98414, 118097, 177146, 265720, 265721, 265724, 265733, 265760, 265841, 266084, 266813, 269000, 275561, 295244, 354293, 531440, 797161, 797162, 797165, 797174, 797201, 797282, 797525, 798254, 800441, 807002, 826685, 885734, 1062881, 1594322, 2391484, 2391485, 2391488, 2391497, 2391524, 2391605, 2391848, 2392577, 2394764, 2401325, 2421008, 2480057, 2657204, 3188645, 4782968
};

static const int _LEN_SPH[] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
};

static const int _LEN_CART[] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};

static int compareints (const void* a, const void* b)
{
    return (*(int*)a - *(int*)b);
}


static int get_label_index(int key)
{
    int *ptr_item = (int*) bsearch(&key, LABELS_INT, LEN_LABELS, sizeof(int), compareints);
    if (ptr_item != NULL) {
        return (int) (ptr_item - LABELS_INT);
    }
    else {
        return -1;
    }
}


static int get_ndigits(int number, int base)
{
    int n = 0;
    while (number) {
        number /= base;
        n += 1;
    }
    return n;
}


static int promote_xyz(int index, int xyz, int order)
{
    int label = LABELS_INT[index];
    int i, n;

    switch (xyz) {
        case 0:
            return index;
            break;
        case 1:
            n = get_ndigits(label, 3);
            for (i = 0; i < order; i++) {
                label += IPOW(3, n + i);
            }
            return get_label_index(label);
            break;
        case 2:
            label *= IPOW(3, order);
            for (i = 0; i < order; i++) {
                label += 2 * IPOW(3, i);
            }
            return get_label_index(label);
            break;
        default:
            return -1;
    }
}


static void dxpy(int n, double* x, int incx, double* y, int incy)
{// y += x
    size_t i;
    size_t offx=0, offy=0;
    for (i = 0; i < n; i++) {
        y[offy] += x[offx];
        offx += incx;
        offy += incy;
    }
}


static double contract_ij_ij(double* x, double* y, int comp,
                             int nrow_y, int ldy, int i0, int j0,
                             int nrow, int ncol)
{// x in F order, y in C order
    int irow, jcol, ic;
    size_t size_y = (size_t) nrow_y * ldy;
    double sum = 0;
    double *py;
    for (ic = 0; ic < comp; ic++) {
        py = y + ic * size_y + i0 * ldy + j0;
        for (irow = 0; irow < nrow; irow++) {
            for (jcol = 0; jcol < ncol; jcol++) {
                sum += x[irow+jcol*nrow] * py[jcol];
            }
            py += ldy;
        }
        x += nrow * ncol;
    }
    return sum;
}


static double contract_ij_ji(double* x, double* y, int comp,
                             int nrow_y, int ldy, int i0, int j0,
                             int nrow, int ncol)
{// x in F order, y in C order
    int irow, jcol, ic;
    size_t size_y = (size_t) nrow_y * ldy;
    double sum = 0;
    double *py;
    for (ic = 0; ic < comp; ic++) {
        py = y + ic * size_y + j0 * ldy + i0;
        for (jcol = 0; jcol < ncol; jcol++) {
            for (irow = 0; irow < nrow; irow++) {
                sum += x[irow+jcol*nrow] * py[irow];
            }
            py += ldy;
        }
        x += nrow * ncol;
    }
    return sum;
}


static void GTOint2c_bra_r0_deriv(
        int (*intor)(), double (*contract)(),
        double* vjp, double* ybar, int comp, int ndim,
        int ish, int jsh,
        int *shls_slice, int *ao_loc, CINTOpt *opt,
        int *atm, int natm, int *bas, int nbas, double *env,
        double* cache, size_t cache_of)
{// (i'|j)
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int naoi_all = ao_loc[ish1] - ao_loc[ish0];
    const int naoj_all = ao_loc[jsh1] - ao_loc[jsh0];

    ish += ish0;
    jsh += jsh0;
    int i0 = ao_loc[ish] - ao_loc[ish0];
    int j0 = ao_loc[jsh] - ao_loc[jsh0];

    const int naoi = ao_loc[ish+1] - ao_loc[ish];
    const int naoj = ao_loc[jsh+1] - ao_loc[jsh];
    int dims[] = {naoi, naoj};
    int shls[] = {ish, jsh};
    double *mat = cache + cache_of;
    // mat in F order
    (*intor)(mat, dims, shls,
             atm, natm, bas, nbas, env, opt, cache);

    int ic;
    int iatm = bas[ATOM_OF+ish*BAS_SLOTS];
    double *ptr_vjp = vjp + iatm * ndim;
    for (ic = 0; ic < ndim; ic++) {
        // minus sign for nuclear derivative
        ptr_vjp[ic] -= (*contract)(mat, ybar, comp,
                                   naoi_all, naoj_all,
                                   i0, j0, naoi, naoj);
        mat += naoi * naoj * comp;
    }
}


static void GTOint2c_bra_rc_deriv(
        int (*intor)(), double (*contract)(),
        double* vjp, double* ybar, int comp, int ndim,
        int ish, int jsh,
        int *shls_slice, int *ao_loc, CINTOpt *opt,
        int *atm, int natm, int *bas, int nbas, double *env,
        double* cache, size_t cache_of)
{// (i'|j)
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int naoi_all = ao_loc[ish1] - ao_loc[ish0];
    const int naoj_all = ao_loc[jsh1] - ao_loc[jsh0];

    ish += ish0;
    jsh += jsh0;
    int i0 = ao_loc[ish] - ao_loc[ish0];
    int j0 = ao_loc[jsh] - ao_loc[jsh0];

    const size_t naoi = ao_loc[ish+1] - ao_loc[ish];
    const size_t naoj = ao_loc[jsh+1] - ao_loc[jsh];
    int dims[] = {naoi, naoj};
    int shls[] = {ish, jsh};
    double *mat = cache + cache_of;
    // mat in F order
    (*intor)(mat, dims, shls,
             atm, natm, bas, nbas, env, opt, cache);

    int ic;
    //FIXME the order of comp and ndim may be wrong
    for (ic = 0; ic < ndim; ic++) {
        vjp[ic] += (*contract)(mat, ybar, comp,
                               naoi_all, naoj_all,
                               i0, j0, naoi, naoj);
        mat += naoi * naoj * comp;
    }
}


static void GTOint2c_bra_exp_deriv(
        int (*intor)(), double (*contract)(),
        double* vjp, double* ybar,
        int* shlmap_c2u, int* es_of,
        int ish, int jsh,
        int *shls_slice, int *ao_loc, int *ao_loc_cart, CINTOpt *opt,
        int *atm, int natm, int *bas, int nbas, double *env,
        int cart, int order, double* cache, size_t cache_of)
{// (i'|j)
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const size_t naoj_all = ao_loc[jsh1] - ao_loc[jsh0];

    ish += ish0;
    jsh += jsh0;
    int i0 = ao_loc[ish] - ao_loc[ish0];
    int j0 = ao_loc[jsh] - ao_loc[jsh0];
    int ies = es_of[ish];

    int nprim_i = bas[NPRIM_OF+ish*BAS_SLOTS];
    int nctr_i = bas[NCTR_OF+ish*BAS_SLOTS];
    int nctr_j = bas[NCTR_OF+jsh*BAS_SLOTS];
    int li = bas[ANG_OF+ish*BAS_SLOTS];
    int lj = bas[ANG_OF+jsh*BAS_SLOTS];
    int lk = li + order;
    int ptr_coeff = bas[PTR_COEFF+ish*BAS_SLOTS];

    int ni, nj, ni_cart, nj_cart;
    ni = ni_cart = _LEN_CART[li];
    nj = nj_cart = _LEN_CART[lj];
    if (!cart) {
        ni = _LEN_SPH[li];
        nj = _LEN_SPH[lj];
    }
    int nk_cart = _LEN_CART[lk];

    int nij = ni * nj;
    int nij_cart = ni_cart * nj_cart;
    int naoj = ao_loc[jsh+1] - ao_loc[jsh];
    int naoj_cart = ao_loc_cart[jsh+1] - ao_loc_cart[jsh];

    int dims[] = {nk_cart, naoj_cart};
    int shls[2];

    double cnorm;
    if (li == 0) {
        cnorm = NORM_S;
    } else if (li == 1) {
        cnorm = NORM_P;
    } else {
        cnorm = 1.;
    }
    
    int i, j, k;
    int ix, iy, iz, ixyz;
    double *mat = cache + cache_of;
    double *gcart = mat + CACHESIZE;
    double *gsph = gcart;
    if (!cart) gsph = gcart + CACHESIZE;
    double *buf1 = gsph + CACHESIZE;

    int ksh = shlmap_c2u[ish] + ksh0;
    for (i = 0; i < nprim_i; i++) {
        assert (lk == bas[ANG_OF+ksh*BAS_SLOTS]);
        shls[0] = ksh;
        shls[1] = jsh;
        // mat in F order
        (*intor)(mat, dims, shls,
                 atm, natm, bas, nbas, env, opt, cache);

        memset(gcart, 0, sizeof(double) * ni_cart * naoj_cart);
        for (ixyz = 0; ixyz < ni_cart; ixyz++) {
            ix = promote_xyz(ixyz, 0, order);
            iy = promote_xyz(ixyz, 1, order);
            iz = promote_xyz(ixyz, 2, order);
            dxpy(naoj_cart, mat+ix, nk_cart, gcart+ixyz, ni_cart);
            dxpy(naoj_cart, mat+iy, nk_cart, gcart+ixyz, ni_cart);
            dxpy(naoj_cart, mat+iz, nk_cart, gcart+ixyz, ni_cart);
        }

        if (!cart) {
            double *ptr_gcart = gcart;
            double *ptr_gsph = gsph;
            double *tmp1;
            for (j = 0; j < nctr_j; j++) {
                tmp1 = CINTc2s_bra_sph(buf1, nj_cart, ptr_gcart, li);
                ptr_gsph = CINTc2s_ket_sph1(ptr_gsph, tmp1, ni, ni, lj);
                ptr_gcart += nij_cart;
                ptr_gsph += nij;
            }
        }

        for (k = 0; k < nctr_i; k++) {
            double sum = (*contract)(gsph, ybar,
                                     naoj_all, i0 + k*ni, j0,
                                     ni, naoj);
            double c = -cnorm * env[ptr_coeff + k*nprim_i + i];
            vjp[ies] += sum * c;
        }
        ksh += 1;
        ies += 1;
    }
}


static void GTOint2c_bra_coeff_deriv(
        int (*intor)(), double (*contract)(),
        double* vjp, double* ybar,
        int* shlmap_c2u, int* cs_of,
        int ish, int jsh,
        int *shls_slice, int *ao_loc, CINTOpt *opt,
        int *atm, int natm, int *bas, int nbas, double *env,
        int cart, double* cache, size_t cache_of)
{// (i'|j)
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const size_t naoj_all = ao_loc[jsh1] - ao_loc[jsh0];

    ish += ish0;
    jsh += jsh0;
    int i0 = ao_loc[ish] - ao_loc[ish0];
    int j0 = ao_loc[jsh] - ao_loc[jsh0];
    int ics = cs_of[ish];

    int nprim_i = bas[NPRIM_OF+ish*BAS_SLOTS];
    int nctr_i = bas[NCTR_OF+ish*BAS_SLOTS];
    int li = bas[ANG_OF+ish*BAS_SLOTS];
    int ni = (cart == 0) ? _LEN_SPH[li] : _LEN_CART[li];
    int naoj = ao_loc[jsh+1] - ao_loc[jsh];

    int i, k;
    int dims[] = {ni, naoj};
    int shls[2];
    int ksh = shlmap_c2u[ish] + ksh0;
    double *mat = cache + cache_of;
    for (i = 0; i < nprim_i; i++) {
        assert (li == bas[ANG_OF+ksh*BAS_SLOTS]);
        shls[0] = ksh;
        shls[1] = jsh;
        // mat in F order
        (*intor)(mat, dims, shls,
                 atm, natm, bas, nbas, env, opt, cache);

        for (k = 0; k < nctr_i; k++) {
            double sum = (*contract)(mat, ybar,
                                     naoj_all, i0 + k*ni, j0,
                                     ni, naoj);
            vjp[ics + k*nprim_i] += sum;
        }
        ksh += 1;
        ics += 1;
    }
}


void GTOint2c_r0_vjp(int (*intor)(), double* vjp, double* ybar,
                     int comp, int ndim, int hermi, int *shls_slice,
                     int *ao_loc, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    int shls_slice_ji[] = {jsh0, jsh1, ish0, ish1};
    size_t cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                          atm, natm, bas, nbas, env);
    size_t cache_of = cache_size;
    cache_size += CACHESIZE;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    double *vjp_loc;
    if (thread_id == 0) {
        vjp_loc = vjp;
    } else {
        vjp_loc = calloc(natm*ndim, sizeof(double));
    }

    int i, ij, ish, jsh;
    double *cache = malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(dynamic, 4)
    for (ij = 0; ij < nish*njsh; ij++) {
        ish = ij / njsh;
        jsh = ij % njsh;

        GTOint2c_bra_r0_deriv(
            intor, contract_ij_ij, vjp_loc, ybar, comp, ndim,
            ish, jsh, shls_slice, ao_loc, opt,
            atm, natm, bas, nbas, env, cache, cache_of);

        if (hermi == 0) {
            GTOint2c_bra_r0_deriv(
                intor, contract_ij_ji, vjp_loc, ybar, comp, ndim,
                jsh, ish, shls_slice_ji, ao_loc, opt,
                atm, natm, bas, nbas, env, cache, cache_of);
        }
    }
    free(cache);

    if (thread_id != 0) {
        for (i = 0; i < natm*ndim; i++) {
            #pragma omp atomic
            vjp[i] += vjp_loc[i];
        }
        free(vjp_loc);
    }
}
}


void GTOint2c_rc_vjp(int (*intor)(), double* vjp, double* ybar,
                     int comp, int ndim, int hermi, int *shls_slice,
                     int *ao_loc, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    int shls_slice_ji[] = {jsh0, jsh1, ish0, ish1};
    size_t cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                          atm, natm, bas, nbas, env);
    size_t cache_of = cache_size;
    cache_size += CACHESIZE;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    double *vjp_loc;
    if (thread_id == 0) {
        vjp_loc = vjp;
    } else {
        vjp_loc = calloc(ndim, sizeof(double));
    }

    int i, ij, ish, jsh;
    double *cache = malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(dynamic, 4)
    for (ij = 0; ij < nish*njsh; ij++) {
        ish = ij / njsh;
        jsh = ij % njsh;

        GTOint2c_bra_rc_deriv(
            intor, contract_ij_ij, vjp_loc, ybar, comp, ndim,
            ish, jsh, shls_slice, ao_loc, opt,
            atm, natm, bas, nbas, env, cache, cache_of);

        if (hermi == 0) {
            GTOint2c_bra_rc_deriv(
                intor, contract_ij_ji, vjp_loc, ybar, comp, ndim,
                jsh, ish, shls_slice_ji, ao_loc, opt,
                atm, natm, bas, nbas, env, cache, cache_of);
        }
    }

    free(cache);

    if (thread_id != 0) {
        for (i = 0; i < ndim; i++) {
            #pragma omp atomic
            vjp[i] += vjp_loc[i];
        }
        free(vjp_loc);
    }
}
}


void GTOint2c_exp_vjp(int (*intor)(), //intor is always *_cart
                     double* vjp, int nes, double* ybar,
                     int* shlmap_c2u, int* es_of,
                     int comp, int hermi,
                     int *shls_slice, int *ao_loc, int *ao_loc_cart, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     int cart, int order)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    int shls_slice_ji[] = {jsh0, jsh1, ish0, ish1, ksh0, ksh1};
    size_t cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                          atm, natm, bas, nbas, env);
    size_t cache_of = cache_size;
    cache_size += CACHESIZE * 4;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    double *vjp_loc;
    if (thread_id == 0) {
        vjp_loc = vjp;
    } else {
        vjp_loc = calloc(nes, sizeof(double));
    }

    int i, ij, ish, jsh;
    double *cache = malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(dynamic, 4)
    for (ij = 0; ij < nish*njsh; ij++) {
        ish = ij / njsh;
        jsh = ij % njsh;

        GTOint2c_bra_exp_deriv(
            intor, contract_ij_ij, vjp_loc, ybar, shlmap_c2u, es_of,
            ish, jsh, shls_slice, ao_loc, ao_loc_cart, opt,
            atm, natm, bas, nbas, env, cart, order, cache, cache_of);

        if (hermi == 0) {
            GTOint2c_bra_exp_deriv(
                intor, contract_ij_ji, vjp_loc, ybar, shlmap_c2u, es_of,
                jsh, ish, shls_slice_ji, ao_loc, ao_loc_cart, opt,
                atm, natm, bas, nbas, env, cart, order, cache, cache_of);
        }
    }

    free(cache);

    if (thread_id != 0) {
        for (i = 0; i < nes; i++) {
            #pragma omp atomic
            vjp[i] += vjp_loc[i];
        }
        free(vjp_loc);
    }
}
}


void GTOint2c_coeff_vjp(int (*intor)(),
                        double* vjp, int ncs, double* ybar,
                        int* shlmap_c2u, int* cs_of,
                        int comp, int hermi,
                        int *shls_slice, int *ao_loc, CINTOpt *opt,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        int cart)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    int shls_slice_ji[] = {jsh0, jsh1, ish0, ish1, ksh0, ksh1};
    size_t cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                          atm, natm, bas, nbas, env);
    size_t cache_of = cache_size;
    cache_size += CACHESIZE;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    double *vjp_loc;
    if (thread_id == 0) {
        vjp_loc = vjp;
    } else {
        vjp_loc = calloc(ncs, sizeof(double));
    }

    int i, ij, ish, jsh;
    double *cache = malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(dynamic, 4)
    for (ij = 0; ij < nish*njsh; ij++) {
        ish = ij / njsh;
        jsh = ij % njsh;

        GTOint2c_bra_coeff_deriv(
            intor, contract_ij_ij, vjp_loc, ybar, shlmap_c2u, cs_of,
            ish, jsh, shls_slice, ao_loc, opt,
            atm, natm, bas, nbas, env, cart, cache, cache_of);

        if (hermi == 0) {
            GTOint2c_bra_coeff_deriv(
                intor, contract_ij_ji, vjp_loc, ybar, shlmap_c2u, cs_of,
                jsh, ish, shls_slice_ji, ao_loc, opt,
                atm, natm, bas, nbas, env, cart, cache, cache_of);
        }
    }

    free(cache);

    if (thread_id != 0) {
        for (i = 0; i < ncs; i++) {
            #pragma omp atomic
            vjp[i] += vjp_loc[i];
        }
        free(vjp_loc);
    }
}
}
