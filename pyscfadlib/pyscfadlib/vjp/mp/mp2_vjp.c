#include <stdlib.h>
#include "config.h"
#include "vhf/fblas.h"

#if defined _OPENMP
#define LOCK_T          omp_lock_t
#define LOCK_INIT(l)    omp_init_lock(l)
#define LOCK_DESTROY(l) omp_destroy_lock(l)
#define LOCK(l)         omp_set_lock(l)
#define UNLOCK(l)       omp_unset_lock(l)
#else
#define LOCK_T          char
#define LOCK_INIT(l)
#define LOCK_DESTROY(l)
#define LOCK(l)
#define UNLOCK(l)
#endif

// Pairs i >= j of nocc orbitals, ordered by rounds of the circle method,
// in which each orbital appears once, so that consecutive pairs mostly
// update different rows of iaL_bar. The pairs i == j come last.
static int _gen_pairs(int *pairs, int nocc)
{
        const int m = nocc + (nocc & 1);
        int r, k, a, b, n = 0;
        for (r = 0; r < m - 1; r++) {
                for (k = 0; k < m / 2; k++) {
                        if (k == 0) {
                                a = m - 1;
                                b = r;
                        } else {
                                a = (r + k) % (m - 1);
                                b = (r - k + m - 1) % (m - 1);
                        }
                        if (a < nocc && b < nocc) {
                                pairs[2*n  ] = a > b ? a : b;
                                pairs[2*n+1] = a > b ? b : a;
                                n++;
                        }
                }
        }
        for (a = 0; a < nocc; a++) {
                pairs[2*n  ] = a;
                pairs[2*n+1] = a;
                n++;
        }
        return n;
}

// VJP of the DF-RMP2 energy terms computed by pyscf's MP2_contract_d,
//   ed = sum_{ij,ab} V_ij,ab t_ij,ab,  ex = -sum_{ij,ab} V_ij,ba t_ij,ab,
//   V_ij = iaL[i] iaL[j]^T,  t_ij,ab = V_ij,ab / (e_i + e_j - e_a - e_b),
// for the cotangents ed_bar and ex_bar. With
//   G_ij,ab = 2 (ed_bar t_ij,ab - ex_bar t_ij,ba),
//   R_ij,ab = -t_ij,ab (ed_bar V_ij,ab - ex_bar V_ij,ba) / (e_i + e_j - e_a - e_b),
// and as G_ji = G_ij^T, R_ji = R_ij^T,
//   iaL_bar[k] = 2 sum_j G_kj iaL[j],
//   e_occ_bar[k] = 2 sum_{j,ab} R_kj,ab,  e_vir_bar[a] = -2 sum_{kj,b} R_kj,ab.
// Each pair i >= j computes V_ij once and updates the rows i and j of
// iaL_bar, each under the lock of that row.
// With the amplitudes t2[i,j,a,b] = t_ij,ab as an output too, the cotangent
// T = t2_bar adds S_ab / (e_i + e_j - e_a - e_b) to G_ij,ab and
// -S_ab t_ij,ab / (e_i + e_j - e_a - e_b) to R_ij,ab, with
// S_ab = (T_ij,ab + T_ji,ba) / 2 so that G_ji = G_ij^T and R_ji = R_ij^T
// still hold for the sums over both pairs (i,j) and (j,i).
// iaL: (nocc, nvir, naux) in C order, as in MP2_contract_d.
// If not NULL, the energy terms (ed, ex) are added to e_out, the amplitudes
// are written to t2_out, and t2_bar is included, the latter two of shape
// (nocc, nocc, nvir, nvir) in C order.
static void _contract_d_vjp(double *e_out, double *t2_out, const double *t2_bar,
                            double *iaL_bar, double *e_occ_bar, double *e_vir_bar,
                            const double ed_bar, const double ex_bar,
                            const double *iaL, const double *e_occ, const double *e_vir,
                            const int nocc, const int nvir, const int naux)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        const double D2 = 2;
        const size_t nvv = (size_t)nvir * nvir;
        const size_t nvx = (size_t)nvir * naux;

        int *pairs = malloc(sizeof(int) * nocc * (nocc + 1));
        const int npair = _gen_pairs(pairs, nocc);
        LOCK_T *locks = malloc(sizeof(LOCK_T) * nocc);
        int k;
        for (k = 0; k < nocc; k++) {
                LOCK_INIT(locks + k);
        }

#pragma omp parallel
{
        double *v = malloc(sizeof(double) * nvv * 3);
        double *t = v + nvv;
        double *g = t + nvv;
        double *eocc = calloc(nocc + nvir, sizeof(double));
        double *evir = eocc + nocc;
        double eij, d, r, ra, rsum, vt, vtx, s;
        double ed = 0;
        double ex = 0;
        double *t2ij, *t2ji;
        const double *t2b_ij = NULL;
        const double *t2b_ji = NULL;
        int ip, i, j, a, b;

#pragma omp for schedule(dynamic)
        for (ip = 0; ip < npair; ip++) {
                i = pairs[2*ip  ];
                j = pairs[2*ip+1];
                // v[a,b] = sum_L iaL[i,a,L] iaL[j,b,L]
                dgemm_(&TRANS_T, &TRANS_N, &nvir, &nvir, &naux,
                       &D1, iaL + j * nvx, &naux, iaL + i * nvx, &naux,
                       &D0, v, &nvir);
                eij = e_occ[i] + e_occ[j];
                for (a = 0; a < nvir; a++) {
                for (b = 0; b < nvir; b++) {
                        t[a*nvir+b] = v[a*nvir+b] / (eij - e_vir[a] - e_vir[b]);
                } }

                if (t2_out != NULL) {
                        // t2[i,j] = t_ij, t2[j,i] = t_ij^T
                        t2ij = t2_out + ((size_t)i * nocc + j) * nvv;
                        t2ji = t2_out + ((size_t)j * nocc + i) * nvv;
                        for (a = 0; a < nvir; a++) {
                        for (b = 0; b < nvir; b++) {
                                t2ij[a*nvir+b] = t[a*nvir+b];
                                t2ji[b*nvir+a] = t[a*nvir+b];
                        } }
                }
                if (t2_bar != NULL) {
                        t2b_ij = t2_bar + ((size_t)i * nocc + j) * nvv;
                        t2b_ji = t2_bar + ((size_t)j * nocc + i) * nvv;
                }

                rsum = 0;
                for (a = 0; a < nvir; a++) {
                        ra = 0;
                        for (b = 0; b < nvir; b++) {
                                d = eij - e_vir[a] - e_vir[b];
                                g[a*nvir+b] = 2 * (ed_bar * t[a*nvir+b] - ex_bar * t[b*nvir+a]);
                                r = -t[a*nvir+b] * (ed_bar * v[a*nvir+b] - ex_bar * v[b*nvir+a]) / d;
                                if (t2_bar != NULL) {
                                        s = .5 * (t2b_ij[a*nvir+b] + t2b_ji[b*nvir+a]);
                                        g[a*nvir+b] += s / d;
                                        r -= s * t[a*nvir+b] / d;
                                }
                                ra += r;
                                if (i != j) {
                                        // the column sums from R_ji = R_ij^T
                                        evir[b] -= 2 * r;
                                }
                        }
                        evir[a] -= 2 * ra;
                        rsum += ra;
                }
                eocc[i] += 2 * rsum;

                if (e_out != NULL) {
                        // the pair j > i gives the same terms as i > j
                        vt = 0;
                        vtx = 0;
                        for (a = 0; a < nvir; a++) {
                        for (b = 0; b < nvir; b++) {
                                vt += v[a*nvir+b] * t[a*nvir+b];
                                vtx += v[b*nvir+a] * t[a*nvir+b];
                        } }
                        if (i != j) {
                                vt *= 2;
                                vtx *= 2;
                        }
                        ed += vt;
                        ex -= vtx;
                }

                // iaL_bar[i,a,L] += 2 sum_b g[a,b] iaL[j,b,L]
                LOCK(locks + i);
                dgemm_(&TRANS_N, &TRANS_N, &naux, &nvir, &nvir,
                       &D2, iaL + j * nvx, &naux, g, &nvir,
                       &D1, iaL_bar + i * nvx, &naux);
                UNLOCK(locks + i);
                if (i != j) {
                        eocc[j] += 2 * rsum;
                        // iaL_bar[j,b,L] += 2 sum_a g[a,b] iaL[i,a,L]
                        LOCK(locks + j);
                        dgemm_(&TRANS_N, &TRANS_T, &naux, &nvir, &nvir,
                               &D2, iaL + i * nvx, &naux, g, &nvir,
                               &D1, iaL_bar + j * nvx, &naux);
                        UNLOCK(locks + j);
                }
        }

#pragma omp critical
        {
                for (i = 0; i < nocc; i++) {
                        e_occ_bar[i] += eocc[i];
                }
                for (a = 0; a < nvir; a++) {
                        e_vir_bar[a] += evir[a];
                }
                if (e_out != NULL) {
                        e_out[0] += ed;
                        e_out[1] += ex;
                }
        }
        free(v);
        free(eocc);
}
        for (k = 0; k < nocc; k++) {
                LOCK_DESTROY(locks + k);
        }
        free(locks);
        free(pairs);
}

void MP2_contract_d_vjp(double *iaL_bar, double *e_occ_bar, double *e_vir_bar,
                        const double ed_bar, const double ex_bar,
                        const double *iaL, const double *e_occ, const double *e_vir,
                        const int nocc, const int nvir, const int naux)
{
        _contract_d_vjp(NULL, NULL, NULL, iaL_bar, e_occ_bar, e_vir_bar,
                        ed_bar, ex_bar, iaL, e_occ, e_vir, nocc, nvir, naux);
}

// MP2_contract_d_vjp which, if not NULL, also adds the energy terms
// (ed, ex) to e_out, writes the amplitudes to t2_out, and includes the
// cotangent t2_bar of the amplitudes, so that the energy and its VJP for
// a cotangent known in advance take one pass, three dgemm per pair,
// instead of four.
void MP2_contract_d_energy_vjp(double *e_out, double *t2_out, const double *t2_bar,
                               double *iaL_bar, double *e_occ_bar, double *e_vir_bar,
                               const double ed_bar, const double ex_bar,
                               const double *iaL, const double *e_occ, const double *e_vir,
                               const int nocc, const int nvir, const int naux)
{
        _contract_d_vjp(e_out, t2_out, t2_bar, iaL_bar, e_occ_bar, e_vir_bar,
                        ed_bar, ex_bar, iaL, e_occ, e_vir, nocc, nvir, naux);
}
