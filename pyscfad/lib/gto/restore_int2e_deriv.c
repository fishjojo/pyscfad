#include <stdlib.h>
#include <string.h>
#include "config.h"

#define BLOCK_DIM    120

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#define TRIU_LOOP(I, J) \
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) \
                for (I = 0, j1 = MIN(j0+BLOCK_DIM, n); I < j1; I++) \
                        for (J = MAX(I,j0); J < j1; J++)


void NPdsymm_triu(int n, double *mat, int hermi)
{
        size_t i, j, j0, j1;

        if (hermi == HERMITIAN || hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = mat[j*n+i];
                }
        } else {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = -mat[j*n+i];
                }
        }
}

void NPdunpack_tril(int n, double *tril, double *mat, int hermi)
{
        size_t i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        mat[i*n+j] = tril[ij];
                }
        }
        if (hermi) {
                NPdsymm_triu(n, mat, hermi);
        }
}

void restore_int2e_deriv(double* out, double* eri, int* aoslices, int natm, int nao)
{
    const size_t n2 = nao * nao;
    const size_t n3 = nao*nao*nao;
    const size_t n4 = nao*nao*nao*nao;
    const size_t npair = nao*(nao+1)/2;
    const size_t nnpair = nao * npair;
    const size_t n2npair = n2 * npair;

#pragma omp parallel
{
    int i, j, x;
    int iatm, xyz, p0, p1;
    double *data, *buf;
#pragma omp for nowait schedule(dynamic)
    for (x = 0; x < natm*3; x++){
        iatm = x / 3;
        xyz = x % 3;
        p0 = aoslices[iatm*4+2];
        p1 = aoslices[iatm*4+3];
        for (i = p0; i < p1; i++){
            for (j = 0; j < nao; j++){
                data = eri + xyz * n2npair + i * nnpair + j * npair;
                buf = out + x * n4 + i * n3 + j * n2;
                NPdunpack_tril(nao, data, buf, 1);
            }
        }
    }
}
}
