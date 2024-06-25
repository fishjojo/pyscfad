#include <stdlib.h>
#include "config.h"

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

static void dsymm_triu(int n, double *tril, double *mat, int hermi)
{
    size_t i, j, ij;
    if (hermi == HERMITIAN || hermi == SYMMETRIC) {
        for (ij = 1, j = 1; j < n; j++) {
            for (i = 0; i < j; i++, ij++) {
                tril[ij] += mat[i*n+j];
            }
            ij++;
        }
    } else {
        for (ij = 1, j = 1; j < n; j++) {
            for (i = 0; i < j; i++, ij++) {
                tril[ij] -= mat[i*n+j];
            }
            tril[ij++] = 0;
        }
    }
}

void NPdunpack_tril_vjp(int n, double *tril, double *mat, int hermi)
{
    size_t i, j, ij;
    for (ij = 0, i = 0; i < n; i++) {
        for (j = 0; j <= i; j++, ij++) {
            tril[ij] = mat[i*n+j];
        }
    }
    if (hermi) {
        dsymm_triu(n, tril, mat, hermi);
    }

}

void NPdunpack_tril_2d_vjp(int count, int n, double *tril, double *mat, int hermi)
{
    #pragma omp parallel
    {
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
        #pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
            NPdunpack_tril_vjp(n, tril+n2*ic, mat+nn*ic, hermi);
        }
    }
}

void NPdpack_tril_vjp(int n, double *tril, double *mat)
{
    size_t i, j, ij;
    for (ij = 0, i = 0; i < n; i++) {
        for (j = 0; j <= i; j++, ij++) {
            mat[i*n+j] = tril[ij];
        }
    }
}

void NPdpack_tril_2d_vjp(int count, int n, double *tril, double *mat)
{
    #pragma omp parallel
    {
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
        #pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
            NPdpack_tril_vjp(n, tril+n2*ic, mat+nn*ic);
        }
    }
}
