#include <stdlib.h>

void pack_tril(int n, double *tril, double *mat)
{
    size_t i, j, ij;
    for (ij = 0, i = 0; i < n; i++) {
        for (j = 0; j < i; j++, ij++) {
            tril[ij] += mat[i*n+j];
            tril[ij] += mat[j*n+i];
        }
        tril[ij] += mat[i*n+i];
        ij++;
    }
}
