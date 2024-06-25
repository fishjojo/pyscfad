#ifndef HAVE_DEFINED_VJPUTIL_H
#define HAVE_DEFINED_VJPUTIL_H

void omp_dsum_reduce_inplace(double **vec, size_t count);

void pack_tril(int n, double *tril, double *mat);

#endif
