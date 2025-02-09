#include <stdlib.h>
#include "config.h"

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))

void omp_dsum_reduce_inplace(double **vec, size_t count)
{
        unsigned int nthreads = omp_get_num_threads();
        unsigned int thread_id = omp_get_thread_num();
        size_t blksize = (count + nthreads - 1) / nthreads;
        size_t start = thread_id * blksize;
        size_t end = MIN(start + blksize, count);
        double *dst = vec[0];
        double *src;
        size_t it, i;
#pragma omp barrier
        for (it = 1; it < nthreads; it++) {
                src = vec[it];
                for (i = start; i < end; i++) {
                        dst[i] += src[i];
                }
        }
#pragma omp barrier
}
