/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include "stdlib.h"
#include "np_helper.h"

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

void NPzhermi_triu(int n, double complex *mat, int hermi)
{
        size_t i, j, j0, j1;

        if (hermi == HERMITIAN) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = conj(mat[j*n+i]);
                }
        } else if (hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = mat[j*n+i];
                }
        } else {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = -conj(mat[j*n+i]);
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

