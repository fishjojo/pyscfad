# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import scipy

def logm(A, disp=True, real=False):
    """Compute matrix logarithm.

    Compute the matrix logarithm ensuring that it is real
    when `A` is nonsingular and each Jordan block of `A` belonging
    to negative eigenvalue occurs an even number of times.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose logarithm to evaluate
    real : bool, default=False
        If `True`, compute a real logarithm of a real matrix if it exists.
        Otherwise, call `scipy.linalg.logm`.

    See Also
    --------
    scipy.linalg.logm
    """
    if real and numpy.isreal(A).all():
        A = numpy.real(A)
        # perform the real Schur decomposition
        t, q = scipy.linalg.schur(A)

        n = A.shape[0]
        idx = 0
        real_output = True
        while idx < n:
            # final block reached for an odd number of orbitals
            if idx == n - 1:
                # single positive block
                if t[idx,idx] > 0:
                    t[idx,idx] = numpy.log(t[idx,idx])
                # single negative block
                else:
                    real_output = False
                    break
            else:
                diag = numpy.isclose(t[idx,idx+1], 0) and numpy.isclose(t[idx+1,idx], 0)
                # single positive block
                if t[idx,idx] > 0 and diag:
                    t[idx,idx] = numpy.log(t[idx,idx])
                # pair of two negative blocks
                elif (
                    t[idx,idx] < 0
                    and diag
                    and numpy.isclose(t[idx,idx], t[idx + 1,idx + 1])
                ):
                    log_lambda = numpy.log(-t[idx,idx])
                    t[idx:idx+2,idx:idx+2] = numpy.array(
                        [[log_lambda, numpy.pi], [-numpy.pi, log_lambda]]
                    )
                    idx += 1
                # antisymmetric 2x2 block
                elif (
                    numpy.isclose(t[idx,idx], t[idx + 1,idx + 1])
                    and numpy.isclose(t[idx + 1,idx], -t[idx,idx + 1])
                ):
                    log_comp = numpy.log(complex(t[idx,idx], t[idx,idx + 1]))
                    t[idx:idx+2,idx:idx+2] = numpy.array(
                        [
                            [numpy.real(log_comp), numpy.imag(log_comp)],
                            [-numpy.imag(log_comp), numpy.real(log_comp)],
                        ],
                    )
                    idx += 1
                else:
                    real_output = False
                    break
            idx += 1

        if not real_output:
            return scipy.linalg.logm(A, disp=disp)

        F = q @ t @ q.T

        # NOTE copied from scipy
        errtol = 1000 * numpy.finfo("d").eps
        # TODO use a better error approximation
        errest = scipy.linalg.norm(scipy.linalg.expm(F) - A, 1) / scipy.linalg.norm(A, 1)
        if disp:
            if not numpy.isfinite(errest) or errest >= errtol:
                print("logm result may be inaccurate, approximate err =", errest)
            return F
        else:
            return F, errest

    else:
        return scipy.linalg.logm(A, disp=disp)

