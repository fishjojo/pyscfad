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

from functools import reduce
import numpy
import scipy
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg import gmres as scipy_gmres
from pyscfad import ops

def _matvec_to_scipy(matvec, b):
    def _matvec(x):
        Ax = matvec(x.reshape(b.shape)).ravel()
        # NOTE result may not be writable
        # (required by scipy>=1.12), so make a copy
        return numpy.array(ops.to_numpy(Ax))
    A = LinearOperator((b.size, b.size), matvec=_matvec, dtype=b.dtype)
    return A

def gmres(A_or_matvec, b, x0=None, *,
          tol=1e-05, atol=None, restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    if x0 is not None:
        x0 = x0.ravel()

    if callable(A_or_matvec):
        A = _matvec_to_scipy(A_or_matvec, b)
    else:
        A = numpy.array(ops.to_numpy(A_or_matvec))

    if scipy.__version__ < '1.12':
        u, info = scipy_gmres(A, b.ravel(), x0=x0, tol=tol, atol=atol,
                              restart=restart, maxiter=maxiter, M=M,
                              callback=callback, callback_type=callback_type)
    else:
        u, info = scipy_gmres(A, b.ravel(), x0=x0, rtol=tol, atol=atol,
                              restart=restart, maxiter=maxiter, M=M,
                              callback=callback, callback_type=callback_type)
    if info > 0:
        raise RuntimeError(f'scipy gmres failed to converge in {info} iterations.')
    elif info < 0:
        raise RuntimeError('scipy gmres failed.')
    return u.reshape(b.shape), info


def gmres_safe(A_or_matvec, b, x0=None, *,
               tol=1e-05, atol=None, restart=None, maxiter=None, M=None,
               callback=None, callback_type=None, cond=1e-6):
    if callable(A_or_matvec):
        A = _matvec_to_scipy(A_or_matvec, b)
    else:
        A = numpy.array(ops.to_numpy(A_or_matvec))

    k = 3
    v_null = None
    while True:
        if k > b.size:
            raise RuntimeError
        w, v = eigsh(A, k=k, which='SM')
        if numpy.all(abs(w) >= cond) and w[-1] > 0:
            break
        elif numpy.any(abs(w) < cond) and w[-1] >= cond:
            v_null = v[:,abs(w)<cond]
            break
        else:
            k += 3
            continue

    if v_null is None:
        return gmres(A_or_matvec, b, x0=x0,
                     tol=tol, atol=atol, restart=restart, maxiter=maxiter, M=M,
                     callback=callback, callback_type=callback_type)

    # pylint: disable=unnecessary-lambda-assignment
    _proj_to_null = lambda u: v_null @ (v_null.T @ u)
    _proj_to_range = lambda u: u - _proj_to_null(u)
    if callable(A_or_matvec):
        # pylint: disable=function-redefined
        def _matvec(u):
            u_range = _proj_to_range(u)
            Au = A_or_matvec(u_range.reshape(b.shape)).ravel()
            Au = numpy.array(ops.to_numpy(Au))
            return _proj_to_range(Au) + _proj_to_null(u)
        A = LinearOperator((b.size, b.size), matvec=_matvec)
    else:
        P_null = numpy.dot(v_null, v_null.T)
        P_range = numpy.eye(A_or_matvec.shape[0]) - P_null
        A = reduce(numpy.dot, (P_range, A_or_matvec, P_range)) + P_null

    if x0 is not None:
        x0 = x0.ravel()
    u, info = scipy_gmres(A, b.ravel(), x0=x0, tol=tol,
                          restart=restart, maxiter=maxiter, M=M, atol=atol,
                          callback=callback, callback_type=callback_type)

    if info > 0:
        raise RuntimeError(f'scipy gmres failed to converge in {info} iterations.')
    elif info < 0:
        raise RuntimeError('scipy gmres failed.')

    u = _proj_to_range(u)
    return u.reshape(b.shape), info
