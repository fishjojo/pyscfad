from functools import reduce
import numpy
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg import gmres as scipy_gmres

def gmres(A_or_matvec, b, x0=None, *,
          tol=1e-05, atol=None, restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    b_shape = b.shape
    if x0 is not None:
        x0 = x0.ravel()

    if callable(A_or_matvec):
        def _matvec(u):
            return A_or_matvec(u.reshape(b_shape)).ravel()
        A = LinearOperator((b.size, b.size), matvec=_matvec)
    else:
        A = A_or_matvec

    u, info = scipy_gmres(A, b.ravel(), x0=x0, tol=tol,
                          restart=restart, maxiter=maxiter, M=M, atol=atol,
                          callback=callback, callback_type=callback_type)
    if info > 0:
        raise RuntimeError(f'scipy gmres failed to converge in {info} iterations.')
    elif info < 0:
        raise RuntimeError('scipy gmres failed.')
    return u.reshape(b_shape), info


def gmres_safe(A_or_matvec, b, x0=None, *,
               tol=1e-05, atol=None, restart=None, maxiter=None, M=None,
               callback=None, callback_type=None, cond=1e-6):
    b_shape = b.shape

    if callable(A_or_matvec):
        def _matvec(u):
            return A_or_matvec(u.reshape(b_shape)).ravel()
        A = LinearOperator((b.size, b.size), matvec=_matvec)
    else:
        A = A_or_matvec

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
    _proj_to_null = lambda u: numpy.dot(v_null, numpy.dot(v_null.T, u))
    _proj_to_range = lambda u: u - _proj_to_null(u)
    if callable(A_or_matvec):
        # pylint: disable=function-redefined
        def _matvec(u):
            u_range = _proj_to_range(u)
            Au = A_or_matvec(u_range.reshape(b_shape)).ravel()
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
    return u.reshape(b_shape), info
