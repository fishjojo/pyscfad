from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres

def gmres(A_or_matvec, b, x0=None, *,
          tol=1e-05, atol=None, restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    b_shape = b.shape
    if x0 is not None:
        x0 = x0.flatten()

    if callable(A_or_matvec):
        def _matvec(u):
            return A_or_matvec(u.reshape(b_shape)).flatten()
        A = LinearOperator((b.size, b.size), matvec=_matvec)
    else:
        A = A_or_matvec

    u, info = scipy_gmres(A, b.flatten(), x0=x0, tol=tol,
                          restart=restart, maxiter=maxiter, M=M, atol=atol,
                          callback=callback, callback_type=callback_type)
    if info > 0:
        raise RuntimeError(f'scipy gmres failed to converge in {info} iterations.')
    elif info < 0:
        raise RuntimeError('scipy gmres failed.')
    return u.reshape(b_shape), info
