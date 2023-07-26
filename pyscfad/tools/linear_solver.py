from functools import partial
import numpy
from scipy.sparse.linalg import LinearOperator
from jaxopt.linear_solve import solve_gmres as jaxopt_gmres
from pyscfad.scipy.sparse.linalg import gmres as pyscfad_gmres

def precond_by_hdiag(h_diag, thresh=1e-12):
    h_diag = numpy.array(h_diag, dtype=float).ravel()
    h_diag[abs(h_diag) < thresh] = 1.
    n = h_diag.size
    M = LinearOperator((n, n), lambda x: x / h_diag)
    return M

def gen_gmres_with_default_kwargs(tol=1e-5, atol=1e-12, maxiter=30):
    from pyscfad import config
    if config.moleintor_opt:
        gmres = partial(pyscfad_gmres,
                        tol=tol, atol=atol, maxiter=maxiter)
    else:
        gmres = partial(jaxopt_gmres,
                        tol=tol, atol=atol, maxiter=maxiter,
                        solve_method='incremental')
    return gmres

gen_gmres = gen_gmres_with_default_kwargs
