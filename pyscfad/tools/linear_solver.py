from functools import partial
import numpy
from scipy.sparse.linalg import LinearOperator
from jax.scipy.sparse.linalg import gmres as jax_gmres
from pyscfad.scipy.sparse.linalg import gmres as pyscfad_gmres
from pyscfad.scipy.sparse.linalg import gmres_safe

class GMRESDisp:
    def __init__(self, disp=False):
        self.disp = disp
        self._iter = 0

    @property
    def iter(self):
        return self._iter

    def __call__(self, pr_norm):
        self._iter += 1
        if self.disp:
            print(f'gmres cycle {self.iter}: residual norm = {pr_norm}.')

def precond_by_hdiag(h_diag, thresh=1e-12):
    h_diag = numpy.array(h_diag, dtype=float).ravel()
    h_diag[abs(h_diag) < thresh] = 1.
    n = h_diag.size
    M = LinearOperator((n, n), lambda x: x / h_diag)
    return M

def gen_gmres_with_default_kwargs(tol=1e-6, atol=1e-12, maxiter=30,
                                  x0=None, M=None, restart=20,
                                  callback=None, callback_type=None,
                                  safe=False, **kwargs):
    from pyscfad import config
    if config.moleintor_opt:
        if safe:
            gmres = partial(gmres_safe,
                            tol=tol, atol=atol, maxiter=maxiter,
                            x0=x0, M=M, restart=restart,
                            callback=callback, callback_type=callback_type,
                            **kwargs)
        else:
            gmres = partial(pyscfad_gmres,
                            tol=tol, atol=atol, maxiter=maxiter,
                            x0=x0, M=M, restart=restart,
                            callback=callback, callback_type=callback_type)
    else:
        gmres = partial(jax_gmres,
                        tol=tol, atol=atol, maxiter=maxiter,
                        solve_method='incremental',
                        x0=x0, M=M, restart=restart)
    return gmres

gen_gmres = gen_gmres_with_default_kwargs
