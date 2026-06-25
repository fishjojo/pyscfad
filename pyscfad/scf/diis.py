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
"""
DIIS
"""
from typing import NamedTuple, Any
import jax
from jax import tree
from pyscf.scf import diis as pyscf_cdiis
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit, vmap
from pyscfad import lib
from pyscfad import pytree
from pyscfad.lib import logger
from pyscfad.scf.anderson import (
    RIDGE_TOL,
    minimize_residual,
    update_history,
    _tree_scale_sum,
    _tree_sub,
)

Array = Any

class CDIIS(lib.diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename, Corth=Corth)
        self.incore = True

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        # no need to trace error vectors
        errvec = ops.stop_grad(errvec)
        logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(errvec))
        f_prev = kwargs.get('f_prev', None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        else:
            f = f*(1-self.damp) + f_prev*self.damp
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

@jit
def get_err_vec_orig(s, d, f):
    def _get_errvec(s, d, f):
        sdf = s @ d @ f
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature='(i,j),(i,j),(i,j)->(k)')(s, d, f)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0),
                      signature='(i,j),(i,j)->(k)')(s, d, f)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

@jit
def get_err_vec_orth(s, d, f, Corth):
    def _get_errvec(s, d, f, c):
        sdf = c.conj().T @ s @ d @ f @ c
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f, Corth)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature='(i,j),(i,j),(i,j),(i,j)->(k)')(s, d, f, Corth)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0,0),
                      signature='(i,j),(i,j),(i,j)->(k)')(s, d, f, Corth)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

SCFDIIS = SCF_DIIS = CDIIS


class DIISState(NamedTuple):
    cycle: int
    param_hist: Any
    res_hist: Any
    res_gram: Array


class DIIS(pytree.PytreeNode):
    r"""Jittable Pulay DIIS.

    Shares the constrained least-squares machinery of
    :class:`~pyscfad.scf.anderson.Anderson` (a Gram matrix of error vectors and
    the Lagrange-constrained minimisation :func:`~pyscfad.scf.anderson.minimize_residual`),
    but extrapolates the *pure* DIIS combination :math:`\sum_i c_i p_i` of the
    stored Fock matrices rather than the Anderson update. The error vector for
    each stored Fock is the change in the iterate, ``param - param_last``, so no
    overlap/density-matrix commutator is required and the whole update is
    traceable under ``jax.jit``.

    Parameters
    ----------
    param: parameters to be mixed (e.g. the packed Fock matrix).
    space: size of the DIIS subspace.
    ridge: ridge regularisation added to the Gram matrix for stability.
    start_cycle: first cycle (0-based) at which extrapolation is applied.
    """
    _dynamic_attr = ["state"]

    def __init__(
        self,
        param: Any,
        space: int = 8,
        ridge: float = RIDGE_TOL,
        start_cycle: int = 1,
    ):
        self.space = space
        self.ridge = ridge
        self.start_cycle = start_cycle
        self.state = self.init_state(param)

    def init_state(
        self,
        param: Any,
    ) -> NamedTuple:
        m = self.space
        param_hist = tree.map(lambda x: np.tile(x, [m]+[1]*x.ndim), param)
        res_hist = tree.map(np.zeros_like, param_hist)
        res_gram = np.zeros((m, m), dtype=np.floatx)
        return DIISState(
            cycle=np.asarray(0, dtype=int),
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )

    def update(
        self,
        param: Any,
        param_last: Any,
    ) -> Any:
        state = self.state
        cycle = state.cycle

        pos = np.mod(cycle, self.space)
        residual = _tree_sub(param, param_last)
        param_hist, res_hist, res_gram = \
            update_history(pos, state.param_hist, state.res_hist,
                           state.res_gram, param, residual)

        next_state = DIISState(
            cycle=cycle+1,
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )

        def _diis_step(param, state):
            alpha = minimize_residual(state.res_gram, self.ridge)
            return _tree_scale_sum(alpha[1:], state.param_hist)

        def _use_param(param, state):
            return param

        start_cycle = jax.lax.select(
            np.greater_equal(self.start_cycle+1, self.space),
            self.start_cycle+1,
            self.space,
        )
        extrapolated = jax.lax.cond(
            np.greater_equal(cycle+1, start_cycle),
            _diis_step,
            _use_param,
            param,
            next_state,
        )

        self.state = next_state
        return extrapolated
