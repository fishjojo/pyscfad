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
from pyscf.scf import diis as pyscf_cdiis
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit, vmap
from pyscfad import lib
from pyscfad.lib import logger

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

SCFDIIS = SCF_DIIS = DIIS = CDIIS


# ---------------------------------------------------------------------------
# Jittable Pulay (commutator) DIIS, structured like ``scf/anderson.py`` so it
# can live inside a ``jax.lax.while_loop`` SCF as a pytree.
# ---------------------------------------------------------------------------
import jax
from jax import tree
from pyscfad import pytree
from pyscfad.scf.anderson import (
    AndersonState,
    RIDGE_TOL,
    minimize_residual,
    update_history,
    _tree_scale_sum,
)

def get_diis_errvec(s, d, f):
    r"""DIIS commutator error vector :math:`e = SDF - (SDF)^\dagger`.

    Broadcasts over a leading spin axis, so it handles both the RHF
    ``(nao, nao)`` and UHF ``(2, nao, nao)`` Fock/density layouts.
    """
    sdf = np.matmul(np.matmul(s, d), f)
    return sdf - np.swapaxes(sdf.conj(), -1, -2)

class Pulay(pytree.PytreeNode):
    r"""Pulay's DIIS using the commutator error vector.

    Mirrors :class:`~pyscfad.scf.anderson.Anderson`: the error-vector Gram
    matrix is accumulated in a fixed-size circular buffer and the DIIS
    equations are solved by :func:`~pyscfad.scf.anderson.minimize_residual`.
    Unlike Anderson, the residual is the externally supplied commutator (not
    ``param - param_last``) and the extrapolation is the pure DIIS combination
    :math:`F = \sum_i c_i F_i` (no residual mixing).

    Parameters
    ----------
    param: a Fock matrix (template defining the stored shape/dtype).
    space: size of the DIIS subspace.
    ridge: ridge regularization for the (near-singular) DIIS matrix.
    start_cycle: starting cycle for extrapolation.
    """
    _dynamic_attr = ["state"]

    def __init__(
        self,
        param,
        space=8,
        ridge=RIDGE_TOL,
        start_cycle=1,
    ):
        self.space = space
        self.ridge = ridge
        self.start_cycle = start_cycle
        self.state = self.init_state(param)

    def init_state(self, param):
        m = self.space
        param_hist = tree.map(lambda x: np.tile(x, [m]+[1]*x.ndim), param)
        res_hist = tree.map(np.zeros_like, param_hist)
        res_gram = np.zeros((m, m), dtype=np.floatx)
        return AndersonState(
            cycle=np.asarray(0, dtype=int),
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )

    def update(self, param, residual):
        state = self.state
        cycle = state.cycle
        pos = np.mod(cycle, self.space)
        param_hist, res_hist, res_gram = update_history(
            pos, state.param_hist, state.res_hist, state.res_gram, param, residual)
        next_state = AndersonState(
            cycle=cycle+1,
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )

        def _diis_step(param, st):
            alpha = minimize_residual(st.res_gram, self.ridge)
            return _tree_scale_sum(alpha[1:], st.param_hist)

        def _use_param(param, st):
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
