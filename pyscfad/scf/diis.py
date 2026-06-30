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
    minimize_residual,
    RIDGE_TOL,
    _tree_set,
    _tree_vdot,
    _tree_scale_sum,
)

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

# Legacy (non-jittable) PySCF-based DIIS, used by ``pyscfad.scf.hf``.
SCFDIIS = SCF_DIIS = CDIIS


class DIISState(NamedTuple):
    cycle: int
    fock_hist: Any
    err_hist: Any
    err_gram: Any


class DIIS(pytree.PytreeNode):
    r"""Jittable Pulay (commutator) DIIS.

    Mirrors :class:`pyscfad.scf.anderson.Anderson`: the mixer state is carried
    as array-only data through ``jax.lax`` control flow so the SCF iteration can
    be jitted and vmapped. The error vector is the (optionally orthonormalised)
    commutator :math:`SDF - FDS` (see :func:`get_err_vec`), and the extrapolated
    Fock matrix is :math:`\sum_i c_i F_i`, where the Pulay coefficients ``c``
    solve the standard DIIS linear system via the same
    :func:`~pyscfad.scf.anderson.minimize_residual` used by Anderson mixing.

    Parameters
    ----------
    fock: initial Fock matrix (``(nao, nao)`` or ``(2, nao, nao)`` for UHF).
    err: initial error vector (matching :func:`get_err_vec`).
    space: size of the DIIS subspace.
    ridge: ridge regularization for the (possibly singular) linear system.
    start_cycle: starting cycle for extrapolation.
    """
    _dynamic_attr = ["state"]

    def __init__(
        self,
        fock: Any,
        err: Any,
        space: int = 8,
        ridge: float = RIDGE_TOL,
        start_cycle: int = 1,
    ):
        self.space = space
        self.ridge = ridge
        self.start_cycle = start_cycle
        self.state = self.init_state(fock, err)

    def init_state(self, fock: Any, err: Any) -> NamedTuple:
        m = self.space
        fock_hist = tree.map(lambda x: np.tile(x, [m] + [1] * x.ndim), fock)
        err_hist = tree.map(lambda x: np.zeros((m,) + x.shape, dtype=x.dtype), err)
        err_gram = np.zeros((m, m), dtype=np.floatx)
        return DIISState(
            cycle=np.asarray(0, dtype=int),
            fock_hist=fock_hist,
            err_hist=err_hist,
            err_gram=err_gram,
        )

    def update(self, fock: Any, err: Any) -> Any:
        state = self.state
        cycle = state.cycle
        pos = np.mod(cycle, self.space)

        fock_hist = _tree_set(state.fock_hist, pos, fock)
        err_hist = _tree_set(state.err_hist, pos, err)
        new_row = jax.vmap(_tree_vdot, in_axes=(0, None))(err_hist, err)
        err_gram = state.err_gram.at[pos, :].set(new_row).at[:, pos].set(new_row)

        next_state = DIISState(
            cycle=cycle + 1,
            fock_hist=fock_hist,
            err_hist=err_hist,
            err_gram=err_gram,
        )

        def _extrapolate(fock, st):
            alpha = minimize_residual(st.err_gram, self.ridge)
            return _tree_scale_sum(alpha[1:], st.fock_hist)

        def _keep(fock, st):
            return fock

        start_cycle = jax.lax.select(
            np.greater_equal(self.start_cycle + 1, self.space),
            self.start_cycle + 1,
            self.space,
        )
        extrapolated = jax.lax.cond(
            np.greater_equal(cycle + 1, start_cycle),
            _extrapolate,
            _keep,
            fock,
            next_state,
        )

        self.state = next_state
        return extrapolated
