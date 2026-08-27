# Copyright 2026 The PySCFAD Authors
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
Jittable :mod:`~pyscfad.lib.diis` module.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import operator

from jax import tree, vmap

from pyscfad import numpy as np
from pyscfad import pytree

if TYPE_CHECKING:
    from typing import Any
    from pyscfad.typing import Array

# Singular values below this fraction of the largest one are dropped from the
# extrapolation, mirroring the eigenvalue screening of
# :meth:`pyscfad.lib.diis.DIIS.extrapolate`.
RCOND = 1e-14

def _tree_sub(xs, ys):
    return tree.map(operator.sub, xs, ys)

def _tree_set(xs, idx, vals):
    return tree.map(lambda x, val: x.at[idx].set(val), xs, vals)

def _tree_vdot(xs, ys):
    vdots = tree.map(np.vdot, xs, ys)
    return tree.reduce(operator.add, vdots)

def _tree_scale_sum(a, xs):
    return tree.map(lambda x: np.tensordot(a, x, axes=1), xs)

def solve_coefficients(
    ovlp: Array,
    mask: Array,
) -> Array:
    r"""Coefficients of the DIIS extrapolation.

    The extrapolated vector :math:`\sum_i c_i x_i` minimizes the norm of
    :math:`\sum_i c_i e_i` subject to :math:`\sum_i c_i = 1`, whose stationary
    conditions are the bordered linear system

    .. math::
        \begin{pmatrix} 0 & 1^T \\ 1 & S \end{pmatrix}
        \begin{pmatrix} \lambda \\ c \end{pmatrix} =
        \begin{pmatrix} 1 \\ 0 \end{pmatrix},
        \qquad S_{ij} = \langle e_i, e_j \rangle .

    Parameters:
        ovlp: Gram matrix :math:`S` of the error vectors, of the full
            subspace size.
        mask: Flags the slots of ``ovlp`` that hold an error vector.

    Returns:
        Coefficients over the full subspace, zero on the unused slots.

    Notes:
        The system is kept at its full (static) size, which a masked-out slot
        ``i`` joins as the trivial equation :math:`c_i = 0`: its row of ``S``
        is replaced by that of the identity and its entry in the constraint
        row by zero. A single error vector therefore gives :math:`c = (1)`,
        i.e. no extrapolation, so no separate minimum-subspace branch is
        needed.

        :math:`S` is normalized by its largest diagonal element first. That
        leaves :math:`c` unchanged -- it only rescales :math:`\lambda` -- but
        makes the least-squares cutoff relative, which matters because the
        error vectors go to zero as the iterations converge.
    """
    nvec = ovlp.shape[0]
    dtype = ovlp.dtype
    eye = np.eye(nvec, dtype=dtype)

    scale = np.max(np.where(mask, np.diagonal(ovlp).real, 0.))
    scale = np.where(scale > 0, scale, 1.).astype(ovlp.real.dtype)
    valid = mask[:,None] & mask[None,:]
    s = np.where(valid, ovlp / scale, eye)

    row = mask.astype(dtype)
    h = np.block([[np.zeros((1, 1), dtype=dtype), row[None,:]],
                  [row[:,None],                   s          ]])
    g = np.zeros(nvec+1, dtype=dtype).at[0].set(1)
    c = np.linalg.lstsq(h, g, rcond=RCOND)[0]
    return c[1:]

class DIISState(NamedTuple):
    cycle: Array
    x_hist: Any
    err_hist: Any
    ovlp: Array
    x_prev: Any

class DIISLite(pytree.PytreeNode):
    r"""Pulay's DIIS (direct inversion in the iterative subspace).

    The jittable counterpart of :class:`pyscfad.lib.diis.DIIS`: the history is
    a fixed-size circular buffer held in the (pytree) state instead of a
    dictionary of buffers, and the subspace equations are solved at a static
    size, so an instance can be carried through :func:`jax.lax.while_loop`.

    Parameters:
        x: Vector to be extrapolated, used to size the history. An arbitrary
            pytree of arrays, e.g. the ``(t1, t2)`` coupled-cluster
            amplitudes.
        space: Size of the subspace.

    Notes:
        ``x`` also seeds the previous-vector slot, so the first
        :meth:`update` already has an error vector -- unlike
        :class:`pyscfad.lib.diis.DIIS`, whose first call only records ``x``.
        Both start extrapolating from more than one vector, one iteration
        apart.
    """
    _dynamic_attr = ["state"]

    def __init__(
        self,
        x: Any,
        space: int = 6,
    ):
        self.space = space
        self.state = self.init_state(x)

    def init_state(
        self,
        x: Any,
    ) -> DIISState:
        m = self.space
        empty = lambda a: np.zeros((m,)+a.shape, dtype=a.dtype)
        dtype = np.result_type(*[a.dtype for a in tree.leaves(x)])
        return DIISState(
            cycle=np.asarray(0, dtype=int),
            x_hist=tree.map(empty, x),
            err_hist=tree.map(empty, x),
            ovlp=np.zeros((m, m), dtype=dtype),
            x_prev=x,
        )

    def update(
        self,
        x: Any,
        xerr: Any | None = None,
    ) -> Any:
        """Extrapolate the vector.

        Parameters:
            x: New vector.
            xerr: Error vector. Defaults to the difference between ``x`` and
                the vector returned by the previous :meth:`update`.

        Returns:
            The extrapolated vector.

        Notes:
            This function has a side effect on ``self.state``.
        """
        state = self.state
        cycle = state.cycle
        m = self.space
        pos = np.mod(cycle, m)

        if xerr is None:
            xerr = _tree_sub(x, state.x_prev)

        x_hist = _tree_set(state.x_hist, pos, x)
        err_hist = _tree_set(state.err_hist, pos, xerr)

        # ovlp[i,pos] = <e_i, e_pos>, and the Gram matrix is Hermitian
        col = vmap(_tree_vdot, in_axes=(0, None))(err_hist, xerr)
        ovlp = state.ovlp.at[:,pos].set(col).at[pos,:].set(col.conj())

        mask = np.arange(m) < np.minimum(cycle+1, m)
        c = solve_coefficients(ovlp, mask)
        x_new = _tree_scale_sum(c, x_hist)

        self.state = DIISState(
            cycle=cycle+1,
            x_hist=x_hist,
            err_hist=err_hist,
            ovlp=ovlp,
            x_prev=x_new,
        )
        return x_new
