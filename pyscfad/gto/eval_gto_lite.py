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
Lightweight :mod:`~pyscfad.gto.eval_gto` module: AO values on grids for
:class:`~pyscfad.gto.MoleLite`, traceable and differentiable w.r.t. the
AO centers, the basis-set parameters and the grid coordinates.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial
from types import SimpleNamespace

import numpy
from jax.custom_derivatives import SymbolicZero
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_helper import (
    aoslices_in_range,
    resolve_bas_concrete,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfad.gto._basis_deriv import basis_jvp_cs_ao, basis_jvp_exp_ao
from pyscfad.gto.eval_gto import _XYZ_ID, _MAX_DERIV_ORDER

if TYPE_CHECKING:
    from pyscfad.typing import Array
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

def _deriv_order(eval_name: str) -> int:
    return int(eval_name.split("_deriv")[1]) if "_deriv" in eval_name else 0

def _bas_layout(eval_name, atm, bas, shls_slice, ao_loc, basis_array_metadata):
    """Static shell structure, shell range and AO offsets."""
    bas_or_meta = bas if basis_array_metadata is None else basis_array_metadata
    bas_conc = resolve_bas_concrete(bas_or_meta, len(atm))
    if shls_slice is None:
        shls_slice = (0, len(bas_conc))
    if ao_loc is None:
        ao_loc = make_loc(bas_conc, eval_name)
    return bas_conc, tuple(int(i) for i in shls_slice), ao_loc

def _eval_gto_numpy(eval_name, shls_slice, atm, bas, env, grid_coords):
    mol = SimpleNamespace(_atm=atm, _bas=bas, _env=env)
    return pyscf_eval_gto(mol, eval_name, grid_coords, shls_slice=shls_slice)

# ``env`` is a function of ``r0``, ``exp`` and ``ctr_coeff``, whose tangents
# carry its derivatives. It is not a nondiff argument nevertheless, as JAX
# rejects traced nondiff arguments under lax control flow, e.g., custom_root.
@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "eval_name",
        "atm",
        "bas",
        "shls_slice",
        "ao_loc",
        "basis_array_metadata",
    ),
)
def eval_gto(
    eval_name: str,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    r0: Array,
    exp: Array,
    ctr_coeff: Array,
    grid_coords: Array,
    shls_slice: tuple[int, int] | None = None,
    ao_loc: numpy.ndarray | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # for padding
) -> Array:
    """AO values on grids.

    Parameters:
        eval_name: ``GTOval_sph`` or ``GTOval_cart``, optionally with
            the suffix ``_deriv{n}`` for the AO derivatives up to order ``n``.
        grid_coords: Grid coordinates, ``(ngrids, 3)``.
        shls_slice: Start and end shells of the AOs.

    Returns:
        AO values of shape ``(ngrids, nao)``, or ``(comp, ngrids, nao)``
        with derivatives, laid out as :func:`pyscf.gto.eval_gto.eval_gto`.
    """
    _, shls_slice, ao_loc = _bas_layout(eval_name, atm, bas, shls_slice, ao_loc,
                                        basis_array_metadata)
    shape = (len(grid_coords), int(ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]))
    order = _deriv_order(eval_name)
    if order > 0:
        shape = ((order+1) * (order+2) * (order+3) // 6,) + shape

    return ops.pure_callback(
        ops.cached_partial(_eval_gto_numpy, eval_name, shls_slice),
        ops.ShapeDtypeStruct(shape, numpy.float64),
        atm, bas, env, grid_coords,
        vmap_method="sequential",
    )

def _eval_gto_jvp(
    eval_name, atm, bas, shls_slice, ao_loc, basis_array_metadata,
    primals, tangents,
):
    env, r0, exp, ctr_coeff, grid_coords = primals
    _, r0_dot, exp_dot, ctr_coeff_dot, grid_coords_dot = tangents
    bas_conc, shls_slice, ao_loc = _bas_layout(eval_name, atm, bas, shls_slice, ao_loc,
                                               basis_array_metadata)

    def ao_values(eval_name, bas=bas, env=env, ctr_coeff=ctr_coeff,
                  shls_slice=shls_slice, ao_loc=ao_loc,
                  basis_array_metadata=basis_array_metadata):
        return eval_gto(eval_name, atm, bas, env, r0, exp, ctr_coeff, grid_coords,
                        shls_slice=shls_slice, ao_loc=ao_loc,
                        basis_array_metadata=basis_array_metadata)

    ao = ao_values(eval_name)
    jvp = np.zeros_like(ao)

    move_r0 = not isinstance(r0_dot, SymbolicZero)
    move_grids = not isinstance(grid_coords_dot, SymbolicZero)
    if move_r0 or move_grids:
        order = _deriv_order(eval_name)
        if order >= _MAX_DERIV_ORDER:
            raise NotImplementedError(f"Derivatives of {eval_name}")
        # gradients of all the components, (3,) + ao.shape
        ao1 = ao_values(eval_name.split("_deriv")[0] + f"_deriv{order+1}")
        dao = ao1[numpy.hstack(_XYZ_ID[:order+1])].reshape((3,) + ao.shape)

        if move_grids:
            jvp += np.einsum("x...gi,gx->...gi", dao, grid_coords_dot)
        if move_r0:
            # an AO moves with its center
            aoslices = aoslices_in_range(bas_conc, ao_loc, len(atm), shls_slice)
            aoidx = np.arange(ao.shape[-1]).reshape((1,) * ao.ndim + (-1,))
            jvp -= _gen_int1e_fill_jvp_r0(dao, r0_dot, aoslices, aoidx)

    cart = "_cart" in eval_name
    if not isinstance(ctr_coeff_dot, SymbolicZero):
        jvp += basis_jvp_cs_ao(
            partial(ao_values, eval_name), atm, bas, env,
            ctr_coeff, ctr_coeff_dot, cart, shls_slice, basis_array_metadata)
    if not isinstance(exp_dot, SymbolicZero):
        jvp += basis_jvp_exp_ao(
            partial(ao_values, eval_name.replace("_sph", "_cart")), atm, bas, env,
            exp, ctr_coeff, exp_dot, cart, shls_slice, basis_array_metadata)
    return ao, jvp

eval_gto.defjvp(_eval_gto_jvp, symbolic_zeros=True)
