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
Lattice GTO integrals using the cuint backend.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from functools import partial
import numpy
import jax
from jax.custom_derivatives import SymbolicZero

from pyscf.gto.mole import ATM_SLOTS, BAS_SLOTS

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.moleintor_lite import (
    _aoslice_by_atom,
    _extract_coords,
)
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from .moleintor_cuint import CuintPlan

@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "Ls_mask",
        "atm",
        "bas",
        "cuint_plan",
        "shls_slice",
        "comp",
        "hermi",
        "ao_loc",
        "trace_coords",
        "trace_basis",
        "aoslices",
    ),
)
def _lattice_intor(
    intor_name: str,
    Ls: ArrayLike,
    Ls_mask: ArrayLike,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    cuint_plan: CuintPlan,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    ao_loc: ArrayLike | None = None,
    trace_coords: bool = False,
    trace_basis: bool = False,
    aoslices: ArrayLike | None = None, # for padding
) -> Array:
    bas = np.asarray(bas).reshape(-1,BAS_SLOTS)
    nbas = bas.shape[0]
    if shls_slice is not None and tuple(shls_slice)[:4] != (0, nbas, 0,  nbas):
        raise NotImplementedError(
            "Computing subblocks of integrals is not supported."
        )
    del bas

    if hermi != 1:
        raise NotImplementedError(
            f"Only hermi=1 is supported, but got hermi={hermi}."
        )

    if intor_name == "int1e_ovlp_sph":
        out = lat_overlap(atm, env, Ls, Ls_mask, cuint_plan)
    else:
        raise NotImplementedError(
            "Integral {intor_name} is not supported."
        )
    return out

def lat_overlap(
    atm: ArrayLike,
    env: ArrayLike,
    Ls: ArrayLike,
    Ls_mask: ArrayLike,
    cuint_plan: CuintPlan,
    deriv: int = 0,
) -> Array:
    atm = np.asarray(atm, dtype=np.int32)
    env = np.asarray(env, dtype=np.float64)
    Ls = np.asarray(Ls, dtype=np.float64).reshape(-1, 3)
    Ls_mask = np.asarray(Ls_mask, dtype=np.int32)

    n_functions = cuint_plan.n_functions
    nL = Ls.shape[0]
    if deriv == 0:
        shape = (nL, n_functions, n_functions)
        target = "cuint_lat_overlap_ffi"
    elif deriv == 1:
        shape = (nL, 3, n_functions, n_functions)
        target = "cuint_lat_overlap_gradient_ffi"
    else:
        raise NotImplementedError
    dtype = np.float64

    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(shape, dtype),
        # FIXME broadcast_all will tile non-mapped inputs to batched shapes,
        # which is needed for out, but a waste for others,
        # esp. for primitive_to_function.
        vmap_method="broadcast_all",
        input_output_aliases={0:0},
    )

    out = np.zeros(shape, dtype)

    is_screened = cuint_plan.is_screened
    n_primitives = cuint_plan.n_primitives
    bas = cuint_plan.bas
    primitive_to_function = cuint_plan.primitive_to_function

    for pair in cuint_plan.pairs:
        out = call(
            out, pair.pair_indices, primitive_to_function,
            atm, bas, env, Ls, Ls_mask,
            i_angular=pair.li,
            j_angular=pair.lj,
            is_screened=is_screened,
            n_pairs=pair.n_pairs,
            n_primitives=n_primitives,
            n_functions=n_functions,
            atm_stride=numpy.int32(atm.size),
            bas_stride=numpy.int32(bas.size),
            env_stride=numpy.int32(env.size),
            n_images=numpy.int32(nL),
            reduce_over_images=numpy.int32(0),
        )
    return out

def _lattice_intor_jvp(
    intor_name, Ls_mask, atm, bas, cuint_plan,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices,
    primals, tangents,
):
    if not intor_name == "int1e_ovlp_sph":
        raise NotImplementedError
    assert hermi == 1

    Ls, env = primals
    Ls_dot, env_dot = tangents

    primal_out = _lattice_intor(
        intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis, aoslices=aoslices,
    )

    tangent_out = np.zeros_like(primal_out)

    if not isinstance(env_dot, SymbolicZero):
        if trace_coords:
            s1a = -lat_overlap(atm, env, Ls, Ls_mask, cuint_plan, deriv=1)
            s1a = s1a.transpose(1,0,2,3)

            env_dot = np.asarray(env_dot, dtype=np.float64)
            coords_dot = _extract_coords(atm, env_dot)

            if shls_slice is None:
                nbas = len(bas)
                shls_slice = (0, nbas, 0, nbas)
            if ao_loc is None:
                _ao_loc = make_loc(bas, intor_name)
            else:
                _ao_loc = ao_loc

            i0, _, j0, _ = shls_slice[:4]
            if aoslices is None:
                aoslices = _aoslice_by_atom(atm, bas, _ao_loc)

            naoi, naoj = s1a.shape[-2:]

            aoidx = np.arange(naoi)
            jvp = _gen_int1e_fill_jvp_r0(s1a, coords_dot, aoslices-_ao_loc[i0], aoidx[None,None,:,None])

            aoidx = np.arange(naoj)
            jvp += _gen_int1e_fill_jvp_r0(-s1a, coords_dot, aoslices-_ao_loc[j0], aoidx[None,None,None,:])

            tangent_out += jvp.reshape(tangent_out.shape)

        if trace_basis:
            raise NotImplementedError("basis set parameter derivative not supported")

    if not isinstance(Ls_dot, SymbolicZero):
        raise NotImplementedError
    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
