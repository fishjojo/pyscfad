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
GTO integrals using the cuint backend.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence
import importlib
from dataclasses import dataclass, field, replace
from functools import partial
import numpy
import jax
from jax.custom_derivatives import SymbolicZero
from pyscf.gto.mole import (
    NPRIM_OF,
    NCTR_OF,
    ANG_OF,
    PTR_EXP,
    PTR_COEFF,
    PTR_COMMON_ORIG,
)
from pyscfad import numpy as np
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_helper import (
#    int1e_get_dr_order,
    int1e_dr1_name,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfad.gto.moleintor_lite import (
    _aoslice_by_atom,
    _extract_coords,
)

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite

_cuint = None
try:
    _cuint = importlib.import_module(
            "._cuint", package="pyscfad_cuda12_plugin",
    )
except ImportError as e:
    raise ImportError(
        "Failed to import '_cuint' from 'pyscfad_cuda12_plugin'. "
    ) from e

if _cuint:
    for _name, _value in _cuint.registrations().items():
        jax.ffi.register_ffi_target(
            _name,
            _value,
            platform="CUDA",
            api_version=1,
        )

@partial(
    jax.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "atm",
        "bas",
        "cuint_plan",
        "shls_slice",
        "comp",
        "hermi",
        "aosym",
        "ao_loc",
        "trace_coords",
        "trace_basis",
        "aoslices",
    ),
)
def getints(
    intor_name: str,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    cuint_plan: CuintPlan,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    aosym: str = "s1",
    ao_loc: ArrayLike | None = None,
    trace_coords: bool = False,
    trace_basis: bool = False,
    aoslices: ArrayLike | None = None, # for padding
) -> Array:
    nbas = len(bas)
    if shls_slice is not None and tuple(shls_slice)[:4] != (0, nbas, 0,  nbas):
        raise NotImplementedError(
            "Computing subblocks of integrals is not supported."
        )
    if hermi != 1:
        raise NotImplementedError(
            f"Only hermi=1 is supported, but got hermi={hermi}."
        )

    atm = np.asarray(atm, dtype=np.int32)
    env = np.asarray(env, dtype=np.float64)
    if intor_name == "int1e_ovlp_sph":
        out = overlap(atm, env, cuint_plan)
    elif intor_name in ("int1e_ovlp_dr10_sph", "int1e_ipovlp_sph"):
        out = overlap(atm, env, cuint_plan, deriv=1)
    elif intor_name == "int1e_r_sph":
        out = dipole(atm, env, cuint_plan)
    elif intor_name == "int1e_r_dr10_sph":
        out = dipole(atm, env, cuint_plan, deriv=1)
    elif intor_name == "int1e_rr_sph":
        out = quadrupole(atm, env, cuint_plan)
    elif intor_name == "int1e_rr_dr10_sph":
        out = quadrupole(atm, env, cuint_plan, deriv=1)
    else:
        raise NotImplementedError(
            "Integral {intor_name} is not supported."
        )
    return out

def getints_jvp(
    intor_name,
    atm,
    bas,
    cuint_plan,
    shls_slice,
    comp,
    hermi,
    aosym,
    ao_loc,
    trace_coords,
    trace_basis,
    aoslices,
    primals,
    tangents,
):
    env, = primals
    env_dot, = tangents
    primal_out = getints(
        intor_name,
        atm, bas, env,
        cuint_plan,
        shls_slice=shls_slice, comp=comp,
        hermi=hermi, aosym=aosym, ao_loc=ao_loc,
        trace_coords=trace_coords,
        trace_basis=trace_basis,
        aoslices=aoslices,
    )

    tangent_out = np.zeros_like(primal_out)
    intor_ip_bra = intor_ip_ket = None
    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)

    if not isinstance(env_dot, SymbolicZero):
        if trace_coords and (intor_ip_bra or intor_ip_ket):
            if intor_name.startswith("int1e_r"):
                rc_deriv = PTR_COMMON_ORIG
            else:
                rc_deriv = None

            tangent_out += _gen_int1e_jvp_r0(
                intor_ip_bra, intor_ip_ket,
                atm, bas, env, env_dot,
                cuint_plan,
                shls_slice, comp, hermi, aosym, ao_loc,
                trace_coords, trace_basis,
                aoslices, rc_deriv,
            ).reshape(tangent_out.shape)

        if trace_basis:
            raise NotImplementedError("basis set parameter derivative not supported")
    return primal_out, tangent_out

getints.defjvp(getints_jvp, symbolic_zeros=True)

def _gen_int1e_jvp_r0(
    intor_a, intor_b,
    atm, bas, env, env_dot,
    cuint_plan,
    shls_slice, comp, hermi, aosym, ao_loc,
    trace_coords, trace_basis,
    aoslices, rc_deriv,
):
    if comp is not None:
        comp = comp * 3

    coords_dot = _extract_coords(atm, env_dot)

    if shls_slice is None:
        nbas = len(bas)
        shls_slice = (0, nbas, 0, nbas)
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_a) if intor_a else make_loc(bas, intor_b)
    else:
        _ao_loc = ao_loc

    i0, _, j0, _ = shls_slice[:4]
    if aoslices is None:
        aoslices = _aoslice_by_atom(atm, bas, _ao_loc)

    if intor_a:
        s1a = -getints(
            intor_a,
            atm, bas, env,
            cuint_plan,
            shls_slice=shls_slice, comp=comp,
            hermi=hermi, aosym=aosym, ao_loc=ao_loc,
            trace_coords=trace_coords, trace_basis=trace_basis,
            aoslices=aoslices,
        )

        naoi, naoj = s1a.shape[-2:]
        s1a = s1a.reshape(3,-1,naoi,naoj)

        aoidx = np.arange(naoi)
        jvp = _gen_int1e_fill_jvp_r0(s1a, coords_dot, aoslices-_ao_loc[i0], aoidx[None,None,:,None])

        if isinstance(rc_deriv, int):
            R0_dot = env_dot[rc_deriv:rc_deriv+3]
            jvp -= np.einsum("xyij,x->yij", s1a, R0_dot)

        if hermi == 1:
            jvp += jvp.transpose(0,2,1)

    elif intor_b:
        raise NotImplementedError

    return jvp

def overlap(atm: Array, env: Array, cuint_plan: CuintPlan, deriv: int = 0) -> Array:
    n_functions = cuint_plan.n_functions
    if deriv == 0:
        shape = (n_functions, n_functions)
        target = "cuint_overlap_ffi"
    elif deriv == 1:
        shape = (3, n_functions, n_functions)
        target = "cuint_overlap_gradient_ffi"
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
            atm, bas, env,
            i_angular=pair.li,
            j_angular=pair.lj,
            is_screened=is_screened,
            n_pairs=pair.n_pairs,
            n_primitives=n_primitives,
            n_functions=n_functions,
            atm_stride=numpy.int32(atm.size),
            bas_stride=numpy.int32(bas.size),
            env_stride=numpy.int32(env.size),
        )
    if deriv == 0:
        out += out.T
    elif deriv == 1:
        out -= out.transpose(0, -1, -2)
    return out

def dipole(atm: Array, env: Array, cuint_plan: CuintPlan, deriv: int = 0) -> Array:
    dtype = np.float64
    n_functions = cuint_plan.n_functions
    if deriv == 0:
        shape = (3, n_functions, n_functions)
        target = "cuint_dipole_ffi"
    elif deriv == 1:
        shape = (9, n_functions, n_functions)
        target = "cuint_dipole_gradient_ffi"

        call_ovlp = jax.ffi.ffi_call(
            "cuint_overlap_ffi",
            jax.ShapeDtypeStruct((n_functions, n_functions), dtype),
            vmap_method="broadcast_all",
            input_output_aliases={0:0},
        )
    else:
        raise NotImplementedError

    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(shape, dtype),
        vmap_method="broadcast_all",
        input_output_aliases={0:0},
    )

    out = np.zeros(shape, dtype)
    if deriv == 1:
        ovlp = np.zeros((n_functions, n_functions), dtype)

    is_screened = cuint_plan.is_screened
    n_primitives = cuint_plan.n_primitives
    bas = cuint_plan.bas
    primitive_to_function = cuint_plan.primitive_to_function

    for pair in cuint_plan.pairs:
        out = call(
            out, pair.pair_indices, primitive_to_function,
            atm, bas, env,
            i_angular=pair.li,
            j_angular=pair.lj,
            is_screened=is_screened,
            n_pairs=pair.n_pairs,
            n_primitives=n_primitives,
            n_functions=n_functions,
            atm_stride=numpy.int32(atm.size),
            bas_stride=numpy.int32(bas.size),
            env_stride=numpy.int32(env.size),
        )

        if deriv == 1:
            ovlp = call_ovlp(
                ovlp, pair.pair_indices, primitive_to_function,
                atm, bas, env,
                i_angular=pair.li,
                j_angular=pair.lj,
                is_screened=is_screened,
                n_pairs=pair.n_pairs,
                n_primitives=n_primitives,
                n_functions=n_functions,
                atm_stride=numpy.int32(atm.size),
                bas_stride=numpy.int32(bas.size),
                env_stride=numpy.int32(env.size),
            )

    if deriv == 0:
        out += out.transpose(0, 2, 1)
    elif deriv == 1:
        out -= out.transpose(0, 2, 1)
        out = -out
        out = out.at[0].subtract(ovlp.T)
        out = out.at[4].subtract(ovlp.T)
        out = out.at[8].subtract(ovlp.T)
    return out

def quadrupole(atm: Array, env: Array, cuint_plan: CuintPlan, deriv: int = 0) -> Array:
    dtype = np.float64
    n_functions = cuint_plan.n_functions
    if deriv == 0:
        shape = (9, n_functions, n_functions)
        target = "cuint_quadrupole_ffi"
    elif deriv == 1:
        shape = (27, n_functions, n_functions)
        target = "cuint_quadrupole_gradient_ffi"

        call_dip = jax.ffi.ffi_call(
            "cuint_dipole_ffi",
            jax.ShapeDtypeStruct((3, n_functions, n_functions), dtype),
            vmap_method="broadcast_all",
            input_output_aliases={0:0},
        )
    else:
        raise NotImplementedError

    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(shape, dtype),
        vmap_method="broadcast_all",
        input_output_aliases={0:0},
    )

    out = np.zeros(shape, dtype)
    if deriv == 1:
        dip = np.zeros((3, n_functions, n_functions), dtype)

    is_screened = cuint_plan.is_screened
    n_primitives = cuint_plan.n_primitives
    bas = cuint_plan.bas
    primitive_to_function = cuint_plan.primitive_to_function

    for pair in cuint_plan.pairs:
        out = call(
            out, pair.pair_indices, primitive_to_function,
            atm, bas, env,
            i_angular=pair.li,
            j_angular=pair.lj,
            is_screened=is_screened,
            n_pairs=pair.n_pairs,
            n_primitives=n_primitives,
            n_functions=n_functions,
            atm_stride=numpy.int32(atm.size),
            bas_stride=numpy.int32(bas.size),
            env_stride=numpy.int32(env.size),
        )

        if deriv == 1:
            dip = call_dip(
                dip, pair.pair_indices, primitive_to_function,
                atm, bas, env,
                i_angular=pair.li,
                j_angular=pair.lj,
                is_screened=is_screened,
                n_pairs=pair.n_pairs,
                n_primitives=n_primitives,
                n_functions=n_functions,
                atm_stride=numpy.int32(atm.size),
                bas_stride=numpy.int32(bas.size),
                env_stride=numpy.int32(env.size),
            )

    if deriv == 0:
        out += out.transpose(0, 2, 1)
        out = out.at[np.array([3, 6, 7])].set(out[np.array([1, 2, 5])])
    elif deriv == 1:
        out -= out.transpose(0, 2, 1)
        out = -out
        dip_T = dip.transpose(0, 2, 1)
        out = out.at[np.array([0, 13, 26])].subtract(2. * dip_T)
        out = out.at[np.array([10, 1, 2])].subtract(dip_T)
        out = out.at[np.array([20, 23, 14])].subtract(dip_T)
        out = out.at[np.array([3, 6, 7, 12, 15, 16, 21, 24, 25])].set(
                        out[np.array([1, 2, 5, 10, 11, 14, 19, 20, 23])])
    return out

@jax.tree_util.register_dataclass
@dataclass
class CuintPlan:
    bas: Array
    primitive_to_function: Array
    n_functions: numpy.int32 = field(metadata={"static": True})
    n_primitives: numpy.int32 = field(metadata={"static": True})
    pairs: list[PairInfo]
    is_screened: numpy.int32 = field(metadata={"static": True})

@jax.tree_util.register_dataclass
@dataclass
class PairInfo:
    li: numpy.int32 = field(metadata={"static": True})
    lj: numpy.int32 = field(metadata={"static": True})
    pair_indices: Array
    n_pairs: numpy.int32 = field(metadata={"static": True})

def cuint_merge_plans(plans: Sequence[CuintPlan]) -> tuple[CuintPlan, CuintPlan]:
    """Merge a sequence of cuint plans for batched calculations.

    The merged plan will have `bas` tiled, while all other attributes
    take the values from the first plan as they are assumed to be the same
    across the batch.

    Parameters:
        plans: cuint plans.

    Returns:
        merged_plans: merged cuint plans.
        vmap_in_axes: tree prefix passed as ``in_axes`` to :func:`jax.vmap`.
    """
    bases = []
    for plan in plans:
        bases.append(plan.bas)

    merged_plans = CuintPlan(
        bas = np.asarray(bases),
        primitive_to_function = plans[0].primitive_to_function,
        n_functions = plans[0].n_functions,
        n_primitives = plans[0].n_primitives,
        pairs = plans[0].pairs,
        is_screened = plans[0].is_screened,
    )

    vmap_in_axes = jax.tree.map(lambda x: None, merged_plans)
    vmap_in_axes = replace(vmap_in_axes, bas=0)
    return merged_plans, vmap_in_axes

def cuint_create_plan(mol: MoleLite, screening: bool = False) -> CuintPlan:
    """Create the cuint plan for computing integrals with the cuint backend.

    Parameters:
        mol: molecular information.
        screening: whether integral screening is applied.

    Returns:
        plan: a static cuint plan.
    """
    if mol.cart:
        raise NotImplementedError
    if screening:
        raise NotImplementedError
    else:
        is_screened = 0

    bas = numpy.asarray(mol._bas)
    ao_loc = mol.ao_loc

    ls = bas[:, ANG_OF]
    sort_idx = numpy.argsort(ls)
    sorted_bas = bas[sort_idx]

    sorted_shl_start = ao_loc[:-1][sort_idx]

    nctr = sorted_bas[:, NCTR_OF]
    nprim = sorted_bas[:, NPRIM_OF]
    decontracted_basis = numpy.repeat(sorted_bas, nctr, axis=-2)
    decontracted_basis[..., NCTR_OF] = 1

    _tmp = numpy.arange(numpy.sum(nctr)) - numpy.repeat(numpy.cumsum(numpy.r_[0, nctr[:-1]]), nctr)
    coeff_offset = _tmp * numpy.repeat(nprim, nctr)
    decontracted_basis[..., PTR_COEFF] += coeff_offset

    sorted_shl_start = numpy.repeat(sorted_shl_start, nctr)
    sorted_shl_start += _tmp * numpy.repeat(2 * sorted_bas[:, ANG_OF] + 1, nctr)

    nprim = numpy.repeat(nprim, nctr)
    decontracted_basis = numpy.repeat(decontracted_basis, nprim, axis=-2)

    primitive_offset = (
        numpy.arange(numpy.sum(nprim))
        - numpy.repeat(numpy.cumsum(numpy.r_[0, nprim[:-1]]), nprim)
    )
    decontracted_basis[..., NPRIM_OF] = 1
    decontracted_basis[..., PTR_COEFF] += primitive_offset
    decontracted_basis[..., PTR_EXP] += primitive_offset

    primitive_to_function = numpy.repeat(sorted_shl_start, nprim)

    n_primitives = decontracted_basis.shape[-2]

    angulars = decontracted_basis[:, ANG_OF]
    spikes = numpy.flatnonzero(numpy.diff(angulars)) + 1
    max_angular = len(spikes)
    l_loc = numpy.r_[0, spikes, n_primitives]

    grouped_primitives_ranges = numpy.empty((max_angular+1, 2), dtype=numpy.int32)
    grouped_primitives_ranges[:, 0] = l_loc[:-1]
    grouped_primitives_ranges[:, 1] = l_loc[1:]

    pairs = []

    for i_angular in range(max_angular + 1):
        i_range = grouped_primitives_ranges[i_angular]
        for j_angular in range(i_angular, max_angular + 1):
            j_range = grouped_primitives_ranges[j_angular]

            if screening:
                raise NotImplementedError
            else:
                n_rows = i_range[1] - i_range[0]
                n_cols = j_range[1] - j_range[0]
                if i_angular == j_angular:
                    n_pairs = (n_rows + 1) * n_rows // 2
                else:
                    n_pairs = n_rows * n_cols
                pair_indices = np.array([*i_range, *j_range], dtype=np.int32)

            pairs.append(
                PairInfo(
                    li = numpy.int32(i_angular),
                    lj = numpy.int32(j_angular),
                    pair_indices = pair_indices,
                    n_pairs = numpy.int32(n_pairs),
                )
            )

    plan = CuintPlan(
        bas = np.asarray(decontracted_basis, dtype=np.int32),
        primitive_to_function = np.asarray(primitive_to_function, dtype=np.int32),
        n_functions = numpy.int32(mol.nao),
        n_primitives = numpy.int32(n_primitives),
        pairs = pairs,
        is_screened = numpy.int32(is_screened),
    )
    return plan

