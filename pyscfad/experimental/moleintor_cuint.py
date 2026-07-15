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
from dataclasses import dataclass, field, replace
from functools import partial
import numpy
import jax
from jax.custom_derivatives import SymbolicZero
from pyscf.gto.mole import (
    ATOM_OF,
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
from pyscfad.gto._basis_deriv import (
    _resolve_template,
    _get_maps,
    gather_env_idx,
)
from pyscfad import ops
from pyscfadlib._cuda_plugin import import_plugin_module

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite

# Load the integral module from the CUDA plugin matching jax's CUDA version
# (pyscfad-cuda12-plugin / pyscfad-cuda13-plugin / ...).
_cuint = import_plugin_module("_cuint")

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
        "bas_tmpl",
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
    bas_tmpl: ArrayLike | None = None, # static structure when bas is traced
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
    bas_tmpl,
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
        bas_tmpl=bas_tmpl,
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
                aoslices, rc_deriv, bas_tmpl,
            ).reshape(tangent_out.shape)

        if trace_basis:
            tangent_out += _gen_int1e_jvp_basis(
                intor_name, atm, bas, env, env_dot, hermi, bas_tmpl,
            ).reshape(tangent_out.shape)
    return primal_out, tangent_out

getints.defjvp(getints_jvp, symbolic_zeros=True)


class BasisCrossPlan:
    """Static structural plan for the fake(primitive) x real cross
    integrals used by the basis-set parameter derivatives
    (see :mod:`pyscfad.gto._basis_deriv`).

    The fake shells are one uncontracted shell per (shell, primitive) with
    unit coefficient (an env slot appended at ``ptr_ones``), mapped to a
    primitive-resolved function space ``[0, naof)``; the real shells are
    decontracted per (shell, contraction, primitive) and keep their
    ``ao_loc`` offsets shifted into ``[naof, naof + nao)``. Pairs are
    explicit ("screened") lists, one group per (l_bra, l_ket) combination.

    All structure is built from the concrete template; the env pointer
    columns are filled from the actual (possibly traced) ``bas`` by
    :meth:`make_rows`, so the plan works under jit and vmap over atomic
    numbers.
    """
    def __init__(self, tmpl, ptr_ones):
        tmpl = numpy.asarray(tmpl)
        ls = tmpl[:, ANG_OF]
        nprims = tmpl[:, NPRIM_OF]
        nctrs = tmpl[:, NCTR_OF]
        nls = 2 * ls + 1

        ao_loc = numpy.append(0, numpy.cumsum(nls * nctrs)).astype(numpy.int32)
        fake_loc = numpy.append(0, numpy.cumsum(nls * nprims)).astype(numpy.int32)
        nao = int(ao_loc[-1])
        naof = int(fake_loc[-1])

        fake_rows = []
        fake_fn = []
        fake_shell = []
        fake_prim = []
        real_rows = []
        real_fn = []
        real_shell = []
        real_prim = []
        real_coeff_off = []
        for i in range(len(tmpl)):
            l, nprim, nctr = int(ls[i]), int(nprims[i]), int(nctrs[i])
            nl = 2 * l + 1
            iatm = int(tmpl[i, ATOM_OF])
            for j in range(nprim):
                fake_rows.append([iatm, l, 1, 1, 0, 0, ptr_ones, 0])
                fake_fn.append(fake_loc[i] + j * nl)
                fake_shell.append(i)
                fake_prim.append(j)
            for k in range(nctr):
                for j in range(nprim):
                    real_rows.append([iatm, l, 1, 1, 0, 0, 0, 0])
                    real_fn.append(naof + ao_loc[i] + k * nl)
                    real_shell.append(i)
                    real_prim.append(j)
                    real_coeff_off.append(k * nprim + j)

        nf = len(fake_rows)
        nr = len(real_rows)
        rows = numpy.asarray(fake_rows + real_rows, dtype=numpy.int32)
        prim2fn = numpy.asarray(fake_fn + real_fn, dtype=numpy.int32)
        n_rows = nf + nr

        l_fake = rows[:nf, ANG_OF]
        l_real = rows[nf:, ANG_OF]
        pairs = []
        for la in numpy.unique(l_fake):
            f_idx = numpy.flatnonzero(l_fake == la)
            for lb in numpy.unique(l_real):
                r_idx = nf + numpy.flatnonzero(l_real == lb)
                enc = (f_idx[:, None] * n_rows + r_idx[None, :]).ravel()
                pairs.append(PairInfo(
                    li=numpy.int32(la),
                    lj=numpy.int32(lb),
                    pair_indices=numpy.asarray(enc, dtype=numpy.int32),
                    n_pairs=numpy.int32(enc.size),
                ))

        self.rows_static = rows
        self.primitive_to_function = prim2fn
        self.pairs = pairs
        self.naof = naof
        self.nao = nao
        self.n_functions = naof + nao
        self.n_primitives = n_rows
        self.nf = nf

        # env pointer gather descriptors
        self.row_shell = numpy.asarray(fake_shell + real_shell)
        self.row_prim = numpy.asarray(fake_prim + real_prim)
        self.real_coeff_off = numpy.asarray(real_coeff_off)

        # per fake-function-row shell/primitive and angular momentum
        # (for the solid-harmonic exponent-derivative identity)
        nl_per_fake = 2 * l_fake + 1
        self.fakefn_shell = numpy.repeat(numpy.asarray(fake_shell), nl_per_fake)
        self.fakefn_prim = numpy.repeat(numpy.asarray(fake_prim), nl_per_fake)
        self.l_fake_fn = numpy.repeat(l_fake, nl_per_fake)

    def make_rows(self, bas):
        """The plan's ``bas`` rows with env pointers gathered from the
        actual (possibly traced) molecular ``bas``.
        """
        ptr_exp = bas[:, PTR_EXP]
        ptr_coeff = bas[:, PTR_COEFF]
        row_ptr_exp = ptr_exp[self.row_shell] + self.row_prim
        real_ptr_coeff = ptr_coeff[self.row_shell[self.nf:]] + self.real_coeff_off
        if isinstance(bas, numpy.ndarray):
            rows = self.rows_static.copy()
            rows[:, PTR_EXP] = row_ptr_exp
            rows[self.nf:, PTR_COEFF] = real_ptr_coeff
            return rows
        rows = np.asarray(self.rows_static)
        rows = ops.index_update(rows, ops.index[:, PTR_EXP],
                                np.asarray(row_ptr_exp, dtype=np.int32))
        rows = ops.index_update(rows, ops.index[self.nf:, PTR_COEFF],
                                np.asarray(real_ptr_coeff, dtype=np.int32))
        return rows


_BASIS_CROSS_PLAN_CACHE = {}


def _get_basis_cross_plan(tmpl, ptr_ones):
    key = (tmpl.tobytes(), tmpl.shape, int(ptr_ones))
    plan = _BASIS_CROSS_PLAN_CACHE.get(key)
    if plan is None:
        plan = BasisCrossPlan(tmpl, ptr_ones)
        _BASIS_CROSS_PLAN_CACHE[key] = plan
    return plan


def gen_overlap_cross(
    atm: Array,
    env: Array,
    plan,
    rows: ArrayLike,
    i_deriv: int = 0,
    j_deriv: int = 0,
    pairs: Sequence[PairInfo] | None = None,
) -> Array:
    """Cross overlap (or its bra/ket coordinate derivatives) over the
    explicit pair lists of a :class:`BasisCrossPlan`, via the general-order
    ``cuint_gen_overlap_ffi`` kernel. No symmetrization is applied.

    ``env`` may carry one leading batch dimension (e.g. lattice images);
    ``atm``/``rows`` are then tiled so that every operand carries the same
    batch layout with per-configuration strides — this composes correctly
    with the kernels' native configuration batching under (nested) vmap.
    """
    atm = np.asarray(atm, dtype=np.int32)
    env = np.asarray(env, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int32)
    if pairs is None:
        pairs = plan.pairs

    comp = 3 ** (i_deriv + j_deriv)
    n = int(plan.n_functions)
    if env.ndim == 1:
        shape = (comp, n, n)
    else:
        nbatch = env.shape[0]
        shape = (nbatch, comp, n, n)
        atm = np.broadcast_to(atm[None], (nbatch,) + atm.shape)
        rows = np.broadcast_to(rows[None], (nbatch,) + rows.shape)
    dtype = np.float64

    call = jax.ffi.ffi_call(
        "cuint_gen_overlap_ffi",
        jax.ShapeDtypeStruct(shape, dtype),
        vmap_method="broadcast_all",
        input_output_aliases={0:0},
    )

    atm_stride = atm.shape[-2] * atm.shape[-1]
    bas_stride = rows.shape[-2] * rows.shape[-1]
    out = np.zeros(shape, dtype)
    for pair in pairs:
        out = call(
            out, pair.pair_indices, plan.primitive_to_function,
            atm, rows, env,
            i_angular=pair.li,
            j_angular=pair.lj,
            is_screened=numpy.int32(1),
            n_pairs=pair.n_pairs,
            n_primitives=numpy.int32(plan.n_primitives),
            n_functions=numpy.int32(n),
            atm_stride=numpy.int32(atm_stride),
            bas_stride=numpy.int32(bas_stride),
            env_stride=numpy.int32(env.shape[-1]),
            i_deriv=numpy.int32(i_deriv),
            j_deriv=numpy.int32(j_deriv),
            comp=numpy.int32(comp),
        )
    return out


def _gen_int1e_jvp_basis(
    intor_name,
    atm,
    bas,
    env,
    env_dot,
    hermi,
    bas_tmpl,
):
    """Basis-set parameter (exponent + contraction coefficient) tangent
    for the cuint backend (first order in the basis parameters).

    The exponent term uses the solid-harmonic identity
    ``r_A^2 chi = [lap_A chi + 2 alpha (2l+3) chi] / (4 alpha^2)``,
    with the bra Laplacian evaluated by ``gen_overlap(i_deriv=2)``.
    """
    if intor_name != "int1e_ovlp_sph":
        raise NotImplementedError(
            "Basis-set parameter derivatives on the cuint backend are only "
            f"supported for int1e_ovlp_sph, got {intor_name}."
        )
    if hermi != 1:
        raise NotImplementedError(f"hermi = {hermi}")

    tmpl = _resolve_template(bas, bas_tmpl)
    ptr_ones = env.shape[-1]
    plan = _get_basis_cross_plan(tmpl, ptr_ones)
    naof = plan.naof
    nao = plan.nao
    rows = plan.make_rows(bas)

    # first order in the basis parameters: the cross integrals are
    # evaluated on the (stopped) primal env only
    envc = ops.stop_gradient(np.concatenate(
        [np.asarray(env, dtype=np.float64), np.ones(1, dtype=np.float64)]))

    x0 = gen_overlap_cross(atm, envc, plan, rows)[0, :naof, naof:]
    d2 = gen_overlap_cross(atm, envc, plan, rows, i_deriv=2)
    tr_d2 = (d2[0] + d2[4] + d2[8])[:naof, naof:]

    ptr_exp = bas[:, PTR_EXP]
    alpha_env_idx = ptr_exp[plan.fakefn_shell] + plan.fakefn_prim
    alpha = ops.stop_gradient(env[alpha_env_idx])
    lfac = 2.0 * (2 * plan.l_fake_fn + 3)
    x_exp = -(tr_d2 + (lfac * alpha)[:, None] * x0) / (4.0 * alpha ** 2)[:, None]

    maps = _get_maps(tmpl, "cs", False)
    coeff_env_idx, exp_env_idx = gather_env_idx(bas, maps)
    w_cs = env_dot[coeff_env_idx]
    w_exp = ops.stop_gradient(env[coeff_env_idx]) * env_dot[exp_env_idx]

    t_cs = np.zeros((nao, naof), dtype=x0.dtype)
    t_cs = ops.index_add(t_cs, ops.index[maps.real_rows, maps.fake_rows], w_cs)
    t_exp = np.zeros((nao, naof), dtype=x0.dtype)
    t_exp = ops.index_add(t_exp, ops.index[maps.real_rows, maps.fake_rows], w_exp)

    jvp = np.einsum("ma,av->mv", t_cs, x0) + np.einsum("ma,av->mv", t_exp, x_exp)
    jvp = jvp + jvp.T
    return jvp


def _gen_int1e_jvp_r0(
    intor_a, intor_b,
    atm, bas, env, env_dot,
    cuint_plan,
    shls_slice, comp, hermi, aosym, ao_loc,
    trace_coords, trace_basis,
    aoslices, rc_deriv, bas_tmpl=None,
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
            aoslices=aoslices, bas_tmpl=bas_tmpl,
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

