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
from typing import TYPE_CHECKING, NamedTuple
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
    BAS_SLOTS,
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
    next_coord_deriv,
    _concrete_bas,
    cs_scatter_maps,
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
        "max_coord_deriv",
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
    max_coord_deriv: int | None = None,
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
            f"Integral {intor_name} is not supported."
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
    max_coord_deriv,
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
        max_coord_deriv=max_coord_deriv,
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
                aoslices, rc_deriv, max_coord_deriv,
            ).reshape(tangent_out.shape)

        if trace_basis:
            tangent_out += _gen_int1e_jvp_basis(
                intor_name, atm, bas, env, env_dot, hermi, cuint_plan,
            ).reshape(tangent_out.shape)
    return primal_out, tangent_out

getints.defjvp(getints_jvp, symbolic_zeros=True)


class CrossChunk(NamedTuple):
    """One kernel launch's worth of fake functions.

    The cuint kernels address their output as a square
    ``n_functions x n_functions`` matrix (the component and configuration
    strides are ``n_functions**2``), while the cross blocks are the
    rectangular fake x real ones. Splitting the fake functions into chunks of
    at most ``nao`` of them makes the square the kernels demand no bigger than
    the block that is actually filled.

    Attributes:
        fn_start: First fake function of the chunk (into ``[0, nao_fake)``).
        n_fn: Number of fake functions in the chunk.
        n_functions: Side of the square the kernels write,
            ``max(n_fn, nao)``.
        primitive_to_function: Function index of every plan row for this
            chunk: the chunk's fake rows relative to ``fn_start``, the real
            rows as themselves (the two index spaces are independent -- one
            indexes the output rows, the other its columns).
        pairs: ``{group name: tuple of PairInfo}`` restricted to the chunk.
    """
    fn_start: int
    n_fn: int
    n_functions: numpy.int32
    primitive_to_function: numpy.ndarray
    pairs: dict


def chunk_budget(nao, nl_max):
    """Fake functions per kernel launch.

    ``nao`` of them: the kernels write a square block, so a chunk that wide
    makes the square exactly the (fake x real) block that is kept, with
    nothing allocated around it. A chunk always holds whole rows, hence the
    floor at the widest single row.
    """
    return max(int(nao), int(nl_max))


def _fake_chunks(nl_per_row, budget):
    """Group consecutive fake rows into chunks of at most ``budget``
    functions, as ``(row start, row stop, function start, function count)``.

    A row's functions are contiguous, so the chunks tile ``[0, nao_fake)``.
    """
    chunks = []
    r0 = fn0 = total = 0
    for i, nl in enumerate(nl_per_row):
        if total and total + nl > budget:
            chunks.append((r0, i, fn0, total))
            r0, fn0, total = i, fn0 + total, 0
        total += int(nl)
    if total:
        chunks.append((r0, len(nl_per_row), fn0, total))
    return chunks


class BasisCrossPlan:
    """Static structural plan for the fake(primitive) x real cross
    integrals used by the basis-set parameter derivatives
    (see :mod:`pyscfad.gto._basis_deriv`).

    The fake shells are one uncontracted shell per (shell, primitive) with
    unit coefficient (an env slot appended at ``ptr_ones``), mapped to a
    primitive-resolved function space ``[0, nao_fake)``; the real shells are
    decontracted per (shell, contraction, primitive) and keep their ``ao_loc``
    offsets in ``[0, nao)``. The fake functions index the rows of the cross
    block and the real ones its columns, so the two spaces are independent.
    Pairs are explicit ("screened") lists, one group per (l_bra, l_ket)
    combination, split over :class:`CrossChunk` s.

    All structure is built from ``bas_conc``; the env pointer
    columns are filled from the actual (possibly traced) ``bas`` by
    :meth:`make_rows`, so the plan works under jit and vmap over atomic
    numbers.
    """
    def __init__(self, bas_conc, ptr_ones, budget=None):
        bas_conc = numpy.asarray(bas_conc)
        ls = bas_conc[:, ANG_OF]
        nprims = bas_conc[:, NPRIM_OF]
        nctrs = bas_conc[:, NCTR_OF]
        nls = 2 * ls + 1

        ao_loc = numpy.append(0, numpy.cumsum(nls * nctrs)).astype(numpy.int32)
        fake_loc = numpy.append(0, numpy.cumsum(nls * nprims)).astype(numpy.int32)
        nao = int(ao_loc[-1])
        nao_fake = int(fake_loc[-1])

        fake_rows = []
        fake_fn = []
        fake_shell = []
        fake_prim = []
        real_rows = []
        real_fn = []
        real_shell = []
        real_prim = []
        real_coeff_off = []
        for i in range(len(bas_conc)):
            l, nprim, nctr = int(ls[i]), int(nprims[i]), int(nctrs[i])
            nl = 2 * l + 1
            iatm = int(bas_conc[i, ATOM_OF])
            for j in range(nprim):
                fake_rows.append([iatm, l, 1, 1, 0, 0, ptr_ones, 0])
                fake_fn.append(fake_loc[i] + j * nl)
                fake_shell.append(i)
                fake_prim.append(j)
            for k in range(nctr):
                for j in range(nprim):
                    real_rows.append([iatm, l, 1, 1, 0, 0, 0, 0])
                    real_fn.append(ao_loc[i] + k * nl)
                    real_shell.append(i)
                    real_prim.append(j)
                    real_coeff_off.append(k * nprim + j)

        nf = len(fake_rows)
        nr = len(real_rows)
        rows = numpy.asarray(fake_rows + real_rows, dtype=numpy.int32)
        fake_fn = numpy.asarray(fake_fn, dtype=numpy.int32)
        real_fn = numpy.asarray(real_fn, dtype=numpy.int32)
        n_rows = nf + nr

        l_fake = rows[:nf, ANG_OF]
        l_real = rows[nf:, ANG_OF]
        chunks = []
        nl_fake = 2 * l_fake + 1
        if budget is None:
            budget = chunk_budget(nao, nl_fake.max())
        for r0, r1, fn0, n_fn in _fake_chunks(nl_fake, budget):
            prim2fn = numpy.zeros(n_rows, dtype=numpy.int32)
            prim2fn[r0:r1] = fake_fn[r0:r1] - fn0
            prim2fn[nf:] = real_fn
            pairs = []
            for la in numpy.unique(l_fake[r0:r1]):
                f_idx = r0 + numpy.flatnonzero(l_fake[r0:r1] == la)
                for lb in numpy.unique(l_real):
                    r_idx = nf + numpy.flatnonzero(l_real == lb)
                    enc = (f_idx[:, None] * n_rows + r_idx[None, :]).ravel()
                    pairs.append(PairInfo(
                        li=numpy.int32(la),
                        lj=numpy.int32(lb),
                        pair_indices=numpy.asarray(enc, dtype=numpy.int32),
                        n_pairs=numpy.int32(enc.size),
                    ))
            chunks.append(CrossChunk(
                fn_start=fn0, n_fn=n_fn,
                n_functions=numpy.int32(max(n_fn, nao)),
                primitive_to_function=prim2fn,
                pairs={"cross": tuple(pairs)},
            ))

        self.rows_static = rows
        self.chunks = chunks
        self.nao_fake = nao_fake
        self.nao = nao
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


def _get_basis_cross_plan(bas_conc, ptr_ones, budget=None):
    key = (bas_conc.tobytes(), bas_conc.shape, int(ptr_ones), budget)
    plan = _BASIS_CROSS_PLAN_CACHE.get(key)
    if plan is None:
        plan = BasisCrossPlan(bas_conc, ptr_ones, budget)
        _BASIS_CROSS_PLAN_CACHE[key] = plan
    return plan


def cuint_max_deriv() -> int:
    """Highest total derivative order (``i_deriv + j_deriv``) the installed
    cuint kernels were compiled for.

    cuint dispatches the derivative order at compile time and silently does
    nothing when asked for an order it was not built with, so every caller
    of :func:`gen_overlap_cross` must check this first. Plugins predating the
    ``max_deriv`` export report cuint's ``MAX_DERIV = 2`` default.
    """
    if not _cuint:
        return 0
    fn = getattr(_cuint, "max_deriv", None)
    if fn is None:
        return 2
    return int(fn())


# cuint integrals whose basis-set parameter derivative is implemented, and the
# order of the bra coordinate derivative each carries. The exponent term adds a
# bra Laplacian on top of that (see _gen_int1e_jvp_basis), so the coordinate
# gradient needs kernels compiled for total derivative order 3.
_BASIS_DERIV_ORDER = {
    "int1e_ovlp_sph": 0,
    "int1e_ovlp_dr10_sph": 1,
    "int1e_ipovlp_sph": 1,
}


def _basis_deriv_order(intor_name: str, backend: str = "cuint") -> int:
    """Coordinate-derivative order of ``intor_name``, checking that its
    basis-parameter derivative (and the kernels it needs) are available.
    """
    try:
        n_deriv = _BASIS_DERIV_ORDER[intor_name]
    except KeyError:
        supported = ", ".join(sorted(_BASIS_DERIV_ORDER))
        raise NotImplementedError(
            f"Basis-set parameter derivatives on the {backend} backend are "
            f"only supported for {supported}, got {intor_name}."
        ) from None
    order = n_deriv + 2
    max_deriv = cuint_max_deriv()
    if order > max_deriv:
        raise NotImplementedError(
            f"The basis-set parameter derivative of {intor_name} needs cuint "
            f"overlap kernels of total derivative order {order} (the exponent "
            "term is a Laplacian on top of the integral's own derivatives), "
            f"but the installed CUDA plugin provides {max_deriv}. Rebuild it "
            f"with -DCUINT_MAX_DERIV={order} (see pyscfadlib/plugins/cuda)."
        )
    return n_deriv


def _lap_trace(x: Array, n_deriv: int, axis: int) -> Array:
    """Contract the trailing pair of derivative slots of a
    :func:`gen_overlap_cross` output into a Laplacian.

    ``x`` has ``3**(n_deriv + 2)`` components along ``axis``, enumerating
    ``n_deriv`` gradient slots (slowest) followed by the two Laplacian slots
    (see the component ordering in cuint's ``gen_kernel``); the result keeps
    the ``3**n_deriv`` gradient components.
    """
    shape = x.shape
    x = x.reshape(shape[:axis] + (3 ** n_deriv, 9) + shape[axis+1:])
    idx = (slice(None),) * (axis + 1)
    return x[idx + (0,)] + x[idx + (4,)] + x[idx + (8,)]


def _plan_bas_concrete(cuint_plan, bas) -> numpy.ndarray:
    """The concrete structural ``bas_conc`` recorded on the cuint plan at
    creation (pointer columns zeroed; they are always gathered from the
    runtime ``bas``), falling back to a concrete ``bas``.
    """
    if cuint_plan.bas_conc is not None:
        return numpy.frombuffer(
            cuint_plan.bas_conc, dtype=numpy.int32).reshape(-1, BAS_SLOTS)
    return _concrete_bas(bas)


def gen_overlap_cross(
    atm: Array,
    env: Array,
    plan,
    rows: ArrayLike,
    chunk: CrossChunk,
    i_deriv: int = 0,
    j_deriv: int = 0,
    group: str = "cross",
) -> Array:
    """Cross overlap (or its bra/ket coordinate derivatives) over one chunk's
    explicit pair lists, via the general-order ``cuint_gen_overlap_ffi``
    kernel. No symmetrization is applied.

    The result is the ``(comp, n_functions, n_functions)`` square the kernels
    write; the caller keeps the fake x real (or real x fake) corner of it,
    ``chunk.n_fn`` by ``plan.nao``.

    ``env`` may carry one leading batch dimension (e.g. lattice images);
    ``atm``/``rows`` are then tiled so that every operand carries the same
    batch layout with per-configuration strides — this composes correctly
    with the kernels' native configuration batching under (nested) vmap.
    """
    atm = np.asarray(atm, dtype=np.int32)
    env = np.asarray(env, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int32)
    pairs = chunk.pairs[group]

    comp = 3 ** (i_deriv + j_deriv)
    n = int(chunk.n_functions)
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
            out, pair.pair_indices, chunk.primitive_to_function,
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


def _cross_blocks(atm, env, plan, rows, n_deriv, lap, group="cross",
                  transpose=False):
    """The fake x real cross block over all of a plan's chunks.

    ``lap`` contracts the two extra derivative slots of the exponent
    identity into a Laplacian, per chunk, so only the ``3**n_deriv``
    components of the integral itself are kept. ``transpose`` selects the
    real x fake blocks (the ket direction of the lattice plan).
    """
    blocks = []
    for chunk in plan.chunks:
        x = gen_overlap_cross(atm, env, plan, rows, chunk, group=group,
                              i_deriv=n_deriv if transpose else n_deriv + 2*lap,
                              j_deriv=2 * lap if transpose else 0)
        if transpose:
            x = x[..., :plan.nao, :chunk.n_fn]
        else:
            x = x[..., :chunk.n_fn, :plan.nao]
        if lap:
            x = _lap_trace(x, n_deriv, x.ndim - 3)
        blocks.append(x)
    return np.concatenate(blocks, axis=-1 if transpose else -2)


def _gen_int1e_jvp_basis(
    intor_name,
    atm,
    bas,
    env,
    env_dot,
    hermi,
    cuint_plan,
):
    """Basis-set parameter (exponent + contraction coefficient) tangent
    for the cuint backend (first order in the basis parameters).

    Handles the overlap and its coordinate gradient
    (``int1e_ovlp_dr10``/``int1e_ipovlp``, one bra derivative), so that
    geometry gradients stay differentiable w.r.t. the basis set. The
    exponent term uses the solid-harmonic identity
    ``r_A^2 chi = [lap_A chi + 2 alpha (2l+3) chi] / (4 alpha^2)``,
    differentiated along with the integral: the bra Laplacian comes from
    ``gen_overlap(i_deriv = n_deriv + 2)``, whose leading components are the
    integral's own gradient components.

    The bra cross term determines the tangent completely: the overlap is
    symmetric and its gradient antisymmetric, so the ket term is the (signed)
    transpose of the bra term.

    Note:
        The cross integrals are evaluated on the stopped primal ``env``, so
        this tangent is treated as geometry-independent. A mixed
        coordinate/basis second derivative therefore has to differentiate
        the coordinates first, e.g. ``jacfwd(grad(f, coords), basis)``; the
        CPU path (``moleintor_lite``) has the same restriction.
    """
    n_deriv = _basis_deriv_order(intor_name)
    if hermi != 1:
        raise NotImplementedError(f"hermi = {hermi}")

    bas_conc = _plan_bas_concrete(cuint_plan, bas)
    ptr_ones = env.shape[-1]
    plan = _get_basis_cross_plan(bas_conc, ptr_ones)
    nao_fake = plan.nao_fake
    nao = plan.nao
    rows = plan.make_rows(bas)

    # first order in the basis parameters: the cross integrals are
    # evaluated on the (stopped) primal env only
    envc = ops.stop_gradient(np.concatenate(
        [np.asarray(env, dtype=np.float64), np.ones(1, dtype=np.float64)]))

    ptr_exp = bas[:, PTR_EXP]
    alpha_env_idx = ptr_exp[plan.fakefn_shell] + plan.fakefn_prim
    alpha = ops.stop_gradient(env[alpha_env_idx])
    lfac = 2.0 * (2 * plan.l_fake_fn + 3)

    maps = cs_scatter_maps(bas_conc, False)
    # the cs maps enumerate (shell, contraction, primitive, function) with
    # coeff_off = contraction * nprim + primitive, so the primitive offset
    # the exponent term needs is coeff_off modulo nprim of that shell
    coeff_env_idx = bas[:, PTR_COEFF][maps.entry_shell] + maps.coeff_off
    prim_off = maps.coeff_off % bas_conc[maps.entry_shell, NPRIM_OF]
    exp_env_idx = bas[:, PTR_EXP][maps.entry_shell] + prim_off
    w_cs = env_dot[coeff_env_idx]
    w_exp = ops.stop_gradient(env[coeff_env_idx]) * env_dot[exp_env_idx]

    dtype = np.float64
    t_cs = np.zeros((nao, nao_fake), dtype=dtype)
    t_cs = ops.index_add(t_cs, ops.index[maps.real_rows, maps.fake_rows], w_cs)
    t_exp = np.zeros((nao, nao_fake), dtype=dtype)
    t_exp = ops.index_add(t_exp, ops.index[maps.real_rows, maps.fake_rows], w_exp)

    # the fake functions are covered one chunk at a time, so the square block
    # the kernels insist on writing is only nao x nao and the exponent term's
    # 3**(n_deriv+2) components live for one chunk each -- what is kept is the
    # (fake x real) block the contraction needs, with the Laplacian already
    # traced out of it
    x0 = _cross_blocks(atm, envc, plan, rows, n_deriv, lap=False)
    tr_d2 = _cross_blocks(atm, envc, plan, rows, n_deriv, lap=True)

    x_exp = -(tr_d2 + (lfac * alpha)[:, None] * x0) / (4.0 * alpha ** 2)[:, None]
    jvp = (np.einsum("ma,...av->...mv", t_cs, x0)
           + np.einsum("ma,...av->...mv", t_exp, x_exp))
    # ket term: + transpose for the (symmetric) overlap, - for its
    # (antisymmetric) gradient
    jvp = jvp + (-1) ** n_deriv * np.swapaxes(jvp, -1, -2)
    return jvp


def _gen_int1e_jvp_r0(
    intor_a, intor_b,
    atm, bas, env, env_dot,
    cuint_plan,
    shls_slice, comp, hermi, aosym, ao_loc,
    trace_coords, trace_basis,
    aoslices, rc_deriv, max_coord_deriv=None,
):
    if comp is not None:
        comp = comp * 3

    # see pyscfad.gto._basis_deriv.next_coord_deriv: with max_coord_deriv=1 the
    # nested integrals keep their basis derivative but stop tracing
    # coordinates, so the second coordinate derivative -- which this backend
    # does not implement -- is never requested.
    nested_coord_deriv, nested_trace_coords = next_coord_deriv(
        max_coord_deriv, trace_coords
    )

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
            trace_coords=nested_trace_coords, trace_basis=trace_basis,
            aoslices=aoslices, max_coord_deriv=nested_coord_deriv,
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
    # concrete structural _bas snapshot (raw int32 bytes, pointer columns
    # zeroed) for the basis-parameter derivatives; bytes keep the static
    # field hashable and value-compared in jit cache keys
    bas_conc: bytes | None = field(default=None, metadata={"static": True})

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
        bas_conc = plans[0].bas_conc,
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

    # structural snapshot for the basis-parameter derivatives; the env
    # pointers are gathered from the runtime bas, so zero them here to
    # make bas_conc (and the plan caches keyed on it) canonical
    bas_conc = bas.copy()
    bas_conc[:, [PTR_EXP, PTR_COEFF]] = 0

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
        bas_conc = bas_conc.tobytes(),
    )
    return plan

