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

from pyscf.gto.mole import (
    ATOM_OF,
    ANG_OF,
    NPRIM_OF,
    NCTR_OF,
    PTR_EXP,
    PTR_COEFF,
    PTR_COORD,
    BAS_SLOTS,
)

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.moleintor_lite import (
    _aoslice_by_atom,
    _extract_coords,
)
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfad.gto._basis_deriv import _resolve_template
from .moleintor_cuint import PairInfo, gen_overlap_cross

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
        "bas_tmpl",
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
    bas_tmpl: ArrayLike | None = None, # static structure when bas is traced
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

class LatBasisCrossPlan:
    """Static structural plan for the per-image basis-derivative cross
    integrals.

    The stored lattice primal ``H_L`` is the raw kernel accumulation over
    the ordered pair set of :func:`cuint_create_plan` (l-sorted primitives;
    for equal angular momenta the upper triangle including a halved
    diagonal), with the ket primitive of each pair shifted by the image
    vector ``L``. Its exact basis tangent is assembled from four cross
    blocks between "fake" (unit-coefficient) and real primitives over a
    doubled system whose ket copy is displaced by ``L``:

    - bra terms: pairs ``(fake_p, real_q)`` for ``(p, q)`` in the primal
      pair set (diagonal pairs kept in a separate group),
    - ket terms: pairs ``(real_p, fake_q)`` for the same ordered pairs,

    where the diagonal (``p == q``) contributions enter with weight 1/2
    from each direction (``<d(chi)|chi(r-L)> != <chi|d(chi)(r-L)>`` for
    ``L != 0``).

    Row layout: ``[fake_bra, real_bra, fake_ket, real_ket]``, each with one
    row per (shell, contraction, primitive); bra rows reference the
    original atoms, ket rows reference a second atom copy whose
    coordinates live in an appended env block. Function spaces: fake
    ``[0, naof)``, real ``[naof, naof + nao)``.

    All structure is built from the concrete template; the env pointer
    columns are filled from the actual (possibly traced) ``bas`` by
    :meth:`make_rows`.
    """
    def __init__(self, tmpl, natm, nenv):
        tmpl = numpy.asarray(tmpl)
        nbas = len(tmpl)
        ls = tmpl[:, ANG_OF]
        nprims = tmpl[:, NPRIM_OF]
        nctrs = tmpl[:, NCTR_OF]
        nls = 2 * ls + 1

        ao_loc = numpy.append(0, numpy.cumsum(nls * nctrs)).astype(numpy.int32)
        fake_loc = numpy.append(0, numpy.cumsum(nls * nctrs * nprims)).astype(numpy.int32)
        nao = int(ao_loc[-1])
        naof = int(fake_loc[-1])

        ptr_ones = int(nenv)
        ptr_coords2 = ptr_ones + 1

        fake_rows = []
        real_rows = []
        fake_fn = []
        real_fn = []
        prim_shell = []
        prim_j = []
        prim_coeff_off = []
        map_fake_rows = []
        map_real_rows = []
        map_entry_shell = []
        map_coeff_off = []
        map_prim_off = []
        for i in range(nbas):
            l, nprim, nctr = int(ls[i]), int(nprims[i]), int(nctrs[i])
            nl = 2 * l + 1
            iatm = int(tmpl[i, ATOM_OF])
            for k in range(nctr):
                for j in range(nprim):
                    fake_rows.append([iatm, l, 1, 1, 0, 0, ptr_ones, 0])
                    real_rows.append([iatm, l, 1, 1, 0, 0, 0, 0])
                    f0 = fake_loc[i] + (k * nprim + j) * nl
                    fake_fn.append(f0)
                    real_fn.append(naof + ao_loc[i] + k * nl)
                    prim_shell.append(i)
                    prim_j.append(j)
                    prim_coeff_off.append(k * nprim + j)
                    m = numpy.arange(nl)
                    map_fake_rows.append(f0 + m)
                    map_real_rows.append(ao_loc[i] + k * nl + m)
                    map_entry_shell.append(numpy.full(nl, i))
                    map_coeff_off.append(numpy.full(nl, k * nprim + j))
                    map_prim_off.append(numpy.full(nl, j))

        npr = len(fake_rows)
        fake_rows = numpy.asarray(fake_rows, dtype=numpy.int32)
        real_rows = numpy.asarray(real_rows, dtype=numpy.int32)
        ket_fake = fake_rows.copy()
        ket_real = real_rows.copy()
        ket_fake[:, ATOM_OF] += natm
        ket_real[:, ATOM_OF] += natm
        rows = numpy.vstack([fake_rows, real_rows, ket_fake, ket_real])
        prim2fn = numpy.asarray(fake_fn + real_fn + fake_fn + real_fn,
                                dtype=numpy.int32)
        n_rows = 4 * npr

        # ordered primal pair set: primitives enumerated with shells
        # l-sorted exactly as in cuint_create_plan
        prim_l = numpy.repeat(ls, nctrs * nprims)
        shell_sort = numpy.argsort(ls)
        order = []
        for i in shell_sort:
            i0 = int(numpy.sum((nctrs * nprims)[:i]))
            order.extend(range(i0, i0 + int(nctrs[i] * nprims[i])))
        order = numpy.asarray(order, dtype=numpy.int64)
        order_l = prim_l[order]

        uls = numpy.unique(order_l)
        lbounds = {int(l): (numpy.searchsorted(order_l, l, "left"),
                            numpy.searchsorted(order_l, l, "right"))
                   for l in uls}

        def _groups(pair_p, pair_q, la, lb, bra_kind):
            # bra_kind "fake": (fake_bra_p, real_ket_q)
            # bra_kind "real": (real_bra_p, fake_ket_q)
            if len(pair_p) == 0:
                return None
            if bra_kind == "fake":
                enc = pair_p * n_rows + (3 * npr + pair_q)
            else:
                enc = (npr + pair_p) * n_rows + (2 * npr + pair_q)
            return PairInfo(
                li=numpy.int32(la),
                lj=numpy.int32(lb),
                pair_indices=numpy.asarray(enc, dtype=numpy.int32),
                n_pairs=numpy.int32(len(enc)),
            )

        pairs_bra_off = []
        pairs_ket_off = []
        pairs_bra_diag = []
        pairs_ket_diag = []
        for la in uls:
            a0, a1 = lbounds[int(la)]
            ra = order[a0:a1]
            # same-l: strict upper triangle in the sorted positions
            iu, ju = numpy.triu_indices(len(ra), k=1)
            g = _groups(ra[iu], ra[ju], la, la, "fake")
            if g: pairs_bra_off.append(g)
            g = _groups(ra[iu], ra[ju], la, la, "real")
            if g: pairs_ket_off.append(g)
            # diagonal pairs
            g = _groups(ra, ra, la, la, "fake")
            if g: pairs_bra_diag.append(g)
            g = _groups(ra, ra, la, la, "real")
            if g: pairs_ket_diag.append(g)
            for lb in uls:
                if lb <= la:
                    continue
                b0, b1 = lbounds[int(lb)]
                rb = order[b0:b1]
                pp, qq = numpy.meshgrid(ra, rb, indexing="ij")
                g = _groups(pp.ravel(), qq.ravel(), la, lb, "fake")
                if g: pairs_bra_off.append(g)
                g = _groups(pp.ravel(), qq.ravel(), la, lb, "real")
                if g: pairs_ket_off.append(g)

        self.rows_static = rows
        self.primitive_to_function = prim2fn
        self.n_functions = naof + nao
        self.n_primitives = n_rows
        self.naof = naof
        self.nao = nao
        self.npr = npr
        self.natm = int(natm)
        self.ptr_coords2 = ptr_coords2
        self.pairs_bra_off = pairs_bra_off
        self.pairs_ket_off = pairs_ket_off
        self.pairs_bra_diag = pairs_bra_diag
        self.pairs_ket_diag = pairs_ket_diag

        # env pointer gather descriptors (per (shell, ctr, prim) row)
        self.prim_shell = numpy.asarray(prim_shell)
        self.prim_j = numpy.asarray(prim_j)
        self.prim_coeff_off = numpy.asarray(prim_coeff_off)

        self.map_fake_rows = numpy.concatenate(map_fake_rows)
        self.map_real_rows = numpy.concatenate(map_real_rows)
        self.map_entry_shell = numpy.concatenate(map_entry_shell)
        self.map_coeff_off = numpy.concatenate(map_coeff_off)
        self.map_prim_off = numpy.concatenate(map_prim_off)
        # per fake-function-row shell/primitive and angular momentum
        nl_per_prim = 2 * fake_rows[:, ANG_OF] + 1
        self.fakefn_shell = numpy.repeat(self.prim_shell, nl_per_prim)
        self.fakefn_prim = numpy.repeat(self.prim_j, nl_per_prim)
        self.l_fake_fn = numpy.repeat(fake_rows[:, ANG_OF], nl_per_prim)

    def make_rows(self, bas):
        """The plan's doubled ``bas`` rows with env pointers gathered from
        the actual (possibly traced) cell ``bas``.
        """
        npr = self.npr
        ptr_exp = bas[:, PTR_EXP][self.prim_shell] + self.prim_j
        ptr_coeff = bas[:, PTR_COEFF][self.prim_shell] + self.prim_coeff_off
        if isinstance(bas, numpy.ndarray):
            rows = self.rows_static.copy()
            rows[:, PTR_EXP] = numpy.tile(ptr_exp, 4)
            rows[npr:2*npr, PTR_COEFF] = ptr_coeff
            rows[3*npr:, PTR_COEFF] = ptr_coeff
            return rows
        rows = np.asarray(self.rows_static)
        rows = ops.index_update(rows, ops.index[:, PTR_EXP],
                                np.tile(np.asarray(ptr_exp, dtype=np.int32), 4))
        rows = ops.index_update(rows, ops.index[npr:2*npr, PTR_COEFF],
                                np.asarray(ptr_coeff, dtype=np.int32))
        rows = ops.index_update(rows, ops.index[3*npr:, PTR_COEFF],
                                np.asarray(ptr_coeff, dtype=np.int32))
        return rows


_LAT_BASIS_CROSS_PLAN_CACHE = {}


def _get_lat_basis_cross_plan(tmpl, natm, nenv):
    key = (tmpl.tobytes(), tmpl.shape, int(natm), int(nenv))
    plan = _LAT_BASIS_CROSS_PLAN_CACHE.get(key)
    if plan is None:
        plan = LatBasisCrossPlan(tmpl, natm, nenv)
        _LAT_BASIS_CROSS_PLAN_CACHE[key] = plan
    return plan


def _gen_int1e_jvp_basis(
    intor_name, Ls, Ls_mask, atm, bas, env, env_dot, ao_loc, bas_tmpl,
):
    """Basis-set parameter tangent of the per-image lattice integrals
    on the cuint backend (first order in the basis parameters).

    The exponent term uses the solid-harmonic identity
    ``r_A^2 chi = [lap_A chi + 2 alpha (2l+3) chi] / (4 alpha^2)`` with
    the Laplacian from ``gen_overlap`` (``i_deriv``/``j_deriv`` = 2 on the
    fake side). All cross integrals run per image through the kernels'
    native configuration batching (the ket atom copy is displaced by L).
    """
    del ao_loc
    if intor_name != "int1e_ovlp_sph":
        raise NotImplementedError(
            "Basis-set parameter derivatives on the cuint lattice backend "
            f"are only supported for int1e_ovlp_sph, got {intor_name}."
        )
    tmpl = _resolve_template(bas, bas_tmpl)
    natm = atm.shape[0]
    nenv = env.shape[-1]
    plan = _get_lat_basis_cross_plan(tmpl, natm, nenv)
    naof = plan.naof
    nao = plan.nao
    rows = plan.make_rows(bas)

    Ls = Ls.reshape(-1, 3)
    nL = Ls.shape[0]

    # doubled system: ket atom copy displaced by L, coordinates in an
    # appended env block; evaluated on the (stopped) primal env only
    atm2 = np.concatenate([np.asarray(atm, dtype=np.int32)] * 2, axis=0)
    ptr2 = plan.ptr_coords2 + 3 * np.arange(natm, dtype=np.int32)
    atm2 = ops.index_update(atm2, ops.index[natm:, PTR_COORD], ptr2)

    env = np.asarray(env, dtype=np.float64)
    coords = _extract_coords(atm, env)
    coords_l = (coords[None, :, :] + np.asarray(Ls, dtype=np.float64)[:, None, :])
    env2 = np.concatenate(
        [
            np.broadcast_to(env, (nL, nenv)),
            np.ones((nL, 1), dtype=np.float64),
            coords_l.reshape(nL, -1),
        ],
        axis=1,
    )
    env2 = ops.stop_gradient(env2)

    def _blocks(pairs_bra, pairs_ket, i_deriv):
        # bra-direction blocks [0:naof, naof:], with optional bra Laplacian
        xb = gen_overlap_cross(atm2, env2, plan, rows, i_deriv=i_deriv,
                               pairs=pairs_bra)
        # ket-direction blocks [naof:, 0:naof], with the ket Laplacian
        xk = gen_overlap_cross(atm2, env2, plan, rows, j_deriv=i_deriv,
                               pairs=pairs_ket)
        return xb[..., :naof, naof:], xk[..., naof:, :naof]

    x0_bra_off, x0_ket_off = _blocks(plan.pairs_bra_off, plan.pairs_ket_off, 0)
    x0_bra_diag, x0_ket_diag = _blocks(plan.pairs_bra_diag, plan.pairs_ket_diag, 0)
    d2_bra_off, d2_ket_off = _blocks(plan.pairs_bra_off, plan.pairs_ket_off, 2)
    d2_bra_diag, d2_ket_diag = _blocks(plan.pairs_bra_diag, plan.pairs_ket_diag, 2)

    ptr_exp_col = bas[:, PTR_EXP]
    alpha_env_idx = ptr_exp_col[plan.fakefn_shell] + plan.fakefn_prim
    alpha = ops.stop_gradient(env[alpha_env_idx])
    lfac = 2.0 * (2 * plan.l_fake_fn + 3)
    scale = 1.0 / (4.0 * alpha ** 2)
    afac = lfac * alpha

    def _x_exp_bra(d2, x0):
        tr = d2[:, 0] + d2[:, 4] + d2[:, 8]
        return -(tr + afac[None, :, None] * x0[:, 0]) * scale[None, :, None]

    def _x_exp_ket(d2, x0):
        tr = d2[:, 0] + d2[:, 4] + d2[:, 8]
        return -(tr + x0[:, 0] * afac[None, None, :]) * scale[None, None, :]

    x_exp_bra_off = _x_exp_bra(d2_bra_off, x0_bra_off)
    x_exp_bra_diag = _x_exp_bra(d2_bra_diag, x0_bra_diag)
    x_exp_ket_off = _x_exp_ket(d2_ket_off, x0_ket_off)
    x_exp_ket_diag = _x_exp_ket(d2_ket_diag, x0_ket_diag)

    ptr_coeff_col = bas[:, PTR_COEFF]
    coeff_env_idx = ptr_coeff_col[plan.map_entry_shell] + plan.map_coeff_off
    exp_env_idx = ptr_exp_col[plan.map_entry_shell] + plan.map_prim_off

    env_dot = np.asarray(env_dot, dtype=np.float64)
    w_cs = env_dot[coeff_env_idx]
    w_exp = ops.stop_gradient(env[coeff_env_idx]) * env_dot[exp_env_idx]

    t_cs = np.zeros((nao, naof), dtype=np.float64)
    t_cs = ops.index_add(t_cs, ops.index[plan.map_real_rows, plan.map_fake_rows], w_cs)
    t_exp = np.zeros((nao, naof), dtype=np.float64)
    t_exp = ops.index_add(t_exp, ops.index[plan.map_real_rows, plan.map_fake_rows], w_exp)

    def _bra(t, x):
        return np.einsum("ma,lav->lmv", t, x)

    def _ket(x, t):
        return np.einsum("lma,na->lmn", x, t)

    jvp = (
        _bra(t_cs, x0_bra_off[:, 0]) + _ket(x0_ket_off[:, 0], t_cs)
        + 0.5 * (_bra(t_cs, x0_bra_diag[:, 0]) + _ket(x0_ket_diag[:, 0], t_cs))
        + _bra(t_exp, x_exp_bra_off) + _ket(x_exp_ket_off, t_exp)
        + 0.5 * (_bra(t_exp, x_exp_bra_diag) + _ket(x_exp_ket_diag, t_exp))
    )

    Ls_mask = np.asarray(Ls_mask).reshape(-1)
    jvp = np.where(Ls_mask[:, None, None] != 0, jvp,
                   np.zeros((), dtype=jvp.dtype))
    return jvp


def _lattice_intor_jvp(
    intor_name, Ls_mask, atm, bas, cuint_plan,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices, bas_tmpl,
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
        bas_tmpl=bas_tmpl,
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
            jvp = _gen_int1e_fill_jvp_r0(s1a, coords_dot, aoslices-_ao_loc[i0],
                                         aoidx[None,None,:,None])

            aoidx = np.arange(naoj)
            jvp += _gen_int1e_fill_jvp_r0(-s1a, coords_dot, aoslices-_ao_loc[j0],
                                          aoidx[None,None,None,:])

            tangent_out += jvp.reshape(tangent_out.shape)

        if trace_basis:
            tangent_out += _gen_int1e_jvp_basis(
                intor_name, Ls, Ls_mask, atm, bas, env, env_dot, ao_loc,
                bas_tmpl,
            ).reshape(tangent_out.shape)

    if not isinstance(Ls_dot, SymbolicZero):
        # Every ket function in image L is displaced rigidly by L, so
        # dS_L/dL is the ket-center derivative summed over all ket centers.
        # By pair translation invariance this equals minus the bra
        # derivative that the deriv=1 kernel provides.
        s1a = -lat_overlap(atm, env, Ls, Ls_mask, cuint_plan, deriv=1)
        Ls_dot = np.asarray(Ls_dot, dtype=np.float64)
        tangent_out += np.einsum("lxpq,lx->lpq", -s1a, Ls_dot).reshape(
            tangent_out.shape
        )
    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
