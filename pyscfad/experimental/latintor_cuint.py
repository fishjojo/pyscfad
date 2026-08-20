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
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_helper import (
    int1e_dr1_name,
    aoslices_in_range,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from .moleintor_cuint import (
    CrossChunk,
    PairInfo,
    chunk_budget,
    _basis_deriv_order,
    _cross_blocks,
    _fake_chunks,
    _plan_bas_concrete,
)

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
        "env",
        "cuint_plan",
        "shls_slice",
        "comp",
        "hermi",
        "ao_loc",
    ),
)
def _lattice_intor(
    intor_name: str,
    Ls: ArrayLike,
    Ls_mask: ArrayLike,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    cuint_plan: CuintPlan,
    r0: Array,
    exp: Array,
    ctr_coeff: Array,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    ao_loc: numpy.ndarray | None = None,
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
    elif intor_name in ("int1e_ovlp_dr10_sph", "int1e_ipovlp_sph"):
        out = lat_overlap(atm, env, Ls, Ls_mask, cuint_plan, deriv=1)
    else:
        raise NotImplementedError(
            f"Integral {intor_name} is not supported."
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
    coordinates live in an appended env block. The fake ``[0, nao_fake)`` and
    real ``[0, nao)`` function spaces are independent (one indexes the rows of
    a cross block, the other its columns) and the fake one is split over
    :class:`~pyscfad.experimental.moleintor_cuint.CrossChunk` s.

    All structure is built from ``bas_conc``; the env pointer
    columns are filled from the actual (possibly traced) ``bas`` by
    :meth:`make_rows`.
    """
    def __init__(self, bas_conc, natm, nenv, budget=None):
        bas_conc = numpy.asarray(bas_conc)
        nbas = len(bas_conc)
        ls = bas_conc[:, ANG_OF]
        nprims = bas_conc[:, NPRIM_OF]
        nctrs = bas_conc[:, NCTR_OF]
        nls = 2 * ls + 1

        ao_loc = numpy.append(0, numpy.cumsum(nls * nctrs)).astype(numpy.int32)
        fake_loc = numpy.append(0, numpy.cumsum(nls * nctrs * nprims)).astype(numpy.int32)
        nao = int(ao_loc[-1])
        nao_fake = int(fake_loc[-1])

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
            iatm = int(bas_conc[i, ATOM_OF])
            for k in range(nctr):
                for j in range(nprim):
                    fake_rows.append([iatm, l, 1, 1, 0, 0, ptr_ones, 0])
                    real_rows.append([iatm, l, 1, 1, 0, 0, 0, 0])
                    f0 = fake_loc[i] + (k * nprim + j) * nl
                    fake_fn.append(f0)
                    real_fn.append(ao_loc[i] + k * nl)
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
        real_fn = numpy.asarray(real_fn, dtype=numpy.int32)
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

        # the primal ordered pair set, as (la, lb, p, q, group suffix); the
        # encodings are built per chunk below
        base_groups = []
        for la in uls:
            a0, a1 = lbounds[int(la)]
            ra = order[a0:a1]
            # same-l: strict upper triangle in the sorted positions
            iu, ju = numpy.triu_indices(len(ra), k=1)
            base_groups.append((la, la, ra[iu], ra[ju], "off"))
            # diagonal pairs
            base_groups.append((la, la, ra, ra, "diag"))
            for lb in uls:
                if lb <= la:
                    continue
                b0, b1 = lbounds[int(lb)]
                rb = order[b0:b1]
                pp, qq = numpy.meshgrid(ra, rb, indexing="ij")
                base_groups.append((la, lb, pp.ravel(), qq.ravel(), "off"))

        def _pair_info(la, lb, enc):
            return PairInfo(
                li=numpy.int32(la),
                lj=numpy.int32(lb),
                pair_indices=numpy.asarray(enc, dtype=numpy.int32),
                n_pairs=numpy.int32(len(enc)),
            )

        nl_per_prim = 2 * fake_rows[:, ANG_OF] + 1
        chunks = []
        if budget is None:
            budget = chunk_budget(nao, nl_per_prim.max())
        for r0, r1, fn0, n_fn in _fake_chunks(nl_per_prim, budget):
            chunk_prim2fn = numpy.zeros(n_rows, dtype=numpy.int32)
            fake_off = numpy.asarray(fake_fn, dtype=numpy.int32)[r0:r1] - fn0
            chunk_prim2fn[r0:r1] = fake_off               # fake bra copy
            chunk_prim2fn[2*npr+r0:2*npr+r1] = fake_off   # fake ket copy
            chunk_prim2fn[npr:2*npr] = real_fn            # real bra copy
            chunk_prim2fn[3*npr:] = real_fn               # real ket copy
            pairs = {"bra_off": [], "bra_diag": [],
                     "ket_off": [], "ket_diag": []}
            for la, lb, p, q, kind in base_groups:
                # bra direction (fake_bra_p, real_ket_q): chunk on the bra
                m = (p >= r0) & (p < r1)
                if m.any():
                    enc = p[m] * n_rows + (3 * npr + q[m])
                    pairs["bra_" + kind].append(_pair_info(la, lb, enc))
                # ket direction (real_bra_p, fake_ket_q): chunk on the ket
                m = (q >= r0) & (q < r1)
                if m.any():
                    enc = (npr + p[m]) * n_rows + (2 * npr + q[m])
                    pairs["ket_" + kind].append(_pair_info(la, lb, enc))
            chunks.append(CrossChunk(
                fn_start=fn0, n_fn=n_fn,
                n_functions=numpy.int32(max(n_fn, nao)),
                primitive_to_function=chunk_prim2fn,
                pairs={k: tuple(v) for k, v in pairs.items()},
            ))

        self.rows_static = rows
        self.chunks = chunks
        self.n_primitives = n_rows
        self.nao_fake = nao_fake
        self.nao = nao
        self.npr = npr
        self.natm = int(natm)
        self.ptr_coords2 = ptr_coords2

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


def _get_lat_basis_cross_plan(bas_conc, natm, nenv, budget=None):
    key = (bas_conc.tobytes(), bas_conc.shape, int(natm), int(nenv), budget)
    plan = _LAT_BASIS_CROSS_PLAN_CACHE.get(key)
    if plan is None:
        plan = LatBasisCrossPlan(bas_conc, natm, nenv, budget)
        _LAT_BASIS_CROSS_PLAN_CACHE[key] = plan
    return plan


def _lat_basis_cross_setup(intor_name, Ls, atm, bas, env, cuint_plan, r0):
    """The plan, per-call ``rows`` and the doubled system (the ket atom copy
    displaced by L, its coordinates in an appended env block) that the two
    lattice basis tangents share.

    All cross integrals run per image through the kernels' native
    configuration batching, on the **stopped** primal ``env``: the tangent is
    first order in the basis parameters, so a mixed coordinate/basis second
    derivative has to differentiate the coordinates first, e.g.
    ``jacfwd(grad(f, coords), basis)``.
    """
    n_deriv = _basis_deriv_order(intor_name, backend="cuint lattice")
    bas_conc = _plan_bas_concrete(cuint_plan, bas)
    natm = atm.shape[0]
    nenv = env.shape[-1]
    Ls = Ls.reshape(-1, 3)
    nL = Ls.shape[0]
    plan = _get_lat_basis_cross_plan(bas_conc, natm, nenv)
    rows = plan.make_rows(bas)

    atm2 = np.concatenate([np.asarray(atm, dtype=np.int32)] * 2, axis=0)
    ptr2 = plan.ptr_coords2 + 3 * np.arange(natm, dtype=np.int32)
    atm2 = ops.index_update(atm2, ops.index[natm:, PTR_COORD], ptr2)

    env = np.asarray(env, dtype=np.float64)
    coords = np.asarray(r0, dtype=np.float64).reshape(-1, 3)
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
    return n_deriv, plan, rows, atm2, env2


def _lat_weighted_blocks(atm2, env2, plan, rows, n_deriv, lap):
    """Bra- and ket-direction cross blocks summed over the pair groups with
    their weights.

    The integral's own ``n_deriv`` derivatives always sit on the bra, which is
    the fake shell in the bra-direction blocks and the real shell in the
    ket-direction ones. ``lap`` applies the Laplacian of the exponent
    identity to the fake shell (already traced).

    The primal halves the diagonal primitive pairs (cuint's ``OVLP_SPELL``),
    so their cross terms enter with weight 1/2 as well; folding that in here
    rather than after the contraction keeps the (tangent-batched) contraction
    outputs down to one per direction.
    """
    def _blocks(kind):
        xb = _cross_blocks(atm2, env2, plan, rows, n_deriv, lap,
                           group="bra_" + kind)
        xk = _cross_blocks(atm2, env2, plan, rows, n_deriv, lap,
                           group="ket_" + kind, transpose=True)
        return xb, xk

    xb_off, xk_off = _blocks("off")
    xb_diag, xk_diag = _blocks("diag")
    return xb_off + 0.5 * xb_diag, xk_off + 0.5 * xk_diag


def _lat_scatter(plan, w):
    """The primitive-to-contracted scatter matrix of one weight vector."""
    t = np.zeros((plan.nao, plan.nao_fake), dtype=np.float64)
    return ops.index_add(t, ops.index[plan.map_real_rows, plan.map_fake_rows], w)


def _lat_contract(t, x_bra, x_ket, Ls_mask):
    """Both cross directions contracted and masked to the live images."""
    jvp = (np.einsum("ma,lcav->lcmv", t, x_bra)
           + np.einsum("lcma,na->lcmn", x_ket, t))
    Ls_mask = np.asarray(Ls_mask).reshape(-1)
    return np.where(Ls_mask[:, None, None, None] != 0, jvp,
                    np.zeros((), dtype=jvp.dtype))


def _gen_int1e_jvp_cs(
    intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
    r0, ctr_coeff, ctr_coeff_dot,
):
    """Contraction-coefficient tangent of the per-image lattice integrals on
    the cuint backend.

    The integrals are linear in the contraction coefficients, so the tangent
    is the primitive cross block itself.
    """
    n_deriv, plan, rows, atm2, env2 = _lat_basis_cross_setup(
        intor_name, Ls, atm, bas, env, cuint_plan, r0)

    ptr_coeff0 = env.shape[-1] - ctr_coeff.shape[-1]
    coeff_env_idx = bas[:, PTR_COEFF][plan.map_entry_shell] + plan.map_coeff_off
    t = _lat_scatter(plan, ctr_coeff_dot[coeff_env_idx - ptr_coeff0])

    x_bra, x_ket = _lat_weighted_blocks(atm2, env2, plan, rows, n_deriv, 0)
    return _lat_contract(t, x_bra, x_ket, Ls_mask)


def _gen_int1e_jvp_exp(
    intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
    r0, exp, ctr_coeff, exp_dot,
):
    """Exponent tangent of the per-image lattice integrals on the cuint
    backend.

    ``d/da exp(-a r_A^2)`` brings down ``-r_A^2``, which the solid-harmonic
    identity ``r_A^2 chi = [lap_A chi + 2 a (2l+3) chi] / (4 a^2)`` turns
    into a Laplacian of the same shell -- two extra derivative slots on the
    fake side, on top of the integral's own bra derivative.
    """
    n_deriv, plan, rows, atm2, env2 = _lat_basis_cross_setup(
        intor_name, Ls, atm, bas, env, cuint_plan, r0)

    ptr_coeff0 = env.shape[-1] - ctr_coeff.shape[-1]
    ptr_exp0 = ptr_coeff0 - exp.shape[-1]
    ptr_exp_col = bas[:, PTR_EXP]
    coeff_env_idx = bas[:, PTR_COEFF][plan.map_entry_shell] + plan.map_coeff_off
    exp_env_idx = ptr_exp_col[plan.map_entry_shell] + plan.map_prim_off
    c = ops.stop_gradient(ctr_coeff[coeff_env_idx - ptr_coeff0])
    t = _lat_scatter(plan, c * exp_dot[exp_env_idx - ptr_exp0])

    alpha_env_idx = ptr_exp_col[plan.fakefn_shell] + plan.fakefn_prim
    alpha = ops.stop_gradient(env[alpha_env_idx])
    afac = 2.0 * (2 * plan.l_fake_fn + 3) * alpha
    scale = 1.0 / (4.0 * alpha ** 2)

    x0_bra, x0_ket = _lat_weighted_blocks(atm2, env2, plan, rows, n_deriv, 0)
    tr_bra, tr_ket = _lat_weighted_blocks(atm2, env2, plan, rows, n_deriv, 1)
    x_bra = -(tr_bra + afac[None, None, :, None] * x0_bra) \
        * scale[None, None, :, None]
    x_ket = -(tr_ket + x0_ket * afac[None, None, None, :]) \
        * scale[None, None, None, :]

    return _lat_contract(t, x_bra, x_ket, Ls_mask)


def _lattice_intor_jvp(
    intor_name, Ls_mask, atm, bas, env, cuint_plan,
    shls_slice, comp, hermi, ao_loc,
    primals, tangents,
):
    assert hermi == 1

    Ls, r0, exp, ctr_coeff = primals
    Ls_dot, r0_dot, exp_dot, ctr_coeff_dot = tangents

    primal_out = _lattice_intor(
        intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
        r0, exp, ctr_coeff,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
    )

    tangent_out = np.zeros_like(primal_out)

    # the bra-derivative integral drives both geometry tangents; going
    # through _lattice_intor (rather than calling the kernel directly) keeps
    # it differentiable w.r.t. the basis-set parameters, which is what makes
    # mixed coordinate/basis derivatives available
    need_ip = (not isinstance(r0_dot, SymbolicZero)
               or not isinstance(Ls_dot, SymbolicZero))
    if need_ip:
        intor_ip_bra = int1e_dr1_name(intor_name)[0]
        s1a = -_lattice_intor(
            intor_ip_bra, Ls, Ls_mask, atm, bas, env, cuint_plan,
            r0, exp, ctr_coeff,
            shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        )

    if not isinstance(r0_dot, SymbolicZero):
        s1a_x = s1a.transpose(1,0,2,3)

        # 'bas' is traced on the padded path; the concrete shell structure
        # comes from the plan, which recorded it at creation
        bas_conc = _plan_bas_concrete(cuint_plan, bas)

        if shls_slice is None:
            nbas = len(bas_conc)
            shls_slice = (0, nbas, 0, nbas)
        if ao_loc is None:
            _ao_loc = make_loc(bas_conc, intor_name)
        else:
            _ao_loc = ao_loc

        i0, i1, j0, j1 = shls_slice[:4]
        aoslices_bra = aoslices_in_range(bas_conc, _ao_loc, len(atm), (i0, i1))
        if (j0, j1) == (i0, i1):
            aoslices_ket = aoslices_bra
        else:
            aoslices_ket = aoslices_in_range(bas_conc, _ao_loc, len(atm),
                                             (j0, j1))

        naoi, naoj = s1a_x.shape[-2:]

        aoidx = np.arange(naoi)
        jvp = _gen_int1e_fill_jvp_r0(s1a_x, r0_dot, aoslices_bra,
                                     aoidx[None,None,:,None])

        aoidx = np.arange(naoj)
        jvp += _gen_int1e_fill_jvp_r0(-s1a_x, r0_dot, aoslices_ket,
                                      aoidx[None,None,None,:])

        tangent_out += jvp.reshape(tangent_out.shape)

    if not isinstance(exp_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_exp(
            intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
            r0, exp, ctr_coeff, exp_dot,
        ).reshape(tangent_out.shape)

    if not isinstance(ctr_coeff_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_cs(
            intor_name, Ls, Ls_mask, atm, bas, env, cuint_plan,
            r0, ctr_coeff, ctr_coeff_dot,
        ).reshape(tangent_out.shape)

    if not isinstance(Ls_dot, SymbolicZero):
        # Every ket function in image L is displaced rigidly by L, so
        # dS_L/dL is the ket-center derivative summed over all ket centers.
        # By pair translation invariance this equals minus the bra
        # derivative that the deriv=1 kernel provides.
        tangent_out += np.einsum("lxpq,lx->lpq", -s1a, Ls_dot).reshape(
            tangent_out.shape
        )
    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
