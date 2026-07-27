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
Shared helpers for basis-set parameter (exponent and contraction
coefficient) derivatives of one-electron integrals in the array-level
integral paths (:mod:`pyscfad.gto.moleintor_lite` and friends).

The derivatives are formulated as cross integrals between a "fake" basis
with one uncontracted shell per primitive Gaussian and the original basis,
followed by a scatter contraction that is linear in the ``env`` tangents:

- Contraction coefficients: integrals are linear in the coefficients, so
  ``dI/dc`` is the cross integral of the corresponding primitive.
- Exponents: ``d/da exp(-a r^2)`` brings down ``-r^2 = -(x^2+y^2+z^2)``,
  realized by raising the fake-shell angular momentum by two and summing
  the promoted Cartesian components (evaluated in Cartesian, transformed
  back to spherical at the end).

The machinery separates the **static structure** of the basis (angular
momenta, numbers of primitives and contractions per shell) from the
**env pointers** (``PTR_EXP``/``PTR_COEFF``):

- The structure is taken from the static ``basis_array_metadata`` if given
  (for :class:`~pyscfad.ml.gto.MolePad` the ``ls``/``nprim``/``nctr`` of the
  :class:`~pyscfad.ml.gto.BasisArray`, every atom carrying the
  same shell template) or, when ``bas`` is concrete
  (:class:`~pyscfad.gto.MoleLite`), from ``bas`` itself. Shapes,
  scatter row maps and pair structures are built from it with numpy at
  trace time.
- The env pointers are gathered from ``bas`` with array ops, so ``bas``
  may be a traced array (e.g. under ``jit`` or ``vmap`` over atomic
  numbers in ML training).

The tangent computation stays linear in ``env_dot`` with the integral
evaluations applied to the primal ``env`` only, so reverse-mode
differentiation (transposition) works.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy
import scipy.linalg

from pyscf.gto.mole import (
    ATOM_OF,
    ANG_OF,
    NPRIM_OF,
    NCTR_OF,
    PTR_EXP,
    PTR_COEFF,
    cart2sph,
)

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto._pyscf_moleintor import make_loc
from pyscfad.gto._moleintor_helper import index_prompt_xyz

if TYPE_CHECKING:
    from collections.abc import Callable
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

# libcint common normalization factors for s and p orbitals
_S_NORM = 0.282094791773878143
_P_NORM = 0.488602511902919921


def next_coord_deriv(
    max_coord_deriv: int | None,
    trace_coords: bool,
) -> tuple[int | None, bool]:
    """Budget for the integrals *inside* a coordinate-tangent term.

    A coordinate JVP term is one order in the nuclear coordinates (or lattice
    shifts); the integrals it is built from are differentiated again only for
    second- and higher-order geometry derivatives. ``max_coord_deriv`` is the
    highest such order the caller will take (``None``: no limit, the default).
    With ``max_coord_deriv=1`` -- forces and stress, which may still be
    differentiated with respect to basis-set parameters -- the nested integrals
    stop tracing coordinates, so the pure (R,R) blocks (``int1e_ovlp_dr20``,
    ``dr11`` and the lattice analogues) are never requested, while the mixed
    coordinate/basis blocks are unaffected. That saves work on every step and
    is what lets the GPU backends, which implement only the first coordinate
    derivative, take basis-parameter gradients of forces.

    Returns ``(budget for the nested call, whether it traces coordinates)``.
    """
    if max_coord_deriv is None:
        return None, trace_coords
    remaining = int(max_coord_deriv) - 1
    return remaining, trace_coords and remaining > 0


def _concrete_bas(bas) -> numpy.ndarray:
    try:
        return numpy.asarray(bas)
    except Exception as exc:
        raise NotImplementedError(
            "Basis parameter derivatives require the static basis "
            "structure. When 'bas' is traced (e.g. under jit), "
            "pass the static basis metadata."
        ) from exc


def _resolve_bas_concrete(bas, basis_array_metadata=None) -> numpy.ndarray:
    """The concrete structural template: tiled from the static
    ``basis_array_metadata`` if given (``ls`` per shell of one atom and
    ``nprim``/``nctr``; every atom carries the same shell template, and
    the number of atoms follows from the static shape of ``bas``), else
    a concrete ``bas``. Only the ``ANG_OF``/``NPRIM_OF``/``NCTR_OF``
    (and ``ATOM_OF``) columns of ``bas_conc`` are used;
    env pointers are always gathered from ``bas``.
    """
    if basis_array_metadata is not None:
        meta = basis_array_metadata
        ls = numpy.asarray(meta.ls, dtype=numpy.int32)
        natm = bas.shape[0] // ls.size
        bas_conc = numpy.zeros(bas.shape, dtype=numpy.int32)
        bas_conc[:, ATOM_OF] = numpy.repeat(numpy.arange(natm), ls.size)
        bas_conc[:, ANG_OF] = numpy.tile(ls, natm)
        bas_conc[:, NPRIM_OF] = numpy.int32(meta.nprim)
        bas_conc[:, NCTR_OF] = numpy.int32(meta.nctr)
        return bas_conc
    return _concrete_bas(bas)


def make_fake_bas(
    bas_conc: numpy.ndarray,
    ptr_ones: int,
    order: int = 0,
) -> numpy.ndarray:
    nprim = bas_conc[:, NPRIM_OF]
    nbas_fake = numpy.sum(nprim)

    fake_bas = numpy.zeros((nbas_fake, bas_conc.shape[1]), dtype=numpy.int32)
    fake_bas[:, ATOM_OF] = numpy.repeat(bas_conc[:, ATOM_OF], nprim)
    fake_bas[:, ANG_OF] = numpy.repeat(bas_conc[:, ANG_OF], nprim) + order
    fake_bas[:, NPRIM_OF] = 1
    fake_bas[:, NCTR_OF] = 1
    fake_bas[:, PTR_COEFF] = ptr_ones
    return fake_bas


def fake_prim_maps(bas_conc: numpy.ndarray):
    """Per fake-shell (shell index, primitive index) of ``bas_conc``."""
    nprim = bas_conc[:, NPRIM_OF]
    nbas_fake = numpy.sum(nprim)
    nbas = len(bas_conc)
    prim_shell = numpy.repeat(numpy.arange(nbas), nprim)
    prim_off = numpy.arange(nbas_fake) - numpy.repeat(
        numpy.cumsum(nprim) - nprim, nprim)
    return prim_shell, prim_off


def fake_shl_loc(bas_conc: numpy.ndarray) -> numpy.ndarray:
    """First fake shell of every real shell. The fake basis holds one shell
    per primitive in shell order, so the fake shells of a real shell range
    ``[sh0, sh1)`` are the contiguous range
    ``[fake_shl_loc[sh0], fake_shl_loc[sh1])``.
    """
    return numpy.append(0, numpy.cumsum(bas_conc[:, NPRIM_OF]))


def _resolve_shls_slice(shls_slice, nbas: int, hermi: int):
    """Bra/ket shell ranges of the requested integral block."""
    if shls_slice is None:
        return 0, nbas, 0, nbas
    i0, i1, j0, j1 = (int(x) for x in shls_slice[:4])
    if hermi == 1 and (i0, i1) != (j0, j1):
        raise NotImplementedError(
            "Basis parameter derivatives with hermi = 1 require a diagonal "
            f"shell block, got shls_slice = {tuple(shls_slice[:4])}."
        )
    return i0, i1, j0, j1


def make_fake_basc(
    bas_conc: numpy.ndarray,
    bas: ArrayLike,
    ptr_ones: int,
    order: int = 0,
):
    fake_bas = make_fake_bas(bas_conc, ptr_ones, order=order)
    basc_conc = numpy.vstack([fake_bas, bas_conc]).astype(numpy.int32)

    prim_shell, prim_off = fake_prim_maps(bas_conc)
    ptr_exp_fake = bas[:, PTR_EXP][prim_shell] + prim_off
    if isinstance(bas, numpy.ndarray):
        nbas_fake = len(fake_bas)
        basc = basc_conc.copy()
        basc[:nbas_fake, PTR_EXP] = ptr_exp_fake
        basc[nbas_fake:] = bas
    else:
        fake_bas = np.asarray(fake_bas)
        fake_bas = ops.index_update(fake_bas, ops.index[:, PTR_EXP],
                                    np.asarray(ptr_exp_fake, dtype=np.int32))
        basc = np.vstack([fake_bas, np.asarray(bas, dtype=np.int32)])
    return basc, basc_conc


class CsMaps(NamedTuple):
    fake_rows: numpy.ndarray
    real_rows: numpy.ndarray
    entry_shell: numpy.ndarray
    coeff_off: numpy.ndarray
    nao_fake: int
    nao: int


class ExpMaps(NamedTuple):
    fake_rows: numpy.ndarray
    real_rows: numpy.ndarray
    entry_shell: numpy.ndarray
    coeff_off: numpy.ndarray
    prim_off: numpy.ndarray
    norm_fac: numpy.ndarray
    nao_fake: int
    nao: int


def _nf(ls, cart):
    if cart:
        return (ls + 1) * (ls + 2) // 2
    else:
        return 2 * ls + 1


def _hstack_int(arrays):
    """:func:`numpy.hstack` tolerating an empty shell range."""
    if not arrays:
        return numpy.zeros(0, dtype=numpy.int64)
    return numpy.hstack(arrays)


def cs_scatter_maps(
    bas_conc: numpy.ndarray,
    cart: bool,
    shl_range: tuple[int, int] | None = None,
) -> CsMaps:
    """Static index maps scattering primitive cross-integral rows into
    the coefficient-tangent contraction.

    One entry per (shell i, contraction k, primitive j, function m):
    ``T[real_rows, fake_rows] += env_dot[ptr_coeff[entry_shell] + coeff_off]``.

    ``shl_range`` restricts the entries to the shells ``[sh0, sh1)``; the
    row indices are then relative to the first (fake and real) AO of that
    shell range, while ``entry_shell`` stays an absolute shell index.
    """
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    ls = bas_conc[sh0:sh1, ANG_OF]
    nprims = bas_conc[sh0:sh1, NPRIM_OF]
    nctrs = bas_conc[sh0:sh1, NCTR_OF]
    nfs = _nf(ls, cart)

    fake_offs = numpy.append(0, numpy.cumsum(nprims * nfs))
    real_offs = numpy.append(0, numpy.cumsum(nctrs * nfs))

    fake_rows = []
    real_rows = []
    entry_shell = []
    coeff_off = []
    for i in range(sh1 - sh0):
        nf, nprim, nctr = nfs[i], nprims[i], nctrs[i]
        k, j, m = numpy.mgrid[0:nctr, 0:nprim, 0:nf]
        fake_rows.append((fake_offs[i] + j * nf + m).ravel())
        real_rows.append((real_offs[i] + k * nf + m).ravel())
        entry_shell.append(numpy.full(nctr * nprim * nf, sh0 + i))
        coeff_off.append((k * nprim + j).ravel())

    return CsMaps(
        fake_rows=_hstack_int(fake_rows),
        real_rows=_hstack_int(real_rows),
        entry_shell=_hstack_int(entry_shell),
        coeff_off=_hstack_int(coeff_off),
        nao_fake=int(fake_offs[-1]),
        nao=int(real_offs[-1]),
    )


def exp_scatter_maps(
    bas_conc: numpy.ndarray,
    shl_range: tuple[int, int] | None = None,
) -> ExpMaps:
    """Static index maps for the exponent tangent (Cartesian only).

    One entry per (shell i, contraction k, primitive j, function m,
    promotion direction d in {x, y, z}); the fake rows address the
    ``l+2`` Cartesian functions promoted by ``x^2``/``y^2``/``z^2``
    (:func:`index_prompt_xyz`), and the weight is
    ``-norm_fac * env[coeff] * env_dot[exp]``.

    ``shl_range`` restricts the entries to the shells ``[sh0, sh1)``, as
    in :func:`cs_scatter_maps`.
    """
    bas_conc = numpy.asarray(bas_conc)
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    ls = bas_conc[sh0:sh1, ANG_OF]
    nprims = bas_conc[sh0:sh1, NPRIM_OF]
    nctrs = bas_conc[sh0:sh1, NCTR_OF]
    nfs = _nf(ls, True)
    nf2s = _nf(ls + 2, True)

    fake_offs = numpy.append(0, numpy.cumsum(nprims * nf2s))
    real_offs = numpy.append(0, numpy.cumsum(nctrs * nfs))

    fake_rows = []
    real_rows = []
    entry_shell = []
    coeff_off = []
    prim_off = []
    norm_fac = []
    for i in range(sh1 - sh0):
        l = int(ls[i])
        nf, nf2 = int(nfs[i]), int(nf2s[i])
        nprim, nctr = int(nprims[i]), int(nctrs[i])
        promoted = numpy.asarray(index_prompt_xyz(l, 2))  # (3, nf)
        if l == 0:
            fac = _S_NORM
        elif l == 1:
            fac = _P_NORM
        else:
            fac = 1.0

        k, j, d, m = numpy.mgrid[0:nctr, 0:nprim, 0:3, 0:nf]
        fake_rows.append((fake_offs[i] + j * nf2 + promoted[d, m]).ravel())
        real_rows.append((real_offs[i] + k * nf + m).ravel())
        entry_shell.append(numpy.full(nctr * nprim * 3 * nf, sh0 + i))
        coeff_off.append((k * nprim + j).ravel())
        prim_off.append((j + 0 * k).ravel())
        norm_fac.append(numpy.full(nctr * nprim * 3 * nf, fac))

    return ExpMaps(
        fake_rows=_hstack_int(fake_rows),
        real_rows=_hstack_int(real_rows),
        entry_shell=_hstack_int(entry_shell),
        coeff_off=_hstack_int(coeff_off),
        prim_off=_hstack_int(prim_off),
        norm_fac=numpy.concatenate(norm_fac) if norm_fac else numpy.zeros(0),
        nao_fake=int(fake_offs[-1]),
        nao=int(real_offs[-1]),
    )


def cart2sph_mat(
    bas_conc: numpy.ndarray,
    shl_range: tuple[int, int] | None = None,
) -> numpy.ndarray:
    """Static Cartesian-to-spherical transformation matrix
    (``(nao_cart, nao_sph)``) of the shells ``[sh0, sh1)``, equivalent to
    :meth:`pyscf.gto.MoleBase.cart2sph_coeff` but built from ``bas_conc``
    alone.
    """
    bas_conc = numpy.asarray(bas_conc)
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    blocks = []
    for i in range(sh0, sh1):
        c = cart2sph(int(bas_conc[i, ANG_OF]), normalized="sp")
        blocks.extend([c] * int(bas_conc[i, NCTR_OF]))
    if not blocks:
        return numpy.zeros((0, 0))
    return scipy.linalg.block_diag(*blocks)


def basis_jvp_cs(
    intor_cross: Callable,
    bas: ArrayLike,
    env: Array,
    env_dot: Array,
    cart: bool,
    hermi: int,
    shls_slice: tuple[int, ...] | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # used for padding
) -> Array:
    """Contraction coefficient contribution to the ``env`` tangent.

    Parameters:
        intor_cross: ``intor_cross(basc, envc, shls_slice, ao_loc)``
            evaluating the cross integrals on the primal ``envc``,
            returning an array of shape ``(..., nrow, ncol)`` (leading
            dims: integral components and/or lattice images / k-points).
            It must not symmetrize (``hermi = 0``), the cross blocks
            being rectangular.
        hermi: 1 computes the bra term and adds its conjugate transpose;
            0 computes the ket cross block explicitly.
        shls_slice: Bra/ket shell ranges ``(i0, i1, j0, j1)`` of the
            requested integral block (the full matrix if None); ``hermi = 1``
            requires a diagonal block.

    Returns:
        Tangent contribution of shape ``(..., naoi, naoj)``.
    """
    assert hermi in (0, 1), f"hermi={hermi} not supported"

    nbas = len(bas)
    bas_conc = _resolve_bas_concrete(bas, basis_array_metadata)
    i0, i1, j0, j1 = _resolve_shls_slice(shls_slice, nbas, hermi)
    ptr_ones = env.shape[-1]
    basc, basc_conc = make_fake_basc(bas_conc, bas, ptr_ones)
    nbas_fake = len(basc) - nbas
    envc = np.concatenate([env, np.ones(1, dtype=env.dtype)])
    ao_loc = make_loc(basc_conc, "cart" if cart else "sph")
    fake_loc = fake_shl_loc(bas_conc).tolist()

    def contract_mat(sh0, sh1):
        """``d(AO of shells [sh0, sh1))/dc`` contracted with ``env_dot``."""
        maps = cs_scatter_maps(bas_conc, cart, (sh0, sh1))
        coeff_env_idx = bas[:, PTR_COEFF][maps.entry_shell] + maps.coeff_off
        w = env_dot[coeff_env_idx]
        t = np.zeros((maps.nao, maps.nao_fake), dtype=env.dtype)
        return ops.index_add(t, ops.index[maps.real_rows, maps.fake_rows], w)

    t_bra = contract_mat(i0, i1)
    s_bra = intor_cross(basc, envc,
                        (fake_loc[i0], fake_loc[i1],
                         nbas_fake + j0, nbas_fake + j1), ao_loc)
    jvp = np.einsum("mp,...pn->...mn", t_bra, s_bra)
    if hermi == 1:
        jvp += np.swapaxes(jvp, -1, -2).conj()
    elif hermi == 0:
        t_ket = t_bra if (j0, j1) == (i0, i1) else contract_mat(j0, j1)
        s_ket = intor_cross(basc, envc,
                            (nbas_fake + i0, nbas_fake + i1,
                             fake_loc[j0], fake_loc[j1]), ao_loc)
        jvp += np.einsum("...mp,np->...mn", s_ket, t_ket)
    return jvp


def basis_jvp_exp(
    intor_cross: Callable,
    bas: ArrayLike,
    env: Array,
    env_dot: Array,
    need_c2s: bool,
    hermi: int,
    shls_slice: tuple[int, ...] | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # used for padding
) -> Array:
    """Exponent part of the ``env`` tangent.

    ``intor_cross`` is as in :func:`basis_jvp_cs`, but must evaluate the
    **Cartesian** variant of the integral (the fake shells carry ``l+2``);
    when ``need_c2s`` is True the result is transformed back to spherical
    on both AO indices. ``shls_slice`` selects the integral block as in
    :func:`basis_jvp_cs`.
    """
    assert hermi in (0, 1), f"hermi={hermi} not supported"

    nbas = len(bas)
    bas_conc = _resolve_bas_concrete(bas, basis_array_metadata)
    i0, i1, j0, j1 = _resolve_shls_slice(shls_slice, nbas, hermi)
    ptr_ones = env.shape[-1]
    basc, basc_conc = make_fake_basc(bas_conc, bas, ptr_ones, order=2)
    nbas_fake = len(basc) - nbas
    envc = np.concatenate([env, np.ones(1, dtype=env.dtype)])
    ao_loc = make_loc(basc_conc, "cart")
    fake_loc = fake_shl_loc(bas_conc).tolist()

    def contract_mat(sh0, sh1):
        """``d(Cartesian AO of shells [sh0, sh1))/da`` times ``env_dot``."""
        maps = exp_scatter_maps(bas_conc, (sh0, sh1))
        coeff_env_idx = bas[:, PTR_COEFF][maps.entry_shell] + maps.coeff_off
        exp_env_idx = bas[:, PTR_EXP][maps.entry_shell] + maps.prim_off
        c = env[coeff_env_idx]
        w = -(maps.norm_fac * c) * env_dot[exp_env_idx]
        t = np.zeros((maps.nao, maps.nao_fake), dtype=env.dtype)
        return ops.index_add(t, ops.index[maps.real_rows, maps.fake_rows], w)

    t_bra = contract_mat(i0, i1)
    s_bra = intor_cross(basc, envc,
                       (fake_loc[i0], fake_loc[i1],
                        nbas_fake + j0, nbas_fake + j1), ao_loc)
    jvp = np.einsum("ma,...av->...mv", t_bra, s_bra)
    if hermi == 1:
        jvp = jvp + np.conj(np.swapaxes(jvp, -1, -2))
    elif hermi == 0:
        t_ket = t_bra if (j0, j1) == (i0, i1) else contract_mat(j0, j1)
        s_ket = intor_cross(basc, envc,
                           (nbas_fake + i0, nbas_fake + i1,
                            fake_loc[j0], fake_loc[j1]), ao_loc)
        jvp = jvp + np.einsum("...ma,na->...mn", s_ket, t_ket)

    if need_c2s:
        c2s_bra = np.asarray(cart2sph_mat(bas_conc, (i0, i1)), dtype=env.dtype)
        c2s_ket = (c2s_bra if (j0, j1) == (i0, i1) else
                   np.asarray(cart2sph_mat(bas_conc, (j0, j1)), dtype=env.dtype))
        jvp = np.einsum("pi,...pq,qj->...ij", c2s_bra, jvp, c2s_ket)
    return jvp
