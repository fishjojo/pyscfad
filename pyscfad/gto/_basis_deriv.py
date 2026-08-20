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
Helpers for basis-set parameter (exponent and contraction
coefficient) derivatives of one-electron integrals for
:class:`~pyscfad.gto.MoleLite`.

The derivatives are formulated as cross integrals between a "fake" basis
with primitive Gaussians and the original basis,
followed by a scatter contraction that is linear in the ``env`` tangents:

- Contraction coefficients: integrals are linear in the coefficients, so
  ``dI/dc`` is the cross integral of the corresponding primitive.
- Exponents: ``d/da exp(-a r^2)`` brings down ``-r^2 = -(x^2+y^2+z^2)``,
  realized by raising the fake-shell angular momentum by two and summing
  the promoted Cartesian components. This has to be evaluated in
  Cartesian: ``r^2`` times a solid harmonic of degree ``l`` is not a
  solid harmonic of degree ``l+2``.
  The Cartesian-to-spherical transformation of the
  differentiated index is folded into the scatter weights;
  the other index is transformed afterwards.

The machinery separates the **static structure** of the basis (angular
momenta, numbers of primitives and contractions per shell) from the
**env pointers** (``PTR_EXP``/``PTR_COEFF``):

- The structure is taken from the static ``basis_array_metadata`` if given
  (for :class:`~pyscfad.ml.gto.MolePad`) or, when ``bas`` is concrete
  (:class:`~pyscfad.gto.MoleLite`), from ``bas`` itself. Shapes,
  scatter row maps and pair structures are built from it with numpy at
  trace time.
- The ``env`` pointers are gathered from ``bas`` with array ops,
  so ``bas`` can be a traced array.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy

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
from pyscfad.gto._moleintor_helper import (
    index_prompt_xyz,
    resolve_bas_concrete,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

# libcint common normalization factors for s and p orbitals
_S_NORM = 0.282094791773878143
_P_NORM = 0.488602511902919921


def make_primitive_bas(
    bas_conc: numpy.ndarray,
    ptr_ones: int, # pointer to the contraction coefficient (1.0)
    order: int = 0, # raise angular momentum by order
) -> numpy.ndarray:
    nprim = bas_conc[:,NPRIM_OF]
    nbas_prim = numpy.sum(nprim) # each primitive is a shell

    prim_bas = numpy.zeros((nbas_prim, bas_conc.shape[1]), dtype=numpy.int32)
    prim_bas[:,ATOM_OF] = numpy.repeat(bas_conc[:,ATOM_OF], nprim)
    prim_bas[:,ANG_OF] = numpy.repeat(bas_conc[:,ANG_OF], nprim) + order
    prim_bas[:,NPRIM_OF] = 1
    prim_bas[:,NCTR_OF] = 1
    prim_bas[:,PTR_COEFF] = ptr_ones
    return prim_bas


def primitive_bas_to_bas_maps(
    bas_conc: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Per primitive-shell (shell index, primitive index) of ``bas``.

    Returns:
        prim_to_bas: primitive shell to contracted shell mapping.
        prim_off: offests of primitive shells in each contracted shell.
    """
    nprim = bas_conc[:,NPRIM_OF]
    nbas_prim = numpy.sum(nprim)
    nbas = len(bas_conc)
    prim_to_bas = numpy.repeat(numpy.arange(nbas), nprim)
    prim_off = numpy.arange(nbas_prim) - numpy.repeat(
        numpy.cumsum(nprim) - nprim, nprim)
    return prim_to_bas, prim_off


def primitive_shell_loc(bas_conc: numpy.ndarray) -> numpy.ndarray:
    """First primitive shell of every real shell.

    The primitive shells of a real shell range
    ``[sh0, sh1)`` are the contiguous range
    ``[primitive_shell_loc[sh0], primitive_shell_loc[sh1])``.
    """
    return numpy.append(0, numpy.cumsum(bas_conc[:,NPRIM_OF]))


def _resolve_shls_slice(
    shls_slice: tuple[int, ...],
    nbas: int,
    hermi: int
) -> tuple[int, ...]:
    """Bra/ket shell ranges of the integral block."""
    if shls_slice is None:
        return (0, nbas, 0, nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    if hermi == 1:
        assert (i0, i1) == (j0, j1)
    return i0, i1, j0, j1


def conc_prim_bas(
    bas_conc: numpy.ndarray,
    bas: numpy.ndarray | Array,
    ptr_ones: int,
    order: int = 0,
):
    prim_bas = make_primitive_bas(bas_conc, ptr_ones, order=order)
    basc_conc = numpy.vstack([prim_bas, bas_conc]).astype(numpy.int32)

    prim_shell, prim_off = primitive_bas_to_bas_maps(bas_conc)
    ptr_exp_prim = bas[:,PTR_EXP][prim_shell] + prim_off
    if isinstance(bas, numpy.ndarray):
        nbas_prim = len(prim_bas)
        basc = basc_conc.copy()
        basc[:nbas_prim,PTR_EXP] = ptr_exp_prim
        basc[nbas_prim:] = bas
    else:
        prim_bas = np.asarray(prim_bas)
        prim_bas = ops.index_update(prim_bas, ops.index[:,PTR_EXP],
                                    np.asarray(ptr_exp_prim, dtype=np.int32))
        basc = np.vstack([prim_bas, np.asarray(bas, dtype=np.int32)])
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
    fac: numpy.ndarray
    nao_fake: int
    nao: int


class C2sMaps(NamedTuple):
    fake_rows: numpy.ndarray
    real_rows: numpy.ndarray
    nao: int


def _contract_bra(s, maps, w, dtype):
    """Contract ``s`` along its bra index (axis -2) with the sparse matrix
    ``w[k]`` at ``(maps.real_rows[k], maps.fake_rows[k])``, i.e. gather the
    source rows, weight them and accumulate into the ``maps.nao``
    destination rows.
    """
    out = np.zeros(s.shape[:-2] + (maps.nao, s.shape[-1]), dtype=dtype)
    return ops.index_add(out, ops.index[..., maps.real_rows, :],
                         w[:, None] * s[..., maps.fake_rows, :])


def _contract_ket(s, maps, w, dtype):
    """:func:`_contract_bra` for the ket index (axis -1)."""
    out = np.zeros(s.shape[:-1] + (maps.nao,), dtype=dtype)
    return ops.index_add(out, ops.index[..., maps.real_rows],
                         w * s[..., maps.fake_rows])


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
    """
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    ls = bas_conc[sh0:sh1,ANG_OF]
    nprims = bas_conc[sh0:sh1,NPRIM_OF]
    nctrs = bas_conc[sh0:sh1,NCTR_OF]
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
    cart: bool,
    shl_range: tuple[int, int] | None = None,
) -> ExpMaps:
    """Static index maps for the exponent tangent.

    One entry per (shell i, contraction k, primitive j, promotion
    direction d in {x, y, z}, Cartesian function m, output function n):
    the fake rows address the ``l+2`` Cartesian functions promoted by
    ``x^2``/``y^2``/``z^2`` (:func:`index_prompt_xyz`), the real rows
    address the output functions of the differentiated shell, and the
    weight is ``-fac * ctr_coeff[coeff] * exp_dot[exp]``.

    The cross integrals are always Cartesian,
    but the differentiated index is returned in the AO basis:
    with ``cart=False`` the Cartesian-to-spherical coefficients are
    folded into ``fac``. Entries whose transformation coefficient
    vanishes are dropped.
    """
    bas_conc = numpy.asarray(bas_conc)
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    ls = bas_conc[sh0:sh1, ANG_OF]
    nprims = bas_conc[sh0:sh1, NPRIM_OF]
    nctrs = bas_conc[sh0:sh1, NCTR_OF]
    nf2s = _nf(ls + 2, True)
    nouts = _nf(ls, cart)

    fake_offs = numpy.append(0, numpy.cumsum(nprims * nf2s))
    real_offs = numpy.append(0, numpy.cumsum(nctrs * nouts))

    fake_rows = []
    real_rows = []
    entry_shell = []
    coeff_off = []
    prim_off = []
    facs = []
    for i in range(sh1 - sh0):
        l = int(ls[i])
        nf, nf2 = _nf(l, True), int(nf2s[i])
        nout = int(nouts[i])
        nprim, nctr = int(nprims[i]), int(nctrs[i])
        promoted = numpy.asarray(index_prompt_xyz(l, 2))  # (3, nf)
        # libcint's common normalization factor of the s and p shells is
        # carried by the real shell, but not by the l+2 fake shell
        if l == 0:
            norm_fac = _S_NORM
        elif l == 1:
            norm_fac = _P_NORM
        else:
            norm_fac = 1.0
        c2s = (numpy.eye(nf) if cart else
               numpy.asarray(cart2sph(l, normalized="sp")))  # (nf, nout)
        m, n = numpy.nonzero(c2s)
        c = c2s[m, n] * norm_fac

        k, j, d, e = numpy.mgrid[0:nctr, 0:nprim, 0:3, 0:m.size]
        fake_rows.append((fake_offs[i] + j * nf2 + promoted[d, m[e]]).ravel())
        real_rows.append((real_offs[i] + k * nout + n[e]).ravel())
        entry_shell.append(numpy.full(k.size, sh0 + i))
        coeff_off.append((k * nprim + j).ravel())
        prim_off.append((j + 0 * k).ravel())
        facs.append(c[e].ravel())

    return ExpMaps(
        fake_rows=_hstack_int(fake_rows),
        real_rows=_hstack_int(real_rows),
        entry_shell=_hstack_int(entry_shell),
        coeff_off=_hstack_int(coeff_off),
        prim_off=_hstack_int(prim_off),
        fac=numpy.concatenate(facs) if facs else numpy.zeros(0),
        nao_fake=int(fake_offs[-1]),
        nao=int(real_offs[-1]),
    )


def cart2sph_scatter_maps(
    bas_conc: numpy.ndarray,
    shl_range: tuple[int, int] | None = None,
) -> tuple[C2sMaps, numpy.ndarray]:
    """Static index maps and coefficients of the Cartesian-to-spherical
    transformation of the shells ``[sh0, sh1)``.

    The transformation is block diagonal over (shell, contraction), so it
    is applied with :func:`_contract_bra`/:func:`_contract_ket` -- one
    entry per (shell, contraction, Cartesian function m, spherical
    function n) with a non-vanishing coefficient -- rather than as
    a product with the dense ``(nao_cart, nao_sph)`` matrix
    (e.g. :meth:`pyscf.gto.MoleBase.cart2sph_coeff`).
    """
    bas_conc = numpy.asarray(bas_conc)
    sh0, sh1 = (0, len(bas_conc)) if shl_range is None else shl_range
    ls = bas_conc[sh0:sh1, ANG_OF]
    nctrs = bas_conc[sh0:sh1, NCTR_OF]
    nfs = _nf(ls, True)
    nsphs = _nf(ls, False)

    cart_offs = numpy.append(0, numpy.cumsum(nctrs * nfs))
    sph_offs = numpy.append(0, numpy.cumsum(nctrs * nsphs))

    cart_rows = []
    sph_rows = []
    facs = []
    for i in range(sh1 - sh0):
        nf, nsph, nctr = int(nfs[i]), int(nsphs[i]), int(nctrs[i])
        c2s = numpy.asarray(cart2sph(int(ls[i]), normalized="sp"))
        m, n = numpy.nonzero(c2s)

        k, e = numpy.mgrid[0:nctr, 0:m.size]
        cart_rows.append((cart_offs[i] + k * nf + m[e]).ravel())
        sph_rows.append((sph_offs[i] + k * nsph + n[e]).ravel())
        facs.append(c2s[m, n][e].ravel())

    maps = C2sMaps(
        fake_rows=_hstack_int(cart_rows),
        real_rows=_hstack_int(sph_rows),
        nao=int(sph_offs[-1]),
    )
    return maps, (numpy.concatenate(facs) if facs else numpy.zeros(0))


def basis_jvp_cs(
    intor_cross: Callable,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    ctr_coeff: Array,
    ctr_coeff_dot: Array,
    cart: bool,
    hermi: int,
    shls_slice: tuple[int, ...] | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # used for padding
) -> Array:
    assert hermi in (0, 1), f"hermi={hermi} not supported"
    natm = len(atm)
    nbas = len(bas)
    if basis_array_metadata is None:
        bas_conc = resolve_bas_concrete(bas)
    else:
        bas_conc = resolve_bas_concrete(basis_array_metadata, natm)
    i0, i1, j0, j1 = _resolve_shls_slice(shls_slice, nbas, hermi)

    ptr_ones = env.shape[-1]
    basc, basc_conc = conc_prim_bas(bas_conc, bas, ptr_ones)
    nbas_prim = len(basc) - nbas

    ptr_coeff0 = env.shape[-1] - ctr_coeff.shape[-1]
    ctr_coeffc = np.append(ctr_coeff, 1.0)
    envc = np.append(env, 1.0)

    ao_loc = make_loc(basc_conc, "cart" if cart else "sph")
    prim_shl_loc = primitive_shell_loc(bas_conc)

    def weights(sh0, sh1):
        """Primitive to contracted coefficient mapping.
        """
        maps = cs_scatter_maps(bas_conc, cart, (sh0, sh1))
        coeff_env_idx = bas[:,PTR_COEFF][maps.entry_shell] + maps.coeff_off
        w = ctr_coeff_dot[coeff_env_idx - ptr_coeff0]
        return maps, w

    def cross(shls):
        """The cross integrals of the shell block, with the static shell
        structure of the concatenated basis.
        """
        return intor_cross(basc, envc, ctr_coeffc, shls, ao_loc, basc_conc)

    maps_bra, w_bra = weights(i0, i1)

    shls = (prim_shl_loc[i0], prim_shl_loc[i1], nbas_prim + j0, nbas_prim + j1)
    jvp = _contract_bra(cross(shls), maps_bra, w_bra, env.dtype)

    if hermi == 1:
        jvp += np.swapaxes(jvp, -1, -2).conj()
    elif hermi == 0:
        if (j0, j1) == (i0, i1):
            maps_ket, w_ket = maps_bra, w_bra
        else:
            maps_ket, w_ket = weights(j0, j1)

        shls = (nbas_prim + i0, nbas_prim + i1, prim_shl_loc[j0], prim_shl_loc[j1])
        jvp += _contract_ket(cross(shls), maps_ket, w_ket, env.dtype)
    return jvp


def basis_jvp_exp(
    intor_cross: Callable,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    exp: Array,
    ctr_coeff: Array,
    exp_dot: Array,
    cart: bool,
    hermi: int,
    shls_slice: tuple[int, ...] | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # used for padding
) -> Array:
    assert hermi in (0, 1), f"hermi={hermi} not supported"
    natm = len(atm)
    nbas = len(bas)
    if basis_array_metadata is None:
        bas_conc = resolve_bas_concrete(bas)
    else:
        bas_conc = resolve_bas_concrete(basis_array_metadata, natm)
    i0, i1, j0, j1 = _resolve_shls_slice(shls_slice, nbas, hermi)

    ptr_ones = env.shape[-1]
    basc, basc_conc = conc_prim_bas(bas_conc, bas, ptr_ones, order=2)
    nbas_prim = len(basc) - nbas

    ptr_coeff0 = env.shape[-1] - ctr_coeff.shape[-1]
    ptr_exp0 = ptr_coeff0 - exp.shape[-1]
    ctr_coeffc = np.append(ctr_coeff, 1.0)
    envc = np.append(env, 1.0)

    ao_loc = make_loc(basc_conc, "cart")
    prim_shl_loc = primitive_shell_loc(bas_conc)

    def weights(sh0, sh1):
        """``-r^2`` brought down by ``d/da``, times the contraction
        coefficient of the primitive.
        """
        maps = exp_scatter_maps(bas_conc, cart, (sh0, sh1))
        coeff_env_idx = bas[:,PTR_COEFF][maps.entry_shell] + maps.coeff_off
        exp_env_idx = bas[:,PTR_EXP][maps.entry_shell] + maps.prim_off
        c = ctr_coeff[coeff_env_idx - ptr_coeff0]
        w = -(maps.fac * c) * exp_dot[exp_env_idx - ptr_exp0]
        return maps, w

    def cross(shls):
        """The cross integrals of the shell block, with the static shell
        structure of the concatenated basis.
        """
        return intor_cross(basc, envc, ctr_coeffc, shls, ao_loc, basc_conc)

    def c2s(sh0, sh1):
        maps, fac = cart2sph_scatter_maps(bas_conc, (sh0, sh1))
        return maps, np.asarray(fac, dtype=env.dtype)

    maps_bra, w_bra = weights(i0, i1)

    shls = (prim_shl_loc[i0], prim_shl_loc[i1], nbas_prim + j0, nbas_prim + j1)
    jvp = _contract_bra(cross(shls), maps_bra, w_bra, env.dtype)
    if not cart:
        c2s_ket = c2s(j0, j1)
        jvp = _contract_ket(jvp, *c2s_ket, env.dtype)

    if hermi == 1:
        jvp += np.swapaxes(jvp, -1, -2).conj()
    elif hermi == 0:
        if (j0, j1) == (i0, i1):
            maps_ket, w_ket = maps_bra, w_bra
        else:
            maps_ket, w_ket = weights(j0, j1)

        shls = (nbas_prim + i0, nbas_prim + i1, prim_shl_loc[j0], prim_shl_loc[j1])
        jvp_ket = _contract_ket(cross(shls), maps_ket, w_ket, env.dtype)
        if not cart:
            c2s_bra = c2s_ket if (j0, j1) == (i0, i1) else c2s(i0, i1)
            jvp_ket = _contract_bra(jvp_ket, *c2s_bra, env.dtype)
        jvp += jvp_ket
    return jvp
