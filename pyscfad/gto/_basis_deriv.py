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

- The structure is taken from a concrete template array ``bas_tmpl``
  (for :class:`~pyscfad.ml.gto.MolePad` the per-atom shell template of
  the :class:`~pyscfad.ml.gto.BasisArray`, which is identical for every
  element; for :class:`~pyscfad.gto.MoleLite` ``bas`` itself). Shapes,
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
from functools import lru_cache
import numpy
import scipy.linalg

from pyscf.gto.mole import (
    ATOM_OF,
    ANG_OF,
    NPRIM_OF,
    NCTR_OF,
    KAPPA_OF,
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

# libcint common normalization factors for s and p orbitals
_S_NORM = 0.282094791773878143
_P_NORM = 0.488602511902919921


def _concrete_bas(bas) -> numpy.ndarray:
    try:
        return numpy.asarray(bas)
    except Exception as exc:
        raise NotImplementedError(
            "Basis-set parameter derivatives require the static basis "
            "structure. When `bas` is traced (e.g. under jit or vmap over "
            "atomic numbers), pass a concrete structural template through "
            "the `bas_tmpl` argument."
        ) from exc


def _resolve_template(bas, bas_tmpl) -> numpy.ndarray:
    """The concrete structural template: ``bas_tmpl`` if given, else a
    concrete ``bas``. Only the ``ANG_OF``/``NPRIM_OF``/``NCTR_OF``
    (and ``ATOM_OF``/``KAPPA_OF``) columns of the template are used;
    env pointers are always gathered from ``bas``.
    """
    if bas_tmpl is not None:
        return numpy.asarray(bas_tmpl)
    return _concrete_bas(bas)


def make_fake_bas(
    tmpl: numpy.ndarray,
    ptr_ones: int,
    order: int = 0,
) -> numpy.ndarray:
    """Structural template of the fake basis with one uncontracted shell
    per primitive.

    Each primitive of each shell becomes its own ``nprim = nctr = 1``
    shell on the same atom whose contraction coefficient points at
    ``ptr_ones`` (an ``env`` slot holding 1.0, shared by all fake shells).
    The ``PTR_EXP`` column is a placeholder filled from the actual ``bas``
    by :func:`make_fake_basc`.

    Parameters:
        tmpl: Concrete structural ``bas`` template.
        ptr_ones: ``env`` index of the appended 1.0 coefficient.
        order: Increment added to the angular momenta (2 for the
            exponent-derivative fake basis).
    """
    tmpl = numpy.asarray(tmpl)
    nprim = tmpl[:, NPRIM_OF]
    nbasf = int(nprim.sum())

    fake_bas = numpy.zeros((nbasf, tmpl.shape[1]), dtype=numpy.int32)
    fake_bas[:, ATOM_OF] = numpy.repeat(tmpl[:, ATOM_OF], nprim)
    fake_bas[:, ANG_OF] = numpy.repeat(tmpl[:, ANG_OF], nprim) + order
    fake_bas[:, NPRIM_OF] = 1
    fake_bas[:, NCTR_OF] = 1
    fake_bas[:, KAPPA_OF] = numpy.repeat(tmpl[:, KAPPA_OF], nprim)
    fake_bas[:, PTR_COEFF] = ptr_ones
    return fake_bas


def fake_prim_maps(tmpl: numpy.ndarray):
    """Per fake-shell (shell index, primitive index) of the template."""
    tmpl = numpy.asarray(tmpl)
    nprim = tmpl[:, NPRIM_OF]
    nbasf = int(nprim.sum())
    prim_shell = numpy.repeat(numpy.arange(len(tmpl)), nprim)
    prim_off = numpy.arange(nbasf) - numpy.repeat(
        numpy.cumsum(nprim) - nprim, nprim)
    return prim_shell, prim_off


def make_fake_basc(
    tmpl: numpy.ndarray,
    bas: ArrayLike,
    ptr_ones: int,
    order: int = 0,
):
    """Concatenated (fake + real) basis and its structural template.

    The returned ``basc`` carries the env pointers gathered from ``bas``
    (and may therefore be a traced array); ``basc_tmpl`` is the concrete
    structural counterpart used for shapes and ``ao_loc``.
    """
    fake_tmpl = make_fake_bas(tmpl, ptr_ones, order=order)
    basc_tmpl = numpy.vstack([fake_tmpl, tmpl]).astype(numpy.int32)

    prim_shell, prim_off = fake_prim_maps(tmpl)
    ptr_exp_fake = bas[:, PTR_EXP][prim_shell] + prim_off
    if isinstance(bas, numpy.ndarray):
        basc = basc_tmpl.copy()
        basc[:len(fake_tmpl), PTR_EXP] = ptr_exp_fake
        basc[len(fake_tmpl):] = bas
    else:
        fake = np.asarray(fake_tmpl)
        fake = ops.index_update(fake, ops.index[:, PTR_EXP],
                                np.asarray(ptr_exp_fake, dtype=np.int32))
        basc = np.concatenate([fake, np.asarray(bas, dtype=np.int32)], axis=0)
    return basc, basc_tmpl


class CsMaps(NamedTuple):
    fake_rows: numpy.ndarray
    real_rows: numpy.ndarray
    entry_shell: numpy.ndarray
    coeff_off: numpy.ndarray
    prim_off: numpy.ndarray
    naof: int
    nao: int


class ExpMaps(NamedTuple):
    fake_rows: numpy.ndarray
    real_rows: numpy.ndarray
    entry_shell: numpy.ndarray
    coeff_off: numpy.ndarray
    prim_off: numpy.ndarray
    norm_fac: numpy.ndarray
    naof: int
    nao: int


def _nl(ls, cart):
    if cart:
        return (ls + 1) * (ls + 2) // 2
    else:
        return 2 * ls + 1


def cs_scatter_maps(tmpl: numpy.ndarray, cart: bool) -> CsMaps:
    """Static index maps scattering primitive cross-integral rows into
    the coefficient-tangent contraction.

    One entry per (shell i, contraction k, primitive j, function m):
    ``T[real_rows, fake_rows] += env_dot[ptr_coeff[entry_shell] + coeff_off]``.
    """
    tmpl = numpy.asarray(tmpl)
    ls = tmpl[:, ANG_OF]
    nprims = tmpl[:, NPRIM_OF]
    nctrs = tmpl[:, NCTR_OF]
    nls = _nl(ls, cart)

    fake_offs = numpy.append(0, numpy.cumsum(nprims * nls))
    real_offs = numpy.append(0, numpy.cumsum(nctrs * nls))

    fake_rows = []
    real_rows = []
    entry_shell = []
    coeff_off = []
    prim_off = []
    for i in range(len(tmpl)):
        nl, nprim, nctr = int(nls[i]), int(nprims[i]), int(nctrs[i])
        k, j, m = numpy.mgrid[0:nctr, 0:nprim, 0:nl]
        fake_rows.append((fake_offs[i] + j * nl + m).ravel())
        real_rows.append((real_offs[i] + k * nl + m).ravel())
        entry_shell.append(numpy.full(nctr * nprim * nl, i))
        coeff_off.append((k * nprim + j).ravel())
        prim_off.append((j + 0 * k).ravel())

    return CsMaps(
        fake_rows=numpy.concatenate(fake_rows),
        real_rows=numpy.concatenate(real_rows),
        entry_shell=numpy.concatenate(entry_shell),
        coeff_off=numpy.concatenate(coeff_off),
        prim_off=numpy.concatenate(prim_off),
        naof=int(fake_offs[-1]),
        nao=int(real_offs[-1]),
    )


def exp_scatter_maps(tmpl: numpy.ndarray) -> ExpMaps:
    """Static index maps for the exponent tangent (Cartesian only).

    One entry per (shell i, contraction k, primitive j, function m,
    promotion direction d in {x, y, z}); the fake rows address the
    ``l+2`` Cartesian functions promoted by ``x^2``/``y^2``/``z^2``
    (:func:`index_prompt_xyz`), and the weight is
    ``-norm_fac * env[coeff] * env_dot[exp]``.
    """
    tmpl = numpy.asarray(tmpl)
    ls = tmpl[:, ANG_OF]
    nprims = tmpl[:, NPRIM_OF]
    nctrs = tmpl[:, NCTR_OF]
    nls = _nl(ls, True)
    nl1s = _nl(ls + 2, True)

    fake_offs = numpy.append(0, numpy.cumsum(nprims * nl1s))
    real_offs = numpy.append(0, numpy.cumsum(nctrs * nls))

    fake_rows = []
    real_rows = []
    entry_shell = []
    coeff_off = []
    prim_off = []
    norm_fac = []
    for i in range(len(tmpl)):
        l = int(ls[i])
        nl, nl1 = int(nls[i]), int(nl1s[i])
        nprim, nctr = int(nprims[i]), int(nctrs[i])
        promoted = numpy.asarray(index_prompt_xyz(l, 2))  # (3, nl)
        if l == 0:
            fac = _S_NORM
        elif l == 1:
            fac = _P_NORM
        else:
            fac = 1.0

        k, j, d, m = numpy.mgrid[0:nctr, 0:nprim, 0:3, 0:nl]
        fake_rows.append((fake_offs[i] + j * nl1 + promoted[d, m]).ravel())
        real_rows.append((real_offs[i] + k * nl + m).ravel())
        entry_shell.append(numpy.full(nctr * nprim * 3 * nl, i))
        coeff_off.append((k * nprim + j).ravel())
        prim_off.append((j + 0 * k).ravel())
        norm_fac.append(numpy.full(nctr * nprim * 3 * nl, fac))

    return ExpMaps(
        fake_rows=numpy.concatenate(fake_rows),
        real_rows=numpy.concatenate(real_rows),
        entry_shell=numpy.concatenate(entry_shell),
        coeff_off=numpy.concatenate(coeff_off),
        prim_off=numpy.concatenate(prim_off),
        norm_fac=numpy.concatenate(norm_fac),
        naof=int(fake_offs[-1]),
        nao=int(real_offs[-1]),
    )


def cart2sph_mat(tmpl: numpy.ndarray) -> numpy.ndarray:
    """Static Cartesian-to-spherical transformation matrix
    (``(nao_cart, nao_sph)``), equivalent to
    :meth:`pyscf.gto.MoleBase.cart2sph_coeff` but built from the
    structural template alone.
    """
    tmpl = numpy.asarray(tmpl)
    blocks = []
    for i in range(len(tmpl)):
        c = cart2sph(int(tmpl[i, ANG_OF]), normalized="sp")
        blocks.extend([c] * int(tmpl[i, NCTR_OF]))
    return scipy.linalg.block_diag(*blocks)


@lru_cache(64)
def _cached_maps(key, kind, cart):
    tmpl = numpy.frombuffer(key[0], dtype=numpy.dtype(key[2])).reshape(key[1])
    if kind == "cs":
        return cs_scatter_maps(tmpl, cart)
    elif kind == "exp":
        return exp_scatter_maps(tmpl)
    elif kind == "c2s":
        return cart2sph_mat(tmpl)
    else:
        raise KeyError(kind)


def _get_maps(tmpl, kind, cart=False):
    key = (tmpl.tobytes(), tmpl.shape, tmpl.dtype.str)
    return _cached_maps(key, kind, cart)


def gather_env_idx(bas, maps):
    """Env indices of the coefficient and exponent of every scatter entry,
    gathered from ``bas`` (which may be traced).
    """
    coeff_env_idx = bas[:, PTR_COEFF][maps.entry_shell] + maps.coeff_off
    exp_env_idx = bas[:, PTR_EXP][maps.entry_shell] + maps.prim_off
    return coeff_env_idx, exp_env_idx


def basis_jvp_cs(
    eval_cross: Callable,
    bas: ArrayLike,
    bas_tmpl: ArrayLike | None,
    env: ArrayLike,
    env_dot: ArrayLike,
    cart: bool,
    hermi: int,
) -> Array:
    """Contraction-coefficient part of the ``env`` tangent.

    Parameters:
        eval_cross: ``eval_cross(basc, envc, shls_slice, ao_loc)``
            evaluating the cross integrals on the primal ``envc``,
            returning an array of shape ``(..., nrow, ncol)`` (leading
            dims: integral components and/or lattice images / k-points).
        hermi: 1 computes the bra term and adds its conjugate transpose;
            0 computes the ket cross block explicitly.

    Returns:
        Tangent contribution of shape ``(..., nao, nao)``.
    """
    tmpl = _resolve_template(bas, bas_tmpl)
    ptr_ones = env.shape[-1]
    basc, basc_tmpl = make_fake_basc(tmpl, bas, ptr_ones)
    nbasf = len(basc_tmpl) - len(tmpl)
    nbas = len(tmpl)
    envc = np.concatenate([env, np.ones(1, dtype=env.dtype)])
    ao_loc = make_loc(basc_tmpl, "cart" if cart else "sph")

    maps = _get_maps(tmpl, "cs", cart)
    coeff_env_idx, _ = gather_env_idx(bas, maps)
    w = env_dot[coeff_env_idx]
    t = np.zeros((maps.nao, maps.naof), dtype=env.dtype)
    t = ops.index_add(t, ops.index[maps.real_rows, maps.fake_rows], w)

    s_bra = eval_cross(basc, envc, (0, nbasf, nbasf, nbasf + nbas), ao_loc)
    jvp = np.einsum("ma,...av->...mv", t, s_bra)
    if hermi == 1:
        jvp = jvp + np.conj(np.swapaxes(jvp, -1, -2))
    elif hermi == 0:
        s_ket = eval_cross(basc, envc, (nbasf, nbasf + nbas, 0, nbasf), ao_loc)
        jvp = jvp + np.einsum("...ma,na->...mn", s_ket, t)
    else:
        raise NotImplementedError(f"hermi = {hermi}")
    return jvp


def basis_jvp_exp(
    eval_cross: Callable,
    bas: ArrayLike,
    bas_tmpl: ArrayLike | None,
    env: ArrayLike,
    env_dot: ArrayLike,
    need_c2s: bool,
    hermi: int,
) -> Array:
    """Exponent part of the ``env`` tangent.

    ``eval_cross`` must evaluate the **Cartesian** variant of the integral
    (the fake shells carry ``l+2``); when ``need_c2s`` is True the result
    is transformed back to spherical on both AO indices.
    """
    tmpl = _resolve_template(bas, bas_tmpl)
    ptr_ones = env.shape[-1]
    basc, basc_tmpl = make_fake_basc(tmpl, bas, ptr_ones, order=2)
    nbasf = len(basc_tmpl) - len(tmpl)
    nbas = len(tmpl)
    envc = np.concatenate([env, np.ones(1, dtype=env.dtype)])
    ao_loc = make_loc(basc_tmpl, "cart")

    maps = _get_maps(tmpl, "exp")
    coeff_env_idx, exp_env_idx = gather_env_idx(bas, maps)
    c = env[coeff_env_idx]
    w = -(maps.norm_fac * c) * env_dot[exp_env_idx]
    t = np.zeros((maps.nao, maps.naof), dtype=env.dtype)
    t = ops.index_add(t, ops.index[maps.real_rows, maps.fake_rows], w)

    s_bra = eval_cross(basc, envc, (0, nbasf, nbasf, nbasf + nbas), ao_loc)
    jvp = np.einsum("ma,...av->...mv", t, s_bra)
    if hermi == 1:
        jvp = jvp + np.conj(np.swapaxes(jvp, -1, -2))
    elif hermi == 0:
        s_ket = eval_cross(basc, envc, (nbasf, nbasf + nbas, 0, nbasf), ao_loc)
        jvp = jvp + np.einsum("...ma,na->...mn", s_ket, t)
    else:
        raise NotImplementedError(f"hermi = {hermi}")

    if need_c2s:
        c2s = np.asarray(_get_maps(tmpl, "c2s"), dtype=env.dtype)
        jvp = np.einsum("pi,...pq,qj->...ij", c2s, jvp, c2s)
    return jvp
