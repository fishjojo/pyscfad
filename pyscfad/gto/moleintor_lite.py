# Copyright 2025-2026 The PySCFAD Authors
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

from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial, lru_cache
import numpy

from jax.custom_derivatives import SymbolicZero
from pyscf.gto.mole import (
    ATOM_OF,
    CHARGE_OF,
    PTR_COORD,
    PTR_COMMON_ORIG,
    PTR_RINV_ORIG,
)

from pyscfad import ops
from pyscfad import numpy as np
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    _get_intor_and_comp,
    _INTOR_FUNCTIONS,
)
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int2e_get_dr_order,
    int1e_dr1_name,
    int2e_dr1_name,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array

def _get_shape_ints2c(
    intor_name: str,
    bas: numpy.ndarray,
    comp: int,
    shls_slice: tuple[int, ...] | None,
    ao_loc: numpy.ndarray | None,
) -> tuple[int, ...]:
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)

    i0, i1, j0, j1 = shls_slice[:4]
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    shape = (naoi, naoj)
    if comp > 1:
        shape = (comp,) + shape
    return shape

def _get_shape_ints3c(
    intor_name: str,
    bas: numpy.ndarray,
    comp: int,
    shls_slice: tuple[int, ...] | None,
    aosym: str,
    ao_loc: numpy.ndarray | None,
) -> tuple[int, ...]:
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ("s1",):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        shape = (naoi, naoj, naok)
    else:
        aosym = "s2ij"
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok)
    if comp > 1:
        shape = (comp,) + shape
    return shape

def _get_shape_ints4c(
    intor_name: str,
    bas: numpy.ndarray,
    comp: int,
    shls_slice: tuple[int, ...] | None,
    aosym: str,
    ao_loc: numpy.ndarray | None,
) -> tuple[int, ...]:
    nbas = bas.shape[0]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)

    if aosym == "s8":
        assert comp == 1
        assert shls_slice is None
        nao = int(ao_loc[-1])
        nao_pair = nao*(nao+1)//2
        shape = (nao_pair*(nao_pair+1)//2,)
    else:
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
        elif len(shls_slice) == 4:
            shls_slice = shls_slice + (0, nbas, 0, nbas)
        i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        naok = ao_loc[k1] - ao_loc[k0]
        naol = ao_loc[l1] - ao_loc[l0]
        if aosym in ("s4", "s2ij"):
            nij = [naoi * (naoi + 1) // 2]
        else:
            nij = [naoi, naoj]
        if aosym in ("s4", "s2kl"):
            nkl = [naok * (naok + 1) // 2]
        else:
            nkl = [naok, naol]
        shape = tuple(nij + nkl)
        if comp > 1:
            shape = (comp,) + shape
    return shape

def _get_shape(
    intor_name: str,
    bas: numpy.ndarray,
    comp: int,
    shls_slice: tuple[int, ...] | None,
    aosym: str,
    ao_loc: numpy.ndarray | None,
) -> tuple[int, ...]:
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    if (intor_name.startswith("int1e") or
        intor_name.startswith("ECP") or
        intor_name.startswith("int2c2e")):
        return _get_shape_ints2c(intor_name, bas, comp, shls_slice, ao_loc)
    elif (intor_name.startswith("int2e") or
          intor_name.startswith("int4c1e")):
        return _get_shape_ints4c(intor_name, bas, comp, shls_slice, aosym, ao_loc)
    elif intor_name.startswith("int3c"):
        return _get_shape_ints3c(intor_name, bas, comp, shls_slice, aosym, ao_loc)
    else:
        raise KeyError(f"Unknown intor {intor_name}")

def _getints_impl(intor_name, atm, bas, env, aosym="s1",
                  shls_slice=None, comp=None, hermi=0, ao_loc=None):
    """Host callback evaluating the integral through PySCF's engine.

    The full ``int2e`` is computed with 8-fold permutation symmetry and
    unpacked afterwards, which is ~8x cheaper than the plain ``s1`` fill.
    """
    from pyscfad.gto._pyscf_moleintor import getints as _pyscf_getints
    # pure_callback may hand over jax arrays; PySCF's engine needs numpy.
    atm = numpy.asarray(atm, dtype=numpy.int32)
    bas = numpy.asarray(bas, dtype=numpy.int32)
    env = numpy.asarray(env, dtype=numpy.float64)
    if ao_loc is not None:
        ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
    fname = intor_name.replace("_sph", "").replace("_cart", "")
    if (fname == "int2e" and aosym == "s1"
            and shls_slice is None and comp in (None, 1)):
        from pyscf import ao2mo
        eri8 = _pyscf_getints(intor_name, atm, bas, env,
                              None, comp, hermi, "s8", ao_loc)
        if ao_loc is None:
            nao = int(make_loc(bas, intor_name)[-1])
        else:
            nao = int(ao_loc[-1])
        return ao2mo.restore("s1", eri8, nao)
    return _pyscf_getints(intor_name, atm, bas, env,
                          shls_slice, comp, hermi, aosym, ao_loc)

@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "atm",
        "bas",
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
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    aosym: str = "s1",
    ao_loc: ArrayLike | None = None,
    trace_coords: bool = False,
    trace_basis: bool = False,
    aoslices: ArrayLike | None = None, # for padding
) -> Array:
    shape = _get_shape(
        intor_name,
        bas,
        comp,
        shls_slice,
        aosym,
        ao_loc,
    )

    result_shape_dtypes = ops.ShapeDtypeStruct(shape, np.float64)

    out = ops.pure_callback(
        partial(_getints_impl, intor_name, aosym=aosym),
        result_shape_dtypes,
        atm,
        bas,
        env,
        sharding=None,
        vmap_method="sequential",
        shls_slice=shls_slice,
        comp=comp,
        hermi=hermi,
        ao_loc=ao_loc,
    )
    return out

def getints_jvp(
    intor_name,
    atm,
    bas,
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
        atm,
        bas,
        env,
        shls_slice=shls_slice,
        comp=comp,
        hermi=hermi,
        aosym=aosym,
        ao_loc=ao_loc,
        trace_coords=trace_coords,
        trace_basis=trace_basis,
        aoslices=aoslices,
    )

    # NOTE a custom_jvp rule must not return the input's SymbolicZero as the
    # output tangent (jax rejects it on aval mismatch); use concrete zeros.
    if isinstance(env_dot, SymbolicZero):
        return primal_out, np.zeros_like(primal_out)

    if trace_basis:
        raise NotImplementedError("basis set parameter derivative not supported")

    if not trace_coords:
        return primal_out, np.zeros_like(primal_out)

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    is_int1e = fname.startswith("int1e") or fname.startswith("int2c2e")
    is_int2e = fname.startswith("int2e")

    if is_int1e:
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)
        _check_deriv_available(intor_ip_bra)
        if hermi == 0:
            _check_deriv_available(intor_ip_ket)
        if intor_name.startswith("int1e_rinv"):
            rc_deriv = PTR_RINV_ORIG
        elif intor_name.startswith("int1e_r"):
            rc_deriv = PTR_COMMON_ORIG
        else:
            rc_deriv = None

        jvp = _gen_int1e_jvp_r0(
            intor_ip_bra,
            intor_ip_ket,
            atm,
            bas,
            env,
            env_dot,
            shls_slice,
            comp,
            hermi,
            aosym,
            ao_loc,
            trace_coords,
            trace_basis,
            aoslices,
            rc_deriv,
        )

        # int1e_nuc additionally depends on the nuclear positions through the
        # 1/|r - R_A| operator; add that via the rinv-at-nucleus trick.
        if "nuc" in intor_name:
            jvp = jvp + _gen_int1e_nuc_jvp_rc(
                intor_ip_bra.replace("nuc", "rinv"),
                intor_ip_ket.replace("nuc", "rinv"),
                atm,
                bas,
                env,
                env_dot,
                shls_slice,
                comp,
                hermi,
                aosym,
                ao_loc,
                trace_coords,
                trace_basis,
                aoslices,
            )

        return primal_out, jvp.reshape(primal_out.shape)

    if is_int2e:
        jvp = _int2e_jvp_r0(
            intor_name,
            atm,
            bas,
            env,
            env_dot,
            shls_slice,
            aosym,
            ao_loc,
            trace_coords,
            trace_basis,
            aoslices,
        )
        return primal_out, jvp.reshape(primal_out.shape)

    raise NotImplementedError(f"Autodiff not implemented for {intor_name}")

getints.defjvp(getints_jvp, symbolic_zeros=True)

def _gen_int1e_jvp_r0(
    intor_a: str,
    intor_b: str,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    env_dot: ArrayLike,
    shls_slice: tuple[int, ...] | None,
    comp: int | None,
    hermi: int,
    aosym: str,
    ao_loc: ArrayLike | None,
    trace_coords: bool,
    trace_basis: bool,
    aoslices: ArrayLike | None = None,
    rc_deriv: int | None = None,
) -> Array:
    if comp is not None:
        comp = comp * 3

    s1a = -getints(
        intor_a,
        atm,
        bas,
        env,
        shls_slice=shls_slice,
        comp=comp,
        hermi=0,
        aosym=aosym,
        ao_loc=ao_loc,
        trace_coords=trace_coords,
        trace_basis=trace_basis,
        aoslices=aoslices,
    )
    naoi, naoj = s1a.shape[1:]
    s1a = s1a.reshape(3,-1,naoi,naoj)

    coords_dot = _extract_coords(atm, env_dot)

    if shls_slice is None:
        nbas = len(bas)
        shls_slice = (0, nbas, 0, nbas)
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_a)
    else:
        _ao_loc = ao_loc

    i0, _, j0, _ = shls_slice[:4]
    if aoslices is None:
        aoslices = _aoslice_by_atom(atm, bas, _ao_loc)
    aoidx = np.arange(naoi)
    jvp = _gen_int1e_fill_jvp_r0(s1a, coords_dot, aoslices-_ao_loc[i0], aoidx[None,None,:,None])

    if isinstance(rc_deriv, int):
        R0_dot = env_dot[rc_deriv:rc_deriv+3]
        jvp -= np.einsum("xyij,x->yij", s1a, R0_dot)

    if hermi == 0:
        s1b = -getints(
            intor_b,
            atm,
            bas,
            env,
            shls_slice=shls_slice,
            comp=comp,
            hermi=0,
            aosym=aosym,
            ao_loc=ao_loc,
            trace_coords=trace_coords,
            trace_basis=trace_basis,
            aoslices=aoslices,
        )
        s1b = _move_ket_deriv_axis(s1b, intor_b, naoi, naoj)

        aoidx = np.arange(naoj)
        jvp += _gen_int1e_fill_jvp_r0(s1b, coords_dot, aoslices-_ao_loc[j0],
                                      aoidx[None,None,None,:])

        if isinstance(rc_deriv, int):
            R0_dot = env_dot[rc_deriv:rc_deriv+3]
            jvp -= np.einsum("xyij,x->yij", s1b, R0_dot)

    elif hermi == 1:
        jvp += jvp.transpose(0,2,1)
    return jvp

def _gen_int1e_nuc_jvp_rc(
    intor_a: str,
    intor_b: str,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    env_dot: ArrayLike,
    shls_slice: tuple[int, ...] | None,
    comp: int | None,
    hermi: int,
    aosym: str,
    ao_loc: ArrayLike | None,
    trace_coords: bool,
    trace_basis: bool,
    aoslices: ArrayLike | None = None,
) -> Array:
    """Operator (nuclear-position) contribution to the ``int1e_nuc`` jvp.

    Uses the rinv-at-nucleus trick: for each nucleus ``A`` the rinv origin is
    placed at ``R_A``, the rinv derivative integral is evaluated, scaled by
    ``-Z_A`` and contracted with the tangent of ``R_A``. ``intor_a``/``intor_b``
    are the rinv bra/ket derivative integral names.
    """
    if comp is not None:
        comp = comp * 3

    nbas = len(bas)
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_a)
    else:
        _ao_loc = ao_loc
    i0, i1, j0, j1 = shls_slice[:4]
    naoi = int(_ao_loc[i1] - _ao_loc[i0])
    naoj = int(_ao_loc[j1] - _ao_loc[j0])

    coords = _extract_coords(atm, env)
    coords_dot = _extract_coords(atm, env_dot)
    charges = np.asarray(atm[:, CHARGE_OF], dtype=env.dtype)
    natm = len(atm)

    jvp = np.zeros((1, naoi, naoj), dtype=env.dtype)
    for ia in range(natm):
        env_ia = ops.index_update(
            env, ops.index[PTR_RINV_ORIG:PTR_RINV_ORIG+3], coords[ia])
        vrinv = getints(
            intor_a,
            atm,
            bas,
            env_ia,
            shls_slice=shls_slice,
            comp=comp,
            hermi=0,
            aosym=aosym,
            ao_loc=ao_loc,
            trace_coords=trace_coords,
            trace_basis=trace_basis,
            aoslices=aoslices,
        ).reshape(3, -1, naoi, naoj)
        if hermi == 0:
            s1b = getints(
                intor_b,
                atm,
                bas,
                env_ia,
                shls_slice=shls_slice,
                comp=comp,
                hermi=0,
                aosym=aosym,
                ao_loc=ao_loc,
                trace_coords=trace_coords,
                trace_basis=trace_basis,
                aoslices=aoslices,
            )
            s1b = _move_ket_deriv_axis(s1b, intor_b, naoi, naoj)
            vrinv = vrinv + s1b
        vrinv = vrinv * (-charges[ia])
        jvp = jvp + np.einsum("xyij,x->yij", vrinv, coords_dot[ia])
    if hermi == 1:
        jvp = jvp + jvp.transpose(0, 2, 1)
    return jvp

def _check_deriv_available(intor_name: str):
    fname = intor_name.replace("_sph", "").replace("_cart", "")
    if fname not in _INTOR_FUNCTIONS:
        raise NotImplementedError(
            f"Derivative integral {fname} is not available in pyscfadlib; "
            "the requested derivative order is not supported.")

def _move_ket_deriv_axis(s1b, intor_b, naoi, naoj):
    """Move the newest ket-derivative axis of ``intor_b`` to the front.

    The components of a ``*_drXY`` integral are ordered
    ``[bra derivatives | operator | ket derivatives]``, so the new ket
    derivative axis sits after the ``3**order_a`` bra axes and the operator
    components of the base integral (e.g. 3 for ``int1e_r``). Axes acting on
    the same center commute, so any ket axis can serve as the new one.
    """
    fname_b = intor_b.replace("_sph", "").replace("_cart", "")
    order_a = int1e_get_dr_order(fname_b)[0]
    op_comp = _get_intor_and_comp(fname_b[:-5])[1]
    s1b = s1b.reshape(3**order_a, op_comp, 3, -1, naoi, naoj)
    s1b = s1b.transpose(2, 0, 1, 3, 4, 5).reshape(3, -1, naoi, naoj)
    return s1b

@lru_cache(16)
def _tril_idx(n: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    return numpy.tril_indices(n)

@lru_cache(16)
def _pair_index_matrix(n: int) -> numpy.ndarray:
    """``M[i,j]``: index of pair ``(max(i,j), min(i,j))`` in tril-packed order."""
    i, j = _tril_idx(n)
    mat = numpy.empty((n, n), dtype=numpy.int32)
    mat[i, j] = mat[j, i] = numpy.arange(i.size, dtype=numpy.int32)
    return mat

def _ao_axis_index(nao: int, ndim: int, axis: int) -> numpy.ndarray:
    shape = [1] * ndim
    shape[axis] = nao
    return numpy.arange(nao).reshape(shape)

def _int2e_jvp_r0(
    intor_name: str,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    env_dot: ArrayLike,
    shls_slice: tuple[int, ...] | None,
    aosym: str,
    ao_loc: ArrayLike | None,
    trace_coords: bool,
    trace_basis: bool,
    aoslices: ArrayLike | None = None,
) -> Array:
    """Coordinate jvp of the ``int2e`` family, any derivative order.

    The tangent of a permutation-symmetric integral keeps that symmetry, so
    packed primals (``s2ij``/``s2kl``/``s4``/``s8``) get packed tangents.
    Derivative sub-integrals are evaluated with the highest packing their
    remaining pair symmetry allows, and contributions of centers inside an
    underivatized pair are recovered by index transposition instead of extra
    integral evaluations. Only the full (``shls_slice=None``) range is
    supported.
    """
    nbas = len(bas)
    full_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    if shls_slice is None:
        shls_slice = full_slice
    elif tuple(shls_slice[:8]) != full_slice:
        raise NotImplementedError(
            "int2e jvp with a partial shls_slice is not supported")

    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_name)
    else:
        _ao_loc = ao_loc
    nao = int(_ao_loc[-1])
    npair = nao * (nao + 1) // 2

    coords_dot = _extract_coords(atm, env_dot)
    if aoslices is None:
        aoslices = _aoslice_by_atom(atm, bas, _ao_loc)

    def _eval(intor, off, sub_aosym, tail):
        """-integral with the new derivative axis moved to the front.

        ``off`` is the number of components preceding the new axis in the
        ``[bra1 | bra2 | ket1 | ket2]`` component layout.
        """
        _check_deriv_available(intor)
        e = -getints(
            intor,
            atm,
            bas,
            env,
            shls_slice=shls_slice,
            comp=None,
            hermi=0,
            aosym=sub_aosym,
            ao_loc=ao_loc,
            trace_coords=trace_coords,
            trace_basis=trace_basis,
            aoslices=aoslices,
        )
        e = np.moveaxis(e.reshape((off, 3, -1) + tail), 1, 0)
        return e.reshape((3, -1) + tail)

    def _fill(e, axis):
        aoidx = _ao_axis_index(nao, e.ndim, axis)
        return _gen_int1e_fill_jvp_r0(e, coords_dot, aoslices, aoidx)

    orders = int2e_get_dr_order(intor_name)
    intor1, intor2, intor3, intor4 = int2e_dr1_name(intor_name)
    tril_i, tril_j = _tril_idx(nao)
    pair_idx = _pair_index_matrix(nao)

    if orders == [0, 0, 0, 0]:
        # Undifferentiated ERI: everything follows from the bra-1 derivative
        # and the 8-fold symmetry (ij|kl)=(ji|kl)=(ij|lk)=(kl|ij). The
        # derivative integral itself retains the kl symmetry, so only the
        # kl-packed bra-1 derivative is ever evaluated.
        e1 = _eval(intor1, 1, "s2kl", (nao, nao, npair))
        bs = _fill(e1, 2)[0]                            # (nao, nao, klpair)
        bs = bs + bs.transpose(1, 0, 2)                 # + bra-2 by (ij|kl)=(ji|kl)
        if aosym == "s1":
            jvp = bs[:, :, pair_idx]                    # unpack kl
            return jvp + jvp.transpose(2, 3, 0, 1)      # + ket by (ij|kl)=(kl|ij)
        if aosym == "s2kl":
            ket = bs[tril_i, tril_j]                    # (ijpair, klpair)
            return bs + ket[:, pair_idx].transpose(1, 2, 0)
        if aosym == "s2ij":
            bra = bs[tril_i, tril_j]
            return bra[:, pair_idx] + bs.transpose(2, 0, 1)
        if aosym in ("s4", "s8"):
            bra = bs[tril_i, tril_j]
            jvp = bra + bra.T
            if aosym == "s4":
                return jvp
            return jvp[_tril_idx(npair)]
        raise NotImplementedError(
            f"AD for {intor_name} with aosym={aosym} is not supported")

    # Differentiated ERI: derivatives on the first center of a pair come
    # before derivatives on its second center (names like dr0100 do not
    # exist), and a pair with any derivative loses its permutation symmetry.
    a, b, c, d = orders
    bra_underiv = a == 0 and b == 0
    ket_underiv = c == 0 and d == 0

    if aosym not in ("s1", "s2ij", "s2kl") or \
            (aosym == "s2ij" and not bra_underiv) or \
            (aosym == "s2kl" and not ket_underiv):
        raise NotImplementedError(
            f"AD for {intor_name} with aosym={aosym} is not supported")

    if ket_underiv:
        # bra-side sub-integrals keep the kl symmetry: evaluate them packed.
        tail = (nao, nao, npair)
        cbra = _fill(_eval(intor1, 1, "s2kl", tail), 2)
        cbra = cbra + _fill(_eval(intor2, 3**a, "s2kl", tail), 3)
        c3 = _fill(_eval(intor3, 3**(a + b), "s1", (nao,) * 4), 4)
        if aosym == "s2kl":
            return cbra + c3[..., tril_i, tril_j] + c3[..., tril_j, tril_i]
        return cbra[..., pair_idx] + c3 + c3.transpose(0, 1, 2, 4, 3)

    if bra_underiv:
        # ket-side sub-integrals keep the ij symmetry: evaluate them packed.
        tail = (npair, nao, nao)
        cket = _fill(_eval(intor3, 1, "s2ij", tail), 3)
        cket = cket + _fill(_eval(intor4, 3**c, "s2ij", tail), 4)
        c1 = _fill(_eval(intor1, 1, "s1", (nao,) * 4), 2)
        if aosym == "s2ij":
            return cket + c1[:, tril_i, tril_j] + c1[:, tril_j, tril_i]
        return cket[:, pair_idx] + c1 + c1.transpose(0, 2, 1, 3, 4)

    # both pairs carry derivatives: no symmetry left, evaluate all four.
    tail = (nao,) * 4
    jvp = _fill(_eval(intor1, 1, "s1", tail), 2)
    jvp = jvp + _fill(_eval(intor2, 3**a, "s1", tail), 3)
    jvp = jvp + _fill(_eval(intor3, 3**(a + b), "s1", tail), 4)
    jvp = jvp + _fill(_eval(intor4, 3**(a + b + c), "s1", tail), 5)
    return jvp

def _aoslice_by_atom(
    atm,
    bas,
    ao_loc,
):
    bas_atom = bas[:,ATOM_OF]
    delimiter = numpy.where(bas_atom[0:-1] != bas_atom[1:])[0] + 1
    assert len(atm) == len(delimiter) + 1
    shell_start = numpy.append(0, delimiter)
    shell_end = numpy.append(delimiter, len(bas))
    out = numpy.hstack(
        [
            ao_loc[shell_start].reshape(-1,1),
            ao_loc[shell_end].reshape(-1,1),
        ]
    )
    return out

def _extract_coords(
    atm,
    env,
):
    idx = atm[:, PTR_COORD]
    coords = env[idx[:, None] + np.arange(3)]
    return coords
