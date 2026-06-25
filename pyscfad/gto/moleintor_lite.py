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
from functools import partial
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
from pyscfad.gto._pyscf_moleintor import make_loc, _get_intor_and_comp
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int2e_get_dr_order,
    int1e_dr1_name,
    int2e_dr1_name,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0

# The coordinate-gradient filling rule is identical for the 1e and 2e cases;
# only the broadcasting pattern of ``aoidx`` differs (see the callers below).
_gen_int2e_fill_jvp_r0 = _gen_int1e_fill_jvp_r0

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
    from pyscfad.gto._pyscf_moleintor import getints as callback

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
        partial(callback, intor_name, aosym=aosym),
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

    tangent_out = np.zeros_like(primal_out)

    if isinstance(env_dot, SymbolicZero):
        return primal_out, tangent_out
    if trace_basis:
        raise NotImplementedError("basis set parameter derivative not supported")
    if not trace_coords:
        return primal_out, tangent_out

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    if fname.startswith("int1e"):
        tangent_out += _int1e_coords_jvp(
            intor_name,
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
        ).reshape(tangent_out.shape)
    elif fname.startswith("int2e"):
        tangent_out += _int2e_coords_jvp(
            intor_name,
            atm,
            bas,
            env,
            env_dot,
            shls_slice,
            comp,
            aosym,
            ao_loc,
            trace_coords,
            trace_basis,
            aoslices,
        ).reshape(tangent_out.shape)
    else:
        raise NotImplementedError(f"Autodiff not implemented for {intor_name}")
    return primal_out, tangent_out

getints.defjvp(getints_jvp, symbolic_zeros=True)

def _int1e_coords_jvp(
    intor_name,
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
):
    """JVP of a 2-center 1e integral w.r.t. atomic coordinates.

    The contribution splits into an AO-center part (derivative w.r.t. the
    bra/ket orbital centers, with the operator held fixed) and, for the nuclear
    attraction integral, an operator-center part recovered with the
    rinv-at-nucleus trick.
    """
    is_nuc = "nuc" in intor_name
    if is_nuc and int1e_get_dr_order(intor_name) != [0, 0]:
        raise NotImplementedError(
            "Higher-order derivatives of int1e_nuc are not supported."
        )

    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)

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

    if is_nuc:
        jvp = jvp + _gen_int1e_nuc_jvp_rc(
            intor_ip_bra.replace("nuc", "rinv"),
            intor_ip_ket.replace("nuc", "rinv"),
            atm,
            bas,
            env,
            env_dot,
            shls_slice,
            hermi,
            ao_loc,
            aoslices,
        )
    return jvp

def _int2e_coords_jvp(
    intor_name,
    atm,
    bas,
    env,
    env_dot,
    shls_slice,
    comp,
    aosym,
    ao_loc,
    trace_coords,
    trace_basis,
    aoslices,
):
    """JVP of the 2e integral ``(ij|kl)`` w.r.t. atomic coordinates (``s1``).

    Only the bra-1 (i-center) derivative ``int2e_dr1000`` is evaluated; the
    remaining three centers are recovered from the full permutation symmetry of
    the real ERI tensor, ``(ij|kl) = (ji|kl) = (kl|ij)``.
    """
    if int2e_get_dr_order(intor_name) != [0, 0, 0, 0]:
        raise NotImplementedError(
            "Higher-order derivatives of int2e are not supported."
        )
    nbas = len(bas)
    if shls_slice is not None and tuple(shls_slice) != (0, nbas) * 4:
        raise NotImplementedError(
            "int2e coordinate derivatives with a partial shls_slice "
            "are not supported."
        )

    intor1 = int2e_dr1_name(intor_name)[0]
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_name)
    else:
        _ao_loc = ao_loc
    nao = int(_ao_loc[-1])

    # ``int2e_dr1000`` shares the AO layout of the base integral, so reuse the
    # provided ``ao_loc`` (necessary when ``bas`` is a traced array, e.g. for
    # MolePad, where ``make_loc`` cannot run at trace time).
    eri1 = -getints(
        intor1,
        atm,
        bas,
        env,
        shls_slice=None,
        comp=None,
        hermi=0,
        aosym="s1",
        ao_loc=ao_loc,
        trace_coords=trace_coords,
        trace_basis=trace_basis,
        aoslices=aoslices,
    )

    coords_dot = _extract_coords(atm, env_dot)
    if aoslices is None:
        _aoslices = _aoslice_by_atom(atm, bas, _ao_loc)
    else:
        _aoslices = aoslices
    aoidx = np.arange(nao)[None, :, None, None, None]
    jvp = _gen_int2e_fill_jvp_r0(eri1, coords_dot, _aoslices, aoidx)
    jvp = jvp + jvp.transpose(1, 0, 2, 3)
    jvp = jvp + jvp.transpose(2, 3, 0, 1)
    return jvp

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
        order_a = int1e_get_dr_order(intor_b)[0]
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
        # TODO make it general
        if "int1e_r_" in intor_b or intor_b == "int1e_r":
            s1b = s1b.reshape(3**order_a,3,3,-1,naoi,naoj)
            s1b = s1b.transpose(2,0,1,3,4,5).reshape(3,-1,naoi,naoj)
        elif "int1e_rr_" in intor_b or intor_b == "int1e_rr":
            s1b = s1b.reshape(3**order_a,9,3,-1,naoi,naoj)
            s1b = s1b.transpose(2,0,1,3,4,5).reshape(3,-1,naoi,naoj)
        else:
            s1b = s1b.reshape(3**order_a,3,-1,naoi,naoj)
            s1b = s1b.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

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
    hermi: int,
    ao_loc: ArrayLike | None,
    aoslices: ArrayLike | None = None,
) -> Array:
    """Operator-center contribution to the ``int1e_nuc`` coordinate JVP.

    The nuclear attraction integral depends on the atomic coordinates not only
    through the orbital centers (handled by :func:`_gen_int1e_jvp_r0`) but also
    through the position of each nucleus in the ``1/|r - R_C|`` operator. For
    nucleus ``C`` the operator-center derivative is obtained by placing the
    ``rinv`` origin at ``R_C`` and evaluating ``int1e_rinv_dr10`` (and, for
    ``hermi == 0``, ``int1e_rinv_dr01``), scaled by ``-Z_C``.
    """
    if shls_slice is None:
        nbas = len(bas)
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
    charges = atm[:, CHARGE_OF]

    _, comp = _get_intor_and_comp(intor_a)
    jvp = np.zeros((comp // 3, naoi, naoj), dtype=np.floatx)
    for ia in range(len(atm)):
        env_ia = ops.index_update(
            env,
            ops.index[PTR_RINV_ORIG:PTR_RINV_ORIG+3],
            coords[ia].astype(env.dtype),
        )
        vrinv = getints(
            intor_a,
            atm,
            bas,
            env_ia,
            shls_slice=shls_slice,
            comp=None,
            hermi=0,
            aosym="s1",
            ao_loc=ao_loc,
            trace_coords=False,
            trace_basis=False,
            aoslices=aoslices,
        ).reshape(3, -1, naoi, naoj)
        if hermi == 0:
            order_a = int1e_get_dr_order(intor_b)[0]
            s1b = getints(
                intor_b,
                atm,
                bas,
                env_ia,
                shls_slice=shls_slice,
                comp=None,
                hermi=0,
                aosym="s1",
                ao_loc=ao_loc,
                trace_coords=False,
                trace_basis=False,
                aoslices=aoslices,
            )
            s1b = s1b.reshape(3**order_a, 3, -1, naoi, naoj).transpose(1, 0, 2, 3, 4)
            vrinv = vrinv + s1b.reshape(3, -1, naoi, naoj)
        vrinv = vrinv * (-charges[ia])
        jvp = jvp + np.einsum("xyij,x->yij", vrinv, coords_dot[ia])
    if hermi == 1:
        jvp = jvp + jvp.transpose(0, 2, 1)
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
