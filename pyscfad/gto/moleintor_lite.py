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
    int1e_dr1_name,
    int1e_dr1_ket_comp_to_front,
    int2e_dr1_name,
    int2e_get_dr_order,
    aoslices_in_range,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfad.gto._basis_deriv import (
    basis_jvp_cs,
    basis_jvp_exp,
    basis_jvp_cs_2e,
    basis_jvp_exp_2e,
)

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

def _get_shape_ints2c(
    intor_name: str,
    bas: numpy.ndarray | Array,
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
    bas: numpy.ndarray | Array,
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
    bas: numpy.ndarray | Array,
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
    bas: numpy.ndarray | Array,
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
        "env",
        "shls_slice",
        "comp",
        "hermi",
        "aosym",
        "ao_loc",
        "basis_array_metadata",
    ),
)
def getints(
    intor_name: str,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    r0: Array,
    exp: Array,
    ctr_coeff: Array,
    origin: Array | None = None,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    aosym: str = "s1",
    ao_loc: numpy.ndarray | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # for padding
) -> Array:
    from pyscfad.gto._pyscf_moleintor import getints as callback

    shape = _get_shape(intor_name, bas, comp, shls_slice, aosym, ao_loc)
    result_shape_dtypes = ops.ShapeDtypeStruct(shape, np.float64)

    out = ops.pure_callback(
        partial(callback, intor_name, aosym=aosym),
        result_shape_dtypes,
        atm, bas, env,
        sharding=None, vmap_method="sequential",
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
    )
    return out

def getints_jvp(
    intor_name, atm, bas, env,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
    primals, tangents,
):
    if (intor_name.startswith("int1e") or
        intor_name.startswith("int2c2e")):
        jvp_fn = intor2c_jvp
    elif (intor_name.startswith("int2e") or
          intor_name.startswith("int4c1e")):
        jvp_fn = intor4c_jvp
    else:
        raise NotImplementedError(f"AD for {intor_name} is not supported")

    return jvp_fn(intor_name, atm, bas, env,
                  shls_slice, comp, hermi, aosym, ao_loc,
                  basis_array_metadata, primals, tangents)

getints.defjvp(getints_jvp, symbolic_zeros=True)

def intor2c_jvp(
    intor_name, atm, bas, env,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
    primals, tangents,
):
    r0, exp, ctr_coeff, origin = primals
    r0_dot, exp_dot, ctr_coeff_dot, origin_dot = tangents

    primal_out = getints(
        intor_name, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi,
        aosym=aosym, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )

    tangent_out = np.zeros_like(primal_out)

    if isinstance(r0_dot, SymbolicZero):
        r0_dot = None
    if origin is None or isinstance(origin_dot, SymbolicZero):
        origin_dot = None

    if not (r0_dot is None and origin_dot is None):
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)
        tangent_out += _gen_int1e_jvp_r0(
            intor_ip_bra, intor_ip_ket,
            atm, bas, env,
            r0, exp, ctr_coeff, origin,
            r0_dot, origin_dot,
            shls_slice, comp, hermi, aosym, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

        if "nuc" in intor_name:
            intor_ip_bra = intor_ip_bra.replace("nuc", "rinv")
            intor_ip_ket = intor_ip_ket.replace("nuc", "rinv")
            tangent_out += _gen_int1e_nuc_jvp_rc(
                intor_ip_bra, intor_ip_ket,
                atm, bas, env,
                r0, exp, ctr_coeff, origin, r0_dot,
                shls_slice, comp, hermi, aosym, ao_loc,
                basis_array_metadata,
            ).reshape(tangent_out.shape)

    if not isinstance(exp_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_exp(
            intor_name, atm, bas, env,
            r0, exp, ctr_coeff, origin, exp_dot,
            shls_slice, comp, hermi, aosym, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(ctr_coeff_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_cs(
            intor_name, atm, bas, env,
            r0, exp, ctr_coeff, origin, ctr_coeff_dot,
            shls_slice, comp, hermi, aosym, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    return primal_out, tangent_out

def intor4c_jvp(
    intor_name, atm, bas, env,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
    primals, tangents,
):
    r0, exp, ctr_coeff, origin = primals
    r0_dot, exp_dot, ctr_coeff_dot, _ = tangents

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    orders = int2e_get_dr_order(intor_name)
    base = fname[:-6].rstrip("_") if fname[-6:-4] == "dr" else fname
    if base not in ("int2e", "int4c1e"):
        raise NotImplementedError(f"AD for {intor_name} is not supported")

    if any(orders):
        # an already differentiated integral has lost the permutation
        # symmetry of (ij|kl), which the packed basis tangents rely on
        if not (isinstance(exp_dot, SymbolicZero) and
                isinstance(ctr_coeff_dot, SymbolicZero)):
            raise NotImplementedError(
                f"Basis-set parameter derivatives of {intor_name} "
                "are not supported")
        if aosym not in ("s1", "s2ij", "s2kl", "s4"):
            raise NotImplementedError(
                f"AD for {intor_name} with aosym = {aosym} is not supported")
    elif aosym not in ("s4", "s8"):
        raise NotImplementedError(
            f"AD for {intor_name} with aosym = {aosym} is not supported")

    primal_out = getints(
        intor_name, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi,
        aosym=aosym, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )

    tangent_out = np.zeros_like(primal_out)

    if not isinstance(r0_dot, SymbolicZero):
        if any(orders):
            tangent_out += _gen_int2e_jvp_r0_dr(
                intor_name, orders,
                atm, bas, env,
                r0, exp, ctr_coeff, origin, r0_dot,
                shls_slice, comp, aosym, ao_loc,
                basis_array_metadata,
            )
        else:
            intor_bra, _, intor_ket, _ = int2e_dr1_name(intor_name)
            _check_dr1_available(intor_name, (intor_bra, intor_ket))
            tangent_out += _gen_int2e_jvp_r0(
                intor_bra, intor_ket,
                atm, bas, env,
                r0, exp, ctr_coeff, origin, r0_dot,
                shls_slice, comp, aosym, ao_loc,
                basis_array_metadata,
            )

    if not isinstance(ctr_coeff_dot, SymbolicZero):
        tangent_out += _gen_int2e_jvp_cs(
            intor_name, atm, bas, env,
            r0, exp, ctr_coeff, origin, ctr_coeff_dot,
            shls_slice, comp, aosym, ao_loc,
            basis_array_metadata,
        )

    if not isinstance(exp_dot, SymbolicZero):
        tangent_out += _gen_int2e_jvp_exp(
            intor_name, atm, bas, env,
            r0, exp, ctr_coeff, origin, exp_dot,
            shls_slice, comp, aosym, ao_loc,
            basis_array_metadata,
        )

    return primal_out, tangent_out

def _gen_int1e_jvp_r0(
    intor_a, intor_b,
    atm, bas, env,
    r0, exp, ctr_coeff, origin,
    r0_dot, origin_dot,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata=None,
):
    if comp is not None:
        comp = comp * 3

    s1a = -getints(
        intor_a, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=0,
        aosym=aosym, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )
    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(3,-1,naoi,naoj)

    jvp = None
    if r0_dot is not None:
        if shls_slice is None:
            nbas = len(bas)
            i0, i1, j0, j1 = (0, nbas, 0, nbas)
        else:
            i0, i1, j0, j1 = shls_slice[:4]

        if ao_loc is None:
            _ao_loc = make_loc(bas, intor_a)
        else:
            _ao_loc = ao_loc

        bas_or_meta = bas if basis_array_metadata is None else basis_array_metadata
        aoslices_bra = aoslices_in_range(bas_or_meta, _ao_loc, len(atm), (i0, i1))
        aoidx = np.arange(naoi)
        jvp = _gen_int1e_fill_jvp_r0(s1a, r0_dot, aoslices_bra,
                                     aoidx[None,None,:,None])

    if origin_dot is not None:
        t = -np.einsum("xyij,x->yij", s1a, origin_dot)
        if jvp is None:
            jvp = t
        else:
            jvp += t

    if hermi == 0:
        s1b = -getints(
            intor_b, atm, bas, env,
            r0, exp, ctr_coeff, origin=origin,
            shls_slice=shls_slice, comp=comp, hermi=0,
            aosym=aosym, ao_loc=ao_loc,
            basis_array_metadata=basis_array_metadata,
        )
        s1b = int1e_dr1_ket_comp_to_front(s1b, intor_b)

        if r0_dot is not None:
            aoidx = np.arange(naoj)
            if (j0, j1) == (i0, i1):
                aoslices_ket = aoslices_bra
            else:
                aoslices_ket = aoslices_in_range(bas_or_meta, _ao_loc, len(atm),
                                                 (j0, j1))
            jvp += _gen_int1e_fill_jvp_r0(s1b, r0_dot, aoslices_ket,
                                          aoidx[None,None,None,:])

        if origin_dot is not None:
            jvp -= np.einsum("xyij,x->yij", s1b, origin_dot)

    elif hermi == 1:
        jvp += jvp.transpose(0,2,1)
    return jvp

def _gen_int1e_nuc_jvp_rc(
    intor_a, intor_b,
    atm, bas, env,
    r0, exp, ctr_coeff, origin, r0_dot,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
):
    if comp is not None:
        comp = comp * 3

    def _rinv(rc, rc_dot, z, env):
        env = ops.index_update(env,
                               ops.index[PTR_RINV_ORIG:PTR_RINV_ORIG+3],
                               rc)

        s1a = getints(intor_a,
                      atm, bas, env,
                      r0, exp, ctr_coeff, origin=rc,
                      shls_slice=shls_slice,
                      comp=comp, hermi=0, aosym="s1", ao_loc=ao_loc,
                      basis_array_metadata=basis_array_metadata)
        naoi, naoj = s1a.shape[-2:]
        s1a = s1a.reshape(3,-1,naoi,naoj)

        if hermi == 0:
            s1b = getints(intor_b,
                          atm, bas, env,
                          r0, exp, ctr_coeff, origin=rc,
                          shls_slice=shls_slice,
                          comp=comp, hermi=0, aosym="s1", ao_loc=ao_loc,
                          basis_array_metadata=basis_array_metadata)
            s1b = int1e_dr1_ket_comp_to_front(s1b, intor_b)
            s1a += s1b
        return -z * np.einsum("xyij,x->yij", s1a, rc_dot)

    jvp = ops.vmap(_rinv, (0,0,0,None))(r0, r0_dot, atm[:,CHARGE_OF], env)
    jvp = np.sum(jvp, axis=0)

    if hermi == 1:
        jvp += jvp.transpose(0,2,1)
    return jvp

def _gen_int1e_jvp_cs(
    intor_name, atm, bas, env,
    r0, exp, ctr_coeff, origin, ctr_coeff_dot,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
):
    cart = intor_name.endswith("_cart")

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_ao_loc, cross_bas_conc):
        return getints(
            intor_name, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0,
            aosym="s1", ao_loc=cross_ao_loc,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )
    return basis_jvp_cs(intor_cross, atm, bas, env, ctr_coeff, ctr_coeff_dot,
                        cart, hermi, shls_slice, basis_array_metadata)

def _gen_int1e_jvp_exp(
    intor_name, atm, bas, env,
    r0, exp, ctr_coeff, origin, exp_dot,
    shls_slice, comp, hermi, aosym, ao_loc,
    basis_array_metadata,
):
    cart = intor_name.endswith("_cart")
    if cart:
        intor_cart = intor_name
    elif intor_name.endswith("_sph"):
        intor_cart = intor_name[:-4] + "_cart"
    else:
        # bare names default to spherical
        intor_cart = intor_name + "_cart"

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_ao_loc, cross_bas_conc):
        return getints(
            intor_cart, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0,
            aosym="s1", ao_loc=cross_ao_loc,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )
    return basis_jvp_exp(intor_cross, atm, bas, env, exp, ctr_coeff, exp_dot,
                         cart, hermi, shls_slice, basis_array_metadata)

def _gen_int2e_jvp_r0(
    intor_bra, intor_ket,
    atm, bas, env,
    r0, exp, ctr_coeff, origin, r0_dot,
    shls_slice, comp, aosym, ao_loc,
    basis_array_metadata=None,
):
    """Coordinate tangent of an ``s4``- or ``s8``-packed ``(ij|kl)`` block.

    All four centers of ``(ij|kl)`` contribute,

    ``jvp[ij,kl] = d1[i,j,kl] + d1[j,i,kl] + d3[ij,k,l] + d3[ij,l,k]``,

    where ``d1`` is ``-int2e_dr1000`` with the ket pair packed
    (``aosym='s2kl'``), contracted with the tangent of the center of its
    bra index, and ``d3`` is ``-int2e_dr0010`` with the bra pair packed
    (``aosym='s2ij'``), contracted with the tangent of the center of its
    first ket index. Packing an index pair gathers the lower-triangular
    row/column pairs of the two unpacked axes, which is also where the
    swapped terms come from.

    Packing a pair requires the two shell ranges of that pair to be
    identical. If in addition the bra range equals the ket range, the
    ``(ij|kl) = (kl|ij)`` symmetry turns the ket-side contribution into the
    transpose of the bra-side one, and ``int2e_dr0010`` is not needed.
    ``aosym='s8'`` is that case with the pair indices packed once more,
    i.e. the lower triangle of the ``s4`` matrix.
    """
    assert comp in (None, 1)
    if comp is not None:
        comp = comp * 3

    shls_slice = _resolve_int2e_shls_slice(shls_slice, len(bas), aosym)
    i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
    _ao_loc = make_loc(bas, intor_bra) if ao_loc is None else ao_loc
    naoi = int(_ao_loc[i1] - _ao_loc[i0])
    naok = int(_ao_loc[k1] - _ao_loc[k0])
    bas_or_meta = bas if basis_array_metadata is None else basis_array_metadata

    def eri1(intor, aosym1):
        return -getints(
            intor, atm, bas, env,
            r0, exp, ctr_coeff, origin=origin,
            shls_slice=shls_slice, comp=comp, hermi=0,
            aosym=aosym1, ao_loc=ao_loc,
            basis_array_metadata=basis_array_metadata,
        )

    # (3,naoi,naoj,nkl) -> (naoi,naoj,nkl)
    aoslices_bra = aoslices_in_range(bas_or_meta, _ao_loc, len(atm), (i0, i1))
    aoidx = np.arange(naoi)
    x_bra = _gen_int1e_fill_jvp_r0(eri1(intor_bra, "s2kl"), r0_dot,
                                   aoslices_bra, aoidx[None,:,None,None])

    x_ket = None
    if (k0, k1) != (i0, i1):
        # (3,nij,naok,naol) -> (nij,naok,naol)
        aoslices_ket = aoslices_in_range(bas_or_meta, _ao_loc, len(atm),
                                         (k0, k1))
        aoidx = np.arange(naok)
        x_ket = _gen_int1e_fill_jvp_r0(eri1(intor_ket, "s2ij"), r0_dot,
                                       aoslices_ket, aoidx[None,None,:,None])
    return _int2e_jvp_pairs(x_bra, x_ket, aosym)

def _resolve_int2e_shls_slice(shls_slice, nbas, aosym):
    """The 8-tuple shell ranges of a packed ``(ij|kl)`` block."""
    if shls_slice is None:
        shls_slice = (0, nbas) * 4
    else:
        shls_slice = tuple(shls_slice)
        if len(shls_slice) == 4:
            shls_slice += (0, nbas, 0, nbas)
    i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
    packed_ij = aosym in ("s2ij", "s4", "s8")
    packed_kl = aosym in ("s2kl", "s4", "s8")
    if (packed_ij and (i0, i1) != (j0, j1) or
            packed_kl and (k0, k1) != (l0, l1)):
        raise NotImplementedError(
            f"aosym = {aosym} requires the two shell ranges of a packed "
            f"index pair to be identical, got shls_slice = {shls_slice}")
    if aosym == "s8" and (k0, k1) != (i0, i1):
        raise NotImplementedError(
            "aosym = s8 requires the bra and the ket pair to span the same "
            f"shells, got shls_slice = {shls_slice}")
    return shls_slice

def _int2e_jvp_pairs(x_bra, x_ket, aosym):
    """Assemble the packed tangent from the one-slot terms.

    ``x_bra[i,j,kl]`` carries the derivative of the first bra index and
    ``x_ket[ij,k,l]`` that of the first ket index. The two remaining slots
    follow from ``(ij|kl) = (ji|kl) = (ij|lk)``, i.e. from summing the two
    orderings of a pair as it is packed. When the bra and the ket span the
    same shells, ``(ij|kl) = (kl|ij)`` gives the ket-side term as the
    transpose of the bra-side one and ``x_ket`` is not needed; ``s8`` is
    that case with the pair indices packed once more.
    """
    idx_i, idx_j = numpy.tril_indices(x_bra.shape[0])
    jvp = x_bra[idx_i,idx_j] + x_bra[idx_j,idx_i]

    if x_ket is not None:
        idx_k, idx_l = numpy.tril_indices(x_ket.shape[1])
        return jvp + x_ket[:,idx_k,idx_l] + x_ket[:,idx_l,idx_k]
    if aosym == "s8":
        idx_p, idx_q = numpy.tril_indices(jvp.shape[0])
        return jvp[idx_p,idx_q] + jvp[idx_q,idx_p]
    return jvp + jvp.T

def _check_dr1_available(intor_name, intors):
    """The derivative integrals ``libcint`` actually provides.
    """
    for intor in intors:
        if intor is None:
            continue
        fname = intor.replace("_sph", "").replace("_cart", "")
        if fname not in _INTOR_FUNCTIONS:
            raise NotImplementedError(
                f"AD for {intor_name} is not supported: libcint does not "
                f"provide {fname}")

def _int2e_comp_to_front(d, off):
    """Move the derivative index of ``int2e_dr...`` to the front of the
    component axis.

    ``libcint`` inserts it as the leading component of the group of its own
    slot, i.e. behind the ``off = 3**(orders of the earlier slots)``
    components those slots contribute.
    """
    ao = d.shape[1:]
    d = d.reshape((off, 3, -1) + ao)
    return np.moveaxis(d, 1, 0).reshape((3, -1) + ao)

def _select_int2e_packed(x, aosym, naoi, naok):
    """Select from an unpacked ``(comp,i,j,k,l)`` tangent the elements a
    packed ``aosym`` keeps. Packing selects elements rather than
    symmetrizing them, so the tangent of a packed integral is the selection
    of the unpacked tangent.
    """
    if aosym in ("s2ij", "s4"):
        idx_i, idx_j = numpy.tril_indices(naoi)
        x = x[:,idx_i,idx_j]
    if aosym in ("s2kl", "s4"):
        idx_k, idx_l = numpy.tril_indices(naok)
        x = x[...,idx_k,idx_l]
    return x

def _gen_int2e_jvp_r0_dr(
    intor_name, orders,
    atm, bas, env,
    r0, exp, ctr_coeff, origin, r0_dot,
    shls_slice, comp, aosym, ao_loc,
    basis_array_metadata=None,
):
    """Coordinate tangent of an ``int2e_dr...`` block, i.e. of a second or
    higher coordinate derivative.

    An already differentiated integral no longer carries the permutation
    symmetry of ``(ij|kl)``, so each of the four centers needs its own
    derivative integral. ``libcint`` only provides the canonical ones: the
    ``j`` term is the ``i`` term transposed as long as the bra is
    undifferentiated, and likewise the ``l`` term for the ket. The terms are
    therefore assembled unpacked and the packed elements of the primal are
    selected at the end (:func:`_select_int2e_packed`).
    """
    intors = list(int2e_dr1_name(intor_name))
    if orders[0] == 0 and orders[1] == 0:
        intors[1] = None # i <-> j symmetric
    if orders[2] == 0 and orders[3] == 0:
        intors[3] = None # k <-> l symmetric
    _check_dr1_available(intor_name, intors)

    shls_slice = _resolve_int2e_shls_slice(shls_slice, len(bas), aosym)
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_name)
    else:
        _ao_loc = ao_loc
    naos = [int(_ao_loc[shls_slice[2*i+1]] - _ao_loc[shls_slice[2*i]])
            for i in range(4)]
    for slot, pair in ((1, "bra"), (3, "ket")):
        if intors[slot] is None and naos[slot] != naos[slot-1]:
            raise NotImplementedError(
                f"the {pair} index pair of {intor_name} spans different "
                "shell ranges, and libcint does not provide "
                f"{int2e_dr1_name(intor_name)[slot]}")

    bas_or_meta = bas if basis_array_metadata is None else basis_array_metadata
    if comp is not None:
        comp = comp * 3
    # components of the slots ahead of each slot
    offs = numpy.cumprod([1] + [3**order for order in orders])

    def term(slot):
        d = -getints(
            intors[slot], atm, bas, env,
            r0, exp, ctr_coeff, origin=origin,
            shls_slice=shls_slice, comp=comp, hermi=0,
            aosym="s1", ao_loc=ao_loc,
            basis_array_metadata=basis_array_metadata,
        )
        d = _int2e_comp_to_front(d, int(offs[slot]))
        aoslices = aoslices_in_range(bas_or_meta, _ao_loc, len(atm),
                                    (shls_slice[2*slot], shls_slice[2*slot+1]))
        # (3, comp, i, j, k, l): the AO axis of the slot is 2 + slot
        aoidx_shape = [1] * 6
        aoidx_shape[2+slot] = -1
        aoidx = np.arange(naos[slot]).reshape(aoidx_shape)
        return _gen_int1e_fill_jvp_r0(d, r0_dot, aoslices, aoidx)

    jvp = term(0)
    if intors[1] is None:
        jvp += jvp.transpose(0,2,1,3,4)
    else:
        jvp += term(1)

    jvp_ket = term(2)
    if intors[3] is None:
        jvp_ket += jvp_ket.transpose(0,1,2,4,3)
    else:
        jvp_ket += term(3)

    return _select_int2e_packed(jvp + jvp_ket, aosym, naos[0], naos[2])

def _gen_int2e_jvp_cs(
    intor_name, atm, bas, env,
    r0, exp, ctr_coeff, origin, ctr_coeff_dot,
    shls_slice, comp, aosym, ao_loc,
    basis_array_metadata,
):
    del ao_loc # the cross integrals bring their own
    cart = intor_name.endswith("_cart")
    shls_slice = _resolve_int2e_shls_slice(shls_slice, len(bas), aosym)

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_aosym,
                    cross_ao_loc, cross_bas_conc):
        return getints(
            intor_name, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0,
            aosym=cross_aosym, ao_loc=cross_ao_loc,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )

    def slot_term(slot):
        return basis_jvp_cs_2e(intor_cross, atm, bas, env,
                               ctr_coeff, ctr_coeff_dot, cart, slot,
                               shls_slice, basis_array_metadata)

    x_bra = slot_term(0)
    x_ket = None
    if shls_slice[4:6] != shls_slice[0:2]:
        x_ket = slot_term(2)
    return _int2e_jvp_pairs(x_bra, x_ket, aosym)

def _gen_int2e_jvp_exp(
    intor_name, atm, bas, env,
    r0, exp, ctr_coeff, origin, exp_dot,
    shls_slice, comp, aosym, ao_loc,
    basis_array_metadata,
):
    del ao_loc # the cross integrals bring their own
    cart = intor_name.endswith("_cart")
    if cart:
        intor_cart = intor_name
    elif intor_name.endswith("_sph"):
        intor_cart = intor_name[:-4] + "_cart"
    else:
        # bare names default to spherical
        intor_cart = intor_name + "_cart"

    shls_slice = _resolve_int2e_shls_slice(shls_slice, len(bas), aosym)

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_aosym,
                    cross_ao_loc, cross_bas_conc):
        return getints(
            intor_cart, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0,
            aosym=cross_aosym, ao_loc=cross_ao_loc,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )

    def slot_term(slot):
        return basis_jvp_exp_2e(intor_cross, atm, bas, env,
                                exp, ctr_coeff, exp_dot, cart, slot,
                                shls_slice, basis_array_metadata)

    x_bra = slot_term(0)
    x_ket = None
    if shls_slice[4:6] != shls_slice[0:2]:
        x_ket = slot_term(2)
    return _int2e_jvp_pairs(x_bra, x_ket, aosym)

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
