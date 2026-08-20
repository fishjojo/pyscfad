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
    PTR_COORD,
)

from pyscfad import ops
from pyscfad import numpy as np
from pyscfad.gto._pyscf_moleintor import make_loc, _get_intor_and_comp
from pyscfad.gto._moleintor_helper import (
    int1e_dr1_name,
    int1e_dr1_ket_comp_to_front,
    aoslices_in_range,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfad.gto._basis_deriv import (
    basis_jvp_cs,
    basis_jvp_exp,
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
    if (not intor_name.startswith("int1e") or
        "nuc" in intor_name):
        raise NotImplementedError(f"Autodiff not implemented for {intor_name}")

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

getints.defjvp(getints_jvp, symbolic_zeros=True)

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
