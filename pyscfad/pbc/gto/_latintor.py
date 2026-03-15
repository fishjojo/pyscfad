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

from functools import partial
import ctypes
import numpy
from jax.custom_derivatives import SymbolicZero

from pyscf import lib
from pyscf.gto.mole import conc_env

from pyscfad.typing import ArrayLike, Array
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.moleintor_lite import (
    _get_shape,
    _aoslice_by_atom,
    _extract_coords,
)
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int1e_dr1_name,
)
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    _get_intor_and_comp,
)
from pyscfad.pbc.gto._pbcintor_lite import (
    _atom_coords,
    _gen_int1e_fill_jvp_r0,
)
from pyscfadlib import libcgto_vjp as libcgto

@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "rcut",
        "atm",
        "bas",
        "shls_slice",
        "comp",
        "hermi",
        "ao_loc",
        "trace_coords",
        "trace_basis",
        "aoslices",
    ),
)
def _lattice_intor(
    intor_name: str,
    rcut: float,
    Ls: ArrayLike,
    atm: ArrayLike,
    bas: ArrayLike,
    env: ArrayLike,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
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
        "s1",
        ao_loc,
    )

    shape = (len(Ls),) + shape
    result_shape_dtypes = ops.ShapeDtypeStruct(shape, np.float64)

    out = ops.pure_callback(
        partial(_lattice_intor_impl_cpu, intor_name),
        result_shape_dtypes,
        rcut, Ls, atm, bas, env, shls_slice, comp, hermi, ao_loc,
        vmap_method="sequential",
    )
    return out

def _lattice_intor_impl_cpu(
    intor_name, rcut, Ls, atm, bas, env,
    shls_slice=None, comp=None, hermi=0, ao_loc=None,
):
    intor_name, comp = _get_intor_and_comp(intor_name, comp)

    Ls = numpy.asarray(Ls, dtype=numpy.float64, order="C").reshape(-1,3)
    nL = len(Ls)
    atm = numpy.asarray(atm, dtype=numpy.int32, order="C")
    bas = numpy.asarray(bas, dtype=numpy.int32, order="C")
    env = numpy.asarray(env, dtype=numpy.float64, order="C")
    nbas = bas.shape[0]

    atm, bas, env = conc_env(atm, bas, env, atm, bas, env)
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    else:
        assert (shls_slice[1] <= nbas and shls_slice[3] <= nbas)

    i0, i1, j0, j1 = shls_slice[:4]
    j0 += nbas
    j1 += nbas

    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
    else:
        # TODO The input ao_loc is for single mol object,
        # need to concatenate it.
        raise NotImplementedError
    ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32, order="C")

    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    out = numpy.zeros((nL,comp,naoi,naoj), dtype=numpy.float64)

    r = _atom_coords(atm, env)
    rr = r[:,None] - r
    dist_max = numpy.linalg.norm(rr, axis=2).max()
    Ls_mask = numpy.linalg.norm(Ls, axis=1) < (rcut + dist_max)
    Ls_mask = numpy.asarray(Ls_mask, dtype=bool, order="C")

    if hermi == 0:
        aosym = "s1"
    else:
        aosym = "s2"

    fill = getattr(libcgto, "LATnr2c_fill_" + aosym)
    fintor = getattr(libcgto, intor_name)
    cintopt = lib.c_null_ptr()

    drv = libcgto.LATnr2c_drv
    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        Ls_mask.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
    )

    if comp == 1:
        out = out[:,0]
    return out

def _gen_int1e_jvp_r0(
    intor_a, intor_b, rcut, Ls, atm, bas, env, env_dot,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices=None,
):
    Ls = Ls.reshape(-1,3)
    nL = len(Ls)

    if comp is not None:
        comp = comp * 3

    s1a = -_lattice_intor(
        intor_a, rcut, Ls, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices,
    )

    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(nL,3,-1,naoi,naoj)
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
    jvp = _gen_int1e_fill_jvp_r0(s1a, coords_dot, aoslices-_ao_loc[i0],
                                 aoidx[None,None,None,:,None])

    order_a = int1e_get_dr_order(intor_b)[0]
    s1b = -_lattice_intor(
        intor_b, rcut, Ls, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices,
    )
    s1b = s1b.reshape(nL,3**order_a,3,-1,naoi,naoj)
    s1b = s1b.transpose(0,2,1,3,4,5).reshape(nL,3,-1,naoi,naoj)

    aoidx = np.arange(naoj)
    jvp += _gen_int1e_fill_jvp_r0(s1b, coords_dot, aoslices-_ao_loc[j0],
                                  aoidx[None,None,None,None,:])
    return jvp

def _lattice_intor_jvp(
    intor_name, rcut, atm, bas,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices,
    primals, tangents,
):
    Ls, env = primals
    Ls_dot, env_dot = tangents

    primal_out = _lattice_intor(
        intor_name, rcut, Ls, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis, aoslices=aoslices,
    )

    tangent_out = np.zeros_like(primal_out)

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    intor_ip_bra = intor_ip_ket = None
    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)

    if not isinstance(env_dot, SymbolicZero):
        if trace_coords and (intor_ip_bra or intor_ip_ket):
            tangent_out += _gen_int1e_jvp_r0(
                intor_ip_bra, intor_ip_ket,
                rcut, Ls, atm, bas, env, env_dot,
                shls_slice, comp, hermi, ao_loc,
                trace_coords, trace_basis, aoslices,
            ).reshape(tangent_out.shape)
        if trace_basis:
            raise NotImplementedError

    if not isinstance(Ls_dot, SymbolicZero):
        raise NotImplementedError

    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
