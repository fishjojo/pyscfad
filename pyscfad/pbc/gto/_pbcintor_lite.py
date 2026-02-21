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
from pyscf.gto.mole import PTR_COORD
from pyscf.gto.mole import conc_env
from pyscf.pbc.gto._pbcintor import libpbc

from pyscfad.typing import Array, ArrayLike
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
from pyscfadlib import libcgto_vjp as libcgto

def _atom_coords(atm, env):
    ptr = atm[:,PTR_COORD]
    c = env[ptr[:,None] + numpy.arange(3)]
    return c

def _get_scaled_atom_coords(coords, a):
    return numpy.dot(coords, numpy.linalg.inv(a))

def _get_lattice_Ls(rcut, atm, env, a, dimension):
    if dimension == 0:
        Ls = scaled_Ls = numpy.zeros((1,3))
        return Ls, scaled_Ls

    r = _atom_coords(atm, env)

    shifts = [[1,0,0],[-1,0,0]]
    if dimension > 1:
        shifts += [[0,1,0],[0,-1,0]]
    if dimension > 2:
        shifts += [[0,0,1],[0,0,-1]]
    shifts = numpy.asarray(shifts) * rcut

    r1 = (r[None,:,:] + shifts[:,None,:]).reshape(-1,3)
    scaled_r1 = numpy.dot(r1, numpy.linalg.inv(a))
    bounds = abs(scaled_r1).max(axis=0)
    bounds = numpy.ceil(bounds).astype(int)

    if dimension == 1:
        Ts = numpy.arange(-bounds[0], bounds[0]+1).reshape(-1,1)
    elif dimension == 2:
        Ts = lib.cartesian_prod((numpy.arange(-bounds[0], bounds[0]+1),
                                 numpy.arange(-bounds[1], bounds[1]+1)))
    elif dimension == 3:
        Ts = lib.cartesian_prod((numpy.arange(-bounds[0], bounds[0]+1),
                                 numpy.arange(-bounds[1], bounds[1]+1),
                                 numpy.arange(-bounds[2], bounds[2]+1)))

    Ls = numpy.dot(Ts[:,:dimension], a[:dimension])

    rr = r[:,None] - r
    dist_max = numpy.linalg.norm(rr, axis=2).max()
    Ls_mask = numpy.linalg.norm(Ls, axis=1) < (rcut + dist_max)
    Ls = Ls[Ls_mask]
    scaled_Ls = Ts[Ls_mask]
    return Ls, scaled_Ls


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
        "dimension",
    ),
)
def _pbc_intor(
    intor_name: str,
    a: ArrayLike,
    kpts: ArrayLike,
    rcut: float,
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
    dimension: int = 3,
) -> Array:
    shape = _get_shape(
        intor_name,
        bas,
        comp,
        shls_slice,
        "s1",
        ao_loc,
    )
    shape = (len(kpts),) + shape
    result_shape_dtypes = ops.ShapeDtypeStruct(shape, np.complex128)

    out = ops.pure_callback(
        partial(_pbc_intor_impl_cpu, intor_name),
        result_shape_dtypes,
        a, kpts, rcut, atm, bas, env, shls_slice, comp, hermi, ao_loc, dimension,
        vmap_method="sequential",
    )
    return out

def _pbc_intor_impl_cpu(
    intor_name, a, kpts, rcut, atm, bas, env,
    shls_slice=None, comp=None, hermi=0, ao_loc=None, dimension=3,
):
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    kpts = numpy.asarray(kpts).reshape(-1,3)
    nkpts = kpts.shape[0]
    a = numpy.asarray(a).reshape(3,3)

    atm = numpy.asarray(atm, dtype=numpy.int32, order="C")
    bas = numpy.asarray(bas, dtype=numpy.int32, order="C")
    env = numpy.asarray(env, dtype=numpy.float64, order="C")
    nbas = bas.shape[0]

    Ls = _get_lattice_Ls(rcut, atm, env, a, dimension)[0]
    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order="C")

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
        # The input ao_loc is for single mol object,
        # need to concatenate it.
        raise NotImplementedError
    ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32, order="C")

    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    out = numpy.empty((nkpts,comp,naoi,naoj), dtype=numpy.complex128)

    if hermi == 0:
        aosym = "s1"
    else:
        aosym = "s2"

    fill = getattr(libpbc, "PBCnr2c_fill_k" + aosym)
    fintor = getattr(libcgto, intor_name)
    cintopt = lib.c_null_ptr()

    drv = libpbc.PBCnr2c_drv
    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
    )

    mat = []
    for k, kpt in enumerate(kpts):
        v = out[k]
        if hermi != 0:
            for ic in range(comp):
                lib.hermi_triu(v[ic], hermi=hermi, inplace=True)
        if comp == 1:
            v = v[0]
        if abs(kpt).sum() < 1e-9:  # gamma_point
            v = v.real
        mat.append(v)
    return numpy.asarray(mat, dtype=numpy.complex128)

def _gen_int1e_jvp_r0(
    intor_a, intor_b, a, kpts, rcut, atm, bas, env, env_dot,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices=None, dimension=3,
):
    kpts = kpts.reshape(-1,3)
    nkpts = kpts.shape[0]

    if comp is not None:
        comp = comp * 3

    s1a = -_pbc_intor(
        intor_a, a, kpts, rcut, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=0, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices, dimension=dimension,
    )

    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(nkpts,3,-1,naoi,naoj)
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

    if hermi == 0:
        order_a = int1e_get_dr_order(intor_b)[0]
        s1b = -_pbc_intor(
            intor_b, a, kpts, rcut, atm, bas, env,
            shls_slice=shls_slice, comp=comp, hermi=0, ao_loc=ao_loc,
            trace_coords=trace_coords, trace_basis=trace_basis,
            aoslices=aoslices, dimension=dimension,
        )
        s1b = s1b.reshape(nkpts,3**order_a,3,-1,naoi,naoj)
        s1b = s1b.transpose(0,2,1,3,4,5).reshape(nkpts,3,-1,naoi,naoj)

        aoidx = np.arange(naoj)
        jvp += _gen_int1e_fill_jvp_r0(s1b, coords_dot, aoslices-_ao_loc[j0],
                                      aoidx[None,None,None,None,:])
    elif hermi == 1:
        jvp += jvp.transpose(0,1,3,2)
    return jvp

def _gen_int1e_fill_jvp_r0(ints, coords_t, aoslices, aoidx):
    def _fill(sl, coord_t):
        mask = (aoidx >= sl[0]) & (aoidx < sl[1])
        grad = np.where(mask, ints, np.array(0, dtype=ints.dtype))
        return np.einsum("kxyij,x->kyij", grad, coord_t)
    jvp = np.sum(ops.vmap(_fill)(aoslices, coords_t), axis=0)
    return jvp

def _pbc_intor_jvp(
    intor_name, rcut, atm, bas,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis,
    aoslices, dimension,
    primals, tangents,
):
    a, kpts, env = primals
    a_dot, kpts_dot, env_dot = tangents

    primal_out = _pbc_intor(
        intor_name, a, kpts, rcut, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices, dimension=dimension,
    )

    tangent_out = np.zeros_like(primal_out)

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    intor_ip_bra = intor_ip_ket = None
    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)

    if not isinstance(env_dot, SymbolicZero):
        if trace_coords and (intor_ip_bra or intor_ip_ket):
            tangent_out += _gen_int1e_jvp_r0(
                intor_ip_bra, intor_ip_ket,
                a, kpts, rcut, atm, bas, env, env_dot,
                shls_slice, comp, hermi, ao_loc,
                trace_coords, trace_basis,
                aoslices, dimension,
            ).reshape(tangent_out.shape)
        if trace_basis:
            raise NotImplementedError

    if not isinstance(a_dot, SymbolicZero):
        raise NotImplementedError

    if not isinstance(kpts_dot, SymbolicZero):
        raise NotImplementedError

    return primal_out, tangent_out

_pbc_intor.defjvp(_pbc_intor_jvp, symbolic_zeros=True)
