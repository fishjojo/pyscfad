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

from __future__ import annotations
from typing import TYPE_CHECKING
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
from pyscfad.gto.moleintor_lite import _get_shape
from pyscfad.gto._basis_deriv import (
    basis_jvp_cs,
    basis_jvp_exp,
)
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int1e_dr1_name,
    aoslices_in_range,
)
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    _get_intor_and_comp,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfadlib import libcgto_vjp as libcgto

if TYPE_CHECKING:
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

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
        "env",
        "ao_loc",
        "dimension",
        "basis_array_metadata",
    ),
)
def _pbc_intor(
    intor_name: str,
    a: ArrayLike,
    kpts: ArrayLike,
    rcut: float,
    atm: numpy.ndarray | Array,
    bas: numpy.ndarray | Array,
    env: Array,
    r0: Array,
    exp: Array,
    ctr_coeff: Array,
    shls_slice: tuple[int, ...] | None = None,
    comp: int | None = None,
    hermi: int = 0,
    ao_loc: ArrayLike | None = None,
    dimension: int = 3,
    basis_array_metadata: BasisArrayMetadata | None = None, # for padding
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
    nbas = bas.shape[0]

    kpts = numpy.asarray(kpts).reshape(-1,3)
    nkpts = kpts.shape[0]

    a = numpy.asarray(a).reshape(3,3)
    Ls = _get_lattice_Ls(rcut, atm, env, a, dimension)[0]
    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order="C")

    atm, bas, env = conc_env(atm, bas, env, atm, bas, env)
    atm = numpy.asarray(atm, dtype=numpy.int32, order="C")
    bas = numpy.asarray(bas, dtype=numpy.int32, order="C")
    env = numpy.asarray(env, dtype=numpy.float64, order="C")

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
        # The input ao_loc is for the single cell; concatenate it for the
        # doubled (bra|ket) environment produced by conc_env above.
        ao_loc = numpy.asarray(ao_loc).ravel()
        nao = ao_loc[-1]
        ao_loc = numpy.concatenate([ao_loc[:-1], nao + ao_loc])
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
    intor_a, intor_b, a, kpts, rcut, atm, bas, env,
    r0, exp, ctr_coeff, r0_dot,
    shls_slice, comp, hermi, ao_loc, dimension,
    basis_array_metadata=None,
):
    kpts = kpts.reshape(-1,3)
    nkpts = kpts.shape[0]

    if comp is not None:
        comp = comp * 3

    s1a = -_pbc_intor(
        intor_a, a, kpts, rcut, atm, bas, env,
        r0, exp, ctr_coeff,
        shls_slice=shls_slice, comp=comp, hermi=0, ao_loc=ao_loc,
        dimension=dimension, basis_array_metadata=basis_array_metadata,
    )

    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(nkpts,3,-1,naoi,naoj)
    s1a = s1a.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

    if shls_slice is None:
        nbas = len(bas)
        shls_slice = (0, nbas, 0, nbas)
    if ao_loc is None:
        _ao_loc = make_loc(bas, intor_a)
    else:
        _ao_loc = ao_loc

    i0, i1, j0, j1 = shls_slice[:4]
    bas_or_meta = bas if basis_array_metadata is None else basis_array_metadata
    aoslices_bra = aoslices_in_range(bas_or_meta, _ao_loc, len(atm), (i0, i1))
    aoidx = np.arange(naoi)
    jvp = _gen_int1e_fill_jvp_r0(s1a, r0_dot, aoslices_bra,
                                 aoidx[None,None,:,None])

    if hermi == 0:
        order_a = int1e_get_dr_order(intor_b)[0]
        s1b = -_pbc_intor(
            intor_b, a, kpts, rcut, atm, bas, env,
            r0, exp, ctr_coeff,
            shls_slice=shls_slice, comp=comp, hermi=0, ao_loc=ao_loc,
            dimension=dimension, basis_array_metadata=basis_array_metadata,
        )
        s1b = s1b.reshape(nkpts,3**order_a,3,-1,naoi,naoj)
        s1b = s1b.transpose(0,2,1,3,4,5).reshape(nkpts,3,-1,naoi,naoj)
        s1b = s1b.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

        if (j0, j1) == (i0, i1):
            aoslices_ket = aoslices_bra
        else:
            aoslices_ket = aoslices_in_range(bas_or_meta, _ao_loc, len(atm),
                                             (j0, j1))
        aoidx = np.arange(naoj)
        jvp += _gen_int1e_fill_jvp_r0(s1b, r0_dot, aoslices_ket,
                                      aoidx[None,None,None,:])
    elif hermi == 1:
        jvp += jvp.transpose(0,2,1).conj()
    return jvp.reshape(nkpts,-1,naoi,naoj)

def _gen_int1e_jvp_cs(
    intor_name, a, kpts, rcut, atm, bas, env,
    r0, exp, ctr_coeff, ctr_coeff_dot,
    shls_slice, comp, hermi, ao_loc, dimension,
    basis_array_metadata,
):
    """Contraction-coefficient tangent of the k-point integrals.

    ``S(k)`` is hermitian, so ``hermi = 1`` adds the conjugate transpose of
    the bra term instead of evaluating the ket cross block; the cross
    integrals themselves are always evaluated with ``hermi = 0``.
    """
    cart = intor_name.endswith("_cart")

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_ao_loc, cross_bas_conc):
        return _pbc_intor(
            intor_name, a, kpts, rcut, atm, basc, envc,
            r0, exp, ctr_coeffc,
            shls_slice=sls, comp=comp, hermi=0, ao_loc=cross_ao_loc,
            dimension=dimension,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )
    return basis_jvp_cs(intor_cross, atm, bas, env, ctr_coeff, ctr_coeff_dot,
                        cart, hermi, shls_slice, basis_array_metadata)


def _gen_int1e_jvp_exp(
    intor_name, a, kpts, rcut, atm, bas, env,
    r0, exp, ctr_coeff, exp_dot,
    shls_slice, comp, hermi, ao_loc, dimension,
    basis_array_metadata,
):
    """Exponent tangent of the k-point integrals.

    ``d/da exp(-a r^2)`` needs the Cartesian variant of the integral (the
    fake shells carry ``l+2``); the differentiated index comes back in the
    requested basis from the scatter.
    """
    cart = intor_name.endswith("_cart")
    if cart:
        intor_cart = intor_name
    elif intor_name.endswith("_sph"):
        intor_cart = intor_name[:-4] + "_cart"
    else:
        # bare names default to spherical
        intor_cart = intor_name + "_cart"

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_ao_loc, cross_bas_conc):
        return _pbc_intor(
            intor_cart, a, kpts, rcut, atm, basc, envc,
            r0, exp, ctr_coeffc,
            shls_slice=sls, comp=comp, hermi=0, ao_loc=cross_ao_loc,
            dimension=dimension,
            basis_array_metadata=cross_bas_conc,
        )
    return basis_jvp_exp(intor_cross, atm, bas, env, exp, ctr_coeff, exp_dot,
                         cart, hermi, shls_slice, basis_array_metadata)


def _pbc_intor_jvp(
    intor_name, rcut, atm, bas, env,
    shls_slice, comp, hermi, ao_loc,
    dimension, basis_array_metadata,
    primals, tangents,
):
    a, kpts, r0, exp, ctr_coeff = primals
    a_dot, kpts_dot, r0_dot, exp_dot, ctr_coeff_dot = tangents

    primal_out = _pbc_intor(
        intor_name, a, kpts, rcut, atm, bas, env,
        r0, exp, ctr_coeff,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        dimension=dimension, basis_array_metadata=basis_array_metadata,
    )

    tangent_out = np.zeros_like(primal_out)

    if not isinstance(a_dot, SymbolicZero):
        raise NotImplementedError
    if not isinstance(kpts_dot, SymbolicZero):
        raise NotImplementedError

    if not isinstance(r0_dot, SymbolicZero):
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)
        tangent_out += _gen_int1e_jvp_r0(
            intor_ip_bra, intor_ip_ket,
            a, kpts, rcut, atm, bas, env,
            r0, exp, ctr_coeff, r0_dot,
            shls_slice, comp, hermi, ao_loc, dimension,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(exp_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_exp(
            intor_name, a, kpts, rcut, atm, bas, env,
            r0, exp, ctr_coeff, exp_dot,
            shls_slice, comp, hermi, ao_loc, dimension,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(ctr_coeff_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_cs(
            intor_name, a, kpts, rcut, atm, bas, env,
            r0, exp, ctr_coeff, ctr_coeff_dot,
            shls_slice, comp, hermi, ao_loc, dimension,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    return primal_out, tangent_out

_pbc_intor.defjvp(_pbc_intor_jvp, symbolic_zeros=True)
