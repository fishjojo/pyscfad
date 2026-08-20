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
from pyscf.gto.mole import conc_env

from pyscfad.typing import ArrayLike, Array
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.moleintor_lite import _get_shape
from pyscfad.gto._basis_deriv import (
    basis_jvp_cs,
    basis_jvp_exp,
    _resolve_shls_slice,
)
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int1e_dr1_name,
    aoslices_in_range,
    resolve_bas_concrete as _resolve_bas_concrete,
)
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    _get_intor_and_comp,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfadlib import libcgto_vjp as libcgto

if TYPE_CHECKING:
    from pyscfad.ml.gto.basis_array import BasisArrayMetadata

@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "Ls_mask",
        "atm",
        "bas",
        "env",
        "shls_slice",
        "comp",
        "hermi",
        "ao_loc",
        "basis_array_metadata",
    ),
)
def _lattice_intor(
    intor_name: str,
    Ls: ArrayLike,
    Ls_mask: ArrayLike,
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
    ao_loc: numpy.ndarray | None = None,
    basis_array_metadata: BasisArrayMetadata | None = None, # for padding
) -> Array:
    shape = _get_shape(intor_name, bas, comp,
                       shls_slice, "s1", ao_loc)

    shape = (len(Ls),) + shape
    result_shape_dtypes = ops.ShapeDtypeStruct(shape, np.float64)

    out = ops.pure_callback(
        partial(_lattice_intor_impl_cpu, intor_name),
        result_shape_dtypes,
        Ls, Ls_mask, atm, bas, env, shls_slice, comp, hermi, ao_loc,
        vmap_method="sequential",
    )
    return out

def _lattice_intor_impl_cpu(
    intor_name, Ls, Ls_mask, atm, bas, env,
    shls_slice=None, comp=None, hermi=0, ao_loc=None,
):
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    nbas = bas.shape[0]

    Ls = numpy.asarray(Ls, dtype=numpy.float64, order="C").reshape(-1,3)
    Ls_mask = numpy.asarray(Ls_mask, dtype=np.int32, order="C")
    nL = len(Ls)

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

    out = numpy.zeros((nL,comp,naoi,naoj), dtype=numpy.float64)

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
    intor_a, intor_b, Ls, Ls_mask,
    atm, bas, env,
    r0, exp, ctr_coeff, origin, r0_dot,
    shls_slice, comp, hermi, ao_loc,
    basis_array_metadata=None,
):
    Ls = Ls.reshape(-1,3)
    nL = len(Ls)

    if comp is not None:
        comp = comp * 3

    s1a = -_lattice_intor(
        intor_a, Ls, Ls_mask, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )

    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(nL,3,-1,naoi,naoj)
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

    order_a = int1e_get_dr_order(intor_b)[0]
    s1b = -_lattice_intor(
        intor_b, Ls, Ls_mask, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )
    s1b = s1b.reshape(nL,3**order_a,3,-1,naoi,naoj)
    s1b = s1b.transpose(0,2,1,3,4,5).reshape(nL,3,-1,naoi,naoj)
    s1b = s1b.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

    if (j0, j1) == (i0, i1):
        aoslices_ket = aoslices_bra
    else:
        aoslices_ket = aoslices_in_range(bas_or_meta, _ao_loc, len(atm),
                                         (j0, j1))
    aoidx = np.arange(naoj)
    jvp += _gen_int1e_fill_jvp_r0(s1b, r0_dot, aoslices_ket,
                                  aoidx[None,None,None,:])
    return jvp.reshape(nL,-1,naoi,naoj)

def _gen_int1e_jvp_Ls(
    intor_b, Ls, Ls_mask, atm, bas, env,
    r0, exp, ctr_coeff, origin, Ls_dot,
    shls_slice, comp, hermi, ao_loc,
    basis_array_metadata=None,
):
    """Tangent of the per-image integrals w.r.t. the lattice shifts ``Ls``.

    Every ket function in image L is displaced rigidly by L, so
    dS_L/dL equals the ket-center derivative summed over all ket centers,
    i.e. the ket-derivative integral itself (no per-atom scatter needed).
    """
    Ls = Ls.reshape(-1,3)
    nL = len(Ls)

    if comp is not None:
        comp = comp * 3

    order_a = int1e_get_dr_order(intor_b)[0]
    s1b = -_lattice_intor(
        intor_b, Ls, Ls_mask, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )
    naoi, naoj = s1b.shape[-2:]
    s1b = s1b.reshape(nL, 3**order_a, 3, -1, naoi, naoj)
    s1b = s1b.transpose(0,2,1,3,4,5).reshape(nL, 3, -1, naoi, naoj)

    jvp = np.einsum("lxcpq,lx->lcpq", s1b, Ls_dot)
    return jvp.reshape(nL, -1, naoi, naoj)


def _s2_fill_mask(jvp, intor_name, atm, bas, ao_loc, shls_slice, hermi,
                  basis_array_metadata):
    """Mask a full tangent down to the ``s2`` lower triangle the primal stores.

    Both cross terms are evaluated explicitly for every image with
    ``hermi = 0``: a per-image transpose would relate different images
    (``S_L^T = S_{-L}``), so the symmetry shortcut the molecular path uses is
    not available here. For ``hermi == 1`` the full tangent is therefore
    masked afterwards to match the s2-fill storage of the primal -- shell-pair
    blocks with bra shell >= ket shell, diagonal shell blocks complete.
    """
    if hermi != 1:
        return jvp

    i0, i1, j0, j1 = _resolve_shls_slice(shls_slice, len(bas), hermi)
    if ao_loc is None:
        bas_or_meta = (bas if basis_array_metadata is None
                       else basis_array_metadata)
        bas_conc = _resolve_bas_concrete(bas_or_meta, len(atm))
        _ao_loc = make_loc(bas_conc, intor_name)
    else:
        _ao_loc = numpy.asarray(ao_loc).ravel()

    row_shell = numpy.repeat(numpy.arange(i0, i1),
                             numpy.diff(_ao_loc[i0:i1+1]))
    col_shell = numpy.repeat(numpy.arange(j0, j1),
                             numpy.diff(_ao_loc[j0:j1+1]))
    mask = row_shell[:,None] >= col_shell[None,:]
    return np.where(mask, jvp, np.zeros((), dtype=jvp.dtype))


def _gen_int1e_jvp_cs(
    intor_name, Ls, Ls_mask, atm, bas, env,
    r0, exp, ctr_coeff, origin, ctr_coeff_dot,
    shls_slice, comp, hermi, ao_loc,
    basis_array_metadata,
):
    """Contraction-coefficient tangent of the per-image lattice integrals."""
    cart = intor_name.endswith("_cart")

    def intor_cross(basc, envc, ctr_coeffc, sls, cross_ao_loc, cross_bas_conc):
        return _lattice_intor(
            intor_name, Ls, Ls_mask, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0, ao_loc=cross_ao_loc,
            # the shell structure of this call is the cross basis, not the
            # outer (padded) one the metadata describes
            basis_array_metadata=cross_bas_conc,
        )

    # hermi = 0: see _s2_fill_mask
    jvp = basis_jvp_cs(intor_cross, atm, bas, env, ctr_coeff, ctr_coeff_dot,
                       cart, 0, shls_slice, basis_array_metadata)
    return _s2_fill_mask(jvp, intor_name, atm, bas, ao_loc, shls_slice, hermi,
                         basis_array_metadata)


def _gen_int1e_jvp_exp(
    intor_name, Ls, Ls_mask, atm, bas, env,
    r0, exp, ctr_coeff, origin, exp_dot,
    shls_slice, comp, hermi, ao_loc,
    basis_array_metadata,
):
    """Exponent tangent of the per-image lattice integrals.

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
        return _lattice_intor(
            intor_cart, Ls, Ls_mask, atm, basc, envc,
            r0, exp, ctr_coeffc, origin=origin,
            shls_slice=sls, comp=comp, hermi=0, ao_loc=cross_ao_loc,
            basis_array_metadata=cross_bas_conc,
        )

    # hermi = 0: see _s2_fill_mask
    jvp = basis_jvp_exp(intor_cross, atm, bas, env, exp, ctr_coeff, exp_dot,
                        cart, 0, shls_slice, basis_array_metadata)
    return _s2_fill_mask(jvp, intor_name, atm, bas, ao_loc, shls_slice, hermi,
                         basis_array_metadata)


def _lattice_intor_jvp(
    intor_name, Ls_mask, atm, bas, env,
    shls_slice, comp, hermi, ao_loc,
    basis_array_metadata,
    primals, tangents,
):
    Ls, r0, exp, ctr_coeff, origin = primals
    Ls_dot, r0_dot, exp_dot, ctr_coeff_dot, _ = tangents

    primal_out = _lattice_intor(
        intor_name, Ls, Ls_mask, atm, bas, env,
        r0, exp, ctr_coeff, origin=origin,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        basis_array_metadata=basis_array_metadata,
    )

    tangent_out = np.zeros_like(primal_out)

    if not isinstance(r0_dot, SymbolicZero):
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)
        tangent_out += _gen_int1e_jvp_r0(
            intor_ip_bra, intor_ip_ket,
            Ls, Ls_mask, atm, bas, env,
            r0, exp, ctr_coeff, origin, r0_dot,
            shls_slice, comp, hermi, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(exp_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_exp(
            intor_name, Ls, Ls_mask, atm, bas, env,
            r0, exp, ctr_coeff, origin, exp_dot,
            shls_slice, comp, hermi, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(ctr_coeff_dot, SymbolicZero):
        tangent_out += _gen_int1e_jvp_cs(
            intor_name, Ls, Ls_mask, atm, bas, env,
            r0, exp, ctr_coeff, origin, ctr_coeff_dot,
            shls_slice, comp, hermi, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    if not isinstance(Ls_dot, SymbolicZero):
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)
        tangent_out += _gen_int1e_jvp_Ls(
            intor_ip_ket,
            Ls, Ls_mask, atm, bas, env,
            r0, exp, ctr_coeff, origin, Ls_dot,
            shls_slice, comp, hermi, ao_loc,
            basis_array_metadata,
        ).reshape(tangent_out.shape)

    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
