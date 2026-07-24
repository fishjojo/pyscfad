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
    _check_basis_deriv_args,
)
from pyscfad.gto._basis_deriv import (
    basis_jvp_cs,
    basis_jvp_exp,
)
from pyscfad.gto._moleintor_helper import (
    int1e_get_dr_order,
    int1e_dr1_name,
)
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    _get_intor_and_comp,
)
from pyscfad.gto._moleintor_jvp import _gen_int1e_fill_jvp_r0
from pyscfadlib import libcgto_vjp as libcgto

@partial(
    ops.custom_jvp,
    nondiff_argnames=(
        "intor_name",
        "Ls_mask",
        "atm",
        "bas",
        "shls_slice",
        "comp",
        "hermi",
        "ao_loc",
        "trace_coords",
        "trace_basis",
        "aoslices",
        "bas_tmpl",
    ),
)
def _lattice_intor(
    intor_name: str,
    Ls: ArrayLike,
    Ls_mask: ArrayLike,
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
    bas_tmpl: ArrayLike | None = None, # static structure when bas is traced
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
    intor_a, intor_b, Ls, Ls_mask, atm, bas, env, env_dot,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices=None, bas_tmpl=None,
):
    Ls = Ls.reshape(-1,3)
    nL = len(Ls)

    if comp is not None:
        comp = comp * 3

    s1a = -_lattice_intor(
        intor_a, Ls, Ls_mask, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices, bas_tmpl=bas_tmpl,
    )

    naoi, naoj = s1a.shape[-2:]
    s1a = s1a.reshape(nL,3,-1,naoi,naoj)
    s1a = s1a.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)
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
                                 aoidx[None,None,:,None])

    order_a = int1e_get_dr_order(intor_b)[0]
    s1b = -_lattice_intor(
        intor_b, Ls, Ls_mask, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices, bas_tmpl=bas_tmpl,
    )
    s1b = s1b.reshape(nL,3**order_a,3,-1,naoi,naoj)
    s1b = s1b.transpose(0,2,1,3,4,5).reshape(nL,3,-1,naoi,naoj)
    s1b = s1b.transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

    aoidx = np.arange(naoj)
    jvp += _gen_int1e_fill_jvp_r0(s1b, coords_dot, aoslices-_ao_loc[j0],
                                  aoidx[None,None,None,:])
    return jvp.reshape(nL,-1,naoi,naoj)

def _gen_int1e_jvp_Ls(
    intor_b, Ls, Ls_mask, atm, bas, env, Ls_dot,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices=None, bas_tmpl=None,
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
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis,
        aoslices=aoslices, bas_tmpl=bas_tmpl,
    )
    naoi, naoj = s1b.shape[-2:]
    s1b = s1b.reshape(nL, 3**order_a, 3, -1, naoi, naoj)
    s1b = s1b.transpose(0,2,1,3,4,5).reshape(nL, 3, -1, naoi, naoj)

    jvp = np.einsum("lxcpq,lx->lcpq", s1b, Ls_dot)
    return jvp.reshape(nL, -1, naoi, naoj)


def _gen_int1e_jvp_basis(
    intor_name, Ls, Ls_mask, atm, bas, env, env_dot,
    shls_slice, comp, hermi, ao_loc, bas_tmpl,
):
    """Basis-set parameter (exponent + contraction coefficient) tangent of
    the per-image lattice integrals (first order in the basis parameters).

    Both the bra and the ket cross terms are computed explicitly for every
    image (per-image transposes would relate different images, ``S_L^T =
    S_{-L}``); for ``hermi == 1`` the tangent is masked to the lower
    triangle, matching the ``s2``-fill storage of the primal.
    """
    from pyscfad.gto._basis_deriv import _resolve_template
    _check_basis_deriv_args(intor_name, bas, shls_slice, "s1", hermi)

    def _eval_cross_factory(name):
        def eval_cross(basc, envc, sls, cross_ao_loc):
            return _lattice_intor(
                name, Ls, Ls_mask, atm, basc, envc,
                shls_slice=sls, comp=comp, hermi=0, ao_loc=cross_ao_loc,
                trace_coords=False, trace_basis=False,
            )
        return eval_cross

    cart = intor_name.endswith("_cart")
    jvp = basis_jvp_cs(_eval_cross_factory(intor_name),
                       bas, bas_tmpl, env, env_dot, cart, hermi=0)

    if cart:
        intor_cart = intor_name
        need_c2s = False
    elif intor_name.endswith("_sph"):
        intor_cart = intor_name[:-4] + "_cart"
        need_c2s = True
    else:
        intor_cart = intor_name + "_cart"
        need_c2s = True
    jvp += basis_jvp_exp(_eval_cross_factory(intor_cart),
                         bas, bas_tmpl, env, env_dot, need_c2s, hermi=0)

    if hermi == 1:
        # the s2 fill stores the shell-pair blocks with
        # bra shell >= ket shell (diagonal shell blocks complete)
        tmpl = _resolve_template(bas, bas_tmpl)
        if ao_loc is None:
            _ao_loc = make_loc(tmpl, intor_name)
        else:
            _ao_loc = numpy.asarray(ao_loc).ravel()
        nbas = len(tmpl)
        shell_of = numpy.repeat(numpy.arange(nbas),
                                numpy.diff(_ao_loc[:nbas+1]))
        mask = shell_of[:,None] >= shell_of[None,:]
        jvp = np.where(mask, jvp, np.zeros((), dtype=jvp.dtype))
    return jvp


def _lattice_intor_jvp(
    intor_name, Ls_mask, atm, bas,
    shls_slice, comp, hermi, ao_loc,
    trace_coords, trace_basis, aoslices, bas_tmpl,
    primals, tangents,
):
    Ls, env = primals
    Ls_dot, env_dot = tangents

    primal_out = _lattice_intor(
        intor_name, Ls, Ls_mask, atm, bas, env,
        shls_slice=shls_slice, comp=comp, hermi=hermi, ao_loc=ao_loc,
        trace_coords=trace_coords, trace_basis=trace_basis, aoslices=aoslices,
        bas_tmpl=bas_tmpl,
    )

    tangent_out = np.zeros_like(primal_out)

    fname = intor_name.replace("_sph", "").replace("_cart", "")
    intor_ip_bra = intor_ip_ket = None
    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor_name)

    if not isinstance(env_dot, SymbolicZero):
        if trace_coords and (intor_ip_bra or intor_ip_ket):
            tangent_out += _gen_int1e_jvp_r0(
                intor_ip_bra, intor_ip_ket,
                Ls, Ls_mask, atm, bas, env, env_dot,
                shls_slice, comp, hermi, ao_loc,
                trace_coords, trace_basis, aoslices, bas_tmpl,
            ).reshape(tangent_out.shape)
        if trace_basis:
            tangent_out += _gen_int1e_jvp_basis(
                intor_name, Ls, Ls_mask, atm, bas, env, env_dot,
                shls_slice, comp, hermi, ao_loc, bas_tmpl,
            ).reshape(tangent_out.shape)

    if not isinstance(Ls_dot, SymbolicZero):
        if not intor_ip_ket:
            raise NotImplementedError(
                f"Lattice-shift derivative not available for {intor_name}."
            )
        tangent_out += _gen_int1e_jvp_Ls(
            intor_ip_ket,
            Ls, Ls_mask, atm, bas, env, Ls_dot,
            shls_slice, comp, hermi, ao_loc,
            trace_coords, trace_basis, aoslices, bas_tmpl,
        ).reshape(tangent_out.shape)

    return primal_out, tangent_out

_lattice_intor.defjvp(_lattice_intor_jvp, symbolic_zeros=True)
