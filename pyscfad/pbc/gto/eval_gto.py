from functools import partial
import numpy
from jax import custom_jvp
from pyscf.gto.moleintor import make_loc
from pyscf.pbc.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.lib import numpy as jnp
from pyscfad.gto.eval_gto import _eval_gto_fill_grad_r0

def eval_gto(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, out=None):
    if "ip" in eval_name:
        return pyscf_eval_gto(cell, eval_name, coords, comp, kpts, kpt,
                              shls_slice, non0tab, ao_loc, out)
    return _eval_gto(cell, eval_name, coords, comp, kpts, kpt,
                     shls_slice, non0tab, ao_loc, out)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,10)))
def _eval_gto(cell, eval_name, coords, comp, kpts, kpt,
              shls_slice, non0tab, ao_loc, out):
    return pyscf_eval_gto(cell, eval_name, coords, comp=comp, kpts=kpts, kpt=kpt,
                          shls_slice=shls_slice, non0tab=non0tab, ao_loc=ao_loc, out=out)

@_eval_gto.defjvp
def _eval_gto_jvp(eval_name, coords, comp, kpts, kpt,
                  shls_slice, non0tab, ao_loc, out, 
                  primals, tangents):
    cell, = primals
    cell_t, = tangents

    primal_out = _eval_gto(cell, eval_name, coords, comp, kpts, kpt,
                           shls_slice, non0tab, ao_loc, out)

    tangent_out = None
    if cell.coords is not None:
        tangent_out = _eval_gto_jvp_r0(cell, cell_t, eval_name, coords, 
                                       comp, kpts, kpt, shls_slice, non0tab, ao_loc)
    if cell.ctr_coeff is not None:
        pass
    if cell.exp is not None:
        pass
    return primal_out, tangent_out


def _eval_gto_jvp_r0(cell, cell_t, eval_name, coords,
                     comp, kpts, kpt, shls_slice, non0tab, ao_loc):
    if "ip" in eval_name:
        raise NotImplementedError("Please use GTOval_cart_deriv1 or \
                                  GTOval_sph_deriv1 instead of GTOval_ip")
    elif "deriv" in eval_name:
        tmp = eval_name.split("deriv", 1)
        order = int(tmp[1])
        intor_ip = tmp[0] + "deriv" + str(order + 1)
    else:
        #intor_ip = eval_name.replace("GTOval", "GTOval_ip")
        if not "cart" in eval_name or not "sph" in eval_name:
            if cell.cart:
                eval_name = eval_name + "_cart"
            else:
                eval_name = eval_name + "_sph"
        intor_ip = eval_name + "_deriv1"
        order = 0

    ao1 = pyscf_eval_gto(cell, intor_ip, coords, None, kpts, kpt,
                         shls_slice, non0tab, ao_loc, out=None)

    single_kpt = False
    if isinstance(ao1, numpy.ndarray):
        ao1 = [ao1,]
        single_kpt = True
    nkpts = len(ao1)

    if not 'PBC' in intor_ip:
        intor_ip = 'PBC' + intor_ip
    ngrids = len(coords)

    tangent_out = []
    for k in range(nkpts):
        grad = _eval_gto_fill_grad_r0(cell, intor_ip, shls_slice, ao_loc, ao1[k], order, ngrids)
        tangent_out_k = jnp.einsum("nxlgi,nx->lgi", grad, cell_t.coords)
        grad = None
        if order == 0:
            tangent_out_k = tangent_out_k[0]
        tangent_out.append(tangent_out_k)
    if single_kpt:
        tangent_out = tangent_out[0]
    return tangent_out
