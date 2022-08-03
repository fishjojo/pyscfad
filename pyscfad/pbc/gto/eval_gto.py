from functools import partial
import numpy
from pyscf.gto.moleintor import make_loc
from pyscf.pbc.gto.eval_gto import _get_intor_and_comp
from pyscf.pbc.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.lib import numpy as np
from pyscfad.lib import stop_grad, custom_jvp, vmap
from pyscfad.gto.eval_gto import _eval_gto_fill_grad_r0

def eval_gto(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, out=None):
    if cell.abc is None:
        fn = eval_gto_diff_cell
    else:
        fn = eval_gto_diff_full
    return fn(cell, eval_name, coords, comp=comp, kpts=kpts, kpt=kpt,
              shls_slice=shls_slice, non0tab=non0tab, ao_loc=ao_loc, out=out)

def eval_gto_diff_full(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
                       shls_slice=None, non0tab=None, ao_loc=None, out=None):
    from pyscfad.gto import mole
    from pyscfad.pbc.gto.cell import shift_bas_center
    if eval_name[:3] == 'PBC':  # PBCGTOval_xxx
        eval_name_mol, comp = _get_intor_and_comp(cell, eval_name[3:], comp)
    else:
        eval_name_mol, comp = _get_intor_and_comp(cell, eval_name, comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    Ls = cell.get_lattice_Ls()
    expkL = np.exp(1j*np.dot(kpts_lst, Ls.T))

    def body(L):
        shifted_cell = shift_bas_center(cell, L)
        ao = mole.eval_gto(shifted_cell, eval_name_mol, coords, comp=comp,
                           shls_slice=shls_slice, non0tab=non0tab,
                           ao_loc=ao_loc, out=None)
        return ao

    # use for loop to reduce memory in fwd-mode AD
    #nk = len(kpts_lst)
    #ng = len(coords)
    #nao = cell.nao
    #if out is None:
    #    if comp == 1:
    #        out = np.zeros((nk,ng,nao), dtype=np.complex128)
    #    else:
    #        out = np.zeros((nk,comp,ng,nao), dtype=np.complex128)
    #for i, L in enumerate(Ls):
    #    ao = body(L)
    #    if comp == 1:
    #        out += expkL[:,i][:,None,None] * ao[None,...]
    #    else:
    #        out += expkL[:,i][:,None,None,None] * ao[None,...]

    aos = np.asarray([body(L) for L in Ls])
    if comp == 1:
        out = np.einsum('kl,lgi->kgi', expkL, aos)
    else:
        out = np.einsum('kl,lcgi->kcgi', expkL, aos)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        out = out[0]
    return out

def eval_gto_diff_cell(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
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
        tangent_out_k = np.einsum("nxlgi,nx->lgi", grad, cell_t.coords)
        grad = None
        if order == 0:
            tangent_out_k = tangent_out_k[0]
        tangent_out.append(tangent_out_k)
    if single_kpt:
        tangent_out = tangent_out[0]
    return tangent_out
