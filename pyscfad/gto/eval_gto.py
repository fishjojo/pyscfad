from functools import partial
import numpy
#import jax

from pyscf.gto import mole as pyscf_mole
from pyscf.gto.moleintor import make_loc
from pyscf.gto.eval_gto import _get_intor_and_comp
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto

from pyscfad import numpy as np
from pyscfad.ops import (
    custom_jvp,
    jit,
    vmap,
)
from pyscfad.gto._moleintor_helper import get_bas_label, promote_xyz
from pyscfad.gto._mole_helper import (
    setup_exp,
    setup_ctr_coeff,
    get_fakemol_exp,
    get_fakemol_cs,
)

_MAX_DERIV_ORDER = 4
#_DERIV_LABEL = []
#for i in range(_MAX_DERIV_ORDER+1):
#    if i == 0:
#        _label = ['',]
#    else:
#        _label = get_bas_label(i)
#    _DERIV_LABEL += _label
#
#order = _MAX_DERIV_ORDER - 1
#_X_ID = [[] for iorder in range(order+1)]
#_Y_ID = [[] for iorder in range(order+1)]
#_Z_ID = [[] for iorder in range(order+1)]
#for iorder in range(order+1):
#    start = (iorder+1) * (iorder+2) * (iorder+3) // 6
#    end = (iorder+2) * (iorder+3) * (iorder+4) // 6
#    for il, label in enumerate(get_bas_label(iorder)):
#        idx_x = _DERIV_LABEL.index(''.join(sorted(label + 'x')), start, end)
#        idx_y = _DERIV_LABEL.index(''.join(sorted(label + 'y')), start, end)
#        idx_z = _DERIV_LABEL.index(''.join(sorted(label + 'z')), start, end)
#        _X_ID[iorder].append(idx_x)
#        _Y_ID[iorder].append(idx_y)
#        _Z_ID[iorder].append(idx_z)
#
#_X_ID = [[1], [4, 5, 6], [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
#_Y_ID = [[2], [5, 7, 8], [11, 13, 14, 16, 17, 18], [21, 23, 24, 26, 27, 28, 30, 31, 32, 33]]
#_Z_ID = [[3], [6, 8, 9], [12, 14, 15, 17, 18, 19], [22, 24, 25, 27, 28, 29, 31, 32, 33, 34]]

_XYZ_ID = [
    numpy.array(
        [[1,],
         [2,],
         [3,],]
    ),
    numpy.array(
        [[4, 5, 6],
         [5, 7, 8],
         [6, 8, 9],]
    ),
    numpy.array(
        [[10, 11, 12, 13, 14, 15],
         [11, 13, 14, 16, 17, 18],
         [12, 14, 15, 17, 18, 19],]
    ),
    numpy.array(
        [[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
         [21, 23, 24, 26, 27, 28, 30, 31, 32, 33],
         [22, 24, 25, 27, 28, 29, 31, 32, 33, 34],]
    ),
]

def eval_gto(mol, eval_name, grid_coords,
             comp=None, shls_slice=None, non0tab=None,
             ao_loc=None, cutoff=None, out=None):
    # FIXME non0tab makes ctr_coeff and exp derivatives wrong
    non0tab=None
    eval_name, comp = _get_intor_and_comp(mol, eval_name, comp)
    return _eval_gto(mol, eval_name, grid_coords,
                     comp, shls_slice, non0tab, ao_loc, cutoff, out)

@partial(custom_jvp, nondiff_argnums=(1,3,4,5,6,7,8))
def _eval_gto(mol, eval_name, grid_coords,
              comp, shls_slice, non0tab, ao_loc, cutoff, out):
    return pyscf_eval_gto(mol, eval_name, grid_coords, comp, shls_slice, non0tab,
                          ao_loc, cutoff, out)

@_eval_gto.defjvp
def _eval_gto_jvp(eval_name, comp, shls_slice, non0tab, ao_loc, cutoff, out,
                  primals, tangents):
    if 'spinor' in eval_name or 'ip' in eval_name:
        raise NotImplementedError

    mol, grid_coords = primals
    mol_t, grid_coords_t = tangents

    primal_out = _eval_gto(mol, eval_name, grid_coords, comp, shls_slice, non0tab,
                           ao_loc, cutoff, out)
    tangent_out = np.zeros_like(primal_out)
    nao = primal_out.shape[-1]

    if mol.coords is not None:
        tangent_out += _eval_gto_jvp_r0(mol, mol_t, eval_name, grid_coords,
                                        comp, shls_slice, non0tab, ao_loc)
    if mol.ctr_coeff is not None:
        tangent_out += _eval_gto_jvp_cs(mol, mol_t, eval_name, grid_coords,
                                        comp, shls_slice, non0tab, ao_loc)
    if mol.exp is not None:
        tangent_out += _eval_gto_jvp_exp(mol, mol_t, eval_name, grid_coords,
                                         comp, shls_slice, non0tab, ao_loc)

    tangent_out += _eval_gto_jvp_r(mol, eval_name, grid_coords, grid_coords_t,
                                   comp, shls_slice, non0tab, ao_loc, nao)
    return primal_out, tangent_out

def _eval_gto_jvp_r(mol, eval_name, grid_coords, grid_coords_t,
                    comp, shls_slice, non0tab, ao_loc, nao):
    if 'deriv' not in eval_name:
        order = 0
        new_eval = eval_name + '_deriv1'
    else:
        tmp = eval_name.split('deriv', 1)
        order = int(tmp[1])
        new_eval = tmp[0] + 'deriv' + str(order + 1)
    if order + 1 > _MAX_DERIV_ORDER:
        raise NotImplementedError

    ao1 = _eval_gto(mol, new_eval, grid_coords, None, shls_slice, non0tab,
                    ao_loc, None, None)

    @jit
    def _contract(ao1, grid_coords_t):
        tangent_out = []
        for iorder in range(order+1):
            tmp = 0
            for i in range(3):
                tmp += np.einsum('ygi,g->ygi',
                                 ao1[_XYZ_ID[iorder][i]],
                                 grid_coords_t[:,i])
            tangent_out.append(tmp)
        tangent_out = np.concatenate(tangent_out)
        return tangent_out

    tangent_out = _contract(ao1, grid_coords_t)
    if order == 0:
        tangent_out = tangent_out[0]
    return tangent_out

def _eval_gto_dot_grad_tangent_r0(mol, mol_t, intor,
                                  shls_slice, ao_loc, ao1,
                                  order, ngrids):
    coords_t = mol_t.coords
    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    sh0, sh1 = shls_slice
    if ao_loc is None:
        ao_loc = make_loc(mol._bas, intor)
    ao_start = ao_loc[sh0]
    ao_end = ao_loc[sh1]
    nao = ao_end - ao_start
    atmlst = numpy.asarray(range(mol.natm))
    aoslices = mol.aoslice_by_atom(ao_loc)

    ids = numpy.zeros((mol.natm, 2))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        if p1 <= ao_start:
            ids[k,0] = ids[k,1] = 0
        else:
            ids[k,0] = max(0, p0 - ao_start)
            ids[k,1] = min(p1, ao_end) - ao_start

    #FIXME scan does not work for reverse mode
    #@jit
    #def fn(ao1, ids):
    #    tangent_out = []
    #    for iorder in range(order+1):
    #        def body(carry, xs):
    #            slices, coords_t = xs
    #            _zero = np.array(0, dtype=ao1.dtype)
    #            idx = np.arange(nao)[None,None,:]
    #            p0, p1 = slices[:]
    #            mask = (idx >= p0) & (idx < p1)
    #            for i in range(3):
    #                carry += np.where(mask, ao1[_XYZ_ID[iorder][i]], _zero) * coords_t[i]
    #            return carry, None
    #        nl = (iorder+1)*(iorder+2)//2
    #        tangent_out.append(jax.lax.scan(body, np.zeros((nl,ngrids,nao)), (ids, coords_t))[0])
    #    tangent_out = np.concatenate(tangent_out)
    #    return tangent_out

    @jit
    def fn(ao1, ids):
        tangent_out = []
        for iorder in range(order+1):
            def body(slices, coords_t):
                _zero = np.array(0, dtype=ao1.dtype)
                idx = np.arange(nao)[None,None,:]
                p0, p1 = slices[:]
                mask = (idx >= p0) & (idx < p1)
                # pylint: disable=cell-var-from-loop
                out  = np.where(mask, ao1[_XYZ_ID[iorder][0]], _zero) * coords_t[0]
                out += np.where(mask, ao1[_XYZ_ID[iorder][1]], _zero) * coords_t[1]
                out += np.where(mask, ao1[_XYZ_ID[iorder][2]], _zero) * coords_t[2]
                return out
            out = vmap(body)(ids, coords_t)
            tangent_out.append(np.sum(out, axis=0))
        tangent_out = np.concatenate(tangent_out)
        return tangent_out
    return fn(-ao1, ids)

def _eval_gto_jvp_r0(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    if 'deriv' not in eval_name:
        order = 0
        new_eval = eval_name + '_deriv1'
    else:
        tmp = eval_name.split('deriv', 1)
        order = int(tmp[1])
        new_eval = tmp[0] + 'deriv' + str(order + 1)
    if order + 1 > _MAX_DERIV_ORDER:
        raise NotImplementedError

    ao1 = _eval_gto(mol, new_eval, grid_coords, None, shls_slice, non0tab,
                    ao_loc, None, None)
    ngrids = len(grid_coords)
    tangent_out = _eval_gto_dot_grad_tangent_r0(mol, mol_t, new_eval,
                                                shls_slice, ao_loc, ao1,
                                                order, ngrids)
    ao1 = None
    if order == 0:
        tangent_out = tangent_out[0]
    return tangent_out

def _eval_gto_jvp_cs(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    from pyscfad.gto.mole import nao_nr_range
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    ngrids = len(grid_coords)

    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    shl0, shl1 = shls_slice

    mol1 = get_fakemol_cs(mol, shls_slice)
    comp = _get_intor_and_comp(mol, eval_name, comp)[1]
    # stop tracing as only 1st order derivatives are non-zero
    ao1 = pyscf_eval_gto(mol1, eval_name, grid_coords, comp, None, non0tab,
                         None, None, None)
    ao1 = ao1.reshape(comp,ngrids,-1)

    _, cs_of, _ = setup_ctr_coeff(mol)

    nao_id0, nao_id1 = nao_nr_range(mol, shl0, shl1)
    nao = nao_id1 - nao_id0

    def _fill_grad():
        grad = numpy.zeros((comp,ngrids,len(ctr_coeff),nao), dtype=ao1.dtype)
        off = 0
        ibas = 0
        for i in range(shl0, shl1):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nbas = (l+1)*(l+2)//2
            else:
                nbas = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = ao1[...,off:(off+nprim*nbas)].reshape(comp,ngrids,nprim,nbas)
            for j in range(nctr):
                grad[...,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),ibas:(ibas+nbas)] += g
                ibas += nbas
            off += nprim*nbas
        return grad
    grad = _fill_grad()
    tangent_out = np.einsum('cgxi,x->cgi', grad, ctr_coeff_t)

    # TODO improve performance
    #@jit
    #def _dot_grad_tangent(ao1, tangent):
    #    tangent_out = np.empty((comp,ngrids,nao), dtype=ao1.dtype)
    #    off = 0
    #    ibas = 0
    #    for i in range(shl0, shl1):
    #        l = mol._bas[i,pyscf_mole.ANG_OF]
    #        if mol.cart:
    #            nbas = (l+1)*(l+2)//2
    #        else:
    #            nbas = 2*l + 1
    #        nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
    #        nctr = mol._bas[i,pyscf_mole.NCTR_OF]
    #        g = ao1[...,off:(off+nprim*nbas)].reshape(comp,ngrids,nprim,nbas)
    #        for j in range(nctr):
    #            out = np.einsum('cgxi,x->cgi', g,
    #                            tangent[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim)])
    #            tangent_out = tangent_out.at[...,ibas:(ibas+nbas)].set(out)
    #            ibas += nbas
    #        off += nprim*nbas
    #    return tangent_out
    #tangent_out = _dot_grad_tangent(ao1, ctr_coeff_t)

    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out

def _eval_gto_jvp_exp(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    if 'sph' in eval_name:
        eval_name = eval_name.replace('sph', 'cart')
        cart = False
    elif 'cart' in eval_name:
        cart = True
    else:
        raise KeyError

    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    shl0, shl1 = shls_slice

    mol1 = get_fakemol_exp(mol, shls_slice=shls_slice)
    comp = _get_intor_and_comp(mol, eval_name, comp)[1]
    # FIXME 1st order derivatives only
    ao1 = pyscf_eval_gto(mol1, eval_name, grid_coords, comp, None, non0tab,
                         None, None, None)

    es, es_of, _env_of = setup_exp(mol)

    ngrids = len(grid_coords)
    nao = pyscf_mole.nao_cart(mol)
    ao1 = ao1.reshape(comp, ngrids, -1)

    # TODO improve performance
    @jit
    def _fill_grad(ao1):
        grad = np.zeros((comp,ngrids,len(es),nao), dtype=ao1.dtype)
        off = 0
        ibas = 0
        for i in range(shl0, shl1):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nbas = (l+1)*(l+2)//2
            nbas1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = ao1[:,:,off:off+nprim*nbas1].reshape(comp, ngrids, nprim, nbas1)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= 0.282094791773878143 # normalization factor for s orbital
                    elif l == 1:
                        c *= 0.488602511902919921 # normalization factor for p orbital
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,:,j,idx_x] + g[:,:,j,idx_y] + g[:,:,j,idx_z]) * c
                        grad = grad.at[:,:,ioff+j,jbas].add(gc)
                        jbas += 1
                ibas += nbas
            off += nprim * nbas1
        return grad

    grad = _fill_grad(ao1)
    tangent_out = np.einsum('cgxi,x->cgi', grad, mol_t.exp)
    if not cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = np.einsum('cgp,pi->cgi', tangent_out, c2s)
    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out
