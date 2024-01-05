from functools import partial
import numpy
#import jax
from pyscf import numpy as np
from pyscf.gto.moleintor import make_loc
from pyscf.gto.eval_gto import _get_intor_and_comp
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.lib import jit, custom_jvp, vmap
#from pyscfad.gto.moleintor import get_bas_label

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
    if 'deriv'+str(_MAX_DERIV_ORDER) in eval_name:
        raise NotImplementedError
    if 'deriv' not in eval_name:
        new_eval = eval_name + '_deriv1'
        order = 0
    else:
        tmp = eval_name.split('deriv', 1)
        order = int(tmp[1])
        new_eval = tmp[0] + 'deriv' + str(order + 1)

    ng = grid_coords.shape[0]
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
    atmlst = np.asarray(range(mol.natm))
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
    if 'deriv'+str(_MAX_DERIV_ORDER) in eval_name:
        raise NotImplementedError
    if 'deriv' not in eval_name:
        new_eval = eval_name + '_deriv1'
        order = 0
    else:
        tmp = eval_name.split('deriv', 1)
        order = int(tmp[1])
        new_eval = tmp[0] + 'deriv' + str(order + 1)

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
    return 0

def _eval_gto_jvp_exp(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    return 0
