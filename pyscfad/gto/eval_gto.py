from functools import partial
import numpy
from jax import custom_jvp
from pyscf.gto.moleintor import make_loc
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from .moleintor import get_bas_label

_MAX_DERIV_ORDER = 4
_DERIV_LABEL = []
for i in range(_MAX_DERIV_ORDER+1):
    if i == 0:
        _label = ["",]
    else:
        _label = get_bas_label(i)
    _DERIV_LABEL += _label

def eval_gto(mol, eval_name, grid_coords,
             comp=None, shls_slice=None, non0tab=None, ao_loc=None, out=None):

    return _eval_gto(mol, eval_name, grid_coords,
                     comp, shls_slice, non0tab, ao_loc, out)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,8)))
def _eval_gto(mol, eval_name, grid_coords,
              comp, shls_slice, non0tab, ao_loc, out):
    return pyscf_eval_gto(mol, eval_name, grid_coords, comp, shls_slice, non0tab,
                          ao_loc, out)

@_eval_gto.defjvp
def _eval_gto_jvp(eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc, out,
                  primals, tangents):
    mol, = primals
    mol_t, = tangents

    primal_out = _eval_gto(mol, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc, out)
    tangent_out = jnp.zeros_like(primal_out)

    if mol.coords is not None:
        tangent_out += _eval_gto_jvp_r0(mol, mol_t, eval_name, grid_coords,
                                        comp, shls_slice, non0tab, ao_loc)
    if mol.ctr_coeff is not None:
        tangent_out += _eval_gto_jvp_cs(mol, mol_t, eval_name, grid_coords,
                                        comp, shls_slice, non0tab, ao_loc)
    if mol.exp is not None:
        tangent_out += _eval_gto_jvp_exp(mol, mol_t, eval_name, grid_coords,
                                         comp, shls_slice, non0tab, ao_loc)
    return primal_out, tangent_out

def _eval_gto_jvp_r0(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    coords_t = mol_t.coords
    if "deriv"+str(_MAX_DERIV_ORDER) in eval_name:
        raise NotImplementedError
    if "deriv" not in eval_name:
        new_eval = eval_name + "_deriv1"
        order = 0
    else:
        tmp = eval_name.split("deriv", 1)
        order = int(tmp[1])
        new_eval = tmp[0] + "deriv" + str(order + 1)

    nc = (order+1) * (order+2) * (order+3) // 6
    if comp is not None:
        assert comp == nc

    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    sh0, sh1 = shls_slice
    if ao_loc is None:
        ao_loc = make_loc(mol._bas, new_eval)
    ao_start = ao_loc[sh0]
    ao_end = ao_loc[sh1]
    nao = ao_end - ao_start
    ng = len(grid_coords)

    ao1 = - pyscf_eval_gto(mol, new_eval, grid_coords, None, shls_slice, non0tab, ao_loc)
    grad = numpy.zeros([mol.natm,3,nc,ng,nao], dtype=ao1.dtype)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom(ao_loc)

    #if nc == 1:
    #    tangent_out = jnp.zeros((ng,nao))
    #else:
    #    tangent_out = jnp.zeros((nc,ng,nao))
    #for iorder in range(order+1):
    #    for k, ia in enumerate(atmlst):
    #        p0, p1 = aoslices [ia, 2:]
    #        if p1 <= ao_start:
    #            continue
    #        id0 = max(0, p0 - ao_start)
    #        id1 = min(p1, ao_end) - ao_start
    #        if order == 0:
    #            tmp = jnp.einsum('xgi,x->gi', ao1[1:4,:,id0:id1], coords_t[k])
    #            tangent_out = ops.index_add(tangent_out, ops.index[:,id0:id1], tmp)
    #        else:
    #            start0 = iorder * (iorder+1) * (iorder+2) // 6
    #            start = (iorder+1) * (iorder+2) * (iorder+3) // 6
    #            end = (iorder+2) * (iorder+3) * (iorder+4) // 6
    #            for il, label in enumerate(get_bas_label(iorder)):
    #                idx_x = _DERIV_LABEL.index("".join(sorted(label + 'x')), start, end)
    #                idx_y = _DERIV_LABEL.index("".join(sorted(label + 'y')), start, end)
    #                idx_z = _DERIV_LABEL.index("".join(sorted(label + 'z')), start, end)
    #                tmp = (ao1[idx_x,:,id0:id1] * coords_t[k,0]
    #                       + ao1[idx_y,:,id0:id1] * coords_t[k,1]
    #                       + ao1[idx_z,:,id0:id1] * coords_t[k,2])
    #                tangent_out = ops.index_add(tangent_out, ops.index[start0+il,:,id0:id1], tmp)

    for iorder in range(order+1):
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia, 2:]
            if p1 <= ao_start:
                continue
            id0 = max(0, p0 - ao_start)
            id1 = min(p1, ao_end) - ao_start

            start0 = iorder * (iorder+1) * (iorder+2) // 6
            start = (iorder+1) * (iorder+2) * (iorder+3) // 6
            end = (iorder+2) * (iorder+3) * (iorder+4) // 6
            for il, label in enumerate(get_bas_label(iorder)):
                idx_x = _DERIV_LABEL.index("".join(sorted(label + 'x')), start, end)
                idx_y = _DERIV_LABEL.index("".join(sorted(label + 'y')), start, end)
                idx_z = _DERIV_LABEL.index("".join(sorted(label + 'z')), start, end)
                grad[k,0,start0+il,:,id0:id1] += ao1[idx_x,:,id0:id1]
                grad[k,1,start0+il,:,id0:id1] += ao1[idx_y,:,id0:id1]
                grad[k,2,start0+il,:,id0:id1] += ao1[idx_z,:,id0:id1]
    ao1 = None
    tangent_out = jnp.einsum("nxlgi,nx->lgi", grad, coords_t)
    grad = None
    if nc == 1:
        tangent_out = tangent_out[0]
    return tangent_out

def _eval_gto_jvp_cs(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    return 0

def _eval_gto_jvp_exp(mol, mol_t, eval_name, grid_coords, comp, shls_slice, non0tab, ao_loc):
    return 0
