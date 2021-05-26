from functools import partial
import numpy
from jax import custom_jvp
from pyscf.gto.moleintor import make_loc
from pyscf.gto.eval_gto import _get_intor_and_comp
from pyscf.pbc.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.lib import numpy as jnp

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
    intor_ip = eval_name.replace("GTOval", "GTOval_ip")
    comp_ip = 3 # only 1st order derivative available
    ao1 = pyscf_eval_gto(cell, intor_ip, coords, comp_ip, kpts, kpt,
                         shls_slice, non0tab, ao_loc, out=None)

    ng = len(coords)
    single_kpt = False
    if isinstance(ao1, numpy.ndarray):
        ao1 = [ao1,]
        single_kpt = True
    nkpts = len(ao1)

    if shls_slice is None:
        shls_slice = (0, cell.nbas)
    sh0, sh1 = shls_slice
    if ao_loc is None:
        if eval_name[:3] == 'PBC':
            eval_name = eval_name[3:]
        eval_name, comp = _get_intor_and_comp(cell, eval_name, comp)
        eval_name = 'PBC' + eval_name
        ao_loc = make_loc(cell._bas, eval_name)
    ao_start = ao_loc[sh0]
    ao_end = ao_loc[sh1]
    nao = ao_end - ao_start
    atmlst = range(cell.natm)
    aoslices = cell.aoslice_by_atom(ao_loc)

    tangent_out = []
    coords_t = cell_t.coords
    for k in range(nkpts):
        grad = numpy.zeros([cell.natm,comp_ip,ng,nao], dtype=ao1[k].dtype)
        for ia in atmlst:
            p0, p1 = aoslices [ia, 2:]
            if p1 <= ao_start:
                continue
            id0 = max(0, p0 - ao_start)
            id1 = min(p1, ao_end) - ao_start

            grad[ia,...,id0:id1] += -ao1[k][...,id0:id1]
        tangent_out.append(jnp.einsum("nxgi,nx->gi", grad, coords_t))
        grad = None
    if single_kpt:
        tangent_out = tangent_out[0]
    return tangent_out
