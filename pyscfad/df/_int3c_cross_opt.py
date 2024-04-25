from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf.gto.moleintor import (
    ascint3,
    make_loc,
    make_cintopt,
    getints3c,
)

#libcgto = pyscf_lib.load_library('libcgto')
from pyscfadlib import libcgto_vjp as libcgto

@partial(custom_vjp, nondiff_argnums=(2,3,4,5,6))
def int3c_cross(mol, auxmol, intor='int3c2e', comp=1, aosym='s2ij',
                shls_slice=None, out=None):
    pmol = mol + auxmol
    atm = pmol._atm
    bas = pmol._bas
    env = pmol._env
    int3c = ascint3(mol._add_suffix(intor))
    ao_loc = make_loc(bas, int3c)
    cintopt = make_cintopt(atm, bas, env, int3c)
    nbas = mol.nbas
    nauxbas = auxmol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
    ints = getints3c(int3c, atm, bas, env,
                     shls_slice=shls_slice, comp=comp,
                     aosym=aosym, ao_loc=ao_loc, cintopt=cintopt, out=out)
    return ints

def int3c_cross_fwd(mol, auxmol, intor, comp, aosym, shls_slice, out):
    primal_out = int3c_cross(mol, auxmol, intor=intor, comp=comp,
                             aosym=aosym, shls_slice=shls_slice, out=out)
    return primal_out, (mol, auxmol)

def int3c_cross_bwd(intor, comp, aosym, shls_slice, out,
                    res, ybar):
    mol, auxmol = res
    log = logger.new_logger(mol)

    if mol.exp is not None:
        raise NotImplementedError
    if mol.ctr_coeff is not None:
        raise NotImplementedError
    if mol.r0 is not None:
        raise NotImplementedError
    if auxmol.exp is not None:
        raise NotImplementedError
    if auxmol.ctr_coeff is not None:
        raise NotImplementedError
    if auxmol.r0 is not None:
        raise NotImplementedError
    assert intor.startswith('int3c2e') and not 'spinor' in intor
    assert aosym == 's2ij'
    assert comp == 1
    assert shls_slice[5] - shls_slice[4] == auxmol.nbas

    natm = mol.natm
    comp = 3
    vjp = numpy.zeros((natm, comp), order='C', dtype=numpy.double)
    ybar = numpy.asarray(ybar, order='F', dtype=numpy.double)

    pmol = mol + auxmol
    atm = numpy.asarray(pmol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(pmol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(pmol._env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)

    intor1 = 'int3c2e_ip1'
    intor1 = ascint3(mol._add_suffix(intor1))
    ao_loc = make_loc(bas, intor1)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)
    cintopt = make_cintopt(atm, bas, env, intor1)

    drv = libcgto.GTOnr3c_ij_r0_vjp
    fill = getattr(libcgto, 'GTOnr3c_ij_r0_vjp_'+aosym)

    cintopt = make_cintopt(atm, bas, env, intor1)
    drv(getattr(libcgto, intor1), fill,
        vjp.ctypes.data_as(ctypes.c_void_p),
        ybar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), (ctypes.c_int*6)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        c_atm, ctypes.c_int(len(atm)), ctypes.c_int(natm),
        c_bas, ctypes.c_int(len(bas)), c_env)

    intor2 = 'int3c2e_ip2'
    ints1 = int3c_cross(mol, auxmol, intor=intor2, comp=3, aosym='s2ij',
                        shls_slice=shls_slice)

    vjp_aux = numpy.zeros_like(vjp)
    aoslices = auxmol.aoslice_by_atom()
    for ia in range(auxmol.natm):
        p0, p1 = aoslices[ia,2:]
        vjp_aux[ia] += pyscf_lib.einsum('xil,il->x', ints1[:,:,p0:p1], ybar[:,p0:p1])

    _, mol_tree = tree_flatten(mol)
    mol_bar = tree_unflatten(mol_tree, [-vjp,])
    _, auxmol_tree = tree_flatten(auxmol)
    auxmol_bar = tree_unflatten(auxmol_tree, [-vjp_aux,])
    log.timer('int3c_cross_bwd')
    del log
    return (mol_bar, auxmol_bar)

int3c_cross.defvjp(int3c_cross_fwd, int3c_cross_bwd)
