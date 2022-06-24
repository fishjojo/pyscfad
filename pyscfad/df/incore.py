from functools import partial
import numpy
from jax import scipy
from jax import custom_jvp
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.gto.moleintor import getints
from pyscf.df.outcore import _guess_shell_ranges
from pyscf import __config__
from pyscfad.lib import ops
from pyscfad.lib import numpy as np
from . import addons

MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)

@partial(custom_jvp, nondiff_argnums=tuple(range(2,7)))
def int3c_cross(mol, auxmol, intor="int3c2e", comp=1, aosym="s1", out=None, shls_slice=None):
    assert aosym == 's1'
    int3c = gto.moleintor.ascint3(mol._add_suffix(intor))
    nbas = mol.nbas
    nauxbas = auxmol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
    pmol = mol + auxmol
    ints = gto.moleintor.getints3c(int3c, pmol._atm, pmol._bas, pmol._env,
                                   shls_slice=shls_slice, comp=comp,
                                   aosym=aosym, ao_loc=None, cintopt=None, out=out)
    return ints

@int3c_cross.defjvp
def int3c_cross_jvp(intor, comp, aosym, out, shls_slice, primals, tangents):
    mol, auxmol = primals
    mol_dot, auxmol_dot = tangents

    primal_out = int3c_cross(mol, auxmol, intor=intor, comp=comp,
                             aosym=aosym, out=out, shls_slice=shls_slice)
    tangent_out = np.zeros_like(primal_out)

    if intor.startswith("int3c2e") and not "spinor" in intor:
        intor_ip1 = intor.replace("int3c2e", "int3c2e_ip1")
        ints = int3c_cross(mol, auxmol, intor=intor_ip1, comp=3,
                           aosym=aosym, out=None, shls_slice=shls_slice)
        tangent_out_mol = _int3c_fill_grad_r0_ip1(mol, mol_dot, -ints)

        intor_ip2 = intor.replace("int3c2e", "int3c2e_ip2")
        ints = int3c_cross(mol, auxmol, intor=intor_ip2, comp=3,
                           aosym=aosym, out=None, shls_slice=shls_slice)
        tangent_out_auxmol = _int3c_fill_grad_r0_ip2(auxmol, auxmol_dot, -ints)
        tangent_out += tangent_out_mol + tangent_out_auxmol
    else:
        raise NotImplementedError

    if mol.ctr_coeff is not None:
        print("ctr_coeff derivative for int3c2e not implemented")
    if mol.exp is not None:
        print("exp derivative for int3c2e not implemented")

    return primal_out, tangent_out

def _int3c_fill_grad_r0_ip1(mol, mol_dot, ints):
    shape = [mol.natm,] + list(ints.shape)
    grad = np.zeros(shape, dtype=ints.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_update(grad, ops.index[ia,:,p0:p1], ints[:,p0:p1])
    tangent_out = np.einsum('nxijk,nx->ijk', grad, mol_dot.coords)
    tangent_out += tangent_out.transpose(1,0,2)
    return tangent_out

def _int3c_fill_grad_r0_ip2(mol, mol_dot, ints):
    shape = [mol.natm,] + list(ints.shape)
    grad = np.zeros(shape, dtype=ints.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_update(grad, ops.index[ia,...,p0:p1], ints[...,p0:p1])
    tangent_out = np.einsum('nxijk,nx->ijk', grad, mol_dot.coords)
    return tangent_out


def cholesky_eri(mol, auxmol=None, auxbasis='weigend+etb',
                 int3c='int3c2e', aosym='s1', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, verbose=0, fauxe2=None):
    '''
    Note: Only support s1 symmetry.
    '''
    assert comp == 1
    log = logger.new_logger(mol, verbose)
    t0 = (log._t0, log._w0)
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        tag = 'cd'
    # pylint: disable=bare-except
    except: #scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        idx = w > LINEAR_DEP_THR
        low = (v[:,idx] / np.sqrt(w[idx]))
        v = None
        tag = 'eig'
    j2c = None
    naoaux, naux = low.shape
    log.debug('size of aux basis %d', naux)
    log.timer_debug1('2c2e', *t0)

    nao = mol.nao
    if aosym != 's1':
        raise NotImplementedError

    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    ints = int3c_cross(mol, auxmol, intor=int3c, comp=comp, aosym=aosym,
                       out=None, shls_slice=shls_slice)
    ints = ints.reshape((-1,naoaux)).T

    if tag == 'cd':
        cderi = scipy.linalg.solve_triangular(low, ints, lower=True,
                                              overwrite_b=True, check_finite=False)
    else:
        cderi = np.dot(low.T, ints)

    log.timer('cholesky_eri', *t0)
    del log
    return cderi.reshape(-1,nao,nao)
