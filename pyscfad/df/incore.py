from functools import partial
from jax import scipy
from jax import custom_jvp
from jax import numpy as np
from pyscf.lib import logger
from pyscf import gto
from pyscf.df.outcore import _guess_shell_ranges
from pyscf import __config__
from pyscfad import ops
from pyscfad.ops import vmap, jit
from pyscfad import config
from . import addons, _int3c_cross_opt

MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)

@partial(custom_jvp, nondiff_argnums=tuple(range(2,7)))
def int3c_cross(mol, auxmol, intor='int3c2e', comp=1, aosym='s1',
                shls_slice=None, out=None):
    assert aosym == 's1'
    pmol = mol + auxmol
    atm = pmol._atm
    bas = pmol._bas
    env = pmol._env
    int3c = gto.moleintor.ascint3(mol._add_suffix(intor))
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    nbas = mol.nbas
    nauxbas = auxmol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
    ints = gto.moleintor.getints3c(int3c, atm, bas, env,
                                   shls_slice=shls_slice, comp=comp,
                                   aosym=aosym, ao_loc=ao_loc, cintopt=cintopt, out=out)
    return ints

@int3c_cross.defjvp
def int3c_cross_jvp(intor, comp, aosym, shls_slice, out,
                    primals, tangents):
    mol, auxmol = primals
    mol_dot, auxmol_dot = tangents
    assert shls_slice[0] == 0 and shls_slice[1] == mol.nbas

    primal_out = int3c_cross(mol, auxmol, intor=intor, comp=comp,
                             aosym=aosym, shls_slice=shls_slice, out=out)
    tangent_out = np.zeros_like(primal_out)

    if intor.startswith('int3c2e') and not 'spinor' in intor:
        intor_ip1 = intor.replace('int3c2e', 'int3c2e_ip1')
        ints = int3c_cross(mol, auxmol, intor=intor_ip1, comp=3,
                           aosym=aosym, shls_slice=shls_slice)
        tangent_out_mol = _int3c_fill_grad_r0_ip1(mol, mol_dot, -ints)

        intor_ip2 = intor.replace('int3c2e', 'int3c2e_ip2')
        ints = int3c_cross(mol, auxmol, intor=intor_ip2, comp=3,
                           aosym=aosym, shls_slice=shls_slice)
        tangent_out_auxmol = _int3c_fill_grad_r0_ip2(auxmol, auxmol_dot, -ints)
        tangent_out += tangent_out_mol + tangent_out_auxmol
    else:
        raise NotImplementedError

    if mol.ctr_coeff is not None:
        raise NotImplementedError('ctr_coeff derivative for int3c2e not implemented')
    if mol.exp is not None:
        raise NotImplementedError('exp derivative for int3c2e not implemented')
    return primal_out, tangent_out

@jit
def _int3c_fill_grad_r0_ip1(mol, mol_dot, ints):
    shape = [mol.natm,] + list(ints.shape)
    grad = np.zeros(shape, dtype=ints.dtype)
    aoslices = mol.aoslice_by_atom()
    shape = [1,] * ints.ndim
    shape[1] = ints.shape[1]
    idx = np.arange(shape[1])
    idx = idx.reshape(shape)
    #for ia in range(mol.natm):
    #    p0, p1 = aoslices[ia,2:]
    #    grad = ops.index_update(grad, ops.index[ia,:,p0:p1], ints[:,p0:p1])
    def body(slices):
        p0, p1 = slices[:]
        mask = (idx >= p0) & (idx < p1)
        return np.where(mask, ints, np.array(0, dtype=ints.dtype))
    grad = vmap(body)(aoslices[:,2:].reshape(-1,2))
    tangent_out = np.einsum('nxijk,nx->ijk', grad, mol_dot.coords)
    tangent_out += tangent_out.transpose(1,0,2)
    return tangent_out

@jit
def _int3c_fill_grad_r0_ip2(mol, mol_dot, ints):
    shape = [mol.natm,] + list(ints.shape)
    grad = np.zeros(shape, dtype=ints.dtype)
    aoslices = mol.aoslice_by_atom()
    shape = [1,] * ints.ndim
    shape[-1] = ints.shape[-1]
    idx = np.arange(shape[-1])
    idx = idx.reshape(shape)
    #for ia in range(mol.natm):
    #    p0, p1 = aoslices[ia,2:]
    #    grad = ops.index_update(grad, ops.index[ia,...,p0:p1], ints[...,p0:p1])
    def body(slices):
        p0, p1 = slices[:]
        mask = (idx >= p0) & (idx < p1)
        return np.where(mask, ints, np.array(0, dtype=ints.dtype))
    grad = vmap(body)(aoslices[:,2:].reshape(-1,2))
    tangent_out = np.einsum('nxijk,nx->ijk', grad, mol_dot.coords)
    return tangent_out


def cholesky_eri(mol, auxmol=None, auxbasis='weigend+etb',
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, verbose=0, fauxe2=None):
    if comp != 1:
        raise NotImplementedError

    log = logger.new_logger(mol, verbose)
    t0 = (log._t0, log._w0)

    if not config.moleintor_opt:
        log.warn('int3c2e symmetry turned off')
        aosym = 's1'

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
    if aosym == 's1':
        nao_pair = nao * nao
    else:
        nao_pair = nao * (nao+1) // 2

    cderi = np.empty((naux, nao_pair))

    max_words = max_memory*1e6/8 - low.size - cderi.size
    # Divide by 3 because scipy.linalg.solve may create a temporary copy for
    # ints and return another copy for results
    buflen = min(max(int(max_words/naoaux/comp/3), 8), nao_pair)
    if not config.moleintor_opt:
        # subshells not supported
        buflen = nao_pair
    shranges = _guess_shell_ranges(mol, buflen, aosym)
    log.debug1('shranges = %s', shranges)

    p1 = 0
    for istep, sh_range in enumerate(shranges):
        log.debug('int3c2e [%d/%d], AO [%d:%d], nrow = %d',
                  istep+1, len(shranges), *sh_range)

        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

        if config.moleintor_opt:
            ints = _int3c_cross_opt.int3c_cross(
                               mol, auxmol, intor=int3c, comp=comp, aosym=aosym,
                               shls_slice=shls_slice)
        else:
            ints = int3c_cross(mol, auxmol, intor=int3c, comp=comp, aosym=aosym,
                               shls_slice=shls_slice)
        ints = ints.reshape((-1,naoaux)).T

        p0, p1 = p1, p1 + nrow
        if tag == 'cd':
            cderi = ops.index_update(cderi, ops.index[:,p0:p1],
                                     scipy.linalg.solve_triangular(
                                        low, ints, lower=True,
                                        overwrite_b=True, check_finite=False))
        else:
            cderi = ops.index_update(cderi, ops.index[:,p0:p1],
                                     np.dot(low.T, ints))

    log.timer('cholesky_eri', *t0)
    del log
    return cderi
