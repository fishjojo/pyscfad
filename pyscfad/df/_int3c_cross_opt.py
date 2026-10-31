# Copyright 2021-2026 The PySCFAD Authors
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
from scipy.linalg import blas
from jax import custom_vjp
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from pyscf import lib
from pyscfad.lib import logger
from pyscfad.lib import _cotangent_buffer
from pyscfad.gto._pyscf_moleintor import (
    ascint3,
    make_loc,
    make_cintopt,
    getints3c
)

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
    # NOTE (..., naux, nao_pair), the C-order view of the F-order output,
    # so that neither the integrals nor their cotangent are transposed
    return ints.swapaxes(-1,-2)

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
    # NOTE (naux, nao_pair) in C order, the layout the driver reads
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

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

    # NOTE by translational invariance, the derivative of (ij|P) w.r.t. the
    # center of P is minus those w.r.t. the centers of i and j. So the ip1
    # contractions, run over the aux shells of one aux atom at a time,
    # give the cotangent of that aux atom as well.
    vjp_aux = numpy.zeros((auxmol.natm, comp))
    for ia, (b0, b1, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        if b0 == b1:
            continue
        vjp_ia = numpy.zeros_like(vjp)
        drv(getattr(libcgto, intor1), fill,
            vjp_ia.ctypes.data_as(ctypes.c_void_p),
            ybar[p0:p1].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp),
            (ctypes.c_int*6)(*shls_slice[:4], shls_slice[4]+b0, shls_slice[4]+b1),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            c_atm, ctypes.c_int(len(atm)), ctypes.c_int(natm),
            c_bas, ctypes.c_int(len(bas)), c_env)
        vjp += vjp_ia
        vjp_aux[ia] = -vjp_ia.sum(axis=0)

    _, mol_tree = tree_flatten(mol)
    mol_bar = tree_unflatten(mol_tree, [-vjp,])
    _, auxmol_tree = tree_flatten(auxmol)
    auxmol_bar = tree_unflatten(auxmol_tree, [-vjp_aux,])
    log.timer('int3c_cross_bwd')
    del log
    return (mol_bar, auxmol_bar)

int3c_cross.defvjp(int3c_cross_fwd, int3c_cross_bwd)


def _trsm_inplace(low, b, trans=False):
    '''b = low^-1 b (low^-T b if trans) in place, for the lower triangular
    low and b of shape (naux, n) in C order.'''
    # NOTE dtrsm(overwrite_b=1) writes into b even if it is read-only,
    # e.g., a numpy view of a JAX array
    assert b.flags.writeable, 'b is overwritten in place'
    # (low^-1 b)^T = b^T low^-T, with b^T in F order
    bt = b.T
    x = blas.dtrsm(1., low, bt, side=1, lower=1, trans_a=int(not trans),
                   overwrite_b=1)
    if x.ctypes.data != bt.ctypes.data:
        bt[:] = x
    return b

def _pair_ranges(mol, intor, shranges):
    '''AO shell slices and ranges of packed AO pairs of shranges'''
    ao_loc = make_loc(mol._bas, ascint3(mol._add_suffix(intor)))
    p1 = 0
    for bstart, bend, _ in shranges:
        i0 = int(ao_loc[bstart])
        i1 = int(ao_loc[bend])
        p0, p1 = p1, p1 + (i1*(i1+1) - i0*(i0+1)) // 2
        yield (bstart, bend, 0, mol.nbas), p0, p1

def _int3c2e_cholesky(mol, auxmol, low, intor, shranges):
    pmol = mol + auxmol
    atm = pmol._atm
    bas = pmol._bas
    env = pmol._env
    int3c = ascint3(mol._add_suffix(intor))
    ao_loc = make_loc(bas, int3c)
    cintopt = make_cintopt(atm, bas, env, int3c)
    nbas = mol.nbas
    nao = int(ao_loc[nbas])
    naux = int(ao_loc[nbas+auxmol.nbas]) - nao
    low = numpy.asarray(low, order='F')

    cderi = numpy.empty((naux, nao*(nao+1)//2))
    for shls_slice, p0, p1 in _pair_ranges(mol, intor, shranges):
        shls_slice = shls_slice + (nbas, nbas+auxmol.nbas)
        # the integrals of all AO pairs go to cderi directly
        out = cderi if p1 - p0 == cderi.shape[1] else None
        # (nij, naux) in F order
        ints = getints3c(int3c, atm, bas, env, shls_slice=shls_slice, comp=1,
                         aosym='s2ij', ao_loc=ao_loc, cintopt=cintopt, out=out)
        _trsm_inplace(low, ints.T)
        if out is None:
            cderi[:,p0:p1] = ints.T
        ints = None
    return cderi

@partial(custom_vjp, nondiff_argnums=(3,4))
def int3c2e_cholesky(mol, auxmol, low, intor, shranges):
    '''The density fitting tensor low^-1 (P|ij), (naux, nao_pair) in C order,
    with low the lower triangular Cholesky factor of the 2c2e integrals and
    shranges the ranges of AO shells (see pyscf.df.outcore._guess_shell_ranges).

    Fuses int3c_cross and the triangular solve so that the tensor, and its
    cotangent in the backward pass, are each built in place in one array.
    The consumers add the cotangent to the buffer of the tensor in place
    (see pyscfad.lib._cotangent_buffer).
    '''
    return _int3c2e_cholesky(mol, auxmol, low, intor, shranges)

def int3c2e_cholesky_fwd(mol, auxmol, low, intor, shranges):
    mol, auxmol, low = tree_map(lambda x: x.value, (mol, auxmol, low),
                                is_leaf=lambda x: isinstance(x, CustomVJPPrimal))
    cderi = _int3c2e_cholesky(mol, auxmol, low, intor, shranges)
    _cotangent_buffer.register(cderi)
    return cderi, (mol, auxmol, low, cderi)

def int3c2e_cholesky_bwd(intor, shranges, res, cderi_bar):
    mol, auxmol, low, cderi = res
    log = logger.new_logger(mol)
    t0 = log.get_t0()
    ints_bar = _cotangent_buffer.pop(cderi)
    if not isinstance(cderi_bar, SymbolicZero):
        if ints_bar is None:
            ints_bar = numpy.array(cderi_bar, dtype=numpy.float64, order='C')
        else:
            ints_bar += numpy.asarray(cderi_bar)
    if ints_bar is None:
        return (None, None, None)

    # cderi = low^-1 ints, so that ints_bar = low^-T cderi_bar
    # and low_bar = -tril(low^-T cderi_bar cderi^T)
    low = numpy.asarray(low, order='F')
    _trsm_inplace(low, ints_bar, trans=True)
    low_bar = -numpy.tril(lib.dot(ints_bar, cderi.T))
    log.timer_debug1('int3c2e_cholesky_bwd triangular solve', *t0)

    nbas = mol.nbas
    mol_bar = auxmol_bar = None
    for shls_slice, p0, p1 in _pair_ranges(mol, intor, shranges):
        shls_slice = shls_slice + (nbas, nbas+auxmol.nbas)
        cts = int3c_cross_bwd(intor, 1, 's2ij', shls_slice, None,
                              (mol, auxmol), ints_bar[:,p0:p1])
        if mol_bar is None:
            mol_bar, auxmol_bar = cts
        else:
            mol_bar, auxmol_bar = tree_map(numpy.add, (mol_bar, auxmol_bar), cts)
    log.timer('int3c2e_cholesky_bwd', *t0)
    del log
    return mol_bar, auxmol_bar, low_bar

int3c2e_cholesky.defvjp(int3c2e_cholesky_fwd, int3c2e_cholesky_bwd,
                        symbolic_zeros=True)
