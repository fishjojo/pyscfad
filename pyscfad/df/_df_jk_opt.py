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
from scipy.linalg import eigh, qr
from scipy.linalg.lapack import dpstrf
from jax import custom_vjp
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax.tree_util import tree_map
from pyscf import lib
from pyscf.df import df_jk as pyscf_df_jk
from pyscfad.lib import logger
from pyscfadlib import libcvhf_vjp as libvhf

libao2mo = lib.load_library('libao2mo')

def _tag_dm(dm, hermi):
    '''Tag a positive semidefinite low-rank dm with its pivoted Cholesky
    factor dm = mo_coeff * mo_coeff.T (mo_occ = 1), so that pyscf builds K
    from the factor, O(naux*nocc*nao^2), instead of from dm, O(naux*nao^3).
    '''
    if hermi != 1 or hasattr(dm, 'mo_coeff'):
        return dm
    dms = numpy.asarray(dm)
    if numpy.iscomplexobj(dms):
        return dm
    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros(dms.shape[:2])
    for k, d in enumerate(dms):
        # default tol: nao * eps * max(diag(d))
        c, piv, rank = dpstrf(d, lower=1)[:3]
        mo_coeff[k,piv-1,:rank] = numpy.tril(c[:,:rank])
        mo_occ[k,:rank] = 1
    return lib.tag_array(numpy.asarray(dm), mo_coeff=mo_coeff, mo_occ=mo_occ)

@partial(custom_vjp, nondiff_argnums=(2,3,4,5))
def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    if with_k:
        dm = _tag_dm(dm, hermi)
    return pyscf_df_jk.get_jk(dfobj, dm, hermi=hermi,
                              with_j=with_j, with_k=with_k,
                              direct_scf_tol=direct_scf_tol)

def get_jk_fwd(dfobj, dm, hermi, with_j, with_k, direct_scf_tol):
    # NOTE with symbolic_zeros, the inputs are CustomVJPPrimal, telling
    # get_jk_bwd which cotangents are needed
    perturbed = (dfobj._cderi.perturbed, dm.perturbed)
    dfobj = tree_map(lambda x: x.value, dfobj,
                     is_leaf=lambda x: isinstance(x, CustomVJPPrimal))
    dm = dm.value
    vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    return (vj, vk), (dfobj, dm, perturbed)

def get_jk_bwd(hermi, with_j, with_k, direct_scf_tol,
               res, ybar):
    dfobj, dm, (eri_perturbed, dm_perturbed) = res
    vj_bar, vk_bar = ybar
    if not with_j or isinstance(vj_bar, SymbolicZero):
        vj_bar = None
    if not with_k or isinstance(vk_bar, SymbolicZero):
        vk_bar = None

    log = logger.new_logger(dfobj)
    dfobj_bar = dm_bar = None
    if eri_perturbed:
        if vk_bar is None or _lowrank_vk_vjp(dm, hermi):
            eri_bar = _eri_bar(dfobj, dm, hermi, vj_bar, vk_bar)
            if dm_perturbed:
                dm_bar = _dm_bar(dfobj, dm, hermi, vj_bar, vk_bar)
        else:
            eri_bar, dm_bar = _eri_dm_bar(dfobj, dm, vj_bar, vk_bar)
        # J/K depend on mol and auxmol only through _cderi (None as zero)
        dfobj_bar = tree_map(lambda x: None, dfobj)
        dfobj_bar._cderi = eri_bar
    elif dm_perturbed:
        dm_bar = _dm_bar(dfobj, dm, hermi, vj_bar, vk_bar)
    if not dm_perturbed:
        dm_bar = None
    log.timer('get_jk_bwd')
    del log
    return (dfobj_bar, dm_bar)

def _dm_bar(dfobj, dm, hermi, vj_bar, vk_bar):
    '''Cotangent of dm alone. As vj and vk are self-adjoint linear maps
    of dm, it is a J/K build with the cotangents as density matrices.
    '''
    dm_bar = numpy.zeros(numpy.shape(dm))
    if vj_bar is not None:
        dm_bar += pyscf_df_jk.get_jk(dfobj, numpy.asarray(vj_bar), hermi=0,
                                     with_k=False)[0]
    if vk_bar is not None:
        dm_bar += _get_k(dfobj, dm, hermi, numpy.asarray(vk_bar))
    return dm_bar

def _get_k(dfobj, dm, hermi, kbar):
    '''K of kbar. In the SCF response, kbar = c y^T + y c^T with c the
    occupied orbitals has rank <= 2*nocc. Its range is found from kbar
    times 2*nocc+10 random vectors, and kbar = q b q^T is split by the
    eigenvalues of the small b into positive and negative parts, whose K
    is built as from occupied orbitals, O(naux*nocc*nao^2), instead of
    from kbar, O(naux*nao^3). The split costs O(nocc*nao^2).
    '''
    nao = kbar.shape[-1]
    kbars = kbar.reshape(-1,nao,nao)
    nset = len(kbars)
    dm = _tag_dm(dm, hermi)
    if numpy.isrealobj(kbars) and hasattr(dm, 'mo_occ'):
        nocc = (dm.mo_occ.reshape(nset,-1) > 0).sum(axis=1)
        tol = nao * numpy.finfo(kbars.dtype).eps
        rng = numpy.random.default_rng(1)
        parts = []
        for kb, n in zip(kbars, nocc):
            nvec = 2*n + 10
            if nvec > nao // 2:
                break
            vec = lib.dot(kb, rng.standard_normal((nao,nvec)))
            q = qr(vec, mode='economic')[0]
            b = lib.dot(q.T, lib.dot(kb, q))
            w, v = eigh(b)
            idx = abs(w) > tol * abs(w).max()
            z, w = lib.dot(q, v[:,idx]), w[idx]
            # NOTE the residual is round-off if the range of kbar is captured
            if abs(kb - lib.dot(z*w, z.T)).max() > 10 * tol * abs(kb).max():
                break
            parts.append((z[:,w>0] * w[w>0]**.5, z[:,w<0] * (-w[w<0])**.5))
        else:
            # kbar = xp xp^T - xn xn^T
            parts = [xp for xp, _ in parts] + [xn for _, xn in parts]
            nmo = max(x.shape[1] for x in parts)
            orbs = numpy.zeros((2*nset,nao,nmo))
            occ = numpy.zeros((2*nset,nmo))
            for k, x in enumerate(parts):
                orbs[k,:,:x.shape[1]] = x
                occ[k,:x.shape[1]] = 1
            dms = lib.tag_array(numpy.zeros((2*nset,nao,nao)), # not used
                                mo_coeff=orbs, mo_occ=occ)
            vk = pyscf_df_jk.get_jk(dfobj, dms, hermi=1, with_j=False)[1]
            return (vk[:nset] - vk[nset:]).reshape(kbar.shape)
    return pyscf_df_jk.get_jk(dfobj, kbar, hermi=0, with_j=False)[1]

def _lowrank_vk_vjp(dm, hermi):
    # dm is factorized as in _tag_dm; df_vk_vjp_lowrank needs pyscfadlib>0.3.4
    return (hermi == 1 and not numpy.iscomplexobj(dm)
            and hasattr(libvhf, 'df_vk_vjp_lowrank'))

def _eri_bar(dfobj, dm, hermi, vj_bar, vk_bar):
    '''Cotangent of _cderi alone. The K part is built from the pivoted
    Cholesky factor of dm, O(naux*nocc*nao^2).
    '''
    dms = numpy.asarray(dm)
    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if vj_bar is not None:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

        vj_bar = numpy.asarray(vj_bar).reshape(-1,nao,nao)
        vj_bar_tril = lib.pack_tril(vj_bar + vj_bar.conj().transpose(0,2,1))
        vj_bar_tril[:,idx*(idx+1)//2+idx] *= .5

    if vk_bar is not None:
        vk_bar = numpy.asarray(vk_bar).reshape(-1,nao,nao)
        s = vk_bar + vk_bar.transpose(0,2,1)
        dm = _tag_dm(dm, hermi)
        mo_coeff = dm.mo_coeff.reshape(nset,nao,-1)
        mo_occ = dm.mo_occ.reshape(nset,-1)
        xs = [numpy.asarray(c[:,occ>0], order='F')
              for c, occ in zip(mo_coeff, mo_occ)]

    #TODO save eri_bar on disk
    nao_pair = nao*(nao+1)//2
    eri_bar = numpy.zeros((dfobj.get_naoaux(), nao_pair))
    # NOTE no unpacked buffer per block; the kernel unpacks one row per thread
    max_memory = dfobj.max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao_pair)))
    p1 = 0
    for eri1 in dfobj.loop(blksize):
        naux = eri1.shape[0]
        p0, p1 = p1, p1 + naux
        if vj_bar is not None:
            rho = dmtril @ eri1.T
            rho_bar = vj_bar_tril @ eri1.T
            lib.dot(rho.T, vj_bar_tril, c=eri_bar[p0:p1], beta=1)
            lib.dot(rho_bar.T, dmtril, c=eri_bar[p0:p1], beta=1)

        if vk_bar is None:
            continue
        for k in range(nset):
            nocc = xs[k].shape[1]
            if nocc == 0:
                continue
            libvhf.df_vk_vjp_lowrank(eri_bar[p0:p1].ctypes.data_as(ctypes.c_void_p),
                                     s[k].ctypes.data_as(ctypes.c_void_p),
                                     xs[k].ctypes.data_as(ctypes.c_void_p),
                                     eri1.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(naux), ctypes.c_int(nao),
                                     ctypes.c_int(nocc))
    return eri_bar

def _eri_dm_bar(dfobj, dm, vj_bar, vk_bar):
    '''Cotangents of _cderi and dm.'''
    fmmm = libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = libao2mo.AO2MOnr_e2_drv
    ftrans = libao2mo.AO2MOtranse2_nr_s2
    vjpdrv = libvhf.df_vk_vjp
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if vj_bar is not None:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

        vj_bar = numpy.asarray(vj_bar).reshape(-1,nao,nao)
        vj_bar_tril = lib.pack_tril(vj_bar + vj_bar.conj().transpose(0,2,1))
        vj_bar_tril[:,idx*(idx+1)//2+idx] *= .5

    #TODO save eri_bar on disk
    eri_bar = numpy.zeros((dfobj.get_naoaux(), nao*(nao+1)//2))
    # NOTE vjpdrv accumulates dms_bar in F order
    dms_bar = [numpy.zeros((nao,nao), order='F') for _ in range(nset)]

    if vk_bar is not None:
        vk_bar = numpy.asarray(vk_bar).reshape(-1,nao,nao)
        vk_bar = [numpy.asarray(x, order='F') for x in vk_bar]
        dms = [numpy.asarray(x, order='F') for x in dms]

    rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
             null, ctypes.c_int(0))
    max_memory = dfobj.max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
    buf = numpy.empty((blksize,nao,nao))
    p1 = 0
    for eri1 in dfobj.loop(blksize):
        naux, nao_pair = eri1.shape
        p0, p1 = p1, p1 + naux
        if vj_bar is not None:
            rho_bar = vj_bar_tril @ eri1.T
            dmtril_bar = rho_bar @ eri1
            for k in range(nset):
                dms_bar[k] += lib.unpack_tril(dmtril_bar[k])

            rho = dmtril @ eri1.T
            lib.dot(rho.T, vj_bar_tril, c=eri_bar[p0:p1], beta=1)
            lib.dot(rho_bar.T, dmtril, c=eri_bar[p0:p1], beta=1)

        if vk_bar is None:
            continue
        for k in range(nset):
            #TODO save buf1 on disk to avoid recomputation
            buf1 = buf[:naux]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 dms[k].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs)

            vjpdrv(eri_bar[p0:p1].ctypes.data_as(ctypes.c_void_p),
                   dms_bar[k].ctypes.data_as(ctypes.c_void_p),
                   vk_bar[k].ctypes.data_as(ctypes.c_void_p),
                   buf1.ctypes.data_as(ctypes.c_void_p),
                   eri1.ctypes.data_as(ctypes.c_void_p),
                   dms[k].ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(naux), ctypes.c_int(nao))

    dm_bar = numpy.asarray(dms_bar).reshape(dm_shape)
    return eri_bar, dm_bar

get_jk.defvjp(get_jk_fwd, get_jk_bwd, symbolic_zeros=True)
