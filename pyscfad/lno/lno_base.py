# Copyright 2023-2026 The PySCFAD Authors
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

import warnings
from functools import reduce
import numpy
from pyscf.mp.mp2 import get_frozen_mask, get_nmo, get_nocc

from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.ops import stop_trace, stop_grad
from pyscfad import scipy
from pyscfad.tools import timer
from pyscfad.ao2mo import _ao2mo
from pyscfad.lno import _checkpointed
from pyscfad.lno.mp2_rdm import make_rdm1_vo, make_rdm1_vo_frag
from pyscfad.lno.tools import autofrag, map_lo_to_frag

USE_CHECKPOINT = True
THRESH_INTERNAL = 1e-10
THRESH_OCC = 1e-6
# Tuned for benzene;
# must be bigger than the energy difference
# between degenerate semicanonical orbitals
SEMICANONICAL_DEG_THRESH = 1e-8
# Anything not bigger than the NO occupation number gap should work
COMPRESS_DEG_THRESH = 1e-12

def kernel(mfcc, orbloc, frag_lolist,
           no_type='ie', eris=None, frag_nonvlist=None):
    if eris is None:
        eris = mfcc.ao2mo()

    if mfcc.dm_corr is True:
        mfcc.dm_corr = make_rdm1_vo(mfcc, eris=eris, ao_repr=True)
    elif mfcc.dm_corr is False:
        mfcc.dm_corr = None

    nfrag = len(frag_lolist)
    if frag_nonvlist is None:
        frag_nonvlist = [[None,None]] * nfrag

    frag_res = [None] * nfrag
    for ifrag in range(nfrag):
        fraglo = numpy.asarray(frag_lolist[ifrag]).ravel()
        orbfragloc = orbloc[:,fraglo]
        frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        frag_res[ifrag] = kernel_1frag(mfcc, eris, orbfragloc, no_type,
                                       frag_target_nocc=frag_target_nocc,
                                       frag_target_nvir=frag_target_nvir)
    return frag_res


def kernel_1frag(mfcc, eris, orbfragloc, no_type,
                 frag_target_nocc=None, frag_target_nvir=None):
    mf = mfcc._scf
    frozen_mask = mfcc.get_frozen_mask()
    thresh_pno = (mfcc.thresh_occ, mfcc.thresh_vir)
    frzfrag, orbfrag = make_fpno1(mfcc, eris, orbfragloc, no_type,
                                  THRESH_INTERNAL, thresh_pno,
                                  frozen_mask=frozen_mask,
                                  frag_target_nocc=frag_target_nocc,
                                  frag_target_nvir=frag_target_nvir)
    if orbfrag is None:
        return (0., 0., 0.)
    frag_res = mfcc.impurity_solve(mf, orbfrag, orbfragloc,
                                   frozen=frzfrag, eris=eris)
    return frag_res

def make_fpno1(mfcc, eris, orbfragloc, no_type, thresh_internal, thresh_external,
               frozen_mask=None, frag_target_nocc=None, frag_target_nvir=None):
    mytimer = timer.Timer()

    mf = mfcc._scf
    mo_occ = mf.mo_occ
    nocc = numpy.count_nonzero(mo_occ > THRESH_OCC)
    nmo = mo_occ.size

    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo()
    moeocc0, moeocc1, moevir1, moevir0 = mfcc.split_moe()
    nocc0, nocc1, nvir1, nvir0 = [m.size for m in [moeocc0,moeocc1,
                                                   moevir1,moevir0]]
    nlo = orbfragloc.shape[1]
    s1e = eris.s1e
    fock = eris.fock
    Lov = eris.Lov

    lovir = False
    if mfcc.use_local_virt:
        lovir = abs(reduce(numpy.dot,
                    (stop_grad(orbfragloc.T),
                     stop_grad(s1e),
                     stop_grad(orbvir1)))).max() > thresh_internal

    if isinstance(thresh_external, float):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    # sanity check for no_type:
    if not lovir and no_type[0] != 'i':
        raise ValueError('Input LOs span only occ but input no_type[0] is not "i".')
    if not lovir and no_type[1] == 'i':
        raise ValueError('Input LOs span only occ but input no_type[1] is "i".')

    # split active occ/vir into internal(1) and external(2)
    m = reduce(np.dot, (orbfragloc.T, s1e, orbocc1))
    uocc1, uocc2 = projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = semicanonicalize(fock, np.dot(orbocc1, uocc1))

    uvir2 = None
    if lovir:
        m = reduce(np.dot, (orbfragloc.T, s1e, orbvir1))
        uvir1, uvir2 = projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = semicanonicalize(fock, np.dot(orbvir1, uvir1))

    # augment virtual space
    uuocc2_corr = uuvir2_corr = None
    if mfcc.dm_corr is not None:
        uuvir2_corr = augment_virt(mfcc.dm_corr, orbfragocc1, orbvir1,
                                   min(thresh_ext_occ, thresh_ext_vir),
                                   s1e, uvir2)
        if uuvir2_corr.shape[-1] == 0:
            uuvir2_corr = None

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1)
    def moe_oV(moefragvir):
        return (moeocc1[:,None] - moefragvir)
    eov = moe_Ov(moeocc1)

    # Construct PT2 dm_vv
    if no_type == 'osv':
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        Lia = eris.get_Ov(u)
        ovov = np.einsum('lIa, lIb->Iab', Lia, Lia)
        eia = moe_Ov(moefragocc1)
        eiajb = eia[:,:,None] + eia[:,None,:]
        dmvv = ovov / eiajb
        if lovir:
            dmvv = np.einsum('ip,Ipq,qj->Iij', uvir2.T, dmvv, uvir2)
        eia = Lia = ovov = eiajb = None
    elif no_type == 'ie' and USE_CHECKPOINT:
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        eia = moe_Ov(moefragocc1)
        ejb = eov
        Lia = eris.get_Ov(u)
        Ljb = Lov
        dmvv, dmoo = _checkpointed.make_mp2_rdm1_ie(Lia, Ljb, eia, ejb)
        if mfcc.dm_corr_frag is True:
            _dmov = make_rdm1_vo_frag(mfcc, dmoo, dmvv,
                                      Lia, Ljb, eia, ejb,
                                      eris=eris, ao_repr=False).T
            uuocc2_corr, uuvir2_corr = augment_ov(_dmov,
                                                  min(thresh_ext_occ, thresh_ext_vir),
                                                  uocc2, uvir2)
            if uuocc2_corr.shape[-1] == 0:
                uuocc2_corr = None
            if uuvir2_corr.shape[-1] == 0:
                uuvir2_corr = None
        eia = ejb = Lia = Ljb = None
    elif no_type[1] == 'r':   # OvOv: IaJc,IbJc->ab
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        ovov = eris.get_OvOv(u)
        eia = ejb = moe_Ov(moefragocc1)
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    elif no_type[1] == 'e': # Ovov: Iajc,Ibjc->ab
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        ovov = eris.get_Ovov(u)
        eia = moe_Ov(moefragocc1)
        ejb = eov
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    else:                   # oVov: iCja,iCjb->ab
        u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
        ovov = eris.get_oVov(u)
        eia = moe_oV(moefragvir1)
        ejb = eov
        e1_or_e2 = 'e2'
        swapidx = 'ij'

    if no_type != 'osv':
        if no_type != 'ie' or not USE_CHECKPOINT:
            eiajb = (eia.ravel()[:,None] + ejb.ravel()).reshape(ovov.shape)
            t2 = ovov / eiajb
            dmvv = make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
            ovov = eiajb = None
        if lovir:
            dmvv = reduce(np.dot, (uvir2.T, dmvv, uvir2))

    # Construct PT2 dm_oo
    if no_type == 'osv':
        u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
        Lia = eris.get_oV(u)
        ovov = np.einsum('liA, ljA->ijA', Lia, Lia)
        eia = moe_oV(moefragvir1)
        eiajb = eia[:,None,:] + eia[None,:,:]
        dmoo = ovov / eiajb
        dmoo = np.einsum('ip,pqA,qj->Aij', uocc2.T, dmoo, uocc2)
        eia = Lia = ovov = eiajb = None
    elif no_type in ['ie','ei']: # ie/ei share same t2
        if no_type[0] == 'e':   # oVov: iAkb,jAkb->ij
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            e1_or_e2 = 'e2'
            swapidx = 'ab'
    else:
        t2 = None
        if no_type[0] == 'r':   # oVoV: iAkB,jAkB->ij
            u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
            ovov = eris.get_oVoV(u)
            eia = ejb = moe_oV(moefragvir1)
            e1_or_e2 = 'e1'
            swapidx = 'ab'
        elif no_type[0] == 'e': # oVov: iAkb,jAkb->ij
            u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
            ovov = eris.get_oVov(u)
            eia = moe_oV(moefragvir1)
            ejb = eov
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
            ovov = eris.get_Ovov(u)
            eia = moe_Ov(moefragocc1)
            ejb = eov
            e1_or_e2 = 'e2'
            swapidx = 'ab'

        eiajb = (eia.ravel()[:,None] + ejb.ravel()).reshape(ovov.shape)
        t2 = ovov / eiajb
        ovov = eiajb = None

    if no_type != 'osv':
        if no_type != 'ie' or not USE_CHECKPOINT:
            dmoo = make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
            t2 = None
        dmoo = reduce(np.dot, (uocc2.T, dmoo, uocc2))

    # Compress external space by PNO
    if frag_target_nocc is not None:
        frag_target_nocc -= orbfragocc1.shape[1]
    if no_type == 'osv':
        orbfragocc2, orbfragocc0 = osv_compression(dmoo, orbocc1, thresh_ext_occ,
                                                   uocc2, frag_target_nocc)
    else:
        if uocc2.shape[-1] == 0:
            orbfragocc12 = orbfragocc1
            orbfragocc0 = np.zeros((orbfragocc12.shape[0],0))
        else:
            orbfragocc2, orbfragocc0 = natorb_compression(dmoo, orbocc1, thresh_ext_occ,
                                                          uocc2, frag_target_nocc,
                                                          uuocc2_corr, mfcc.natorb_occdeg_thresh)
            orbfragocc12 = semicanonicalize(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    if lovir:
        if frag_target_nvir is not None:
            frag_target_nvir -= orbfragvir1.shape[1]
        if no_type == 'osv':
            orbfragvir2, orbfragvir0 = osv_compression(dmvv, orbvir1, thresh_ext_vir,
                                                       uvir2, frag_target_nvir)
        else:
            orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                          uvir2, frag_target_nvir,
                                                          uuvir2_corr, mfcc.natorb_occdeg_thresh)
        orbfragvir12 = semicanonicalize(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
    else:
        orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                      None, frag_target_nvir,
                                                      uuvir2_corr, mfcc.natorb_occdeg_thresh)
        if orbfragvir2.shape[-1] == 0:
            warnings.warn('No virtual orbital is included for this fragment, '
                          'setting correlation energy to zero.')
            return None, None
        else:
            orbfragvir12 = semicanonicalize(fock, orbfragvir2)[1]

    orbfrag = np.hstack([orbocc0, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, orbvir0])
    frzfrag = numpy.hstack([numpy.arange(orbocc0.shape[1]+orbfragocc0.shape[1]),
                            numpy.arange(nocc+orbfragvir12.shape[1],nmo)])

    mytimer.timer('make_fpno1:')
    return frzfrag, orbfrag

def make_rdm1_mp2(t2, kind, e1_or_e2, swapidx):
    r''' Calculate MP2 rdm1 from T2.

    Args:
        t2 (np.ndarray):
            In 'ovov' order.
        kind (str):
            'oo' for oo-block; 'vv' for vv-block
        e1_or_e2 (str):
            Which electron are the free indices on?
            'e1': iakb,jakb -> ij; iajc,ibjc -> ab
            'e2': kaib,kajb -> ij; icja,icjb -> ab
        swapidx (str):
            How is the exchange term handled in einsum?
            'ij': iajb --> jaib
            'ab': iajb --> ibja
    '''
    if kind not in ['oo','vv']:
        raise KeyError('kind must be "oo" or "vv".')
    if e1_or_e2 not in ['e1','e2']:
        raise KeyError('e1_or_e2 must be "e1" or "e2".')
    if swapidx not in ['ij','ab']:
        raise KeyError('swapidx must be "ij" or "ab".')

    def swapped(s, swapidx):
        assert len(s) == 4
        order = [2,1,0,3] if swapidx == 'ij' else [0,3,2,1]
        return ''.join([s[i] for i in order])

    if kind == 'oo':
        if e1_or_e2 == 'e1':
            ids0 = 'iakb'
            ids1 = 'jakb'
        else:
            ids0 = 'kaib'
            ids1 = 'kajb'
        ids2 = 'ij'
    else:
        if e1_or_e2 == 'e1':
            ids0 = 'iajc'
            ids1 = 'ibjc'
        else:
            ids0 = 'icja'
            ids1 = 'icjb'
        ids2 = 'ab'
    ids0x = swapped(ids0, swapidx)
    ids1x = swapped(ids1, swapidx)

    merge_ids = lambda s0,s1,s2: '->'.join([','.join([s0,s1]),s2])
    dm = (np.einsum(merge_ids(ids0 , ids1 , ids2), t2, t2)*2 -
          np.einsum(merge_ids(ids0 , ids1x, ids2), t2, t2)   -
          np.einsum(merge_ids(ids0x, ids1 , ids2), t2, t2)   +
          np.einsum(merge_ids(ids0x, ids1x, ids2), t2, t2)*2) * 0.5
    return dm


def augment_ov(dmov, thresh, prj_occ=None, prj_vir=None):
    if prj_occ is not None:
        dmov = np.dot(prj_occ.T, dmov)
    if prj_vir is not None:
        dmov = np.dot(dmov, prj_vir)
    u, sigma, vt = scipy.linalg.svd(dmov)
    idx = numpy.where(abs(sigma) > thresh)[0]
    v = vt.conj().T
    return u[:,idx], v[:,idx]

def augment_virt(dm_corr, orbo, orbv, thresh, s1e=None, prj=None):
    nocc = orbo.shape[-1]
    dm_corr_ov = transform_rdm1(dm_corr, orbo, orbv, s1e)
    if prj is not None:
        dm_corr_ov = np.dot(dm_corr_ov, prj)
    _, sigma, vt = scipy.linalg.svd(dm_corr_ov)
    idx = numpy.where(abs(sigma) > thresh)[0]
    v = vt.conj().T
    return v[:,idx]

def transform_rdm1(dm0, orb1, orb2, s1e=None):
    if s1e is None:
        dm1 = reduce(np.dot, (orb1.conj().T, dm0, orb2))
    else:
        dm1 = reduce(np.dot, (orb1.conj().T, s1e, dm0, s1e, orb2))
    return dm1

def projection_construction(M, thresh):
    r''' Given M_{mu,i} = <mu | i> the ovlp between two orthonormal basis, find
    the unitary rotation |j'> = u_ij |i> so that {|j'>} significantly ovlp with
    {|mu>}.
    '''
    #e, u = scipy.linalg.eigh(np.dot(M.T, M))
    #mask = abs(e) > thresh
    #return u[:,mask], u[:,~mask]
    if M.shape[0] > M.shape[1]:
        v, e, _ = scipy.linalg.svd(M.conj().T)
    else:
        _, e, vt = scipy.linalg.svd(M)
        v = vt.conj().T
    norb = np.count_nonzero(e > thresh)
    return v[:,:norb], v[:,norb:]

def semicanonicalize(fock, orb):
    f = reduce(np.dot, (orb.T, fock, orb))
    if orb.shape[1] == 1:
        moe = f.ravel()
    else:
        moe, u = scipy.linalg.eigh(f, deg_thresh=SEMICANONICAL_DEG_THRESH)
        orb = np.dot(orb, u)
    return moe, orb

def canonical_orth_(S, thr=1e-8):
    '''Löwdin's canonical orthogonalization'''
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = scipy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / np.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = np.dot(np.diag(normlz), X)
    return X

def collocate_unitary(us):
    '''Collocate a few unitary matrices

    Args:
        us : list, tuple
            list of unitary matrices

    Returns:
        us0 : array
            Collocated unitary matrix
        x1 : array
            Unitary matrix which transforms
            to the orthogonal vector space
    '''
    us = np.concatenate(us, axis=1)
    x = canonical_orth_(np.dot(us.T, us))
    us0 = np.dot(us, x)

    us1 = np.eye(us0.shape[0]) - np.dot(us0, us0.T)
    e1, x1 = scipy.linalg.eigh(us1)
    x1 = x1[:, e1 > 0.99]
    assert us0.shape[-1] + x1.shape[-1] == us0.shape[0]
    return us0, x1

def osv_compression(dms, orb, thresh, prj=None, norb_target=None):
    us = []
    for dm in dms:
        e, u = scipy.linalg.eigh(dm, deg_thresh=COMPRESS_DEG_THRESH)
        us.append(u[:, abs(e) > thresh])
    us0, x1 = collocate_unitary(us)

    if prj is not None:
        orb = np.dot(orb, prj)
    orb1x = np.dot(orb, us0)
    orb0x = np.dot(orb, x1)
    return orb1x, orb0x

def natorb_compression(dm, orb, thresh, prj=None, norb_target=None,
                       uuvir2_corr=None, natorb_occdeg_thresh=0):
    e, u = scipy.linalg.eigh(dm, deg_thresh=COMPRESS_DEG_THRESH)
    if norb_target is None:
        idx = numpy.where(abs(e) > thresh)[0]
        if len(idx) > 0 and natorb_occdeg_thresh > 0:
            # NOTE include near degenerate states
            idx_deg = numpy.where(abs(e - e[idx[0]]) < natorb_occdeg_thresh)[0]
            idx = numpy.union1d(idx, idx_deg)
    elif isinstance(norb_target, (int, numpy.integer)):
        if norb_target < 0:
            raise ValueError(f'Target norb is negative: {norb_target}.')
        elif norb_target > e.size:
            raise ValueError(f'Target norb exceeds total number of orbs: {norb_target} > {e.size}')
        order = e.argsort()[::-1]
        idx = order[:norb_target]
    else:
        raise TypeError('Input "norb_target" should be integer type.')
    idxc = numpy.array([i for i in range(e.size) if i not in idx])

    if prj is not None:
        orb = np.dot(orb, prj)

    if uuvir2_corr is not None:
        us = (u[:,idx], uuvir2_corr)
        us0, x1 = collocate_unitary(us)
        orb1x = np.dot(orb, us0)
        orb0x = np.dot(orb, x1)
    else:
        orbx = np.dot(orb, u)
        orb1x = sub_colspace(orbx, idx)
        orb0x = sub_colspace(orbx, idxc)
    return orb1x, orb0x

def sub_colspace(A, idx):
    if idx.size == 0:
        return np.zeros([A.shape[0],0])
    else:
        return A[:,idx]

def get_cholesky_mos(mo_coeff):
    from pyscfad.lo.cholesky import cholesky_mos
    return cholesky_mos(mo_coeff)

def get_iao(mol, mo_coeff, minao='minao', orth=True):
    from pyscfad.lo.orth import vec_lowdin
    from pyscfad.lo.iao import iao as iao_kernel
    c = iao_kernel(mol, mo_coeff, minao=minao)

    if orth:
        s = mol.intor_symmetric('int1e_ovlp')
        c = vec_lowdin(c, s)
    return c

def get_ibo(mol, mo_coeff, init_guess=None,
            conv_tol=1e-10, symmetry=False, options=None):
    return get_pm(mol, mo_coeff, pop_method='ibo',
                  init_guess=init_guess, conv_tol=conv_tol,
                  symmetry=symmetry, options=options)

def get_boys(mol, mo_coeff, init_guess=None,
             conv_tol=1e-10, symmetry=False, options=None):
    from pyscfad.lo.boys import boys
    return boys(mol, mo_coeff, init_guess=init_guess,
                conv_tol=conv_tol, symmetry=symmetry,
                gmres_options=options)

def get_pm(mol, mo_coeff, pop_method='mulliken',
           init_guess=None, conv_tol=1e-10,
           symmetry=False, options=None):
    from pyscfad.lo.pipek import pm
    return pm(mol, mo_coeff, pop_method=pop_method,
              init_guess=init_guess, conv_tol=conv_tol,
              symmetry=symmetry, gmres_options=options)

def mo_splitter(maskact, maskocc, kind='mask'):
    ''' Split MO indices into
        - frozen occupieds
        - active occupieds
        - active virtuals
        - frozen virtuals

    Args:
        maskact (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is active.
        maskocc (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is occupied.
        kind (str):
            Determine the return type.
            'mask'  : return masks each of length nmo
            'index' : return index arrays
            'idx'   : same as 'index'

    Return:
        See the description for input arg 'kind' above.
    '''
    maskfrzocc = ~maskact &  maskocc
    maskactocc =  maskact &  maskocc
    maskactvir =  maskact & ~maskocc
    maskfrzvir = ~maskact & ~maskocc
    if kind == 'mask':
        return maskfrzocc, maskactocc, maskactvir, maskfrzvir
    elif kind in ['index','idx']:
        return [numpy.where(m)[0] for m in [maskfrzocc, maskactocc,
                                            maskactvir, maskfrzvir]]
    else:
        raise ValueError("'kind' must be 'mask' or 'index'(='idx').")


class LNO(pytree.PytreeNode):
    _dynamic_attr = {'_scf', 'mol', 'with_df'}

    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None, **kwargs):
        self._scf = mf
        self.mol = mf.mol
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

        self.frozen = frozen
        self.fock = fock
        self.s1e = s1e

        self.thresh_occ = thresh
        self.thresh_vir = thresh
        self.lo_type = 'iao'
        self.no_type = 'ie'
        self.verbose = self.verbose_imp = mf.mol.verbose

        # Whether to use local virtual orbitals
        self.use_local_virt = True
        # Natural orbitals with occupation number difference smaller than
        # natorb_occdeg_thresh will be added to the correlation space
        self.natorb_occdeg_thresh = 0
        # MP2 (relaxed dm - unrelaxed dm) in AO basis for augmenting virtual space
        self.dm_corr = None
        self.dm_corr_frag = None

        self._nmo = None
        self._nocc = None
        self.mo_occ = mf.mo_occ

    get_nocc = get_nocc
    get_nmo = get_nmo

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    def mo_splitter(self, mo_occ, kind='mask'):
        r''' Return index arrays that split MOs into
            - frozen occupieds
            - active occupieds
            - active virtuals
            - frozen virtuals

        Args:
            kind (str):
                'mask'  : return masks each of length nmo
                'index' : return index arrays
                'idx'   : same as 'index'
        '''
        maskact = self.get_frozen_mask()
        maskocc = mo_occ > THRESH_OCC
        return mo_splitter(maskact, maskocc, kind=kind)

    def split_mo(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_occ is None:
            mo_occ = self._scf.mo_occ
        masks = self.mo_splitter(mo_occ)
        return [mo_coeff[:,m] for m in masks]

    def split_moe(self, mo_energy=None, mo_occ=None):
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if mo_occ is None:
            mo_occ = self._scf.mo_occ
        masks = self.mo_splitter(mo_occ)
        return [mo_energy[m] for m in masks]

    def get_lo(self, mol=None, mo_coeff=None, mo_occ=None,
               lo_type='iao', init_guess=None, symmetry=False,
               options=None):
        if mol is None:
            mol = self._scf.mol
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_occ is None:
            mo_occ = self._scf.mo_occ

        orbocc = self.split_mo(mo_coeff, mo_occ)[1]

        if lo_type.lower() == 'iao':
            orbloc = get_iao(mol, orbocc)
        elif lo_type.lower() == 'ibo':
            orbloc = get_ibo(mol, orbocc, init_guess=init_guess,
                             symmetry=symmetry, options=options)
        elif lo_type.lower() == 'boys':
            orbloc = get_boys(mol, orbocc, init_guess=init_guess,
                              symmetry=symmetry, options=options)
        elif lo_type.lower() == 'pm':
            orbloc = get_pm(mol, orbocc, init_guess=init_guess,
                            symmetry=symmetry, options=options)
        elif lo_type.lower() == 'cholesky':
            orbloc = get_cholesky_mos(orbocc)
        else:
            raise KeyError(f'Unrecognized orbital localization method: {lo_type}.')
        return orbloc

    def ao2mo(self, fock=None, s1e=None):
        if fock is None:
            fock = self.fock
        if s1e is None:
            s1e = self.s1e

        if self.with_df is not None:
            eris = _make_df_eris_incore(self, fock, s1e)
        else:
            raise NotImplementedError
        return eris

    def kernel(self,
               frag_lolist=None,
               frag_wghtlist=None,
               frag_atmlist=None,
               lo_type=None,
               no_type=None,
               frag_nonvlist=None,
               orbloc=None,
               lo_init_guess=None,
               lo_symmetry=False,
               lo_options=None):
        if lo_type is None:
            lo_type = self.lo_type
        if no_type is None:
            no_type = self.no_type
        if orbloc is None:
            orbloc = self.get_lo(lo_type=lo_type, init_guess=lo_init_guess,
                                 symmetry=lo_symmetry, options=lo_options)

        # LO assignment to fragments
        if frag_lolist is None:
            if frag_atmlist is None:
                #log.info('Grouping LOs by single-atom fragments')
                frag_atmlist = stop_trace(autofrag)(self.mol)
            else:
                #log.info('Grouping LOs by user input atom-based fragments')
                pass
            frag_lolist = stop_trace(map_lo_to_frag)(self.mol, orbloc, frag_atmlist,
                                         verbose=self.verbose)
        elif frag_lolist == '1o':
            #log.info('Using single-LO fragment')
            frag_lolist = np.arange(orbloc.shape[1]).reshape(-1,1)
        else:
            #log.info('Using user input LO-fragment assignment')
            pass

        nfrag = len(frag_lolist)
        if frag_wghtlist is None:
            frag_wghtlist = np.ones(nfrag)

        frag_res = kernel(self, orbloc, frag_lolist, no_type=no_type,
                          frag_nonvlist=frag_nonvlist)
        self._post_proc(frag_res, frag_wghtlist)

    def _post_proc(self, frag_res, frag_wghtlist):
        raise NotImplementedError

    get_frozen_mask = get_frozen_mask

def _make_df_eris_incore(mycc, fock=None, s1e=None):
    if fock is None:
        fock = mycc.fock
    if s1e is None:
        s1e = mycc.s1e
    eris = _LNODFINCOREERIS(fock=fock, s1e=s1e)
    eris._common_init_(mycc)
    return eris

class _LNOERIS():
    def __init__(self, fock=None, s1e=None):
        #self.mo_coeff = None
        #self.nocc = None
        #self.h1e = None
        self.s1e = s1e
        #self.vhf = None
        self.fock = fock
        self.Lov = None

    def _common_init_(self, mcc):
        mf = mcc._scf
        if self.s1e is None:
            self.s1e = mf.get_ovlp()
        if self.fock is None:
            h1e = mf.get_hcore()
            vhf = mf.get_veff()
            self.fock = mf.get_fock(h1e=h1e, s1e=self.s1e, vhf=vhf)
            del h1e, vhf

class _LNODFINCOREERIS(_LNOERIS):
    def _common_init_(self, mcc):
        super()._common_init_(mcc)
        orbo, orbv = mcc.split_mo()[1:3]
        nocc = orbo.shape[-1]
        mo_coeff = np.concatenate((orbo, orbv), axis=-1)
        self.Lov = get_Lov(mcc._scf, mo_coeff, nocc)

    def get_Ov(self, u):
        return np.einsum('iI,Lia->LIa', u, self.Lov)

    def get_oV(self, u):
        return np.einsum('aA,Lia->LiA', u, self.Lov)

    @staticmethod
    def _get_eris(Lia, Ljb):
        return np.einsum('Lia,Ljb->iajb', Lia, Ljb)

    def get_Ovov(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, self.Lov)

    def get_OvOv(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, LOv)

    def get_oVov(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, self.Lov)

    def get_oVoV(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, LoV)


def get_Lov(mf, mo_coeff, nocc):
    assert hasattr(mf, 'with_df')
    cderi = mf.with_df._cderi
    naux = cderi.shape[0]
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    ijslice = (0, nocc, nocc, nmo)
    if hasattr(cderi, 'shape'):
        Lov = _ao2mo.nr_e2(cderi, mo_coeff, ijslice, aosym='s2')
        Lov = Lov.reshape((naux, nocc, nvir))
    else:
        raise NotImplementedError
    return Lov
