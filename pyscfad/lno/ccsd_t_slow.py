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

'''Impurity (T) correction (slow version).
'''

from pyscf.lib import logger
from pyscfad import numpy as np
from pyscfad import config_update
from pyscfad import lib
from pyscfad.ops import jit, vmap

def get_ovvv(ovvv, *slices):
    ovw = np.asarray(ovvv[slices])
    nocc, nvir, nvir_pair = ovw.shape
    with config_update('pyscfad_moleintor_opt', False):
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
    nvir1 = ovvv.shape[2]
    return ovvv.reshape(nocc,nvir,nvir1,nvir1)


def kernel(mycc, eris, ulo, t1=None, t2=None, verbose=logger.NOTE):
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    mo_e = eris.mo_energy
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = e_occ[:,None,None] + e_occ[None,:,None] + e_occ[None,None,:]

    eris_vvov = get_ovvv(eris.ovvv).conj().transpose(1,3,0,2)
    eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]

    idx = []
    scal = []
    for a in range(nvir):
        for b in range(nvir):
            for c in range(b+1):
                if b==c:
                    fac = 2
                else:
                    fac = 1
                idx.append([a,b,c])
                scal.append(fac)
    idx = np.asarray(idx, dtype=np.int32)
    scal = np.asarray(scal, dtype=float)

    mat = np.dot(ulo.T, ulo)
    et = _compute_et(mat, t1, t2, eris_vvov, eris_vooo, eris_vvoo,
                     fvo, eijk, e_vir, idx, scal)
    log.info('CCSD(T) correction = %.15g', et)
    return et

@jit
def _compute_et(mat, t1, t2, eris_vvov, eris_vooo, eris_vvoo,
                fvo, eijk, e_vir, idx, scal):
    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    def get_w(a, b, c):
        w = np.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= np.einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w

    def get_v(a, b, c):
        v = np.einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= np.einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v

    def r3(w):
        return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
                - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
                - 2 * w.transpose(1,0,2))

    def body(mat, fac, index):
        a, b, c = index[:]
        d3  = eijk - e_vir[a] - e_vir[b] - e_vir[c]
        d3 *= fac

        wabc = get_w(a, b, c)
        wacb = get_w(a, c, b)
        wbac = get_w(b, a, c)
        wbca = get_w(b, c, a)
        wcab = get_w(c, a, b)
        wcba = get_w(c, b, a)

        WW = (wabc
              + wacb.transpose(0,2,1) + wbac.transpose(1,0,2)
              + wbca.transpose(2,0,1) + wcab.transpose(1,2,0)
              + wcba.transpose(2,1,0))

        vabc = get_v(a, b, c)
        vacb = get_v(a, c, b)
        vbac = get_v(b, a, c)
        vbca = get_v(b, c, a)
        vcab = get_v(c, a, b)
        vcba = get_v(c, b, a)

        VV = (vabc
              + vacb.transpose(0,2,1) + vbac.transpose(1,0,2)
              + vbca.transpose(2,0,1) + vcab.transpose(1,2,0)
              + vcba.transpose(2,1,0))

        ZZ = r3(WW + .5 * VV) / d3
        #et = np.einsum('pjk,ijk->pi', WW, ZZ.conj())
        #return np.dot(et.ravel(), mat.ravel())
        WW_mat = np.einsum('pjk,pi->ijk', WW, mat)
        et = np.einsum('ijk,ijk', WW_mat, ZZ.conj())
        return et

    et = vmap(body, in_axes=(None, 0, 0))(mat, scal, idx)
    et = np.sum(et) / 3. * 2.
    return et


def iterative_kernel(mycc, eris, ulo, t1=None, t2=None,
                     max_cycle=50, tol=1e-8, tolnormt=1e-6,
                     verbose=logger.NOTE):
    from pyscfad.cc import ccsd_t_slow as ccsdt
    mat = np.dot(ulo.T, ulo)

    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2

    t3 = ccsdt.iterative_kernel(
             mycc, eris=eris, t1=t1, t2=t2, t3=None,
             max_cycle=max_cycle, tol=tol,
             tolnormt=tolnormt, verbose=verbose)[2]

    w = ccsdt.get_w_6d(eris, t2)
    w = ccsdt.P6(w)

    nocc = t3.shape[0]
    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = ccsdt.get_d3(eia)
    w /= d3
    tmp = ccsdt.r3_6d(w)*d3
    t3 = np.einsum('pjkabc,pi->ijkabc',t3, mat)
    et = np.einsum('ijkabc,ijkabc', t3, tmp) / 3.
    return et
