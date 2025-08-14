# Copyright 2021-2025 Xing Zhang
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

from functools import reduce
import numpy
from pyscf.prop.polarizability.rhf import Polarizability as pyscf_Polarizability
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import logger
# TODO scipy backend
from jax.scipy.sparse.linalg import gmres

def cphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                   max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_ai = mo_energy[viridx][:,None] - mo_energy[occidx][None,:]
    # e_ai - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    LEVEL_SHIF = 0.1
    diag_0 = e_ai - freq
    diag_1 = e_ai + freq
    diag_0 = ops.index_add(diag_0, diag_0 < LEVEL_SHIF, LEVEL_SHIF)
    diag_1 = ops.index_add(diag_1, diag_1 < LEVEL_SHIF, LEVEL_SHIF)
    diag = (diag_0, diag_1)

    nvir, nocc = e_ai.shape
    mo_coeff = mf.mo_coeff
    nao, nmo = mo_coeff.shape
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    h1 = h1.reshape(-1,nvir,nocc)
    ncomp = h1.shape[0]

    rhs = np.stack((-h1, -h1), axis=1)
    rhs = rhs.reshape(ncomp,nocc*nvir*2)
    mo1base = np.stack((-h1/diag[0],
                        -h1/diag[1]), axis=1)
    mo1base = mo1base.reshape(ncomp,nocc*nvir*2)

    vresp = mf.gen_response(hermi=0)
    def vind(xys):
        nz = len(xys)
        dms = []
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            # *2 for double occupancy
            dmx = reduce(np.dot, (orbv, x  *2, orbo.T))
            dmy = reduce(np.dot, (orbo, y.T*2, orbv.T))
            dms.append(dmx + dmy)  # AX + BY
        dms = np.asarray(dms)

        v1ao = vresp(dms)
        v1vo = np.einsum('xpq,pi,qj->xij', v1ao, orbv, orbo)  # ~c1
        v1ov = np.einsum('xpq,pi,qj->xji', v1ao, orbo, orbv)  # ~c1^T

        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            v1vo = ops.index_add(v1vo, ops.index[i], (e_ai - freq) * x)
            v1ov = ops.index_add(v1ov, ops.index[i], (e_ai + freq) * y)
        v = np.stack((v1vo, v1ov), axis=1)
        return v.reshape(nz,-1)

    mo1 = gmres(vind, rhs, mo1base, tol=tol)[0]
    mo1 = mo1.reshape(-1,2,nvir,nocc)
    log.timer('krylov solver in CPHF', *t0)

    dms = []
    for i in range(ncomp):
        x, y = mo1[i]
        dmx = reduce(np.dot, (orbv, x  *2, orbo.T))
        dmy = reduce(np.dot, (orbo, y.T*2, orbv.T))
        dms.append(dmx + dmy)
    dms = np.asarray(dms)
    mo_e1 = np.einsum('xpq,pi,qj->xij', vresp(dms), orbo, orbo)
    mo1 = (mo1[:,0], mo1[:,1])
    return mo1, mo_e1

def polarizability_with_freq(polobj, freq=None):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]

    charges = ops.to_numpy(mol.atom_charges())
    coords  = ops.to_numpy(mol.atom_coords())
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1 = np.einsum('xpq,pi,qj->xij', int_r, orbv.conj(), orbo)
    mo1 = cphf_with_freq(mf, mo_energy, mo_occ, h1, freq,
                         polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)[0]

    e2 =  np.einsum('xpi,ypi->xy', h1, mo1[0])
    e2 += np.einsum('xpi,ypi->xy', h1, mo1[1])

    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 *= -2
    log.debug('Polarizability tensor with freq %s', freq)
    log.debug('%s', e2)
    return e2


class Polarizability(pyscf_Polarizability):
    def gen_vind(self, mf, mo_coeff, mo_occ):
        vresp = mf.gen_response(hermi=1)
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        nocc = orbo.shape[1]
        nao, nmo = mo_coeff.shape
        def vind(mo1):
            dm1 = np.einsum('xai,pa,qi->xpq', mo1.reshape(-1,nmo,nocc), mo_coeff,
                             orbo.conj())
            dm1 = (dm1 + dm1.transpose(0,2,1).conj()) * 2
            v1mo = np.einsum('xpq,pi,qj->xij', vresp(dm1), mo_coeff.conj(), orbo)
            return v1mo.ravel()
        return vind

    polarizability_with_freq = polarizability_with_freq
