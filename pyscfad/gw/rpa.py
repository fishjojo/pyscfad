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

import numpy
from pyscf.lib import logger, current_memory
from pyscf.mp.mp2 import _mo_without_core, _mo_energy_without_core
from pyscf.gw import rpa as pyscf_rpa
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.ops import vmap, jit
from pyscfad import scf, dft
from pyscfad.df.addons import restore

def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=None, verbose=logger.NOTE):
    mf = rpa._scf
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert rpa.frozen < rpa.nocc

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = pyscf_rpa._get_scaled_legendre_roots(nw)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.RHF(rpa.mol)
    e_hf = rhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)
    return e_hf, e_corr

@jit
def get_rho_response(omega, mo_energy, Lpq):
    '''
    Compute density response function in auxiliary basis at freq iw.
    '''
    nocc = Lpq.shape[1]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = np.einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi

def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    '''
    Compute RPA correlation energy
    '''
    mol = rpa.mol
    mf = rpa._scf
    dm = mf.make_rdm1()
    rks = dft.RKS(mol, xc=mf.xc)
    veff = rks.get_veff(mol, dm)
    h1e = rks.get_hcore(mol)
    s1e = rks.get_ovlp(mol)
    fock = rks.get_fock(h1e, s1e, veff, dm)
    mo_energy, _ = rks.eig(fock, s1e)

    mo_energy = _mo_energy_without_core(rpa, mo_energy)
    nocc = rpa.nocc
    naux = Lpq.shape[0]

    if (mo_energy[nocc] - mo_energy[nocc-1]) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    def body(omega, weight):
        Pi = get_rho_response(omega, mo_energy, Lpq[:, :nocc, nocc:])
        ec_w  = np.log(np.linalg.det(np.eye(naux) - Pi))
        ec_w += np.trace(Pi)
        e_corr_i = 1./(2.*numpy.pi) * ec_w * weight
        return e_corr_i
    e_corr_i = vmap(body)(freqs, wts)
    e_corr = np.sum(e_corr_i)
    return e_corr

class RPA(pytree.PytreeNode, pyscf_rpa.RPA):
    _dynamic_attr = {'_scf', 'mol', 'with_df'}

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40):
        '''
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            nw: interger, grid number
        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        '''
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        log = logger.new_logger(self)
        self.dump_flags()
        self.e_hf, self.e_corr = \
            kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, verbose=self.verbose)

        log.timer('RPA')
        del log
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nao = mo_coeff.shape[0]
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = current_memory()[0]

        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            cderi = restore('s1', self.with_df._cderi, nao)
            Lpq = np.einsum('lpq,pi,qj->lij', cderi, mo_coeff, mo_coeff)
            return Lpq
        else:
            raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')
