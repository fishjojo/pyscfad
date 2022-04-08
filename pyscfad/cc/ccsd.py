from functools import reduce
from pyscf import __config__
from pyscf.cc import ccsd as pyscf_ccsd
from pyscf.mp.mp2 import _mo_without_core
from pyscfad import lib
from pyscfad import util
from pyscfad.lib import numpy as jnp
from pyscfad.gto import mole
from pyscfad.scf import hf

Traced_Attributes = ['_scf',]

def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    vector_t1 = t1.ravel()
    vector_t2 = t2.transpose(0,2,1,3).reshape(nov,nov)[jnp.tril_indices(nov)]
    vector = jnp.concatenate((vector_t1, vector_t2), axis=None)
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, t2

@util.pytree_node(Traced_Attributes, num_args=1)
class CCSD(pyscf_ccsd.CCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        pyscf_ccsd.CCSD.__init__(self, mf, frozen=frozen,
                                 mo_coeff=mo_coeff, mo_occ=mo_occ)
        if self.diis is True:
            self.diis = lib.diis.DIIS(self, self.diis_file, incore=self.incore_complete)
        self.__dict__.update(kwargs)

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscfad.cc import eom_rccsd
        return eom_rccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscfad.cc import eom_rccsd
        return eom_rccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscfad.cc import eom_rccsd
        return eom_rccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

class _ChemistsERIs(pyscf_ccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(jnp.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        """
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD.\n'
                            'CCSD may be difficult to converge. Increasing '
                            'CCSD Attribute level_shift may improve '
                            'convergence.', gap)
        except ValueError:  # gap.size == 0
            pass
        """
        return self
