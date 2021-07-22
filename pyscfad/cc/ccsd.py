from typing import Optional, Any
from functools import reduce
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.mp.mp2 import _mo_without_core
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.gto import mole
from pyscfad.scf import hf

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

@lib.dataclass
class CCSD(ccsd.CCSD):
    _scf: hf.SCF = lib.field(pytree_node=True)
    frozen: Optional[jnp.array] = None
    mo_coeff: Optional[jnp.array] = None
    mo_occ: Optional[jnp.array] = None

    mol: Optional[mole.Mole] = None
    verbose: Optional[int] = None
    stdout: Any = None
    max_memory: Optional[int] = None
    level_shift: Optional[float] = None

    converged: bool = False
    converged_lambda: bool = False
    emp2: Optional[float]= None
    e_hf: Optional[float] = None
    e_corr: Optional[float] = None
    t1: Optional[jnp.array] = None
    t2: Optional[jnp.array] = None
    l1: Optional[jnp.array] = None
    l2: Optional[jnp.array] = None
    _nocc: Optional[int] = None
    _nmo: Optional[int] = None
    chkfile: Any = None

    max_cycle: int = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol: float = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    iterative_damping: float = getattr(__config__, 'cc_ccsd_CCSD_iterative_damping', 1.0)
    conv_tol_normt: float = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)

    diis: Any = getattr(__config__, 'cc_ccsd_CCSD_diis', True)
    diis_space: int = getattr(__config__, 'cc_ccsd_CCSD_diis_space', 6)
    diis_file: Optional[str] = None
    diis_start_cycle: int = getattr(__config__, 'cc_ccsd_CCSD_diis_start_cycle', 0)
    # FIXME: Should we avoid DIIS starting early?
    diis_start_energy_diff: float = getattr(__config__, 'cc_ccsd_CCSD_diis_start_energy_diff', 1e9)

    direct: bool = getattr(__config__, 'cc_ccsd_CCSD_direct', False)
    async_io: bool = getattr(__config__, 'cc_ccsd_CCSD_async_io', True)
    incore_complete: bool = getattr(__config__, 'cc_ccsd_CCSD_incore_complete', False)
    cc2: bool = getattr(__config__, 'cc_ccsd_CCSD_cc2', False)

    def __post_init__(self):
        if self.mo_coeff is None:
            self.mo_coeff = self._scf.mo_coeff
        if self.mo_occ is None: 
            self.mo_occ = self._scf.mo_occ
        if self.mol is None:
            self.mol = self._scf.mol
        if self.verbose is None:
            self.verbose = self.mol.verbose
        if self.stdout is None:
            self.stdout = self.mol.stdout
        if self.max_memory is None:
            self.max_memory = self._scf.max_memory
        if self.level_shift is None:
            self.level_shift = 0
        if self.chkfile is None:
            self.chkfile = self._scf.chkfile

        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        self._keys = set(self.__dict__.keys())

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

class _ChemistsERIs(ccsd._ChemistsERIs):
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
