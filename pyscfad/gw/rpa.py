from typing import Optional, Any
import numpy
from jax import numpy as jnp
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf import df as pyscf_df
from pyscf.gw import rpa as pyscf_rpa
from pyscfad import lib, gto, scf, dft, df


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

    # Compute totol energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    """
    Compute RPA correlation energy
    """
    mol = rpa.mol
    mf = rpa._scf
    dm = mf.make_rdm1()
    rks = dft.RKS(mol, xc=mf.xc)
    veff = rks.get_veff(mol, dm)
    h1e = rks.get_hcore(mol)
    s1e = rks.get_ovlp(mol)
    fock = rks.get_fock(h1e, s1e, veff, dm)
    mo_energy, _ = rks.eig(fock, s1e)

    #mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    mo_energy = pyscf_rpa._mo_energy_without_core(rpa, mo_energy)
    nocc = rpa.nocc
    nw = len(freqs)
    naux = Lpq.shape[0]

    if (mo_energy[nocc] - mo_energy[nocc-1]) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    e_corr = 0.
    for w in range(nw):
        Pi = pyscf_rpa.get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        ec_w = jnp.log(jnp.linalg.det(jnp.eye(naux) - Pi))
        ec_w += jnp.trace(Pi)
        e_corr += 1./(2.*numpy.pi) * ec_w * wts[w]

    return e_corr

@lib.dataclass
class RPA(pyscf_rpa.RPA):
    _scf : scf.hf.SCF = lib.field(pytree_node=True)
    frozen : Optional[int] = None
    mol : Optional[gto.Mole] = lib.field(pytree_node=True, default=None)
    verbose: Optional[int] = None
    stdout: Optional[Any] = None
    max_memory: Optional[int] = None

    with_df : Optional[df.DF] = lib.field(pytree_node=True, default=None)
    mo_energy : Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    mo_coeff : Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    mo_occ : Optional[jnp.array] = None
    e_hf : Optional[float] = None
    e_corr : Optional[float] = None
    e_tot : Optional[float] = None
    _nocc : Optional[int] = None
    _nmo : Optional[int] = None

    def __init__(self, _scf, frozen=None, auxbasis=None, **kwargs):
        self._scf = _scf
        self.frozen = frozen
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        if getattr(self, "mol", None) is None:
            self.mol = self._scf.mol
        if getattr(self, "stdout", None) is None:
            self.stdout = self.mol.stdout
        if getattr(self, "verbose", None) is None:
            self.verbose = self.mol.verbose
        if getattr(self, "max_memory", None) is None:
            self.max_memory = self.mol.max_memory

        if getattr(self, "with_df", None) is None:
            if getattr(self._scf, 'with_df', None):
                self.with_df = self._scf.with_df
            else:
                if auxbasis is None:
                    auxbasis = pyscf_df.addons.make_auxbasis(self.mol, mp2fit=True)
                auxmol = df.addons.make_auxmol(self.mol, auxbasis)
                self.with_df = df.DF(self.mol, auxmol=auxmol)

        if getattr(self, "_nocc", None) is None:
            self._nocc = None
        if getattr(self, "_nmo", None) is None:
            self._nmo = None
        if getattr(self, "mo_energy", None) is None:
            self.mo_energy = self._scf.mo_energy
        if getattr(self, "mo_coeff", None) is None:
            self.mo_coeff = self._scf.mo_coeff
        if getattr(self, "mo_occ", None) is None:
            self.mo_occ = self._scf.mo_occ
        if getattr(self, "e_corr", None) is None:
            self.e_corr = None
        if getattr(self, "e_hf", None) is None:
            self.e_hf = None
        if getattr(self, "e_tot", None) is None:
            self.e_tot = None
        self._keys = set(self.__dict__.keys())

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            nw: interger, grid number
        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = pyscf_rpa._mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = pyscf_rpa._mo_energy_without_core(self, self._scf.mo_energy)

        #cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                        kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, verbose=self.verbose)

        #logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = pyscf_lib.current_memory()[0]

        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = jnp.einsum("lpq,pi,qj->lij", self.with_df._cderi, mo_coeff, mo_coeff)
            return Lpq
        else:
            raise RuntimeError("not enough memory")
