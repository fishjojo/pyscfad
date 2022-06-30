import numpy
from jax import vmap, jit
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf import df as pyscf_df
from pyscf.gw import rpa as pyscf_rpa
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad import gto, scf, dft, df


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

@jit
def get_rho_response(omega, mo_energy, Lpq):
    """
    Compute density response function in auxiliary basis at freq iw.
    """
    nocc = Lpq.shape[1]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = np.einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi

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

@util.pytree_node(['_scf','mol','with_df','mo_energy','mo_coeff'], num_args=1)
class RPA(pyscf_rpa.RPA):
    def __init__(self, mf, frozen=None, auxbasis=None, **kwargs):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        self.with_df = None

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None

        self.__dict__.update(kwargs)
        if self.with_df is None:
            if getattr(self._scf, 'with_df', None):
                self.with_df = self._scf.with_df
            else:
                if auxbasis is None:
                    auxbasis = pyscf_df.addons.make_auxbasis(self.mol, mp2fit=True)
                auxmol = df.addons.make_auxmol(self.mol, auxbasis)
                self.with_df = df.DF(self.mol, auxmol=auxmol)

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

        log = logger.new_logger(self)
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                        kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, verbose=self.verbose)

        log.timer('RPA')
        del log
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = pyscf_lib.current_memory()[0]

        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = np.einsum("lpq,pi,qj->lij", self.with_df._cderi, mo_coeff, mo_coeff)
            return Lpq
        else:
            raise RuntimeError("not enough memory")
