from functools import reduce, partial
from jaxopt import linear_solve
from pyscf import __config__
from pyscf import numpy as np
from pyscf.lib import logger
from pyscf.cc import ccsd as pyscf_ccsd
from pyscf.mp.mp2 import _mo_without_core
from pyscfad import lib
from pyscfad.lib import jit
from pyscfad import util
from pyscfad.gto import mole
from pyscfad.scf import hf
from pyscfad import implicit_diff

CCSD_IMPLICIT_DIFF = getattr(__config__, "pyscfad_ccsd_implicit_diff", False)
# assume 'mol', 'mo_coeff', etc. come from '_scf',
# otherwise they need to be traced
CC_Tracers = ['_scf']
# attributes explicitly appearing in :fun:`update_amps` need to be traced
ERI_Tracers = ['fock', 'mo_energy', #'mol', 'mo_coeff', 'e_hf',
               'oooo', 'ooov', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv']

def _converged_iter(amp, mycc, eris):
    t1, t2 = mycc.vector_to_amplitudes(amp)
    t1, t2 = mycc.update_amps(t1, t2, eris)
    amp = mycc.amplitudes_to_vector(t1, t2)
    return amp

def _iter(amp, mycc, eris, *,
          diis=None, max_cycle=50, tol=1e-8,
          tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)

    t1, t2 = mycc.vector_to_amplitudes(amp)
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)
    cput1 = log.timer('initialize CCSD')

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, diis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    amp = mycc.amplitudes_to_vector(t1, t2)
    return amp, conv

if CCSD_IMPLICIT_DIFF:
    solver = partial(linear_solve.solve_gmres, tol=1e-5,
                     solve_method='incremental', maxiter=30)
    _iter = implicit_diff.custom_fixed_point(
                _converged_iter, solve=solver, has_aux=True)(_iter)

def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    vec = mycc.amplitudes_to_vector(t1, t2)
    vec, conv = _iter(vec, mycc, eris,
                      diis=adiis, max_cycle=max_cycle, tol=tol,
                      tolnormt=tolnormt, verbose=log)
    t1, t2 = mycc.vector_to_amplitudes(vec)
    eccsd = mycc.energy(t1, t2, eris)
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

@jit
def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    vector_t1 = t1.ravel()
    vector_t2 = t2.transpose(0,2,1,3).reshape(nov,nov)[np.tril_indices(nov)]
    vector = np.concatenate((vector_t1, vector_t2), axis=None)
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, t2

@util.pytree_node(CC_Tracers, num_args=1)
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

    def ccsd(self, t1=None, t2=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from pyscfad.cc import ccsd_t_slow
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return ccsd_t_slow.kernel(self, eris, t1, t2, self.verbose)

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

@util.pytree_node(ERI_Tracers)
class _ChemistsERIs(pyscf_ccsd._ChemistsERIs):
    def __init__(self, mol=None, **kwargs):
        pyscf_ccsd._ChemistsERIs.__init__(self, mol=mol)
        self.__dict__.update(kwargs)

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        return self
