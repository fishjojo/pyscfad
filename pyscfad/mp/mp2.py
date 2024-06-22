from functools import wraps
import jax
from pyscf import __config__ as pyscf_config
from pyscf.lib import split_reshape
from pyscf.mp import mp2 as pyscf_mp2
from pyscfad import numpy as np
from pyscfad import util
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad import ops
from pyscfad import ao2mo

WITH_T2 = getattr(pyscf_config, 'mp_mp2_with_t2', True)

@wraps(pyscf_mp2.kernel)
def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = np.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        if hasattr(eris.ovov, 'ndim') and eris.ovov.ndim == 4:
            gi = eris.ovov[i]
        else:
            gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi.conj()/(eia[:,:,None] + eia[i][None,None,:])
        emp2 += np.einsum('jab,jab', t2i, gi) * 2
        emp2 -= np.einsum('jab,jba', t2i, gi)
        if with_t2:
            t2 = ops.index_update(t2, ops.index[i], t2i)

    return emp2.real, t2

def _iterative_kernel(mp, eris, verbose=None):
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    emp2, t2 = mp.init_amps(eris=eris)
    log.info('Init E(MP2) = %.15g', emp2)

    adiis = lib.diis.DIIS(mp)

    conv = False
    for istep in range(mp.max_cycle):
        t2new = mp.update_amps(t2, eris)

        if hasattr(t2new, 'ndim'):
            normt = np.linalg.norm(t2new - t2)
            t2 = None
            t2new = adiis.update(t2new)
        else: # UMP2
            normt = np.linalg.norm([np.linalg.norm(t2new[i] - t2[i])
                                     for i in range(3)])
            t2 = None
            t2shape = [x.shape for x in t2new]
            t2new = np.hstack([x.ravel() for x in t2new])
            t2new = adiis.update(t2new)
            t2new = split_reshape(t2new, t2shape)

        t2, t2new = t2new, None
        emp2, e_last = mp.energy(t2, eris), emp2
        log.info('cycle = %d  E_corr(MP2) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep+1, emp2, emp2 - e_last, normt)
        cput1 = log.timer('MP2 iter', *cput1)
        if abs(emp2-e_last) < mp.conv_tol and normt < mp.conv_tol_normt:
            conv = True
            break
    log.timer('MP2', *cput0)
    del log
    return conv, emp2, t2

@wraps(pyscf_mp2.energy)
def energy(mp, t2, eris):
    nocc, nvir = t2.shape[1:3]
    eris_ovov = np.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    emp2  = np.einsum('ijab,iajb', t2, eris_ovov) * 2
    emp2 -= np.einsum('ijab,ibja', t2, eris_ovov)
    return emp2.real

@wraps(pyscf_mp2.update_amps)
def update_amps(mp, t2, eris):
    #assert (isinstance(eris, _ChemistsERIs))
    nocc, nvir = t2.shape[1:3]
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mp.level_shift

    foo = fock[:nocc,:nocc] - np.diag(mo_e_o)
    fvv = fock[nocc:,nocc:] - np.diag(mo_e_v)
    t2new  = np.einsum('ijac,bc->ijab', t2, fvv)
    t2new -= np.einsum('ki,kjab->ijab', foo, t2)
    t2new = t2new + t2new.transpose(1,0,3,2)

    eris_ovov = np.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    t2new += eris_ovov.conj().transpose(0,2,1,3)
    eris_ovov = None

    eia = mo_e_o[:,None] - mo_e_v
    t2new /= eia[:,None,:,None] + eia[None,:,None,:]
    return t2new

def make_rdm1(mp, t2=None, eris=None, ao_repr=False):
    from pyscfad.cc import ccsd_rdm
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = np.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=ao_repr)

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None:
        t2 = mp.t2
    assert t2 is not None
    nocc = mp.nocc
    nvir = mp.nmo - nocc

    dm1occ = np.zeros((nocc,nocc), dtype=t2.dtype)
    dm1vir = np.zeros((nvir,nvir), dtype=t2.dtype)
    @jax.checkpoint
    def _fn(carry, t2i):
        dm1vir, dm1occ = carry
        l2i = t2i.conj()
        dm1vir += np.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - np.einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += np.einsum('iab,jab->ij', l2i, t2i) * 2 \
                - np.einsum('iab,jba->ij', l2i, t2i)
        return (dm1vir, dm1occ), None
    (dm1vir, dm1occ), _ = jax.lax.scan(_fn, (dm1vir, dm1occ), t2)
    return -dm1occ, dm1vir

@util.pytree_node(['_scf', 'mol'], num_args=1)
class MP2(pyscf_mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        pyscf_mp2.MP2.__init__(self, mf, frozen=frozen,
                               mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        eris = _ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        mo_coeff = eris.mo_coeff

        nocc = self.nocc
        co = np.asarray(mo_coeff[:,:nocc])
        cv = np.asarray(mo_coeff[:,nocc:])
        eris.ovov = ao2mo.general(self._scf._eri, (co,cv,co,cv))
        return eris

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            self.converged, self.e_corr, self.t2 = self._iterative_kernel(eris)

        # TODO SCS-MP2
        self.e_corr_ss = 0
        self.e_corr_os = 0

        self._finalize()
        return self.e_corr, self.t2

    make_rdm1 = make_rdm1
    energy = energy
    update_amps = update_amps
    _iterative_kernel = _iterative_kernel

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

RMP2 = MP2

class _ChemistsERIs(pyscf_mp2._ChemistsERIs):
    def _common_init_(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')

        self.mo_coeff = pyscf_mp2._mo_without_core(mp, mo_coeff)
        self.mol = mp.mol

        if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            self.mo_energy = pyscf_mp2._mo_energy_without_core(mp, mp._scf.mo_energy)
            self.fock = np.diag(self.mo_energy)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            self.fock = self.mo_coeff.conj().T.dot(fockao).dot(self.mo_coeff)
            self.mo_energy = self.fock.diagonal().real
        return self
