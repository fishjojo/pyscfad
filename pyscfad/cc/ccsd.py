from functools import reduce
import numpy
from pyscf.cc import ccsd as pyscf_ccsd
from pyscf.mp.mp2 import _mo_without_core
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import lib
from pyscfad.lib import logger
#from pyscfad.ops import jit
#from pyscfad import util
from pyscfad import config
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.tools.linear_solver import gen_gmres

# assume 'mol', 'mo_coeff', etc. come from '_scf',
# otherwise they need to be traced
#CC_Tracers = ['_scf']
# attributes explicitly appearing in :fun:`update_amps` need to be traced
#ERI_Tracers = ['fock', 'mo_energy', #'mol', 'mo_coeff', 'e_hf',
#               'oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv']

def _converged_iter(amp, mycc, eris):
    t1, t2 = mycc.vector_to_amplitudes(amp)
    t1, t2 = mycc.update_amps(t1, t2, eris)
    amp = mycc.amplitudes_to_vector(t1, t2)
    del t1, t2
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
    t1 = t2 = None
    del log
    return amp, conv


def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    vec = mycc.amplitudes_to_vector(t1, t2)

    vec, conv = make_implicit_diff(_iter, config.ccsd_implicit_diff,
                                   optimality_cond=_converged_iter,
                                   solver=gen_gmres(), has_aux=True)(
                                        vec, mycc, eris,
                                        diis=adiis, max_cycle=max_cycle, tol=tol,
                                        tolnormt=tolnormt, verbose=log)

    t1, t2 = mycc.vector_to_amplitudes(vec)
    eccsd = mycc.energy(t1, t2, eris)
    log.timer('CCSD')
    vec = None
    del adiis, log
    return conv, eccsd, t1, t2


#@jit
def update_amps(mycc, t1, t2, eris):
    if mycc.cc2:
        raise NotImplementedError

    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir+1) // 2
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    t1new = np.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris, t2sym='jiba')
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc] - np.diag(mo_e_o)
    foo += .5 * np.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:] - np.diag(mo_e_v)
    fvv -= .5 * np.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    # begin _add_ovvv_
    eris_vovv = eris.ovvv.transpose(1,0,2)
    # pylint: disable=invalid-unary-operand-type
    wooVV = -np.dot(t1, eris_vovv.reshape(nvir,-1))

    eris_vovv = lib.unpack_tril(eris_vovv.reshape(nvir*nocc,nvir_pair))
    eris_vovv = eris_vovv.reshape(nvir,nocc,nvir,nvir)

    fvv += 2*np.einsum('kc,ckab->ab', t1, eris_vovv)
    fvv -= np.einsum('kc,bkca->ab', t1, eris_vovv)

    if not mycc.direct:
        vvvo = eris_vovv.transpose(0,2,3,1)#.copy()
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        tmp = np.einsum('ijcd,cdbk->ijbk', tau, vvvo)
        t2new -= np.einsum('ka,ijbk->ijab', t1, tmp)

    wVOov = np.einsum('biac,jc->bija', eris_vovv, t1)

    theta = t2.transpose(1,2,0,3) * 2
    theta -= t2.transpose(0,2,1,3)
    t1new += np.einsum('icjb,cjba->ia', theta, eris_vovv)

    wooVV = lib.unpack_tril(wooVV.reshape(nocc**2,nvir_pair))
    wVooV = wooVV.reshape(nocc,nocc,nvir,nvir).transpose(2,1,0,3)
    # end _add_ovvv_

    woooo = np.asarray(eris.oooo).transpose(0,2,1,3).copy()

    eris_ovoo = eris.ovoo
    eris_oovv = eris.oovv
    foo += np.einsum('kc,kcji->ij', 2*t1, eris_ovoo)
    foo += np.einsum('kc,icjk->ij',  -t1, eris_ovoo)
    tmp = np.einsum('la,jaik->lkji', t1, eris_ovoo)
    woooo += tmp + tmp.transpose(1,0,3,2)

    wVOov -= np.einsum('jbik,ka->bjia', eris_ovoo, t1)
    t2new += wVOov.transpose(1,2,0,3)

    wVooV += np.einsum('kbij,ka->bija', eris_ovoo, t1)

    eris_ovvo = eris.ovvo
    t1new -= np.einsum('jb,jiab->ia', t1, eris_oovv)
    wVooV -= eris_oovv.transpose(2,0,1,3)
    wVOov += wVooV*.5  #: bjia + bija*.5

    t2new += (eris_ovvo*0.5).transpose(0,3,1,2)
    eris_voov = eris_ovvo.conj().transpose(1,0,3,2)
    t1new += 2*np.einsum('jb,aijb->ia', t1, eris_voov)

    tmp  = np.einsum('ic,kjbc->ibkj', t1, eris_oovv)
    tmp += np.einsum('bjkc,ic->jbki', eris_voov, t1)
    t2new -= np.einsum('ka,jbki->jiba', t1, tmp)

    fov += np.einsum('kc,aikc->ia', t1, eris_voov) * 2
    fov -= np.einsum('kc,akic->ia', t1, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1*.5, t1)
    if mycc.dcsd:
        tau += t2 * .5
        theta = t2.transpose(1,0,2,3) - t2 * .5
        fvv_t2 = -np.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo_t2 =  np.einsum('aikb,kjab->ij', eris_voov, theta)
    else:
        tau += t2
    theta  = tau.transpose(1,0,2,3) * 2
    theta -= tau
    fvv -= np.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
    foo += np.einsum('aikb,kjab->ij', eris_voov, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        woooo_t2 = np.einsum('ijab,aklb->ijkl', t2, eris_voov)
    else:
        tau += t2
    woooo += np.einsum('ijab,aklb->ijkl', tau, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if not mycc.dcsd:
        tau += t2 * .5
    wVooV += np.einsum('bkic,jkca->bija', eris_voov, tau)

    tmp = np.einsum('jkca,ckib->jaib', t2, wVooV)
    t2new += tmp.transpose(2,0,1,3)
    tmp *= .5
    t2new += tmp.transpose(0,2,1,3)

    wVOov += eris_voov
    eris_VOov = -.5 * eris_voov.transpose(0,2,1,3)
    tau  =  t2.transpose(1,3,0,2) * 2
    tau -=  t2.transpose(0,3,1,2)
    tau1 = -np.einsum('ia,jb->ibja', t1*2, t1)
    tau +=  tau1
    if mycc.dcsd:
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_voov, tau)
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_VOov, tau1)
    else:
        eris_VOov += eris_voov
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_VOov, tau)

    theta  = t2 * 2
    theta -= t2.transpose(1,0,2,3)
    t2new += np.einsum('kica,ckjb->ijab', theta, wVOov)

    theta = t2.transpose(1,0,2,3) * 2 - t2
    t1new += np.einsum('jb,ijba->ia', fov, theta)
    t1new -= np.einsum('jbki,kjba->ia', eris.ovoo, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        t2new += .5 * np.einsum('ijkl,klab->ijab', woooo_t2, tau)
    tau += t2
    t2new += .5 * np.einsum('ijkl,klab->ijab', woooo, tau)

    ft_ij = foo + np.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - np.einsum('ia,ib->ab', .5*t1, fov)
    t2new += np.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= np.einsum('ki,kjab->ijab', ft_ij, t2)

    if mycc.dcsd:
        fvv += fvv_t2
        foo += foo_t2
    t1new += np.einsum('ib,ab->ia', t1, fvv)
    t1new -= np.einsum('ja,ji->ia', t1, foo)
    t2new += t2new.transpose(1,0,3,2)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = eia[:,None,:,None] + eia[None,:,None,:]
    t1new /= eia
    t2new /= eijab
    eia = eijab = None
    return t1new, t2new


def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=None, t2sym=None):
    '''t2sym: whether t2 has the symmetry t2[ijab]==t2[jiba] or
    t2[ijab]==-t2[jiab] or t2[ijab]==-t2[jiba]
    '''
    if t2sym in ('jiba', '-jiba', '-jiab'):
        Ht2tril = _add_vvvv_tril(mycc, t1, t2, eris, with_ovvv=with_ovvv)
        nocc, nvir = t2.shape[1:3]
        Ht2 = _unpack_t2_tril(Ht2tril, nocc, nvir, out, t2sym)
    else:
        raise NotImplementedError
    return Ht2

def _add_vvvv_tril(mycc, t1, t2, eris, out=None, with_ovvv=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    Using symmetry t2[ijab] = t2[jiba] and Ht2[ijab] = Ht2[jiba], compute the
    lower triangular part of  Ht2
    '''
    if with_ovvv is None:
        with_ovvv = mycc.direct
    nocc, nvir = t2.shape[1:3]

    idx = numpy.tril_indices(nocc)
    tau = t2[idx]
    if t1 is not None:
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        tau += tmp[idx]

    if mycc.direct:   # AO-direct CCSD
        raise NotImplementedError
    else:
        assert not with_ovvv
        Ht2tril = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out)
    del idx
    return Ht2tril

def _unpack_t2_tril(t2tril, nocc, nvir, out=None, t2sym='jiba'):
    t2 = np.empty((nocc,nocc,nvir,nvir), dtype=t2tril.dtype)
    idx, idy = numpy.tril_indices(nocc)
    t2 = ops.index_update(t2, ops.index[idx,idy], t2tril)
    if t2sym == 'jiba':
        t2 = ops.index_update(t2, ops.index[idy,idx], t2tril.transpose(0,2,1))
    elif t2sym == '-jiba':
        t2 = ops.index_update(t2, ops.index[idy,idx], -t2tril.transpose(0,2,1))
    elif t2sym == '-jiab':
        t2 = ops.index_update(t2, ops.index[idy,idx], -t2tril)
        t2 = ops.index_update(t2, ops.index[numpy.diag_indices(nocc)], 0)
    del idx, idy
    return t2

#@jit
def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    vector_t1 = t1.ravel()
    vector_t2 = t2.transpose(0,2,1,3).reshape(nov,nov)[numpy.tril_indices(nov)]
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

#@jit
def energy(cc, t1=None, t2=None, eris=None):
    if t1 is None:
        t1 = cc.t1
    if t2 is None:
        t2 = cc.t2
    if eris is None:
        eris = cc.ao2mo()

    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    e = 2*np.einsum('ia,ia', fov, t1)
    tau  = np.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    #if abs(e.imag) > 1e-4:
    #    logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real

#@util.pytree_node(CC_Tracers, num_args=1)
class CCSD(pyscf_ccsd.CCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        pyscf_ccsd.CCSD.__init__(self, mf, frozen=frozen,
                                 mo_coeff=mo_coeff, mo_occ=mo_occ)
        #if self.diis is True:
        #    self.diis = lib.diis.DIIS(self, self.diis_file, incore=self.incore_complete)
        self.__dict__.update(kwargs)

    def init_amps(self, eris=None):
        log = logger.new_logger(self)
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        e_hf = self.e_hf
        if e_hf is None:
            e_hf = self.get_e_hf(mo_coeff=self.mo_coeff)
        mo_e = eris.mo_energy
        nocc = self.nocc
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]

        t1 = eris.fock[:nocc,nocc:] / eia
        eris_ovov = eris.ovov
        t2 = (eris_ovov.transpose(0,2,1,3).conj()
              / (eia[:,None,:,None] + eia[None,:,None,:]))
        emp2  = 2 * np.einsum('ijab,iajb', t2, eris_ovov)
        emp2 -=     np.einsum('jiab,iajb', t2, eris_ovov)
        self.emp2 = emp2.real

        log.info('Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g',
                 e_hf + self.emp2, self.emp2)
        log.timer('init mp2')
        del log
        return self.emp2, t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None

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
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        if config.moleintor_opt:
            from pyscfad.cc import ccsd_t
            return ccsd_t.kernel(self, eris, t1, t2, self.verbose)
        else:
            from pyscfad.cc import ccsd_t_slow
            return ccsd_t_slow.kernel(self, eris, t1, t2, self.verbose)

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
        #                                    partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
        #                                    partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def amplitude_equation(self, t1, t2, eris):
        raise NotImplementedError

    @property
    def dcsd(self):
        return False

    energy = energy
    update_amps = update_amps
    _add_vvvv = _add_vvvv

#@util.pytree_node(ERI_Tracers)
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

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if config.moleintor_opt:
            return pyscf_ccsd._ChemistsERIs.get_ovvv(self, *slices)
        else:
            ovw = np.asarray(self.ovvv[slices])
            nocc, nvir, nvir_pair = ovw.shape
            ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
            nvir1 = ovvv.shape[2]
            return ovvv.reshape(nocc,nvir,nvir1,nvir1)
