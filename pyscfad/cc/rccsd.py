from pyscf import numpy as np
from pyscf.lib import logger
from pyscf.lib import current_memory
from pyscfad import util
from pyscfad.lib import jit
from pyscfad import ao2mo
from pyscfad.cc import ccsd
from pyscfad.cc import rintermediates as imd

@jit
def update_amps(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    mo_oo = np.diagflat(mo_e_o)
    mo_vv = np.diagflat(mo_e_v)

    foo = fock[:nocc,:nocc]#.copy()
    fvv = fock[nocc:,nocc:]#.copy()

    t1new, t2new = cc.amplitude_equation(t1, t2, eris)
    # Move energy terms to the other side
    t1new +=   np.einsum('ac,ic->ia', -mo_vv, t1)
    t1new +=  -np.einsum('ki,ka->ia', -mo_oo, t1)
    if cc.cc2:
        Lvv2 = -np.diagflat(np.diag(fvv))
        tmp = np.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = -np.diagflat(np.diag(foo))
        tmp = np.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        tmp = np.einsum('ac,ijcb->ijab', -mo_vv, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum('ki,kjab->ijab', -mo_oo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)

    eia = mo_e_o[:,None] - mo_e_v
    #eijab = direct_sum('ia,jb->ijab',eia,eia)
    eijab = eia[:,None,:,None] + eia[None,:,None,:]
    t1new /= eia
    t2new /= eijab
    return t1new, t2new

@jit
def amplitude_equation(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]#.copy()
    foo = fock[:nocc,:nocc]#.copy()
    fvv = fock[nocc:,nocc:]#.copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # T1 equation
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = eris.get_ovvv()
    t1new += 2*np.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -np.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*np.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -np.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = eris.ovoo
    t1new +=-2*np.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   np.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*np.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   np.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # T2 equation
    tmp2  = np.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1,3,0,2)
    tmp = np.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = np.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    tmp = np.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3)
    if cc.cc2:
        Woooo2 = eris.oooo.transpose(0,2,1,3)
        Woooo2 += np.einsum('lcki,jc->klij', eris_ovoo, t1)
        Woooo2 += np.einsum('kclj,ic->klij', eris_ovoo, t1)
        Woooo2 += np.einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
        t2new += np.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = np.einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += eris.vvvv.transpose(0,2,1,3)
        t2new += np.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
        #Lvv2 -= np.diagflat(np.diag(fvv))
        tmp = np.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
        #Loo2 -= np.diagflat(np.diag(foo))
        tmp = np.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)

        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        t2new += np.einsum('klij,klab->ijab', Woooo, tau)
        t2new += np.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = np.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
        tmp  = 2.*np.einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   np.einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
    return t1new, t2new

@util.pytree_node(ccsd.CC_Tracers, num_args=1)
class RCCSD(ccsd.CCSD):
    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            raise NotImplementedError

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError

    update_amps = update_amps
    amplitude_equation = amplitude_equation

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    log = logger.new_logger(mycc)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff, compact=False)
        #eri1 = ao2mo.restore(1, eri1, nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc]#.copy()
    eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:]
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc]#.copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:]#.copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:]#.copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc]#.copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]#.copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:]#.copy()
    log.timer('CCSD integral transformation')
    del(log)
    return eris

@util.pytree_node(ccsd.ERI_Tracers)
class _ChemistsERIs(ccsd._ChemistsERIs):
    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv
