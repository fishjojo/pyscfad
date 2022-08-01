'''
Intermediates for restricted CCSD.  Complex integrals are supported.
'''
from pyscf import numpy as np
#from pyscf.lib import logger
#from pyscfad import lib
#from pyscfad.lib import jit

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

#@jit
def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki  = 2*np.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   np.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*np.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   np.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Fki += foo
    return Fki

#@jit
def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac  =-2*np.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   np.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*np.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   np.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Fac += fvv
    return Fac

#@jit
def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc  = 2*np.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   np.einsum('kdlc,ld->kc', eris_ovov, t1)
    Fkc += fov
    return Fkc

### Eqs. (40)-(41) "lambda"
#@jit
def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1, t2, eris) + np.einsum('kc,ic->ki',fov, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2*np.einsum('lcki,lc->ki', eris_ovoo, t1)
    Lki -=   np.einsum('kcli,lc->ki', eris_ovoo, t1)
    return Lki

#@jit
def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1, t2, eris) - np.einsum('kc,ka->ac',fov, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"
#@jit
def cc_Woooo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij  = np.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += np.einsum('kclj,ic->klij', eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wklij += np.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += np.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

#@jit
def cc_Wvvvv(t1, t2, eris):
    # Incore
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd  = np.einsum('kdac,kb->abcd', eris_ovvv,-t1)
    Wabcd -= np.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    return Wabcd

#@jit
def cc_Wvoov(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic  = np.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= np.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakic -= 0.5*np.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*np.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= np.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += np.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

#@jit
def cc_Wvovo(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci  = np.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= np.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5*np.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= np.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

#@jit
def Wooov(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wklid  = np.einsum('ic,kcld->klid', t1, eris_ovov)
    Wklid += np.asarray(eris.ovoo).transpose(2,0,3,1)
    return Wklid

#@jit
def Wvovv(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Walcd  = np.einsum('ka,kcld->alcd',-t1, eris_ovov)
    Walcd += np.asarray(eris.get_ovvv()).transpose(2,0,3,1)
    return Walcd

#@jit
def W1ovvo(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wkaci  = 2*np.einsum('kcld,ilad->kaci', eris_ovov, t2)
    Wkaci +=  -np.einsum('kcld,liad->kaci', eris_ovov, t2)
    Wkaci +=  -np.einsum('kdlc,ilad->kaci', eris_ovov, t2)
    Wkaci += np.asarray(eris.ovvo).transpose(0,2,1,3)
    return Wkaci

#@jit
def W2ovvo(t1, t2, eris):
    Wkaci = np.einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkaci += np.einsum('kcad,id->kaci', eris_ovvv, t1)
    return Wkaci

#@jit
def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

#@jit
def W1ovov(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wkbid = -np.einsum('kcld,ilcb->kbid', eris_ovov, t2)
    Wkbid += np.asarray(eris.oovv).transpose(0,2,1,3)
    return Wkbid

#@jit
def W2ovov(t1, t2, eris):
    Wkbid = np.einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkbid += np.einsum('kcbd,ic->kbid', eris_ovvv, t1)
    return Wkbid

#@jit
def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

#@jit
def Woooo(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wklij  = np.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += np.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij += np.einsum('ldki,jd->klij', eris_ovoo, t1)
    Wklij += np.einsum('kclj,ic->klij', eris_ovoo, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

#@jit
def Wvvvv(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wabcd  = np.einsum('kcld,klab->abcd', eris_ovov, t2)
    Wabcd += np.einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd -= np.einsum('ldac,lb->abcd', eris_ovvv, t1)
    Wabcd -= np.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

#@jit
def Wvvvo(t1, t2, eris, _Wvvvv=None):
    nocc,nvir = t1.shape
    eris_ovvv = np.asarray(eris.get_ovvv())
    # Check if t1=0 (HF+MBPT(2))
    # don't make vvvv if you can avoid it!
    Wabcj  =  -np.einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -np.einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*np.einsum('ldac,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -np.einsum('ldac,ljbd->abcj', eris_ovvv, t2)
    Wabcj +=  -np.einsum('lcad,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -np.einsum('kcbd,jkda->abcj', eris_ovvv, t2)
    eris_ovoo = np.asarray(eris.ovoo)
    Wabcj +=   np.einsum('kclj,lkba->abcj', eris_ovoo, t2)
    Wabcj +=   np.einsum('kclj,lb,ka->abcj', eris_ovoo, t1, t1)
    Wabcj +=  -np.einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    Wabcj += np.asarray(eris_ovvv).transpose(3,1,2,0).conj()
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1, t2, eris)
        Wabcj += np.einsum('abcd,jd->abcj', _Wvvvv, t1)
    return Wabcj

#@jit
def Wovoo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkbij  =   np.einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -np.einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   np.einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*np.einsum('ldki,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=  -np.einsum('ldki,jldb->kbij', eris_ovoo, t2)
    Wkbij +=  -np.einsum('kdli,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=   np.einsum('kcbd,jidc->kbij', eris_ovvv, t2)
    Wkbij +=   np.einsum('kcbd,jd,ic->kbij', eris_ovvv, t1, t1)
    Wkbij +=  -np.einsum('kclj,libc->kbij', eris_ovoo, t2)
    Wkbij +=   np.einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    Wkbij += np.asarray(eris_ovoo).transpose(3,1,2,0).conj()
    return Wkbij

#@jit
def _get_vvvv(eris):
    if eris.vvvv is None and getattr(eris, 'vvL', None) is not None:  # DF eris
        raise NotImplementedError
    #    vvL = np.asarray(eris.vvL)
    #    nvir = int(np.sqrt(eris.vvL.shape[0]*2))
    #    return ao2mo.restore(1, lib.dot(vvL, vvL.T), nvir)
    elif eris.vvvv.ndim == 2:
    #    nvir = int(np.sqrt(eris.vvvv.shape[0]*2))
    #    return ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
        raise NotImplementedError
    else:
        return eris.vvvv
