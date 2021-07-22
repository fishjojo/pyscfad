from typing import Optional, Any
from pyscf import __config__
from pyscf.cc import eom_rccsd as pyscf_eom_rccsd
from pyscfad import lib
from pyscfad.lib import numpy as np
from pyscfad import gto
from pyscfad.cc import ccsd


@lib.dataclass
class EOM(pyscf_eom_rccsd.EOM):
    _cc: ccsd.CCSD = lib.field(pytree_node=True)
    mol: Optional[gto.Mole] = None
    verbose: Optional[int] = None
    stdout: Any = None
    max_memory: Optional[int] = None
    max_space: int = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
    max_cycle: Optional[int] = None
    conv_tol: Optional[float] = None
    partition: Any = getattr(__config__, 'eom_rccsd_EOM_partition', None)
    
    e: Optional[float] = None
    v: Optional[np.array] = None
    nocc: Optional[np.array] = None
    nmo: Optional[np.array] = None
   
    def __post_init__(self):
        if self.mol is None:
            self.mol = self._cc.mol
        if self.verbose is None:
            self.verbose = self._cc.verbose
        if self.stdout is None:
            self.stdout = self._cc.stdout
        if self.max_memory is None:
            self.max_memory = self._cc.max_memory
        if self.max_cycle is None:
            self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', self._cc.max_cycle)
        if self.conv_tol is None:
            self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', self._cc.conv_tol)

        if self.nocc is None:
            self.nocc = self._cc.nocc
        if self.nmo is None:
            self.nmo = self._cc.nmo 
        
        self._keys = set(self.__dict__.keys())

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype)

    if eom.partition == 'mp':
        foo_diag = np.diag(foo)
        fvv_diag = np.diag(fvv)
        Hr2 = - foo_diag[:,None,None] - foo_diag[None,:,None]  + fvv_diag[None,None,:]
    else:
        Lvv_diag = np.diag(imds.Lvv)
        Loo_diag = np.diag(imds.Loo)
        Hr2 = - Loo_diag[:,None,None] - Loo_diag[None,:,None]  + Lvv_diag[None,None,:]
        wij = np.einsum('ijij->ij', imds.Woooo)
        Hr2 += wij[:,:,None]
        wjb = np.einsum('jbbj->jb', imds.Wovvo)
        Hr2 += 2 * wjb[None,:,:]
        Hr2 += -np.einsum('ij,jb->ijb', np.eye(nocc), wjb)
        wjb = np.einsum('jbjb->jb', imds.Wovov)
        Hr2 += -wjb[None,:,:]
        Hr2 += -wjb[:,None,:]
        Hr2 += -2*np.einsum('jibc,ijcb->ijb', imds.Woovv, t2)
        Hr2 += np.einsum('ijbc,ijcb->ijb', imds.Woovv, t2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kbij,k->ijb', imds.Wovoo, r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += np.einsum('bd,ijd->ijb', fvv, r2)
        Hr2 += -np.einsum('ki,kjb->ijb', foo, r2)
        Hr2 += -np.einsum('lj,ilb->ijb', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += np.einsum('bd,ijd->ijb', imds.Lvv, r2)
        Hr2 += -np.einsum('ki,kjb->ijb', imds.Loo, r2)
        Hr2 += -np.einsum('lj,ilb->ijb', imds.Loo, r2)
        Hr2 +=  np.einsum('klij,klb->ijb', imds.Woooo, r2)
        Hr2 += 2*np.einsum('lbdj,ild->ijb', imds.Wovvo, r2)
        Hr2 +=  -np.einsum('kbdj,kid->ijb', imds.Wovvo, r2)
        Hr2 +=  -np.einsum('lbjd,ild->ijb', imds.Wovov, r2) #typo in Ref
        Hr2 +=  -np.einsum('kbid,kjd->ijb', imds.Wovov, r2)
        tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2)
        tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2)
        Hr2 += -np.einsum('c,ijcb->ijb', tmp, imds.t2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMIP(EOM, pyscf_eom_rccsd.EOMIP):
    get_diag = ipccsd_diag
    matvec = ipccsd_matvec

class EOMEA(EOM, pyscf_eom_rccsd.EOMEA):
    pass

class EOMEE(EOM, pyscf_eom_rccsd.EOMEE):
    pass
