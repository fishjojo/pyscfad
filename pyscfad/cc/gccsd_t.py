from functools import reduce
import numpy
from jax import numpy as np
#from jax import jit
from jax import scipy
from pyscf.mp.mp2 import _mo_without_core
from pyscf.lib import logger
from pyscfad.ao2mo import _ao2mo


def spatial2spin(tx, orbspin=None):
    '''Convert T1/T2 of spatial orbital representation to T1/T2 of
    spin-orbital representation
    '''
    if getattr(tx, 'ndim', None) == 2:
        # RCCSD t1 amplitudes
        return spatial2spin((tx,tx), orbspin)
    elif getattr(tx, 'ndim', None) == 4:
        # RCCSD t2 amplitudes
        t2aa = tx - tx.transpose(1,0,2,3)
        return spatial2spin((t2aa,tx,t2aa), orbspin)
    elif len(tx) == 2:  # t1
        t1a, t1b = tx
        nocc_a, nvir_a = t1a.shape
        nocc_b, nvir_b = t1b.shape
    elif len(tx) == 3:  # t2
        t2aa, t2ab, t2bb = tx
        nocc_a, nocc_b, nvir_a, nvir_b = t2ab.shape
    else:
        raise RuntimeError('Unknown T amplitudes')

    if orbspin is None:
        assert nocc_a == nocc_b
        orbspin = numpy.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]

    if len(tx) == 2:  # t1
        t1 = np.zeros((nocc,nvir), dtype=t1a.dtype)
        t1 = t1.at[idxoa[:,None],idxva].add(t1a)
        t1 = t1.at[idxob[:,None],idxvb].add(t1b)
        return t1

    else:
        t2 = np.zeros((nocc**2,nvir**2), dtype=t2aa.dtype)
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxoba = idxob[:,None] * nocc + idxoa
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvba = idxvb[:,None] * nvir + idxva
        idxvbb = idxvb[:,None] * nvir + idxvb
        t2aa = t2aa.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        t2ab = t2ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        t2bb = t2bb.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        t2 = t2.at[idxoaa.ravel()[:,None],idxvaa.ravel()].add(t2aa)
        t2 = t2.at[idxobb.ravel()[:,None],idxvbb.ravel()].add(t2bb)
        t2 = t2.at[idxoab.ravel()[:,None],idxvab.ravel()].add(t2ab)
        t2 = t2.at[idxoba.T.ravel()[:,None],idxvba.T.ravel()].add(t2ab)
        abba = -t2ab
        t2 = t2.at[idxoab.ravel()[:,None],idxvba.T.ravel()].add(abba)
        t2 = t2.at[idxoba.T.ravel()[:,None],idxvab.ravel()].add(abba)
        return t2.reshape(nocc,nocc,nvir,nvir)


class _PhysicistsERIs:
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        # NOTE only works for RHF
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_coeff = _mo_without_core(mycc, mo_coeff)
        nocc = mycc.nocc

        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        orbspin = numpy.array([0,1]*nocc + [0,1]*nvir)
        mo_coeff1 = np.zeros((nao*2,nmo*2), dtype=mo_coeff.dtype)
        mo_coeff1 = mo_coeff1.at[:nao,orbspin==0].set(mo_coeff)
        mo_coeff1 = mo_coeff1.at[nao:,orbspin==1].set(mo_coeff)
        self.mo_coeff = mo_coeff1

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        fockao = scipy.linalg.block_diag(fockao,fockao)
        self.fock = reduce(np.dot, (self.mo_coeff.conj().T, fockao, self.mo_coeff))

        self.nocc = nocc*2
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for GCCSD', gap)
        return self

def _make_df_eris_incore(cc, mo_coeff=None):
    eris = _PhysicistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo_a = eris.mo_coeff[:nao//2]
    mo_b = eris.mo_coeff[nao//2:]

    ijslice = (0, nmo, 0, nmo)
    eri1 = with_df._cderi
    Lpq_a = _ao2mo.nr_e2(eri1, mo_a, ijslice, aosym='s2', mosym='s1').reshape(naux,-1)
    Lpq_b = _ao2mo.nr_e2(eri1, mo_b, ijslice, aosym='s2', mosym='s1').reshape(naux,-1)

    eri  = np.dot(Lpq_a.T, Lpq_a)
    eri += np.dot(Lpq_b.T, Lpq_b)
    eri1 = np.dot(Lpq_a.T, Lpq_b)
    eri += eri1
    eri += eri1.T

    eri1 = Lpq_a = Lpq_b = None

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc]
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:]
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:]
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:]
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc]
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:]
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:]
    return eris

#@jit
def kernel(cc, prj, eris=None, t1=None, t2=None, verbose=logger.NOTE):
    log = logger.new_logger(cc, verbose)
    if t1 is None or t2 is None:
        t1, t2 = cc.t1, cc.t2

    eris = _make_df_eris_incore(cc)
    # NOTE RCCSD -> G
    t1 = spatial2spin(t1)
    t2 = spatial2spin(t2)

    nocc, nvir = t1.shape

    assert prj.shape[1]*2 == nocc
    prj1 = np.empty((prj.shape[0],nocc), prj.dtype)
    prj1 = prj1.at[:,0::2].set(prj)
    prj1 = prj1.at[:,1::2].set(prj)

    bcei = np.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = np.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = np.asarray(eris.oovv).conj().transpose(2,3,0,1)

    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    d3 = (eia[:,None,None,:,None,None] +
          eia[None,:,None,None,:,None] +
          eia[None,None,:,None,None,:])

    t3c  =(np.einsum('jkae,bcei->ijkabc', t2, bcei) -
           np.einsum('imbc,majk->ijkabc', t2, majk))
    t3c  = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
    t3c  = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
    t3d  = np.einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += np.einsum('ai,jkbc->ijkabc', eris.fock[nocc:,:nocc], t2)
    t3d  = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
    t3d  = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)

    t3 = (t3c + t3d) / d3
    t3 = np.einsum('pjkabc,ip->ijkabc', t3.conj(), prj1)

    w = t3c
    w = np.einsum('pjkabc,ip->ijkabc', w, prj1)
    et = np.einsum('ijkabc,ijkabc', t3, w) / 36

    log.info('CCSD(T) correction = %.15g', et)
    return et
