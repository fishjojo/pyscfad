from pyscf.lib import direct_sum
from pyscf.tdscf import rhf as pyscf_tdrhf
from pyscfad import util
from pyscfad import ao2mo
from pyscfad.lib import numpy as np

Traced_Attributes = ['_scf', 'mol']

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    assert mo_coeff.dtype == np.double

    nmo = mo_coeff.shape[-1]
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = np.hstack((orbo,orbv))
    nmo = nocc + nvir

    e_ia = direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = np.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = np.zeros_like(a)

    def add_hf(a, b, hyb=1):
        eri_mo = ao2mo.incore.general(mf._eri, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a  = np.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= np.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb

        b  = np.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= np.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        return a, b

    a_hf, b_hf = add_hf(a,b)
    a += a_hf
    b += b_hf
    return a, b

# pylint: disable=abstract-method
@util.pytree_node(Traced_Attributes, num_args=1)
class TDMixin(pyscf_tdrhf.TDMixin):
    def __init__(self, mf, **kwargs):
        pyscf_tdrhf.TDMixin.__init__(self, mf)
        self.__dict__.update(kwargs)

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        return get_ab(mf)

@util.pytree_node(Traced_Attributes, num_args=1)
class TDA(TDMixin, pyscf_tdrhf.TDA):
    pass

CIS = TDA
