import numpy
from pyscf.lib import logger
from pyscf.scf import hf
from pyscfad import lib
from pyscfad.lib import np_helper as np
from pyscfad.gto import mole

def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == numpy.complex128 or eri.size == nao**4:
        eri = eri.reshape((nao,)*4)
        dms = dm.reshape(-1,nao,nao)
        vj = vk = None
        if with_j:
            vj = np.einsum('ijkl,xji->xkl', eri, dms)
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = np.einsum('ijkl,xjk->xil', eri, dms)
            vk = vk.reshape(dm.shape)
    else:
        raise NotImplementedError
    return vj, vk

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:,mo_occ>0]
    return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm)
    e_coul = np.einsum('ij,ji->', vhf, dm) * .5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    #logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    return (e1+e_coul).real, e_coul


@lib.dataclass
class SCF(hf.SCF):
    mol: mole.Mole = lib.field(pytree_node=True)
    mo_coeff: np.array = lib.field(pytree_node=True, default=None)

    verbose: int = None
    max_memory: int = None
    stdout: type = None
    chkfile: str = None

    mo_energy: np.array = None
    mo_occ: np.array = None
    e_tot: float = None
    converged: bool = None
    callback: type = None
    scf_summary: dict = None
    opt: type = None
    #_eri: np.array = None

    def __post_init__(self):
        # This will reset non-traced attributes to default values
        # NOTE that the default values are defined in the base class
        mf = hf.SCF(self.mol)
        for key, value in mf.__dict__.items():
            if getattr(self, key, None) is None:
                object.__setattr__(self, key, value)

    #def get_init_guess(self, mol=None, key='minao'):
    #    dm = hf.SCF.get_init_guess(self, mol=mol, key=key)
    #    return dm

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is None:
            #object.__setattr__(self, "_eri", mol.intor('int2e'))
            self._eri = self.mol.intor('int2e')
        vj, vk = dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = np.asarray(dm) - dm_last
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return vhf_last + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    energy_elec = energy_elec

RHF = SCF
