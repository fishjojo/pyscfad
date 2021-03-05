import numpy
from pyscf.lib import logger
from pyscf.scf import hf as molhf
from pyscfad.gto import mole
from jax import numpy as jnp
from flax import struct

def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = jnp.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == numpy.complex128 or eri.size == nao**4:
        eri = eri.reshape((nao,)*4)
        dms = dm.reshape(-1,nao,nao)
        vj = vk = None
        if with_j:
            vj = jnp.einsum('ijkl,xji->xkl', eri, dms)
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = jnp.einsum('ijkl,xjk->xil', eri, dms)
            vk = vk.reshape(dm.shape)
    else:
        raise NotImplementedError
    return vj, vk

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:,mo_occ>0]
    return jnp.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = jnp.einsum('ij,ji->', h1e, dm)
    e_coul = jnp.einsum('ij,ji->', vhf, dm) * .5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1.val, e_coul.val)
    return (e1+e_coul).real, e_coul


@struct.dataclass
class SCF(molhf.SCF):
    mol: mole.Mole
    mo_coeff: jnp.array = None
    mo_occ: jnp.array = struct.field(pytree_node=False, default=None)
    _eri: jnp.array = None

    def __post_init__(self):
        mf = molhf.SCF(self.mol)
        for k, v in mf.__dict__.items():
            if not k in self.__dict__:
                object.__setattr__(self, k, v)

    def setattr(self, attr, value):
        object.__setattr__(self, attr, value)

    def get_init_guess(self, mol=None, key='minao'):
        dm = molhf.SCF.get_init_guess(self, mol=mol, key=key)
        return dm

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is None:
            object.__setattr__(self, "_eri", mol.intor('int2e'))
        vj, vk = dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = jnp.asarray(dm) - dm_last
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
