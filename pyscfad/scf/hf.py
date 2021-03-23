from pyscf.scf import hf
import jax
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.gto import mole

def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = jnp.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == jnp.complex128 or eri.size == nao**4:
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


@lib.dataclass
class SCF(hf.SCF):
    mol: mole.Mole = lib.field(pytree_node=True)
    mo_coeff: jnp.array = lib.field(pytree_node=True, default=None)

    verbose: int = None
    max_memory: int = None
    stdout: type = None
    chkfile: str = None

    mo_energy: jnp.array = None
    mo_occ: jnp.array = None
    e_tot: float = None
    converged: bool = None
    callback: type = None
    scf_summary: dict = None
    opt: type = None
    #_eri: jnp.array = None

    init_guess: str = "minao"
    max_cycle: int = 50

    def __post_init__(self):
        # This will reset non-traced attributes to default values
        # NOTE that the default values are defined in the base class
        mf = hf.SCF(self.mol)
        for key, value in mf.__dict__.items():
            if getattr(self, key, None) is None:
                object.__setattr__(self, key, value)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is None:
            self._eri = self.mol.intor('int2e')
        vj, vk = dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        return vj, vk

    def nuc_grad_ad(self):
        """
        Energy gradient wrt nuclear coordinates computed by AD
        """
        dm0 = self.get_init_guess() #avoid tracing through get_init_guess
        jac = jax.jacfwd(self.__class__.kernel)(self, dm0=dm0)
        return jac.mol.coords

RHF = SCF
