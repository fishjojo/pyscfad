import tempfile
from typing import Optional, Any
import jax

from pyscf import __config__
from pyscf.lib import param
from pyscf.scf import hf, diis
from pyscf.scf.hf import MUTE_CHKFILE

from pyscfad import lib, gto
from pyscfad.lib import numpy as jnp

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
    mol: gto.Mole = lib.field(pytree_node=True)
    mo_coeff: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    mo_energy: Optional[jnp.array] = lib.field(pytree_node=True, default=None)

    conv_tol: float = getattr(__config__, 'scf_hf_SCF_conv_tol', 1e-9)
    conv_tol_grad: Optional[float] = getattr(__config__, 'scf_hf_SCF_conv_tol_grad', None)
    max_cycle: int = getattr(__config__, 'scf_hf_SCF_max_cycle', 50)
    init_guess: str = getattr(__config__, 'scf_hf_SCF_init_guess', 'minao')

    DIIS: Any = diis.SCF_DIIS
    diis: Any = getattr(__config__, 'scf_hf_SCF_diis', True)
    diis_space: int = getattr(__config__, 'scf_hf_SCF_diis_space', 8)
    diis_start_cycle: int = getattr(__config__, 'scf_hf_SCF_diis_start_cycle', 1)
    diis_file: Optional[str] = None
    diis_space_rollback: bool = False

    damp: float = getattr(__config__, 'scf_hf_SCF_damp', 0.)
    level_shift: float = getattr(__config__, 'scf_hf_SCF_level_shift', 0.)
    direct_scf: bool = getattr(__config__, 'scf_hf_SCF_direct_scf', True)
    direct_scf_tol: float = getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)
    conv_check: bool = getattr(__config__, 'scf_hf_SCF_conv_check', True)

    verbose: int = None
    max_memory: int = None
    stdout: Any = None

    chkfile: Optional[str] = None
    _chkfile: Any = None

    mo_occ: Optional[jnp.array] = None
    e_tot: float = 0.
    converged: bool = False
    callback: Any = None
    scf_summary: dict = lib.field(default_factory = dict)

    opt: Any = None
    _eri: Optional[jnp.array] = None

    def __post_init__(self):
        if not MUTE_CHKFILE and self.chkfile is None:
            self._chkfile = tempfile.NamedTemporaryFile(dir=param.TMPDIR)
            self.chkfile = self._chkfile.name

        if self.verbose is None:
            self.verbose = self.mol.verbose
        if self.max_memory is None:
            self.max_memory = self.mol.max_memory
        if self.stdout is None:
            self.stdout = self.mol.stdout

        self._keys = set(self.__dict__.keys())

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is None:
            self._eri = self.mol.intor('int2e')
        vj, vk = dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        return vj, vk

    def nuc_grad_ad(self):
        """
        Energy gradient wrt nuclear coordinates computed by AD
        """
        if self.converged:
            dm0 = self.make_rdm1()
            self.reset() # need to reset _eri to get its gradient
        else:
            dm0 = self.get_init_guess() #avoid tracing through get_init_guess
        jac = jax.jacfwd(self.__class__.kernel)(self, dm0=dm0)
        return jac.mol.coords

RHF = SCF
