from functools import partial
import tempfile
from typing import Optional, Any
import jax
from jaxopt import implicit_diff, linear_solve
from pyscf import __config__
from pyscf.lib import param, logger
from pyscf.scf import hf
from pyscf.scf.hf import MUTE_CHKFILE
from pyscfad import lib, gto
from pyscfad.lib import numpy as jnp
from pyscfad.lib import stop_grad
from . import _vhf
from . import diis

SCF_IMPLICIT_KERNEL = getattr(__config__, "pyscfad_scf_implicit_kernel", False)

def _converged_scf(mo_coeff, mf, mo_occ, h1e, s1e,
                   conv_tol=1e-10, conv_tol_grad=None, mf_diis=None):
    mol = getattr(mf, "cell", mf.mol)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    _, mo_coeff_new = mf.eig(fock, s1e)
    return mo_coeff_new - mo_coeff

def _scf(mo_coeff, mf, mo_occ, h1e, s1e, conv_tol=1e-10, conv_tol_grad=None, mf_diis=None):
    mol = getattr(mf, "cell", mf.mol)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'cycle= %d E= %.15g', 1, e_tot)

    if conv_tol_grad is None:
        conv_tol_grad = jnp.sqrt(conv_tol)
    scf_conv = False
    for cycle in range(1,mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(stop_grad(mo_energy), stop_grad(mo_coeff))
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        fock = mf.get_fock(stop_grad(h1e), stop_grad(s1e), stop_grad(vhf), stop_grad(dm))
        norm_gorb = jnp.linalg.norm(mf.get_grad(stop_grad(mo_coeff),
                                    stop_grad(mo_occ), stop_grad(fock)))

        norm_gorb = norm_gorb / jnp.sqrt(norm_gorb.size)
        norm_ddm = jnp.linalg.norm(stop_grad(dm)-stop_grad(dm_last))
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True
        if scf_conv:
            break
    return mo_coeff, scf_conv#, mo_energy, mo_occ, e_tot

if SCF_IMPLICIT_KERNEL:
    _scf = implicit_diff.custom_root(_converged_scf, has_aux=True,
                solve=partial(linear_solve.solve_gmres, tol=1e-6, solve_method='incremental'))(_scf)


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None, dm0=None, **kwargs):
    mol = getattr(mf, "cell", mf.mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    if conv_tol_grad is None:
        conv_tol_grad = jnp.sqrt(conv_tol)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(stop_grad(mo_energy), stop_grad(mo_coeff))
    mo_coeff, scf_conv = _scf(mo_coeff, mf, mo_occ, h1e, s1e, conv_tol, conv_tol_grad, mf_diis)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'Extra cycle E= %.15g', e_tot)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = jnp.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == jnp.complex128 or eri.size == nao**4:
        vj, vk = _dot_eri_dm_nosymm(eri, dm, with_j, with_k)
    else:
        if dm.dtype == jnp.complex128:
            raise NotImplementedError
        vj, vk = _vhf.incore(eri, dm, hermi, with_j, with_k)
    return vj, vk

@partial(jax.jit, static_argnums=(2,3))
def _dot_eri_dm_nosymm(eri, dm, with_j, with_k):
    nao = dm.shape[-1]
    eri = eri.reshape((nao,)*4)
    dms = dm.reshape(-1,nao,nao)
    vj = vk = None
    if with_j:
        vj = jnp.einsum('ijkl,xji->xkl', eri, dms)
        vj = vj.reshape(dm.shape)
    if with_k:
        vk = jnp.einsum('ijkl,xjk->xil', eri, dms)
        vk = vk.reshape(dm.shape)
    return vj, vk

@lib.dataclass
class SCF(hf.SCF):
    # pylint: disable=too-many-instance-attributes
    mol: gto.Mole = lib.field(pytree_node=True)
    mo_coeff: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    mo_energy: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    _eri: Optional[jnp.array] = lib.field(pytree_node=True, default=None)

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
    _built: bool = False

    def __post_init__(self):
        if not MUTE_CHKFILE and self.chkfile is None:
            # pylint: disable=R1732
            self._chkfile = tempfile.NamedTemporaryFile(dir=param.TMPDIR)
            self.chkfile = self._chkfile.name

        if self.verbose is None:
            self.verbose = self.mol.verbose
        if self.max_memory is None:
            self.max_memory = self.mol.max_memory
        if self.stdout is None:
            self.stdout = self.mol.stdout
        if not self._built:
            self._built = True
        self._keys = set(self.__dict__.keys())

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is None:
            self._eri = self.mol.intor('int2e', aosym='s1')
        vj, vk = dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        return vj, vk

    def get_init_guess(self, mol=None, key='minao'):
        if mol is None:
            mol = self.mol
        mol = stop_grad(mol)
        return hf.SCF.get_init_guess(self, mol, key)

    def kernel(self, dm0=None, **kwargs):
        self.build(self.mol)
        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, self.conv_tol, self.conv_tol_grad,
                       dm0=dm0, **kwargs)
        return self.e_tot

    def energy_grad(self, dm0=None, mode="rev"):
        """
        Energy gradient wrt AO parameters computed by AD

        NOTE:
            The attributes of the SCF instance will not be modified
        """
        if dm0 is None:
            try:
                dm0 = self.make_rdm1()
            except:
                pass

        def hf_energy(self, dm0=None):
            self.reset()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

        if mode == "rev":
            jac = jax.jacrev(hf_energy)(self, dm0=dm0)
        else:
            if SCF_IMPLICIT_KERNEL:
                msg = """Forward mode differentiation is not available
                         when applying the implicit function differentiation."""
                raise KeyError(msg)
            jac = jax.jacfwd(hf_energy)(self, dm0=dm0)
        if hasattr(jac,"cell"):
            return jac.cell
        else:
            return jac.mol

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        from pyscfad.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df, only_dfj)

@lib.dataclass
class RHF(SCF, hf.RHF):
    pass
