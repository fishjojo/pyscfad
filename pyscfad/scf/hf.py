from functools import partial
import numpy
from jax import jacrev, jacfwd
from jaxopt import linear_solve
from pyscf import __config__
from pyscf import numpy as np
from pyscf.lib import param, logger
from pyscf.scf import hf as pyscf_hf
from pyscf.scf import chkfile
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL
from pyscfad import lib
from pyscfad.lib import jit
from pyscfad import util
from pyscfad import implicit_diff
from pyscfad import gto
from pyscfad import df
from pyscfad.lib import stop_grad
from pyscfad.scf import _vhf
from pyscfad.scf.diis import SCF_DIIS
from pyscfad.scf._eigh import eigh

SCF_IMPLICIT_DIFF = getattr(__config__, "pyscfad_scf_implicit_diff", False)
Traced_Attributes = ['mol', 'mo_coeff', 'mo_energy', '_eri']

def eig(h, s, x0=None):
    e, c = eigh(h, s, x0)
    return e, c

def _converged_scf(mo_coeff, mf, s1e, h1e, mo_occ):
    mol = getattr(mf, "cell", mf.mol)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    _, mo_coeff = mf.eig(fock, s1e, mo_coeff)
    return mo_coeff

def _scf(mo_coeff, mf, s1e, h1e, mo_occ, *,
         dm0=None, conv_tol=1e-10, conv_tol_grad=None, diis=None,
         dump_chk=True, callback=None, log=None):
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    if log is None:
        log = logger.new_logger(mf)
    scf_conv = False

    mol = getattr(mf, "cell", mf.mol)
    if dm0 is not None:
        dm = dm0
    elif mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
    else:
        raise KeyError("Either dm or mo_coeff and mo_occ need to be given.")

    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)

    cput1 = log.timer('initialize scf')
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(stop_grad(mo_energy), stop_grad(mo_coeff))
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        fock = mf.get_fock(stop_grad(h1e), stop_grad(s1e), stop_grad(vhf), stop_grad(dm))
        norm_gorb = numpy.linalg.norm(mf.get_grad(stop_grad(mo_coeff),
                                      stop_grad(mo_occ), stop_grad(fock)))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(stop_grad(dm)-stop_grad(dm_last))
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = log.timer('cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break
    return mo_coeff, scf_conv, mo_occ, mo_energy

if SCF_IMPLICIT_DIFF:
    solver = partial(linear_solve.solve_gmres, tol=1e-9,
                     solve_method='incremental', maxiter=30)
    _scf = implicit_diff.custom_fixed_point(
                _converged_scf, solve=solver, has_aux=True,
                nondiff_argnums=(4,), use_converged_args={4:2})(_scf)


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    log = logger.new_logger(mf)
    cput0 = (log._t0, log._w0)
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    mol = getattr(mf, "cell", mf.mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    if mf._eri is None:
        if getattr(mf, 'with_df', None) is None:
            mf._eri = mol.intor('int2e', aosym='s1')

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = numpy.linalg.cond(stop_grad(s1e))
    log.debug('cond(S) = %s', cond)
    if cond.max()*1e-17 > conv_tol:
        log.warn('Singularity detected in overlap matrix (condition number = %4.3g). '
                 'SCF may be inaccurate and hard to converge.', cond.max())

    if mf.max_cycle <= 0:
        # Skip SCF iterations. Compute only the total energy of the initial density
        vhf = mf.get_veff(mol, dm)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        log.info('init E= %.15g', e_tot)

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(stop_grad(mo_energy), stop_grad(mo_coeff))
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    #mf.pre_kernel(locals())

    # SCF iteration
    mo_coeff, scf_conv, mo_occ, mo_energy = \
            _scf(mo_coeff, mf, s1e, h1e, mo_occ, dm0=dm,
                 conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
                 diis=mf_diis, dump_chk=dump_chk, callback=callback, log=log)

    # Recompute energy so that energy has the correct gradient from mo_coeff
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    run_extra_cycle = False
    if SCF_IMPLICIT_DIFF and (not conv_check or not scf_conv):
        log.warn('\tAn extra scf cycle is going to be run\n'
                 '\tin order to restore the mo_energy derivatives\n'
                 '\tmissing in implicit differentiation.')
        run_extra_cycle = True

    if (scf_conv and conv_check) or run_extra_cycle:
        # An extra diagonalization, to remove level shift
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(stop_grad(mo_energy), stop_grad(mo_coeff))
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(stop_grad(h1e), stop_grad(s1e),
                           stop_grad(vhf), stop_grad(dm))
        norm_gorb = numpy.linalg.norm(mf.get_grad(stop_grad(mo_coeff),
                                      stop_grad(mo_occ), stop_grad(fock)))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(stop_grad(dm)-stop_grad(dm_last))

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        log.info('Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    log.timer('scf_cycle', *cput0)
    del log
    # A post-processing hook before return
    #mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == np.complex128 or eri.size == nao**4:
        vj, vk = _dot_eri_dm_nosymm(eri, dm, with_j, with_k)
    else:
        if dm.dtype == np.complex128:
            raise NotImplementedError
        vj, vk = _vhf.incore(eri, dm, hermi, with_j, with_k)
    return vj, vk

@partial(jit, static_argnums=(2,3))
def _dot_eri_dm_nosymm(eri, dm, with_j, with_k):
    nao = dm.shape[-1]
    eri = eri.reshape((nao,)*4)
    dms = dm.reshape(-1,nao,nao)
    vj = vk = None
    if with_j:
        vj = np.einsum('ijkl,xji->xkl', eri, dms)
        vj = vj.reshape(dm.shape)
    if with_k:
        vk = np.einsum('ijkl,xjk->xil', eri, dms)
        vk = vk.reshape(dm.shape)
    return vj, vk

@util.pytree_node(Traced_Attributes, num_args=1)
class SCF(pyscf_hf.SCF):
    '''
    A subclass of :class:`pyscf.scf.hf.SCF` where the following
    attributes can be traced.

    Attributes:
        mol : :class:`pyscfad.gto.Mole` object
            Molecular structure and global options.
        mo_coeff : array
            Molecular orbital coefficients.
        mo_energy : array
            Molecular orbital energies.
        _eri : array
            Two electron repulsion integrals.
    '''
    DIIS = SCF_DIIS

    def __init__(self, mol, **kwargs):
        pyscf_hf.SCF.__init__(self, mol)
        self.__dict__.update(kwargs)

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
        return pyscf_hf.SCF.get_init_guess(self, mol, key)

    # pylint: disable=arguments-differ
    def kernel(self, dm0=None, **kwargs):
        self.build(self.mol)
        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, self.conv_tol, self.conv_tol_grad,
                       dm0=dm0, **kwargs)
        return self.e_tot

    def _eigh(self, h, s, x0=None):
        return eig(h, s, x0)

    def eig(self, h, s, x0=None):
        return self._eigh(h, s, x0)

    def energy_grad(self, dm0=None, mode="rev"):
        """
        Energy gradient wrt AO parameters computed by AD.
        In principle, MO response is not needed, and we can just take
        the derivative of eigen decomposition with converged
        density matrix. But here it is implemented in this way to show
        the difference between unrolling for loops and implicit differentiation.

        NOTE:
            The attributes of the SCF instance will not be modified
        """
        if dm0 is None:
            try:
                dm0 = self.make_rdm1()
            except TypeError:
                pass

        def hf_energy(self, dm0=None):
            self.reset()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

        if mode == "rev":
            jac = jacrev(hf_energy)(self, dm0=dm0)
        else:
            if SCF_IMPLICIT_DIFF:
                msg = """Forward mode differentiation is not available
                         when applying the implicit function differentiation."""
                raise KeyError(msg)
            jac = jacfwd(hf_energy)(self, dm0=dm0)
        if hasattr(jac,"cell"):
            return jac.cell
        else:
            return jac.mol

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        return df.density_fit(self, auxbasis, with_df, only_dfj)

@util.pytree_node(Traced_Attributes, num_args=1)
class RHF(SCF, pyscf_hf.RHF):
    pass
