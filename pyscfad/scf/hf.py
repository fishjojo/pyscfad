from functools import partial, reduce, wraps
import numpy
import jax
from pyscf import numpy as np
from pyscf.data import nist
from pyscf.lib import logger, stop_grad, module_method
from pyscf.scf import hf as pyscf_hf
from pyscf.scf import chkfile
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL

from pyscfad import config
from pyscfad import lib
from pyscfad.lib import jit, stop_trace
from pyscfad import util
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad import df
from pyscfad.scf import _vhf
from pyscfad.scf.diis import SCF_DIIS
from pyscfad.scipy.linalg import eigh
from pyscfad.tools.linear_solver import gen_gmres

Traced_Attributes = ['mol', '_eri']#, 'mo_coeff', 'mo_energy']

def _scf_optimality_cond(dm, mf, s1e, h1e):
    mol = getattr(mf, 'cell', mf.mol)
    vhf = mf.get_veff(mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = stop_trace(mf.get_occ)(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    del mo_energy, mo_occ
    return dm


def _scf(dm, mf, s1e, h1e, *,
         conv_tol=1e-10, conv_tol_grad=None, diis=None,
         dump_chk=True, callback=None, log=None):
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    if log is None:
        log = logger.new_logger(mf)
    scf_conv = False
    mol = getattr(mf, 'cell', mf.mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    cput1 = log.timer('initialize scf')

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = stop_trace(mf.get_occ)(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        fock = stop_trace(mf.get_fock)(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(stop_trace(mf.get_grad)(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(stop_grad(dm - dm_last))
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

        cput1 = log.timer(f'cycle= {cycle+1}', *cput1)

        if scf_conv:
            break
    return dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    log = logger.new_logger(mf)
    cput0 = (log._t0, log._w0)
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    mol = getattr(mf, 'cell', mf.mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    # NOTE if use implicit differentiation,
    # the eri derivative will be lost if not computed before SCF iterations.
    if config.scf_implicit_diff:
        if getattr(mf, 'with_df', None) is not None:
            if hasattr(mf.with_df, '_cderi') and mf.with_df._cderi is None:
                mf.with_df.build()
        else:
            if mf._eri is None:
                if config.moleintor_opt:
                    mf._eri = mol.intor('int2e', aosym='s4')
                else:
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
        mo_occ = stop_trace(mf.get_occ)(mo_energy, mo_coeff)
        # hack for ROHF
        mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)
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
    # NOTE if use implicit differentiation, only dm will have gradient.
    dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = \
            make_implicit_diff(_scf, config.scf_implicit_diff,
                    optimality_cond=_scf_optimality_cond,
                    solver=gen_gmres(), has_aux=True)(
                 dm, mf, s1e, h1e,
                 conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
                 diis=mf_diis, dump_chk=dump_chk, callback=callback, log=log)

    run_extra_cycle = False
    if config.scf_implicit_diff and (not conv_check or not scf_conv):
        log.warn('\tAn extra scf cycle is going to be run\n'
                 '\tin order to restore the mo_energy derivatives\n'
                 '\tmissing in implicit differentiation.')
        run_extra_cycle = True

    if (scf_conv and conv_check) or run_extra_cycle:
        # An extra diagonalization, to remove level shift
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = stop_trace(mf.get_occ)(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = stop_trace(mf.get_fock)(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(stop_trace(mf.get_grad)(mo_coeff, mo_occ, fock))
        del fock, vhf
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(stop_grad(dm - dm_last))

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

    # hack for ROHF
    mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)

    log.timer('scf_cycle', *cput0)
    del log
    # A post-processing hook before return
    #mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


@partial(jit, static_argnums=(2,3))
def _dot_eri_dm_s1(eri, dm, with_j, with_k):
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


def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    if eri.dtype == np.complex128 or eri.size == nao**4:
        vj, vk = _dot_eri_dm_s1(eri, dm, with_j, with_k)
    else:
        if eri.dtype == np.complex128:
            raise NotImplementedError
        vj, vk = _vhf.incore(eri, dm, hermi, with_j, with_k)
    return vj, vk


@wraps(pyscf_hf.energy_elec)
def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm).real
    e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    mf.scf_summary['e1'] = stop_grad(e1)
    mf.scf_summary['e2'] = stop_grad(e_coul)
    logger.debug(mf, 'E1 = %s  E_coul = %s',
                 stop_grad(e1), stop_grad(e_coul))
    return e1+e_coul, e_coul


@wraps(pyscf_hf.make_rdm1)
def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:,mo_occ>0]
    dm = np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    return dm


@wraps(pyscf_hf.level_shift)
def level_shift(s, d, f, factor):
    dm_vir = s - reduce(np.dot, (s, d, s))
    return f + dm_vir * factor


@wraps(pyscf_hf.dip_moment)
def dip_moment(mol, dm, unit='Debye', verbose=logger.NOTE, **kwargs):
    log = logger.new_logger(mol, verbose)

    if 'unit_symbol' in kwargs:  # pragma: no cover
        log.warn('Kwarg "unit_symbol" was deprecated. It was replaced by kwarg '
                 'unit since PySCF-1.5.')
        unit = kwargs['unit_symbol']

    if getattr(dm, 'ndim', None) != 2:
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm).real

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == 'DEBYE':
        mol_dip *= nist.AU2DEBYE
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    del log
    return mol_dip


def damping(s, d, f, factor):
    dm_vir = np.eye(s.shape[0]) - np.dot(s, d)
    f0 = reduce(np.dot, (dm_vir, f, d, s))
    f0 = (f0 + f0.conj().T) * (factor/(factor+1.))
    return f - f0


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
            if config.moleintor_opt:
                self._eri = self.mol.intor('int2e', aosym='s4')
            else:
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

    def _eigh(self, h, s):
        return eigh(h, s)

    def eig(self, h, s):
        return self._eigh(h, s)

    def energy_grad(self, dm0=None, mode='rev'):
        '''
        Energy gradient with respect to AO parameters computed by AD.
        In principle, MO response is not needed, and we can just take
        the derivative of eigen decomposition with converged
        density matrix. But here it is implemented in this way to show
        the difference between unrolling for loops and implicit differentiation.

        NOTE:
            The attributes of the SCF instance will not be modified
        '''
        if dm0 is None:
            try:
                dm0 = self.make_rdm1()
            except TypeError:
                pass

        def hf_energy(self, dm0=None):
            self.reset()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

        if mode.lower().startswith('rev'):
            jac = jax.grad(hf_energy)(self, dm0=dm0)
        else:
            if config.scf_implicit_diff:
                msg = """Forward mode differentiation is not available
                         when applying the implicit function differentiation."""
                raise KeyError(msg)
            jac = jax.jacfwd(hf_energy)(self, dm0=dm0)
        if hasattr(jac, 'cell'):
            return jac.cell
        else:
            return jac.mol

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        return df.density_fit(self, auxbasis, with_df, only_dfj)

    @wraps(pyscf_hf.SCF.get_veff)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.direct_scf:
            ddm = np.asarray(dm) - dm_last
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return vhf_last + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm =self.make_rdm1()
        return dip_moment(mol, dm, unit, verbose=verbose, **kwargs)

    make_rdm1 = module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])
    energy_elec = energy_elec


@util.pytree_node(Traced_Attributes, num_args=1)
class RHF(SCF, pyscf_hf.RHF):
    @wraps(pyscf_hf.RHF.get_veff)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * .5
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * .5
            vhf += np.asarray(vhf_last)
        return vhf
