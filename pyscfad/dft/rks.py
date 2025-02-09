from pyscf import __config__
from pyscf.lib import current_memory
from pyscf.lib import logger
from pyscf.dft import rks as pyscf_rks
from pyscf.dft import gen_grid
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.ops import stop_grad
from pyscfad.scf import hf
from pyscfad.dft import numint

class VXC(pytree.PytreeNode):
    _dynamic_attr = {'vxc', 'ecoul', 'exc', 'vj', 'vk'}

    def __init__(self, vxc=None,
                 ecoul=None, exc=None,
                 vj=None, vk=None):
        self.vxc = vxc
        self.ecoul = ecoul
        self.exc = exc
        self.vj = vj
        self.vk = vk

    def __repr__(self):
        return self.vxc.__repr__()

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()

    ks.initialize_grids(mol, dm)

    log = logger.new_logger(ks)
    ground_state = getattr(dm, 'ndim', None) == 2

    ni = ks._numint
    vxc = VXC()
    if hermi == 2:  # because rho = 0
        n, vxc.exc, vxc.vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - current_memory()[0]
        n, vxc.exc, vxc.vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        log.debug('nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm,
                                          max_memory=max_memory)

            vxc.exc += enlc
            vxc.vxc += vnlc
            log.debug('nelec with nlc grids = %s', n)
        log.timer('vxc')

    incremental_jk = (ks._eri is None and ks.direct_scf and
                      getattr(vhf_last, 'vj', None) is not None)

    if incremental_jk:
        _dm = np.asarray(dm) - np.asarray(dm_last)
    else:
        _dm = dm

    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        vj = ks.get_j(mol, _dm, hermi)
        if incremental_jk:
            vj += vhf_last.vj
        vxc.vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if omega == 0:
            vj, vk = ks.get_jk(mol, _dm, hermi)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(mol, _dm, hermi)
            vk = ks.get_k(mol, _dm, hermi, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(mol, _dm, hermi)
            vk = ks.get_k(mol, _dm, hermi, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(mol, _dm, hermi)
            vk *= hyb
            vklr = ks.get_k(mol, _dm, hermi, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        if incremental_jk:
            vj += vhf_last.vj
            vk += vhf_last.vk
        vxc.vxc += vj - vk * .5

        if ground_state:
            vxc.exc -= np.einsum('ij,ji', dm, vk).real * .5 * .5

    if ground_state:
        vxc.ecoul = np.einsum('ij,ji', dm, vj).real * .5
    else:
        vxc.ecoul = None

    del log
    return vxc

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = ks.make_rdm1()
    if h1e is None:
        h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm)
    e2 = vhf.ecoul + vhf.exc
    ks.scf_summary['e1'] = e1.real
    ks.scf_summary['coul'] = vhf.ecoul.real
    ks.scf_summary['exc'] = vhf.exc.real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
    return (e1+e2).real, e2

def prune_small_rho_grids_(ks, mol, dm, grids):
    rho = ks._numint.get_rho(stop_grad(mol),
                             stop_grad(dm),
                             grids,
                             ks.max_memory)
    return grids.prune_by_density_(rho, ks.small_rho_cutoff)

def _dft_common_init_(mf, xc='LDA,VWN'):
    mf.xc = xc
    mf.nlc = ''
    mf.disp = None
    mf.grids = None
    mf.nlcgrids = None
    mf._numint = numint.NumInt()

def _dft_common_post_init_(mf):
    if mf.grids is None:
        mf.grids = gen_grid.Grids(stop_grad(mf.mol))
        mf.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', mf.grids.level)
    if mf.nlcgrids is None:
        mf.nlcgrids = gen_grid.Grids(stop_grad(mf.mol))
        mf.nlcgrids.level = getattr(
            __config__, 'dft_rks_RKS_nlcgrids_level', mf.nlcgrids.level)

class KohnShamDFT(pyscf_rks.KohnShamDFT):
    small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

    # NOTE __init__ is divided into two functions to not trace grid build
    # TODO consider grid response
    __init__ = _dft_common_init_
    __post_init__ = _dft_common_post_init_

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        if getattr(self, 'grids', None) is not None:
            self.grids.reset(mol)
        if getattr(self, 'nlcgrids', None) is not None:
            self.nlcgrids.reset(mol)
        return self

    def initialize_grids(self, mol=None, dm=None):
        if mol is None:
            mol = self.mol

        log = logger.new_logger(self)
        ground_state = getattr(dm, 'ndim', None) == 2
        if self.grids.coords is None:
            self.grids.build(with_non0tab=True)
            if self.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                self.grids = prune_small_rho_grids_(self, mol, dm,
                                                    self.grids)
            t0 = log.timer('setting up grids')
        is_nlc = self.do_nlc()
        if is_nlc and self.nlcgrids.coords is None:
            self.nlcgrids.build(with_non0tab=True)
            if self.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                self.nlcgrids = prune_small_rho_grids_(self, mol, dm,
                                                       self.nlcgrids)
            t0 = log.timer('setting up nlc grids')
        del log
        return self

class RKS(KohnShamDFT, hf.RHF):
    """Subclass of :class:`pyscf.dft.rks.RKS` with traceable attributes.

    Attributes
    ----------
    mol : :class:`pyscfad.gto.Mole`
        :class:`pyscfad.gto.Mole` instance.
    mo_coeff : array
        MO coefficients.
    mo_energy : array
        MO energies.
    _eri : array
        Two-electron repulsion integrals.

    Notes
    -----
    Grid response is not considered with AD.
    """
    def __init__(self, mol, xc='LDA,VWN', **kwargs):
        hf.RHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)
        self.__dict__.update(kwargs)
        # NOTE this has to be after __dict__ update,
        # otherwise stop_grad(mol) won't work.
        # Currently, no grid response is considered.
        KohnShamDFT.__post_init__(self)

    get_veff = get_veff
    energy_elec = energy_elec
    nuc_grad_method = pyscf_rks.RKS.nuc_grad_method #analytic nuclear gradient method
