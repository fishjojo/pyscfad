import numpy
from pyscf import __config__
from pyscf.lib import current_memory
from pyscf.lib import logger
from pyscf.dft import rks as pyscf_rks
from pyscf.dft import gen_grid
from pyscfad import lib
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad.lib import stop_grad
from pyscfad.scf import hf
from pyscfad.dft import numint

@util.pytree_node([])
class VXC():
    def __init__(self, **kwargs):
        self.vxc = None
        self.ecoul = None
        self.exc = None
        self.vj = None
        self.vk = None
        self.__dict__.update(kwargs)

    def __repr__(self):
        return self.vxc.__repr__()

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    log = logger.new_logger(ks)

    ground_state = getattr(dm, "ndim", None) == 2

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
            ks.grids = prune_small_rho_grids_(ks, mol, dm, ks.grids)
        log.timer('setting up grids')
    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                ks.nlcgrids = prune_small_rho_grids_(ks, mol, dm, ks.nlcgrids)
            log.timer('setting up nlc grids')

    ni = ks._numint
    vxc = VXC()
    if hermi == 2:  # because rho = 0
        n, vxc.exc, vxc.vxc = 0, 0., 0.
    else:
        max_memory = ks.max_memory - current_memory()[0]
        n, vxc.exc, vxc.vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        if ks.nlc != '':
            assert 'VV10' in ks.nlc.upper()
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm,
                                      max_memory=max_memory)
            vxc.exc += enlc
            vxc.vxc += vnlc
        log.debug('nelec by numeric integration = %s', n)
        log.timer('vxc')

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc.vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
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
    ks.scf_summary['e1'] = stop_grad(e1).real
    ks.scf_summary['coul'] = stop_grad(vhf.ecoul).real
    ks.scf_summary['exc'] = stop_grad(vhf.exc).real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
    return (e1+e2).real, e2

NELEC_ERROR_TOL = getattr(__config__, 'dft_rks_prune_error_tol', 0.02)
def prune_small_rho_grids_(ks, mol, dm, grids):
    mol = stop_grad(mol)
    dm = stop_grad(dm)
    rho = ks._numint.get_rho(mol, dm, grids, ks.max_memory)
    n = numpy.dot(rho, grids.weights)
    if abs(n-mol.nelectron) < NELEC_ERROR_TOL*n:
        rho *= grids.weights
        idx = abs(rho) > ks.small_rho_cutoff / grids.weights.size
        logger.debug(ks, 'Drop grids %d',
                     grids.weights.size - numpy.count_nonzero(idx))
        grids.coords  = numpy.asarray(grids.coords [idx], order='C')
        grids.weights = numpy.asarray(grids.weights[idx], order='C')
        grids.non0tab = grids.make_mask(mol, grids.coords)
    return grids

def _dft_common_init_(mf, xc='LDA,VWN'):
    mf.xc = xc
    mf.nlc = ''
    mf.grids = None
    mf.nlcgrids = None
    # Use rho to filter grids
    mf.small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)
##################################################
# don't modify the following attributes, they are not input options
    mf._numint = numint.NumInt()
    mf._keys = mf._keys.union(['xc', 'nlc', 'omega', 'grids', 'nlcgrids',
                               'small_rho_cutoff'])

def _dft_common_post_init_(mf):
    if mf.grids is None:
        mf.grids = gen_grid.Grids(stop_grad(mf.mol))
        mf.grids.level = getattr(__config__, 'dft_rks_RKS_grids_level',
                                 mf.grids.level)
    if mf.nlcgrids is None:
        mf.nlcgrids = gen_grid.Grids(stop_grad(mf.mol))
        mf.nlcgrids.level = getattr(__config__, 'dft_rks_RKS_nlcgrids_level',
                                    mf.nlcgrids.level)


class KohnShamDFT(pyscf_rks.KohnShamDFT):
    __init__ = _dft_common_init_
    __post_init__ = _dft_common_post_init_

@util.pytree_node(hf.Traced_Attributes, num_args=1)
class RKS(KohnShamDFT, hf.RHF):
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
