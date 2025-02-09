from pyscf.dft import uks as pyscf_uks
from pyscf.lib import current_memory
from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.scf import uhf
from pyscfad.dft import rks

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    log = logger.new_logger(ks)

    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()

    dm = np.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        logger.warn('Incompatible dm dimension. Treat dm as RHF density matrix.')
        dm = np.repeat(dm[None]*.5, 2, axis=0)

    ks.initialize_grids(mol, dm)

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    ni = ks._numint
    vxc = rks.VXC()
    if hermi == 2:  # because rho = 0
        n, vxc.exc, vxc.vxc = (0, 0), 0, 0
    else:
        max_memory = ks.max_memory - current_memory()[0]
        n, vxc.exc, vxc.vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        log.debug('nelec by numeric integration = %s', n)

        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm[0]+dm[1],
                                          max_memory=max_memory)
            vxc.exc += enlc
            vxc.vxc += vnlc
            log.debug('nelec with nlc grids = %s', n)

        log.timer('vxc')

    incremental_jk = (ks._eri is None and ks.direct_scf and
                      getattr(vhf_last, 'vj', None) is not None)
    if incremental_jk:
        dm_last = np.asarray(dm_last)
        dm = np.asarray(dm)
        assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
        _dm = dm - dm_last
    else:
        _dm = dm
    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        vj = ks.get_j(mol, _dm[0]+_dm[1], hermi)
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
        vj = vj[0] + vj[1]
        if incremental_jk:
            vj += vhf_last.vj
            vk += vhf_last.vk
        vxc.vxc += vj - vk

        if ground_state:
            vxc.exc -=(np.einsum('ij,ji', dm[0], vk[0]).real +
                       np.einsum('ij,ji', dm[1], vk[1]).real) * .5

    if ground_state:
        vxc.ecoul = np.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        vxc.ecoul = None
    return vxc

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm  = ks.make_rdm1()
    if h1e is None:
        h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if getattr(dm, 'ndim', None) != 2:
        dm = dm[0] + dm[1]
    return rks.energy_elec(ks, dm, h1e, vhf)

class UKS(rks.KohnShamDFT, uhf.UHF):
    def __init__(self, mol, xc='LDA,VWN', **kwargs):
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)
        self.__dict__.update(kwargs)
        rks.KohnShamDFT.__post_init__(self)

    get_veff        = get_veff
    energy_elec     = energy_elec
    nuc_grad_method = pyscf_uks.UKS.nuc_grad_method
