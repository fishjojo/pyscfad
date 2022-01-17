import jax
import numpy

from pyscf.dft import uks
from pyscf     import __config__
from pyscf.lib import current_memory
from pyscf.lib import logger

from pyscfad import lib

from pyscfad.lib import numpy as jnp
from pyscfad.lib import stop_grad

from pyscfad.scf import uhf
from .           import rks

@lib.dataclass
class VXC():
    vxc:   jnp.ndarray = None
    ecoul: float       = None
    exc:   float       = None
    vj:    jnp.array   = None
    vk:    jnp.ndarray = None

    def reset(self):
        self.vxc   = None
        self.ecoul = None
        self.exc   = None
        self.vj    = None
        self.vk    = None

    def __repr__(self):
        return self.vxc.__repr__()

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    
    if not isinstance(dm, jnp.ndarray):
        dm = jnp.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = jnp.asarray((dm*.5, dm*.5))
    
    ground_state = (getattr(dm, "ndim", None) == 3) and (dm.shape[0] == 2)

    t0 = (logger.process_clock(), logger.perf_counter())

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.nlcgrids = rks.prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.nlcgrids)
            t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    ni  = ks._numint
    vxc = VXC()
    if hermi == 2:  # because rho = 0
        n, vxc.exc, vxc.vxc = (0,0), 0, 0
    else:
        max_memory          = ks.max_memory - current_memory()[0]
        n, vxc.exc, vxc.vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)

        if ks.nlc != '':
            assert('VV10' in ks.nlc.upper())
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm[0]+dm[1], max_memory=max_memory)
            vxc.exc += enlc
            vxc.vxc += vnlc

        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj  = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj  = ks.get_j(mol, dm[0]+dm[1], hermi)
        vxc.vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr  = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk   += vklr
            vj  = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj     = vj[0] + vj[1]
            vk    *= hyb
            if abs(omega) > 1e-10:
                vklr  = ks.get_k(mol, dm, hermi, omega)
                vklr *= (alpha - hyb)
                vk   += vklr
        vxc.vxc += vj - vk

        if ground_state:
            vxc.exc -=(jnp.einsum('ij,ji', dm[0], vk[0]).real + jnp.einsum('ij,ji', dm[1], vk[1]).real) * .5

    if ground_state:
        vxc.ecoul = jnp.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        vxc.ecoul = None

    return vxc

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None:  dm  = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()

    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    
    return rks.energy_elec(ks, dm[0] + dm[1], h1e, vhf)

@lib.dataclass
class UKS(rks.KohnShamDFT, uhf.UHF):
    def __post_init__(self):
        uhf.UHF.__post_init__(self)
        rks.KohnShamDFT.__post_init__(self)

    get_veff        = get_veff
    energy_elec     = energy_elec
    nuc_grad_method = uks.UKS.nuc_grad_method #analytic nuclear gradient method

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = UKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753
