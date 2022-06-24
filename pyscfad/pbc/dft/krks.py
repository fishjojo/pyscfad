from typing import Optional
import numpy
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc.dft import krks as pyscf_krks
from pyscf.pbc.dft import gen_grid, multigrid
from pyscfad import lib
from pyscfad.lib import numpy as np
from pyscfad.lib import stop_grad
from pyscfad.dft.rks import VXC
from pyscfad.pbc.scf import khf
from pyscfad.pbc.dft import rks

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (getattr(dm, 'ndim', None) == 3 and 
                    kpts_band is None)

# For UniformGrids, grids.coords does not indicate whehter grids are initialized
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, stop_grad(cell), stop_grad(dm), ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 0,
                                        kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    weight = 1./len(kpts)
    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= np.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = VXC(vxc=vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

@lib.dataclass
class KRKS(rks.KohnShamDFT, khf.KRHF):
    def __init__(self, cell, kpts=numpy.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), **kwargs):
        self.cell = cell
        self.xc = xc
        self.kpts = kpts
        self.exxdiv = exxdiv
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if not self._built:
            khf.KRHF.__init__(self, cell, kpts=kpts, exxdiv=exxdiv)
            rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        khf.KRHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        weight = 1./len(h1e_kpts)
        e1 = weight * np.einsum('kij,kji', h1e_kpts, dm_kpts)
        tot_e = e1 + vhf.ecoul + vhf.exc
        self.scf_summary['e1'] = e1.real
        self.scf_summary['coul'] = vhf.ecoul.real
        self.scf_summary['exc'] = vhf.exc.real
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
        return tot_e.real, vhf.ecoul + vhf.exc

    get_rho = pyscf_krks.get_rho
