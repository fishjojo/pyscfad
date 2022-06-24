from typing import Optional
import numpy
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc.dft import rks as pyscf_rks
from pyscf.pbc.dft import gen_grid, multigrid
from pyscf.pbc.dft.rks import prune_small_rho_grids_
from pyscfad import lib
from pyscfad.lib import numpy as np
from pyscfad.lib import stop_grad
from pyscfad.dft import rks as mol_ks
from pyscfad.dft.rks import VXC
from pyscfad.pbc.scf import hf as pbchf
from pyscfad.pbc.dft import numint

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    if cell is None:
        cell = ks.cell
    if dm is None:
        dm = ks.make_rdm1()
    if kpt is None:
        kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
                                       kpt.reshape(1,3), kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    ground_state = (getattr(dm, 'ndim', 0) == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = prune_small_rho_grids_(ks, stop_grad(cell), stop_grad(dm), ks.grids, kpt)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 0,
                                        kpt, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= np.einsum('ij,ji', dm, vk).real * .5 * .5

    if ground_state:
        ecoul = np.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = VXC(vxc=vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc


def _dft_common_init_(mf, xc='LDA,VWN', **kwargs):
    from pyscfad.pbc.scf import khf
    mf.xc = xc
    mf.grids = gen_grid.UniformGrids(mf.cell)
    if isinstance(mf, khf.KSCF):
        mf._numint = numint.KNumInt(mf.kpts)
    else:
        mf._numint = numint.NumInt()
    mf._keys = mf._keys.union(['xc', 'grids', 'small_rho_cutoff'])

@lib.dataclass
class KohnShamDFT(mol_ks.KohnShamDFT):
    __init__ = _dft_common_init_

@lib.dataclass
class RKS(KohnShamDFT, pbchf.RHF):
    kpt: numpy.ndarray = numpy.zeros(3)
    exxdiv: Optional[str] = getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')

    def __init__(self, cell, xc='LDA,VWN', **kwargs):
        self.cell = cell
        self.xc = xc
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if not self._built:
            pbchf.RHF.__init__(self, cell)
            KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbchf.RHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = mol_ks.energy_elec
    get_rho = pyscf_rks.get_rho
