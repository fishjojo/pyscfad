import numpy
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc.dft import rks as pyscf_rks
from pyscf.pbc.dft import gen_grid, multigrid
from pyscf.pbc.dft.rks import prune_small_rho_grids_
from pyscf.pbc.lib.kpts import KPoints

from pyscfad import numpy as np
from pyscfad.ops import stop_grad
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

    log = logger.new_logger(ks)

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
                                       kpt.reshape(1,3), kpts_band,
                                       with_j=True, return_j=False)
        log.debug('nelec by numeric integration = %s', n)
        log.timer('vxc')
        return vxc

    ground_state = (getattr(dm, 'ndim', 0) == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = prune_small_rho_grids_(ks, stop_grad(cell), stop_grad(dm), ks.grids, kpt)
        log.timer('setting up grids')

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 0,
                                        kpt, kpts_band)
        log.debug('nelec by numeric integration = %s', n)
        log.timer('vxc')

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
    del log
    return vxc

def _dft_common_init_(mf, xc='LDA,VWN', **kwargs):
    from pyscfad.pbc.scf import khf
    mf.xc = xc
    mf.grids = None
    mf.nlc = ''
    mf.nlcgrids = None
    mf.small_rho_cutoff = getattr(
        __config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)
    if isinstance(mf, khf.KSCF):
        if isinstance(mf.kpts, KPoints):
            mf._numint = numint.KNumInt(mf.kpts.kpts)
        else:
            mf._numint = numint.KNumInt(mf.kpts)
    else:
        mf._numint = numint.NumInt()

def _dft_common_post_init_(mf):
    from pyscf.pbc.gto import Cell
    if mf.grids is None:
        mf.grids = gen_grid.UniformGrids(mf.cell.view(Cell))
    if mf.nlcgrids is None:
        mf.nlcgrids = gen_grid.UniformGrids(mf.cell.view(Cell))

class KohnShamDFT(mol_ks.KohnShamDFT, pyscf_rks.KohnShamDFT):
    __init__ = _dft_common_init_
    __post_init__ = _dft_common_post_init_

    dump_flags = pyscf_rks.KohnShamDFT.dump_flags

class RKS(KohnShamDFT, pbchf.RHF):
    """Subclass of :class:`pyscf.pbc.dft.rks.RKS` with traceable attributes.

    Attributes
    ----------
    cell : :class:`pyscfad.pbc.gto.Cell`
        :class:`pyscfad.pbc.gto.Cell` instance.
    mo_coeff : array
        MO coefficients.
    mo_energy : array
        MO energies.

    Notes
    -----
    Grid response is not considered with AD.
    """
    def __init__(self, cell, xc='LDA,VWN', kpt=numpy.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.RHF.__init__(self, cell, kpt, exxdiv)
        KohnShamDFT.__init__(self, xc)
        # NOTE this has to be after __dict__ update,
        # otherwise stop_grad(mol) won't work.
        # Currently, no grid response is considered.
        KohnShamDFT.__post_init__(self)

    def dump_flags(self, verbose=None):
        pbchf.RHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = mol_ks.energy_elec
    get_rho = pyscf_rks.get_rho
