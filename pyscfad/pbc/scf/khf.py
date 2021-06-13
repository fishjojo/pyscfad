import sys
import numpy
from pyscf import __config__
from pyscf.pbc.scf import khf as pyscf_khf
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df
from pyscfad.pbc.scf import hf as pbchf

def get_hcore(mf, cell=None, kpts=None):
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    if cell.pseudo:
        nuc = mf.with_df.get_pp(kpts, cell=cell)
    else:
        raise NotImplementedError
        #nuc = lib.asarray(mf.with_df.get_nuc(kpts))
    if len(cell._ecpbas) > 0:
        raise NotImplementedError
        #nuc += lib.asarray(ecp.ecp_int(cell, kpts))
    t = jnp.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    return nuc + t


@lib.dataclass
class KSCF(pbchf.SCF, pyscf_khf.KSCF):
    kpts: numpy.ndarray = numpy.zeros((1,3))
    exx_built: bool = False

    def __init__(self, cell, **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if not self._built:
            mol_hf.SCF.__init__(self, cell)
            mol_hf.SCF.__post_init__(self)
            self.direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', True)
            self.conv_tol_grad = getattr(__config__, 'pbc_scf_KSCF_conv_tol_grad', None)
            self.conv_tol = self.cell.precision * 10
        if self.with_df is None:
            self.with_df = df.FFTDF(self.cell)

        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk'])

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        #cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            raise NotImplementedError
            #vj, vk = self.rsjk.get_jk(dm_kpts, hermi, kpts, kpts_band,
            #                          with_j, with_k, omega, self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                         with_j, with_k, omega, self.exxdiv, cell=cell)
        #logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    get_init_guess = pyscf_khf.KSCF.get_init_guess
    get_hcore = get_hcore
    get_ovlp = pyscf_khf.KSCF.get_ovlp
    get_fock = pyscf_khf.KSCF.get_fock
    get_occ = pyscf_khf.KSCF.get_occ
    energy_elec = pyscf_khf.KSCF.energy_elec
    get_fermi = pyscf_khf.KSCF.get_fermi

    get_veff = pyscf_khf.KSCF.get_veff
    get_j = pyscf_khf.KSCF.get_j
    get_k = pyscf_khf.KSCF.get_k
    get_grad = pyscf_khf.KSCF.get_grad
    make_rdm1 = pyscf_khf.KSCF.make_rdm1
    eig = pyscf_khf.KSCF.eig

KRHF = KSCF
