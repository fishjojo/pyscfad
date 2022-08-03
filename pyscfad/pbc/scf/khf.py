import sys
import numpy
from pyscf import __config__
from pyscf import numpy as np
from pyscf.pbc.scf import khf as pyscf_khf
from pyscfad import lib
from pyscfad import util
from pyscfad.lib import stop_grad
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df
from pyscfad.pbc.scf import hf as pbchf

# TODO add mo_coeff, which requires AD wrt complex numbers
Traced_Attributes = ['cell', 'mo_energy', 'with_df']

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
    t = np.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    return nuc + t


@util.pytree_node(Traced_Attributes, num_args=1)
class KSCF(pbchf.SCF, pyscf_khf.KSCF):
    def __init__(self, cell, kpts=numpy.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell, kpts=kpts)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        #self.kpts = kpts
        self.conv_tol = cell.precision * 10

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk'])
        self.__dict__.update(kwargs)

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

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        cell = stop_grad(cell)
        return pyscf_khf.KSCF.get_init_guess(self, cell, key)

    def eig(self, h_kpts, s_kpts, x0=None):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []
        if x0 is None:
            x0 = [None] * nkpts

        for k in range(nkpts):
            e, c = self._eigh(h_kpts[k], s_kpts[k], x0[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return eig_kpts, mo_coeff_kpts


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
    #eig = pyscf_khf.KSCF.eig

KRHF = KSCF
