import sys
import h5py
import numpy
from jax import numpy as np
from pyscf import __config__
from pyscf.pbc.scf import hf as pyscf_pbc_hf
from pyscf.pbc.scf.hf import _format_jks
#from pyscfad import util
from pyscfad.lib import stop_grad, logger
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df

#Traced_Attributes = ['cell', 'mo_coeff', 'mo_energy', 'with_df']

def get_ovlp(cell, kpt=np.zeros(3)):
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt)
    return s

#@util.pytree_node(Traced_Attributes, num_args=1)
class SCF(mol_hf.SCF, pyscf_pbc_hf.SCF):
    def __init__(self, cell, kpt=numpy.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):#, **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        self.rsjk = None

        self.exxdiv = exxdiv
        self.kpt = kpt
        self.conv_tol = max(cell.precision * 10, 1e-8)

        self._keys = self._keys.union(['cell', 'exxdiv', 'with_df', 'rsjk'])
        #self.__dict__.update(kwargs)

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm)
        return dm

    def get_hcore(self, cell=None, kpt=None):
        if cell is None:
            cell = self.cell
        if kpt is None:
            kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt)
        else:
            raise NotImplementedError
        if len(cell._ecpbas) > 0:
            raise NotImplementedError
        h1 = cell.pbc_intor('int1e_kin', comp=1, hermi=1, kpts=kpt)
        return nuc + h1

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        #if cell is None:
        #    cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt

        cpu0 = (logger.process_clock(), logger.perf_counter())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if self.rsjk:
            raise NotImplementedError
        else:
            vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                         with_j, with_k, omega, exxdiv=self.exxdiv)

        if with_j:
            vj = _format_jks(vj, dm, kpts_band)
        if with_k:
            vk = _format_jks(vk, dm, kpts_band)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None:
            cell = self.cell
        if kpt is None:
            kpt = self.kpt
        return get_ovlp(cell, kpt)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with h5py.File(self.chkfile, 'a') as fh5:
                fh5['scf/kpt'] = stop_grad(self.kpt)
        return self

    get_veff = pyscf_pbc_hf.SCF.get_veff
    energy_nuc = pyscf_pbc_hf.SCF.energy_nuc
    energy_grad = NotImplemented

RHF = SCF

def normalize_dm_(mf, dm):
    cell = mf.cell
    s = stop_grad(mf.get_ovlp(cell))
    if getattr(dm, 'ndim', 0) == 2:
        ne = numpy.einsum('ij,ji->', stop_grad(dm), s).real
    else:
        ne = numpy.einsum('xij,ji->', stop_grad(dm), s).real
    if abs(ne - cell.nelectron).sum() > 1e-7:
        logger.debug(mf, 'Big error detected in the electron number '
                     'of initial guess density matrix (Ne/cell = %g)!\n'
                     '  This can cause huge error in Fock matrix and '
                     'lead to instability in SCF for low-dimensional '
                     'systems.\n  DM is normalized wrt the number '
                     'of electrons %s', ne, cell.nelectron)
        dm *= cell.nelectron / ne
    return dm
