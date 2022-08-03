import sys
import numpy
from pyscf import __config__
from pyscf import numpy as np
from pyscf.lib import logger
from pyscf.pbc.scf import hf as pyscf_pbc_hf
from pyscf.pbc.scf.hf import _format_jks
from pyscfad import lib
from pyscfad import util
from pyscfad.lib import stop_grad
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df

Traced_Attributes = ['cell', 'mo_coeff', 'mo_energy', 'with_df']

@util.pytree_node(Traced_Attributes, num_args=1)
class SCF(mol_hf.SCF, pyscf_pbc_hf.SCF):
    def __init__(self, cell, kpt=numpy.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        self.kpt = kpt
        self.conv_tol = cell.precision * 10
        self.no_incore = True

        self._keys = self._keys.union(['cell', 'exxdiv', 'with_df'])
        self.__dict__.update(kwargs)

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm)
        return dm

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt, cell=cell)
        else:
            raise NotImplementedError
            #nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError
            #nuc += ecp.ecp_int(cell, kpt)
        return nuc + cell.pbc_intor('int1e_kin', 1, 1, kpt)

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        #cpu0 = (logger.process_clock(), logger.perf_counter())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (not omega and kpts_band is None and
            # TODO: generate AO integrals with rsjk algorithm
            not self.rsjk and
            (self.exxdiv == 'ewald' or not self.exxdiv) and
            (self._eri is not None or cell.incore_anyway or
             (not self.direct_scf and self._is_mem_enough())) and not self.no_incore):
            raise NotImplementedError
            #if self._eri is None:
            #    logger.debug(self, 'Building PBC AO integrals incore')
            #    self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            #vj, vk = mol_hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)

            #if with_k and self.exxdiv == 'ewald':
            #    from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
            #    # G=0 is not inculded in the ._eri integrals
            #    _ewald_exxdiv_for_G0(self.cell, kpt, dm.reshape(-1,nao,nao),
            #                         vk.reshape(-1,nao,nao))
        elif self.rsjk:
            raise NotImplementedError
            #vj, vk = self.rsjk.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
            #                          with_j, with_k, omega, exxdiv=self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                         with_j, with_k, omega, exxdiv=self.exxdiv, cell=cell)

        if with_j:
            vj = _format_jks(vj, dm, kpts_band)
        if with_k:
            vk = _format_jks(vk, dm, kpts_band)
        #logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    get_veff = pyscf_pbc_hf.SCF.get_veff
    energy_nuc = pyscf_pbc_hf.SCF.energy_nuc
    energy_grad = mol_hf.SCF.energy_grad

RHF = SCF

def normalize_dm_(mf, dm):
    cell = mf.cell
    s = stop_grad(mf.get_ovlp(cell))
    if getattr(dm, "ndim", 0) == 2:
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
