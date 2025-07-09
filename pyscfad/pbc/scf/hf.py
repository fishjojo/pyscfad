import sys
import h5py
import numpy
from pyscf import __config__
from pyscf.lib import with_doc
from pyscf.pbc.scf import hf as pyscf_pbc_hf
from pyscf.pbc.scf.hf import _format_jks
from pyscfad import numpy as np
from pyscfad.ops import stop_grad, stop_trace
from pyscfad.lib import logger
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df

@with_doc(pyscf_pbc_hf.get_ovlp.__doc__)
def get_ovlp(cell, kpt=np.zeros(3)):
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt)
    return np.asarray(s)

class SCF(mol_hf.SCF, pyscf_pbc_hf.SCF):
    """Subclass of :class:`pyscf.pbc.scf.hf.SCF` with traceable attributes.

    Attributes
    ----------
    cell : :class:`pyscfad.pbc.gto.Cell`
        :class:`pyscfad.pbc.gto.Cell` instance.
    mo_coeff : array
        MO coefficients.
    mo_energy : array
        MO energies.
    """
    _dynamic_attr = ["cell",]
    def __init__(self, cell, kpt=numpy.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        self.rsjk = None

        self.exxdiv = exxdiv
        self.kpt = kpt
        self.conv_tol = max(cell.precision * 10, 1e-8)

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if cell is None:
            cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm, s1e)
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
        h1 = cell.pbc_intor('int1e_kin', hermi=1, kpts=kpt)
        return nuc + h1

    @with_doc(pyscf_pbc_hf.SCF.get_jk.__doc__)
    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt

        log = logger.new_logger(self)
        cpu0 = (log._t0, log._w0)

        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (not omega and kpts_band is None and
            not self.rsjk and
            (self.exxdiv == 'ewald' or not self.exxdiv) and
            (self._eri is not None or cell.incore_anyway or
             self._is_mem_enough())):
            log.warn('pbc.SCF will not construct incore 4-center ERIs.')
        if self.rsjk:
            raise NotImplementedError
        else:
            vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                         with_j, with_k, omega, exxdiv=self.exxdiv)

        if with_j:
            vj = _format_jks(vj, dm, kpts_band)
        if with_k:
            vk = _format_jks(vk, dm, kpts_band)
        log.timer('vj and vk', *cpu0)
        del log
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

    def energy_nuc(self):
        # NOTE always compute nuclear energy to trace it
        return self.cell.energy_nuc()

    def check_sanity(self):
        # TODO need better way to adapt pyscf's check_sanity
        return self

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        logger.info(self, '******** PBC SCF flags ********')
        if hasattr(self, 'kpts'):
            logger.info(self, 'kpts = %s', self.kpts)
        elif hasattr(self, 'kpt'):
            logger.info(self, 'kpt = %s', self.kpt)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        if getattr(self, 'smearing_method', None) is not None:
            logger.info(self, 'Smearing method = %s', self.smearing_method)
        logger.info(self, 'DF object = %s', self.with_df)
        return self

    get_veff = pyscf_pbc_hf.SCF.get_veff
    energy_grad = NotImplemented


class RHF(SCF, pyscf_pbc_hf.RHF):
    pass


@with_doc(pyscf_pbc_hf.normalize_dm_.__doc__)
def normalize_dm_(mf, dm, s1e=None):
    # NOTE not tracing this function as it is mainly used
    # to generate the initial density matrix
    return stop_trace(pyscf_pbc_hf.normalize_dm_)(mf, dm, s1e=s1e)

