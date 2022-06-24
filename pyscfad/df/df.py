import tempfile
from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf.df import df as pyscf_df
from pyscfad import lib
from pyscfad import util
from pyscfad.df import addons, incore, df_jk

@util.pytree_node(['mol', 'auxmol', '_cderi'], num_args=1)
class DF(pyscf_df.DF):
    def __init__(self, mol, auxbasis=None, **kwargs):
        pyscf_df.DF.__init__(self, mol, auxbasis=auxbasis)
        self.__dict__.update(kwargs)

    def build(self):
        log = logger.new_logger(self)

        self.check_sanity()
        self.dump_flags()

        mol = self.mol
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        auxmol = self.auxmol
        nao = mol.nao
        naux = auxmol.nao
        nao_pair = nao*nao

        max_memory = self.max_memory - pyscf_lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if (nao_pair*naux*8/1e6 < .9*max_memory and
            not isinstance(self._cderi_to_save, str)):
            self._cderi = incore.cholesky_eri(mol, auxmol=auxmol,
                                              int3c=int3c, int2c=int2c,
                                              max_memory=max_memory, verbose=log)
        else:
            raise NotImplementedError("Outcore density fitting is not implemented.")
        del log
        return self

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        # resetting auxmol will lose its tracing
        #self.auxmol = None
        self._cderi = None
        if not isinstance(self._cderi_to_save, str):
            # pylint: disable=consider-using-with
            self._cderi_to_save = tempfile.NamedTemporaryFile(dir=pyscf_lib.param.TMPDIR)
        self._vjopt = None
        self._rsh_df = {}
        return self

    def get_naoaux(self):
        # determine naoaux with self._cderi, because DF object may be used as CD
        # object when self._cderi is provided.
        if self._cderi is None:
            self.build()
        return self._cderi.shape[0]

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if self._cderi is None:
            self.build()
        if omega is None:
            return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        else:
            raise NotImplementedError
