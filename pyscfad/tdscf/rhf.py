from typing import Optional, Any
from pyscf.tdscf import rhf as pyscf_tdrhf
from pyscfad import lib
from pyscfad import gto, scf

@lib.dataclass
class TDMixin(pyscf_tdrhf.TDMixin):
    _scf : scf.hf.SCF = lib.field(pytree_node=True)
    mol : Optional[gto.Mole] = lib.field(pytree_node=True, default=None)
    verbose : Optional[int] = None
    stdout : Optional[Any] = None
    max_memory : Optional[int] = None
    chkfile : Optional[Any] = None
    wfnsym : Optional[Any] = None
    converged : Optional[Any] = None
    e : Optional[float] = None
    xy : Optional[Any] = None

    def __post_init__(self):
        if getattr(self, "mol", None) is None:
            self.mol = self._scf.mol
        if getattr(self, "verbose", None) is None:
            self.verbose = self._scf.verbose
        if getattr(self, "stdout", None) is None:
            self.stdout = self._scf.stdout
        if getattr(self, "max_memory", None) is None:
            self.max_memory = self._scf.max_memory
        if getattr(self, "chkfile", None) is None:
            self.chkfile = self._scf.chkfile

        keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift',
                    'max_space', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

@lib.dataclass
class TDA(TDMixin, pyscf_tdrhf.TDA):
    pass

CIS = TDA
