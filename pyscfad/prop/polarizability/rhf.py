from typing import Optional, Any
from pyscf.prop.polarizability.rhf import Polarizability as pyscf_Polarizability
from pyscfad import lib
from pyscfad import gto
from pyscfad import scf

@lib.dataclass
class Polarizability(pyscf_Polarizability):
    _scf : scf.hf.SCF = lib.field(pytree_node=True)
    mol : Optional[gto.Mole] = lib.field(pytree_node=True, default=None)
    verbose : Optional[int] = None
    stdout : Optional[Any] = None
    cphf : Optional[bool] = True
    max_cycle_cphf : Optional[int] = 20
    conv_tol : Optional[float] = 1e-9

    def __post_init__(self):
        if self.mol is None:
            self.mol = self._scf.mol
        if self.verbose is None:
            self.verbose = self.mol.verbose
        if self.stdout is None:
            self.stdout = self.mol.stdout
