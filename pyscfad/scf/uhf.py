from typing import Optional
import jax
from pyscf.scf   import uhf
from pyscfad     import lib
from pyscfad.scf import hf

@lib.dataclass
class UHF(hf.SCF, uhf.UHF):
    _nelec: Optional[int] = None 
    init_guess_breaksym: Optional[bool] = None

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            return self.mol.nelec
    
    @nelec.setter
    def nelec(self, x):
        self._nelec = x
