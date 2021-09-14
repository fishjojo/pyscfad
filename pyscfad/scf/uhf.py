from   functools import partial
import tempfile
from typing import Optional, Any

import jax

from pyscf        import __config__
from pyscf.lib    import param
from pyscf.scf    import uhf, diis

from pyscfad     import lib, gto
from pyscfad.lib import numpy as jnp
from pyscfad.lib import stop_grad

from . import hf

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