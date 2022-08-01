from pyscf import numpy as np
from pyscf.scf import uhf as pyscf_uhf
from pyscfad import util
from pyscfad.scf import hf

@util.pytree_node(hf.Traced_Attributes, num_args=1)
class UHF(hf.SCF, pyscf_uhf.UHF):
    def __init__(self, mol, **kwargs):
        pyscf_uhf.UHF.__init__(self, mol)
        self.__dict__.update(kwargs)

    def eig(self, h, s, x0=None):
        if x0 is not None:
            e_a, c_a = self._eigh(h[0], s, x0[0])
            e_b, c_b = self._eigh(h[1], s, x0[1])
        else:
            e_a, c_a = self._eigh(h[0], s)
            e_b, c_b = self._eigh(h[1], s)
        return np.array((e_a,e_b)), np.array((c_a,c_b))
