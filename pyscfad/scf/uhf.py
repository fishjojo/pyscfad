from pyscf.scf import uhf as pyscf_uhf
from pyscfad import util
from pyscfad.scf import hf

@util.pytree_node(hf.Traced_Attributes, num_args=1)
class UHF(hf.SCF, pyscf_uhf.UHF):
    def __init__(self, mol, **kwargs):
        pyscf_uhf.UHF.__init__(self, mol)
        self.__dict__.update(kwargs)
