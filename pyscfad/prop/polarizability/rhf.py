from pyscf.prop.polarizability.rhf import Polarizability as pyscf_Polarizability
from pyscfad import util

Traced_Attributes = ['_scf', 'mol']

@util.pytree_node(Traced_Attributes, num_args=1)
class Polarizability(pyscf_Polarizability):
    def __init__(self, mf, **kwargs):
        pyscf_Polarizability.__init__(self, mf)
        self.__dict__.update(kwargs)
