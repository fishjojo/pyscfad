from pyscf.tdscf import rhf as pyscf_tdrhf
from pyscfad import util

Traced_Attributes = ['_scf', 'mol']

# pylint: disable=abstract-method
@util.pytree_node(Traced_Attributes, num_args=1)
class TDMixin(pyscf_tdrhf.TDMixin):
    def __init__(self, mf, **kwargs):
        pyscf_tdrhf.TDMixin.__init__(self, mf)
        self.__dict__.update(kwargs)

@util.pytree_node(Traced_Attributes, num_args=1)
class TDA(TDMixin, pyscf_tdrhf.TDA):
    pass

CIS = TDA
