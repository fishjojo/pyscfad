from pyscfad import util
from pyscfad.cc import dfccsd

@util.pytree_node(dfccsd.CC_Tracers, num_args=1)
class RDCSD(dfccsd.RCCSD):
    @property
    def dcsd(self):
        return True
