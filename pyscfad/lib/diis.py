from pyscf import __config__
from pyscf.lib import diis as pyscf_diis
from pyscfad import lib

@lib.dataclass
class DIIS(pyscf_diis.DIIS):
    def __init__(self, dev=None, filename=None,
                 incore=getattr(__config__, 'lib_diis_DIIS_incore', False)):
        pyscf_diis.DIIS.__init__(self, dev=dev, filename=filename, incore=incore)
