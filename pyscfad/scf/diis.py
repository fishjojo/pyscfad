from pyscf.scf import diis as pyscf_cdiis
from pyscfad import lib
from pyscfad.lib import diis

@lib.dataclass
class CDIIS(diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename)

SCFDIIS = SCF_DIIS = DIIS = CDIIS
