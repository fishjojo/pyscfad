import numpy
from pyscf.scf import diis as pyscf_cdiis
from pyscfad import config
from pyscfad.ops import stop_grad
from pyscfad.lib import (
    logger,
    diis,
)

class CDIIS(diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename, Corth=Corth)
        self.incore = True

    def update(self, s, d, f, *args, **kwargs):
        s = numpy.asarray(stop_grad(s))
        d = numpy.asarray(stop_grad(d))
        if config.moleintor_opt and config.scf_implicit_diff:
            # diis is not being traced
            f = numpy.asarray(f)
            return pyscf_cdiis.CDIIS.update(self, s, d, f, *args, **kwargs)
        else:
            # diis may be traced
            errvec = pyscf_cdiis.get_err_vec(
                        s, d, numpy.asarray(stop_grad(f)), self.Corth)
            logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
            xnew = diis.DIIS.update(self, f, xerr=errvec)
            if self.rollback > 0 and len(self._bookkeep) == self.space:
                self._bookkeep = self._bookkeep[-self.rollback:]
            return xnew


SCFDIIS = SCF_DIIS = DIIS = CDIIS
