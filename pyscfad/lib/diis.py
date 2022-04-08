import numpy
import scipy.linalg
from pyscf import __config__
from pyscf.lib import logger
from pyscf.lib import diis as pyscf_diis
from pyscfad.lib import numpy as np

class DIIS(pyscf_diis.DIIS):
    def __init__(self, dev=None, filename=None,
                 incore=getattr(__config__, 'lib_diis_DIIS_incore', False)):
        pyscf_diis.DIIS.__init__(self, dev=dev, filename=filename, incore=incore)

    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise ValueError('No vector found in DIIS object.')

        h = self._H[:nd+1,:nd+1]
        g = numpy.zeros(nd+1, h.dtype)
        g[0] = 1

        w, v = scipy.linalg.eigh(h)
        if numpy.any(abs(w)<1e-14):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w)>1e-14
            c = numpy.dot(v[:,idx]*(1./w[idx]), numpy.dot(v[:,idx].T.conj(), g))
        else:
            try:
                c = numpy.linalg.solve(h, g)
            except numpy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = np.zeros(xi.size, c.dtype)
            xnew += xi * ci
        return xnew
