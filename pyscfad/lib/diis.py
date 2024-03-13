import numpy
import scipy.linalg
from pyscf.lib import prange
from pyscf.lib import diis as pyscf_diis
from pyscf.lib.diis import INCORE_SIZE, BLOCK_SIZE
from pyscfad import numpy as np
from pyscfad.ops import stop_grad
from pyscfad.lib import logger

# pylint: disable=consider-using-f-string
class DIIS(pyscf_diis.DIIS):
    def push_vec(self, x):
        x = x.ravel()

        if len(self._bookkeep) >= self.space:
            self._bookkeep = self._bookkeep[1-self.space:]

        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            key = 'x%d' % (self._head)
            self._store(key, x)
            self._head += 1

        elif self._xprev is None:
            self._xprev = x
            self._store('xprev', x)
            if 'xprev' not in self._buffer:  # not incore
                self._xprev = self._diisfile['xprev']
        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d'%self._head
            xkey = 'x%d'%self._head
            self._store(xkey, x)
            if x.size < INCORE_SIZE or self.incore:
                # no need to trace error vector
                x = numpy.asarray(stop_grad(x))
                xprev = numpy.asarray(stop_grad(self._xprev))
                self._store(ekey, x - xprev)
            else:  # not call _store to reduce memory footprint
                if ekey not in self._diisfile:
                    self._diisfile.create_dataset(ekey, (x.size,), x.dtype)
                edat = self._diisfile[ekey]
                for p0, p1 in prange(0, x.size, BLOCK_SIZE):
                    edat[p0:p1] = x[p0:p1] - self._xprev[p0:p1]
                self._diisfile.flush()
            self._head += 1

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
