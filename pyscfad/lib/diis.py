# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import scipy
from pyscf.lib import diis as pyscf_diis
from pyscf.lib.diis import INCORE_SIZE
from pyscfad import numpy as np
from pyscfad import ops
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
                raise NotImplementedError('outcore diis not supported')
        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d'%self._head
            xkey = 'x%d'%self._head
            self._store(xkey, x)
            if x.size < INCORE_SIZE or self.incore:
                # no need to trace error vector
                self._store(ekey, ops.stop_grad(x - self._xprev))
            else:
                raise NotImplementedError('outcore diis not supported')
            self._head += 1

    def update(self, x, xerr=None):
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        dt = self.get_err_vec(self._head-1)
        if self._H is None:
            self._H = numpy.zeros((self.space+1,self.space+1), dt.dtype)
            self._H[0,1:] = self._H[1:,0] = 1
        for i in range(nd):
            tmp = 0
            dti = self.get_err_vec(i)
            tmp = ops.to_numpy(np.dot(dt.conj(), dti))
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conj()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = self.extrapolate(nd)

            self._store('xprev', xnew)
            if 'xprev' not in self._buffer:  # not incore
                raise NotImplementedError('outcore diis not supported')
        return xnew.reshape(x.shape)

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
                xnew = np.zeros(xi.size, dtype=c.dtype)
            xnew += xi * ci
        return xnew
