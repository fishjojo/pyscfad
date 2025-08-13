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

from jax.scipy.sparse.linalg import gmres
from pyscfad.lib import logger

def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN):
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose)
    else:
        raise NotImplementedError

def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN):
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = e_a[:,None] - e_i[None,:]

    def vind_vo(mo1):
        v  = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v += e_ai * mo1.reshape(h1.shape)
        return -v.ravel()

    mo1 = gmres(vind_vo, h1.ravel(), tol=tol, maxiter=max_cycle)[0]
    return mo1.reshape(h1.shape), None
