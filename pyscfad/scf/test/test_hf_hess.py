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
from jax import jacrev
from pyscfad import scf

def test_hessian(get_n2):
    mol = get_n2
    def ehf(mol):
        mf = scf.RHF(mol)
        e = mf.kernel()
        return e
    hess = jacrev(jacrev(ehf))(mol).coords.coords
    #analytic result
    hess0 = numpy.asarray(
                [[[[ 1.50164118e-03, 0., 0.],
                   [-1.50164118e-03, 0., 0.]],
                  [[0.,  1.50164118e-03, 0.],
                   [0., -1.50164118e-03, 0.]],
                  [[0., 0.,  1.86573234e+00],
                   [0., 0., -1.86573234e+00]]],
                 [[[-1.50164118e-03, 0., 0.],
                   [ 1.50164118e-03, 0., 0.]],
                  [[0., -1.50164118e-03, 0.],
                   [0.,  1.50164118e-03, 0.]],
                  [[0., 0., -1.86573234e+00],
                   [0., 0.,  1.86573234e+00]]]])
    assert(abs(hess - hess0).max()) < 1e-6
