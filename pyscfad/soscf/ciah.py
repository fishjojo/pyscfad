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
# TODO add other backend for expm
from jax.scipy.linalg import expm
from pyscfad import numpy as np
from pyscfad.ops import jit

def pack_uniq_var(mat):
    nmo = mat.shape[-1]
    idx = np.tril_indices(nmo, -1)
    return mat[idx]

@jit
def unpack_uniq_var(v):
    nmo = int(numpy.sqrt(v.size*2)) + 1
    idx = np.tril_indices(nmo, -1)
    mat = np.zeros((nmo,nmo))
    mat = mat.at[idx].set(v)
    return mat - mat.conj().T

def extract_rotation(dr, u0=1):
    dr = unpack_uniq_var(dr)
    return np.dot(u0, expm(dr))

