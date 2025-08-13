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

from functools import partial, reduce
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import scipy

@partial(ops.jit, static_argnums=1)
def lowdin(s, thresh=1e-15):
    e, v = scipy.linalg.eigh(s)
    e_sqrt = np.where(e>thresh, np.sqrt(e), np.inf)
    return np.dot(v/e_sqrt[None,:], v.conj().T)

def vec_lowdin(c, s=1):
    return np.dot(c, lowdin(reduce(np.dot, (c.conj().T, s, c))))

