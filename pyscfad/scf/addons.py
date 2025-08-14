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

from functools import wraps
from pyscf.scf import addons as pyscf_addons
from pyscfad import numpy as np
from pyscfad import scipy

@wraps(pyscf_addons.canonical_orth_)
def canonical_orth_(S, thr=1e-7):
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = scipy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / np.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = np.dot(np.diag(normlz), X)
    return X
