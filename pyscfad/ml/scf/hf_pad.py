# Copyright 2025-2026 The PySCFAD Authors
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
from pyscfad import numpy as np
from pyscfad.scf import hf_lite as hf
from pyscfad.scipy.linalg import eigh

class SCF(hf.SCF):
    # NOTE too large shift will give large errors
    padding_level_shift = 1e6

    def _eigh(self, h, s):
        ao_mask = self.mol.ao_mask
        mask = np.asarray(1 - ao_mask, dtype=np.int32)
        s = s + np.diag(mask)
        h = np.where(np.outer(ao_mask, ao_mask), h, 0.)
        h = h + np.diag(mask) * self.padding_level_shift
        return eigh(h, s)

    def mo_mask(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy

        thr = .99 * self.padding_level_shift
        return np.where(np.greater(mo_energy, thr), False, True)

SCFPad = SCF
