# Copyright 2026 The PySCFAD Authors
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

"""
k-point SCF with padding (for batched periodic calculations).

Follows the same strategy as :mod:`pyscfad.ml.scf.hf_pad`: the padded
(fake) AO block of every k-point Fock matrix is level-shifted far above the
real spectrum, and :meth:`mo_mask` flags the fake MOs so occupations
(:func:`pyscfad.pbc.scf.khf_lite.get_occ`) skip them.
"""
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.pbc.scf import khf_lite
from pyscfad.ml.scf.hf_pad import SCF as SCFPadBase


class KSCF(khf_lite.KSCF):
    padding_level_shift = SCFPadBase.padding_level_shift

    def _eigh(self, h, s, **kwargs):
        ao_mask = self.mol.ao_mask
        mask = np.asarray(1 - ao_mask, dtype=np.int32)
        s = s + np.diag(mask)[None, ...]
        h = np.where(np.outer(ao_mask, ao_mask)[None, ...], h, 0.0)
        h = h + (np.diag(mask) * self.padding_level_shift)[None, ...]
        return ops.vmap(super(khf_lite.KSCF, self)._eigh)(h, s)

    def mo_mask(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        thr = 0.99 * self.padding_level_shift
        return np.where(np.greater(mo_energy, thr), False, True)


KSCFPad = KSCF
