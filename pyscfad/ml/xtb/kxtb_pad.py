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
k-point XTB with padding (for batched periodic calculations).

Follows the same strategy as :mod:`pyscfad.ml.xtb.xtb_pad` /
:mod:`pyscfad.ml.scf.hf_pad`: cells are padded with fake atoms
(``numbers == 0``) and a fixed-shape lattice-image grid, so heterogeneous
solids batch under ``jax.vmap`` with static shapes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad import numpy as np

from pyscfad.xtb import kxtb
from pyscfad.ml.pbc.scf.khf_pad import KSCFPad
from pyscfad.ml.xtb import xtb_pad

if TYPE_CHECKING:
    from pyscfad.typing import Array


class KXTB(kxtb.KXTB, KSCFPad):
    @property
    def tot_electrons(self):
        return xtb_pad.tot_valence_electrons(self.cell, nkpts=len(self.kpts))


class GFN1KXTB(kxtb.GFN1KXTB, KXTB):
    """Padded GFN1-XTB with k-point sampling."""

    def _get_gamma(self) -> Array:
        gamma = kxtb.GFN1KXTB._get_gamma(self)
        shl_mask = self.cell.shl_mask
        return np.where(np.outer(shl_mask, shl_mask), gamma, 0.0)

