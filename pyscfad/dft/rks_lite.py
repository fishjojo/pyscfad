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
Lightweight :mod:`~pyscfad.dft.rks` module.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscf import __config__
from pyscf.dft import rks as pyscf_rks

from pyscfad.scf import hf_lite
from pyscfad.dft import gen_grid, numint, rks

if TYPE_CHECKING:
    from pyscfad.gto import MoleLite

class KohnShamDFT:
    """Kohn-Sham DFT for the lightweight SCF classes.

    Attributes:
        xc: Exchange-correlation functional.
        nlc: Non-local correlation functional.
        grids: Integration grids (:class:`~pyscfad.dft.gen_grid.Grids`),
            built from ``mol`` when first used.
        nlcgrids: Integration grids for ``nlc``.

    Notes:
        - The grids are traceable functions of the atomic coordinates,
          so the derivatives include the grid response.
        - Range-separated hybrid functionals are not supported.
    """
    # pruning grids by the density would make their size data dependent
    small_rho_cutoff = 0

    def __init__(self, mol: MoleLite, xc: str = "LDA,VWN", **kwargs):
        super().__init__(mol, **kwargs)
        self.xc = xc
        self.nlc = ""
        self.grids = gen_grid.Grids(mol)
        self.grids.level = getattr(
            __config__, "dft_rks_RKS_grids_level", self.grids.level)
        self.nlcgrids = gen_grid.Grids(mol)
        self.nlcgrids.level = getattr(
            __config__, "dft_rks_RKS_nlcgrids_level", self.nlcgrids.level)
        self.max_memory = mol.max_memory
        self._numint = numint.NumInt()

    do_nlc = pyscf_rks.KohnShamDFT.do_nlc
    initialize_grids = pyscf_rks.KohnShamDFT.initialize_grids

class RKS(KohnShamDFT, hf_lite.SCF):
    """Restricted Kohn-Sham DFT.
    """
    get_veff = rks.get_veff
    energy_elec = rks.energy_elec

RKSLite = RKS
