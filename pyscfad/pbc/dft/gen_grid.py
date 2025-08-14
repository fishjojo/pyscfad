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
from pyscf.pbc.gto import Cell as pyscf_Cell
from pyscf.pbc.dft import gen_grid as pyscf_gen_grid
from pyscfad import numpy as np
from pyscfad.ops import stop_grad
from pyscfad.pbc.gto.cell import get_uniform_grids

class UniformGrids(pyscf_gen_grid.UniformGrids):
    @property
    def coords(self):
        if self._coords is not None:
            return self._coords
        else:
            return get_uniform_grids(self.cell, self.mesh)

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        else:
            ngrids = numpy.prod(self.mesh)
            weights = np.full((ngrids,), self.cell.vol / ngrids)
            return weights

    def make_mask(self, cell=None, coords=None, relativity=0, shls_slice=None,
                  verbose=None):
        if cell is None:
            cell = self.cell
        if coords is None:
            coords = self.coords
        return pyscf_gen_grid.make_mask(cell.view(pyscf_Cell), stop_grad(coords),
                                        relativity, shls_slice, verbose)
