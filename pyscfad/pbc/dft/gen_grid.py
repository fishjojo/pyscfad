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
