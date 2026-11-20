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
Integration grids built with array operations from the atomic coordinates,
so that they can be traced (``jax.jit``) and differentiated.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy
from pyscf.dft import gen_grid, radi

from pyscfad import numpy as np
from pyscfad.gto.mole import inter_distance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pyscfad.typing import Array
    from pyscfad.gto import MoleLite

def get_partition(
    mol: MoleLite,
    atom_grids_tab: dict,
    radii_adjust: Callable | None = None,
    atomic_radii: numpy.ndarray | None = radi.BRAGG_RADII,
    becke_scheme: Callable = gen_grid.original_becke,
) -> tuple[Array, Array]:
    """Grid coordinates and weights with Becke partitioning.

    See also:
        :func:`pyscf.dft.gen_grid.get_partition`
    """
    if becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError(f"becke_scheme = {becke_scheme}")

    natm = mol.natm
    # the adjusted cell function is g + a[i,j] * (1 - g**2)
    if callable(radii_adjust) and atomic_radii is not None:
        f_radii_adjust = radii_adjust(mol, atomic_radii)
        a = numpy.array([[f_radii_adjust(i, j, 0.) for j in range(natm)]
                         for i in range(natm)])
    else:
        a = numpy.zeros((natm, natm))

    atm_coords = mol.atom_coords()
    atm_dist_inv = np.safe_reciprocal(inter_distance(coords=atm_coords), fill_value=0.)
    offdiag = ~numpy.eye(natm, dtype=bool)

    coords_all = []
    weights_all = []
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        dr = coords[:,None,:] - atm_coords
        grid_dist = np.safe_sqrt(np.einsum("gax,gax->ga", dr, dr))
        g = (grid_dist[:,:,None] - grid_dist[:,None,:]) * atm_dist_inv
        g = g + a * (1 - g**2)
        for _ in range(3):
            g = (3 - g**2) * g * .5
        pbecke = np.prod(np.where(offdiag, .5 * (1 - g), 1.), axis=-1)
        weights = vol * pbecke[:,ia] / pbecke.sum(axis=1)
        coords_all.append(coords)
        weights_all.append(weights)
    return np.vstack(coords_all), np.hstack(weights_all)

class Grids(gen_grid.Grids):
    """Integration grids whose coordinates and weights are traceable
    functions of the atomic coordinates.

    Derivatives therefore include the grid response.

    Notes:
        Unlike :class:`pyscf.dft.gen_grid.Grids`, the grids are neither
        sorted nor padded (``alignment``), and no screening mask
        (``non0tab``) is built.
    """
    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        del with_non0tab, sort_grids
        if mol is None:
            mol = self.mol
        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method, self.level, self.prune, **kwargs)
        self.coords, self.weights = get_partition(
            mol, atom_grids_tab, self.radii_adjust, self.atomic_radii, self.becke_scheme)
        self.non0tab = self.screen_index = None
        return self
