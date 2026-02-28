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
Lightweight :mod:`~pyscfad.pbc.gto.cell` module
"""
from pyscfad.typing import Array, ArrayLike
from pyscfad import numpy as np
from pyscfad.gto import MoleLite
from pyscfad.pbc.gto import cell
from pyscfad.pbc.gto.cell import estimate_rcut

class Cell(MoleLite):
    """Unit cell information.

    Args:
        a: The lattice vectors.
        precision: The integral precision.
        rcut: The cutoff radius for lattice sum.
        dimension: PBC dimensions.
            0: no PBC.
            1: PBC along ``a[0]``.
            2: PBC along ``a[0]`` and ``a[1]``.
            3: PBC along ``a[0]``, ``a[1]``, and ``a[2]``.
    """
    def __init__(
        self,
        a: ArrayLike = np.zeros((3,3)),
        precision: float = 1e-8,
        rcut: float | None = None,
        dimension: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a = np.asarray(a, dtype=float).reshape(3,3)
        self.precision = precision
        if rcut is None:
            rcut = estimate_rcut(self, self.precision)
        self.rcut = rcut
        self.dimension = dimension

    def lattice_vectors(self) -> Array:
        """Unit cell lattice vectors.
        """
        return self.a

    def pbc_intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        kpts: ArrayLike | None = None,
        shls_slice: tuple[int, ...] | None = None,
        **kwargs,
    ) -> Array:
        """Periodic one-electron integrals.

        See Also:
            :func:`~pyscf.pbc.gto.Cell.pbc_intor`

        Notes:
            Unlike pyscf, the argument ``kpt`` is not supported,
            and the returned array always contains one dimension for the k-points.
        """
        from pyscfad.pbc.gto._pbcintor_lite import _pbc_intor

        if "kpt" in kwargs:
            raise KeyError("The argument 'kpt' is not supported, use 'kpts' insdead.")

        if kpts is None:
            kpts = np.zeros((1,3))
        kpts = np.asarray(kpts, float).reshape(-1,3)

        intor_name = self._add_suffix(intor_name)

        out = _pbc_intor(
            intor_name, self.a, kpts, self.rcut,
            self._atm, self._bas, self._env,
            shls_slice=shls_slice, comp=comp, hermi=hermi,
            trace_coords=self.trace_coords, trace_basis=self.trace_basis,
            dimension=self.dimension,
        )
        return out

    make_kpts = cell.Cell.make_kpts
    get_scaled_kpts = cell.Cell.get_scaled_kpts
    get_abs_kpts = cell.Cell.get_abs_kpts
    reciprocal_vectors = cell.Cell.reciprocal_vectors

CellLite = Cell
