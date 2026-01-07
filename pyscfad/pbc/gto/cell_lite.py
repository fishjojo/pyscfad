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

from typing import Any

from pyscfad import numpy as np
from pyscfad.gto import MoleLite
from pyscfad.pbc.gto.cell import estimate_rcut

Array = Any

class Cell(MoleLite):
    """Unit cell information.

    Parameters
    ----------
    a : array
        The lattice vectors.
    precision : float
        The integral precision.
    rcut : float
        The cutoff radius for lattice sum.
    """

    use_loose_rcut = False

    def __init__(
        self,
        a: Array = np.zeros((3,3)),
        precision: float = 1e-8,
        rcut: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a = np.asarray(a, dtype=float).reshape(3,3)
        self.precision = precision
        if rcut is None:
            rcut = estimate_rcut(self, self.precision)
        self.rcut = rcut

    def lattice_vectors(self):
        return self.a

    def pbc_intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        kpts: Array | None = None,
        shls_slice: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Periodic one-electron integrals.

        See Also
        --------
        pyscf.pbc.gto.Cell.pbc_intor

        Notes
        -----
        Unlike the PySCF version, the argument ``kpt`` is not supported,
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
            intor_name,
            self.a,
            kpts,
            self.rcut,
            self._atm,
            self._bas,
            self._env,
            shls_slice=shls_slice,
            comp=comp,
            hermi=hermi,
            trace_coords=self.trace_coords,
        )
        return out

CellLite = Cell
