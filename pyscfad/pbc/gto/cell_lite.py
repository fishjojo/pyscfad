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
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad import numpy as np
from pyscfad.gto import MoleLite
from pyscfad.pbc.gto import cell, _latintor
from pyscfad.pbc.gto.cell import estimate_rcut
from pyscfad.pbc.tools import (
    get_lattice_Ls,
    nimgs_to_lattice_Ls,
)
from pyscfad.experimental.moleintor_cuint import CuintPlan
from pyscfad.experimental import latintor_cuint

if TYPE_CHECKING:
    from pyscfad.typing import Array, ArrayLike

class Cell(MoleLite):
    """Unit cell information.

    Args:
        a: The lattice vectors.
        precision: The integral precision.
        rcut: The cutoff radius for lattice sum.
        nimgs: Number of periodic images.
        dimension: PBC dimensions.
            0: no PBC.
            1: PBC along ``a[0]``.
            2: PBC along ``a[0]`` and ``a[1]``.
            3: PBC along ``a[0]``, ``a[1]``, and ``a[2]``.
    """
    low_dim_ft_type = None

    def __init__(
        self,
        a: ArrayLike = np.zeros((3,3)),
        precision: float = 1e-8,
        rcut: float | None = None,
        nimgs: int | tuple[int, ...] | None = None,
        dimension: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a = np.asarray(a, dtype=np.floatx).reshape(3,3)
        self.precision = precision
        self.dimension = dimension

        if rcut is None and self._bas is not None:
            rcut = estimate_rcut(self, precision)
        self.rcut = rcut
        if nimgs is None and self.rcut is not None:
            Ls = self.get_lattice_Ls()
            scaled_Ls = np.dot(Ls, np.linalg.inv(self.a))
            scaled_Ls = np.rint(scaled_Ls).astype(int)
            nimgs = abs(scaled_Ls).max(axis=0)
        self.nimgs = nimgs
        self._Ls = None

    @property
    def Ls(self) -> Array:
        if self._Ls is None:
            if self.nimgs is None:
                raise RuntimeError("cell.nimgs is not set")
            self._Ls = nimgs_to_lattice_Ls(self)
        return self._Ls

    @Ls.setter
    def Ls(self, val: ArrayLike):
        self._Ls = np.asarray(val, dtype=np.floatx).reshape(-1,3)

    def get_Ls_mask(
        self,
        Ls: ArrayLike | None = None,
        rcut: float | None = None
    ) -> Array:
        if Ls is None:
            Ls = self.Ls
        Ls = np.asarray(Ls, dtype=np.float64).reshape(-1,3)

        if rcut is None:
            rcut = self.rcut
        if rcut is None:
            raise KeyError("cell.rcut not set")

        r = self.atom_coords()
        rr = r[:,None] - r
        dist_max = np.linalg.norm(rr, axis=2).max()
        Ls_mask = np.linalg.norm(Ls, axis=1) < (self.rcut + dist_max)
        Ls_mask = np.asarray(Ls_mask, dtype=np.int32)
        return Ls_mask

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

    def lattice_intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        Ls: ArrayLike | None = None,
        shls_slice: tuple[int, ...] | None = None,
        cuint_plan: CuintPlan | None = None,
    ) -> Array:
        """Lattice one-electron integrals.

        Notes:
            When ``hermi=1``, the CPU backend will output the lower triangle
            (including the diagonal); the cuint backend will output irregular blocks
            with the diagonal blocks halved, which should be used as follows

            .. code-block:: python

                s1e = einsum('kl,lpq->kpq', expkL, s1e_lat)
                s1e = s1e + s1e.transpose(0,2,1).conj()
        """
        intor_name = self._add_suffix(intor_name)

        if Ls is None:
            Ls = self.Ls

        Ls_mask = self.get_Ls_mask(Ls)

        if cuint_plan is None:
            cuint_plan = self.cuint_plan

        if cuint_plan is None:
            out = _latintor._lattice_intor(
                intor_name, Ls, Ls_mask,
                self._atm, self._bas, self._env,
                shls_slice=shls_slice, comp=comp, hermi=hermi,
                trace_coords=self.trace_coords, trace_basis=self.trace_basis,
            )
        else:
            out = latintor_cuint._lattice_intor(
                intor_name, Ls, Ls_mask,
                self._atm, self._bas, self._env, cuint_plan,
                shls_slice=shls_slice, comp=comp, hermi=hermi,
                trace_coords=self.trace_coords, trace_basis=self.trace_basis,
            )
        return out

    make_kpts = cell.Cell.make_kpts
    get_scaled_kpts = cell.Cell.get_scaled_kpts
    get_abs_kpts = cell.Cell.get_abs_kpts
    reciprocal_vectors = cell.Cell.reciprocal_vectors
    get_ewald_params = cell.Cell.get_ewald_params
    vol = cell.Cell.vol
    cutoff_to_mesh = cell.Cell.cutoff_to_mesh
    get_Gv_weights = cell.Cell.get_Gv_weights
    get_scaled_atom_coords = cell.Cell.get_scaled_atom_coords
    get_lattice_Ls = get_lattice_Ls

CellLite = Cell
