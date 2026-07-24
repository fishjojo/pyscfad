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
Unit cell information with padding (for batched calculations).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad import numpy as np
from pyscfad.ml.gto.mole_pad import MolePad, _bas_template
from pyscfad.pbc.gto import cell as _cell
from pyscfad.pbc.gto import _latintor
from pyscfad.experimental import latintor_cuint

if TYPE_CHECKING:
    from pyscfad.typing import Array, ArrayLike
    from pyscfad.ml.gto.basis_array import BasisArray
    from pyscfad.experimental.moleintor_cuint import CuintPlan


def make_image_grid(nimgs, dimension: int = 3):
    """Static integer translation grid ``Ts`` for a fixed number of images.

    ``Ls = Ts @ a`` then gives fixed-shape lattice vectors for any (traced)
    lattice matrix ``a``. ``nimgs`` may be a scalar or per-dimension sequence
    and must be a concrete (host) value.
    """
    import numpy as host_np

    if host_np.isscalar(nimgs):
        bounds = (int(nimgs),) * dimension
    else:
        bounds = tuple(int(n) for n in nimgs)
        assert len(bounds) <= 3

    axes = [host_np.arange(-b, b + 1) for b in bounds]
    axes += [host_np.zeros(1, dtype=int)] * (3 - len(axes))
    Ts = host_np.stack(
        host_np.meshgrid(*axes, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return Ts


class CellPad(MolePad):
    """Padded unit cell (for batched periodic calculations).

    Same padding conventions as :class:`MolePad` (``numbers == 0`` marks
    padding atoms; shells/AOs are padded uniformly per element through the
    :class:`BasisArray`), plus the lattice data:

    Args:
        a: Lattice vectors ``[3, 3]`` (in Bohr; rows are the vectors). May be
            traced (e.g. for batching over solids).
        Ls: Fixed-shape lattice translations ``[nL, 3]``. Build them as
            ``make_image_grid(nimgs) @ a`` with a static, batch-wide ``nimgs``
            so every cell in a batch shares the same ``nL``; per-cell validity
            is handled at integral time through :meth:`get_Ls_mask`.
        rcut: Static lattice-sum cutoff radius (Bohr). Choose the maximum over
            the batch (e.g. from ``pyscfad.pbc.gto.cell.estimate_rcut`` on a
            representative unpadded cell).
    """

    low_dim_ft_type = None

    def __init__(
        self,
        numbers: ArrayLike,
        coords: ArrayLike,
        basis: BasisArray | None = None,
        a: ArrayLike | None = None,
        Ls: ArrayLike | None = None,
        rcut: float | None = None,
        precision: float = 1e-8,
        dimension: int = 3,
        charge: int = 0,
        spin: int = 0,
        cart: bool = False,
        verbose: int = 3,
        trace_coords: bool = False,
        trace_basis: bool = False,
        cuint_plan: CuintPlan | None = None,
        bas0: ArrayLike = None,
        env0: ArrayLike = None,
    ):
        super().__init__(
            numbers,
            coords,
            basis=basis,
            charge=charge,
            spin=spin,
            cart=cart,
            verbose=verbose,
            trace_coords=trace_coords,
            trace_basis=trace_basis,
            cuint_plan=cuint_plan,
            bas0=bas0,
            env0=env0,
        )
        if a is None:
            raise ValueError("CellPad requires the lattice vectors 'a'.")
        if rcut is None:
            raise ValueError(
                "CellPad requires a static 'rcut' (use the batch maximum)."
            )
        if Ls is None:
            raise ValueError(
                "CellPad requires fixed-shape lattice translations 'Ls' "
                "(e.g. make_image_grid(nimgs) @ a with static nimgs)."
            )
        self.a = np.asarray(a, dtype=np.floatx).reshape(3, 3)
        self.precision = precision
        self.dimension = dimension
        self.rcut = rcut
        self.Ls = np.asarray(Ls, dtype=np.floatx).reshape(-1, 3)

    def lattice_vectors(self) -> Array:
        return self.a

    def get_Ls_mask(
        self,
        Ls: ArrayLike | None = None,
        rcut: float | None = None,
    ) -> Array:
        if Ls is None:
            Ls = self.Ls
        Ls = np.asarray(Ls, dtype=np.float64).reshape(-1, 3)
        if rcut is None:
            rcut = self.rcut

        r = self.atom_coords()
        rr = r[:, None] - r
        # squared distances (norm at zero separation has a NaN gradient);
        # padding atoms sit at the origin, which only makes dist_max
        # conservative (more images kept).
        rr2 = np.einsum("ijx,ijx->ij", rr, rr)
        dist_max = np.sqrt(np.max(rr2) + 1e-30)
        Ls_mask = np.linalg.norm(Ls, axis=1) < (rcut + dist_max)
        Ls_mask = np.asarray(Ls_mask, dtype=np.int32)
        return Ls_mask

    def lattice_intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        Ls: ArrayLike | None = None,
        shls_slice: tuple[int, ...] | None = None,
        cuint_plan: CuintPlan | None = None,
    ) -> Array:
        """Lattice one-electron integrals over the padded cell.

        Same backend conventions as
        :meth:`pyscfad.pbc.gto.cell_lite.Cell.lattice_intor` (CPU: lower
        triangle for ``hermi=1``; cuint: halved diagonal blocks).
        """
        intor_name = self._add_suffix(intor_name)

        if Ls is None:
            Ls = self.Ls
        Ls_mask = self.get_Ls_mask(Ls)

        if cuint_plan is None:
            cuint_plan = self.cuint_plan

        ao_loc = self.ao_loc
        aoslices = self.aoslice_by_atom(ao_loc=ao_loc)[:, 2:4]
        # concrete structural view of _bas (see mole_pad._bas_template)
        bas_tmpl = _bas_template(self.basis, self.natm)

        if cuint_plan is None:
            out = _latintor._lattice_intor(
                intor_name, Ls, Ls_mask,
                self._atm, self._bas, self._env,
                shls_slice=shls_slice, comp=comp, hermi=hermi,
                ao_loc=ao_loc,
                trace_coords=self.trace_coords,
                trace_basis=self.trace_basis,
                aoslices=aoslices,
                bas_tmpl=bas_tmpl,
            )
        else:
            out = latintor_cuint._lattice_intor(
                intor_name, Ls, Ls_mask,
                self._atm, self._bas, self._env, cuint_plan,
                shls_slice=shls_slice, comp=comp, hermi=hermi,
                ao_loc=ao_loc,
                trace_coords=self.trace_coords,
                trace_basis=self.trace_basis,
                aoslices=aoslices,
                bas_tmpl=bas_tmpl,
            )
        return out

    # Lattice/k-space helpers shared with the full Cell implementation;
    # they only rely on lattice_vectors()/dimension (all traceable).
    make_kpts = _cell.Cell.make_kpts
    get_scaled_kpts = _cell.Cell.get_scaled_kpts
    get_abs_kpts = _cell.Cell.get_abs_kpts
    reciprocal_vectors = _cell.Cell.reciprocal_vectors
    get_ewald_params = _cell.Cell.get_ewald_params
    vol = _cell.Cell.vol
    cutoff_to_mesh = _cell.Cell.cutoff_to_mesh
    get_Gv_weights = _cell.Cell.get_Gv_weights
    get_scaled_atom_coords = _cell.Cell.get_scaled_atom_coords
