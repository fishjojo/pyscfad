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
Lightweight :mod:`~pyscfad.scf.rohf` module.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

from pyscf.lib import module_method

from pyscfad import numpy as np
from pyscfad.ops import vmap
from pyscfad.scf import hf_lite, uhf_lite, rohf, sp2

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array

def get_occ(
    mf: ROHF,
    mo_energy: ArrayLike | None = None,
    mo_coeff: ArrayLike | None = None,
) -> Array:
    """Doubly occupy the ``nelec[1]`` lowest MOs,
    and singly occupy the next ``nelec[0] - nelec[1]``.
    """
    if mf.sigma is not None and mf.sigma > 0:
        raise NotImplementedError("Smearing for ROHF")
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    mask = mf.mo_mask(mo_energy, mo_coeff)
    return sum(hf_lite.fill_occ(mf, mo_energy, mask, n) for n in mf.nelec)

def get_grad(mo_coeff: ArrayLike, mo_occ: ArrayLike, fock_ao: ArrayLike) -> Array:
    """Orbital gradients from the Roothaan effective Fock matrix.

    See also:
        :func:`pyscf.scf.rohf.get_grad`
    """
    occa = mo_occ > 0
    occb = mo_occ > 1
    uniq_var_a = ~occa[:,None] & occa
    uniq_var_b = ~occb[:,None] & occb
    fock_mo = mo_coeff.conj().T @ fock_ao @ mo_coeff
    return (fock_mo * (uniq_var_a.astype(fock_mo.dtype) + uniq_var_b)).ravel()

class ROHF(uhf_lite.UHF):
    """Restricted open-shell Hartree-Fock.

    The MOs diagonalize the Roothaan effective Fock matrix, while the
    density matrix has the alpha and beta components as in
    :class:`~pyscfad.scf.uhf_lite.UHF`.
    """
    def get_fock(
        self,
        h1e: Array | None = None,
        s1e: Array | None = None,
        vhf: Any | None = None,
        dm: Array | None = None,
        cycle: int = -1,
        diis: Any | None = None,
        diis_start_cycle: int | None = None,
        level_shift_factor: float | None = None,
        damp_factor: float | None = None,
        fock_last: Array | None = None,
    ) -> Array:
        if s1e is None:
            s1e = self.get_ovlp()
        f = rohf.get_roothaan_fock(super().get_fock(h1e, s1e, vhf, dm), dm, s1e)
        return hf_lite.extrapolate_fock(f, s1e, dm[0] + dm[1], diis, fock_last)

    def get_grad(
        self,
        mo_coeff: ArrayLike,
        mo_occ: ArrayLike,
        fock: ArrayLike | None = None,
    ) -> Array:
        if fock is None:
            fock = self.get_fock(dm=self.make_rdm1(mo_coeff, mo_occ))
        return get_grad(mo_coeff, mo_occ, fock)

    def make_rdm1_sp2(self, fock: Array, x: Array) -> Array:
        # the alpha and beta electrons occupy the same effective Fock matrix
        return vmap(sp2.make_rdm1, in_axes=(None, None, 0))(fock, x, np.asarray(self.nelec))

    eig = hf_lite.SCF.eig
    get_occ = get_occ
    make_rdm1 = module_method(rohf.make_rdm1, absences=["mo_coeff", "mo_occ"])

ROHFLite = ROHF
