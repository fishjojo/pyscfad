# Copyright 2025-2026 The PySCFAD Authors
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

from pyscfad import numpy as np
from pyscfad.scf import hf_lite as hf
from pyscfad.scipy.linalg import eigh

class _SCFPadMixin:
    """Padding-aware diagonalization and MO masking.

    Shared by the restricted (:class:`SCF`) and unrestricted (:class:`UHF`)
    padded SCF classes. The padding methods operate on a single spin block
    (a 2D Fock/overlap matrix and a 1D MO-energy vector), so they compose with
    the spin-resolved :class:`~pyscfad.scf.hf_lite.UHF` whose ``eig`` calls
    ``_eigh`` once per spin and whose ``get_occ`` masks each spin separately.
    """
    @property
    def padding_level_shift(self):
        r"""Energy shift applied to the fake (padding) MOs.

        The padding block of the Fock matrix is shifted by this amount so the
        fake MOs sit far above the real spectrum and are never occupied (see
        :meth:`mo_mask`). LAPACK's generalized eigensolver (``sygvd``) is
        backward stable with error :math:`\sim \epsilon \lVert H \rVert`, and
        the shifted padding block makes :math:`\lVert H \rVert \sim`
        ``padding_level_shift``; the *real* eigenvalues are therefore perturbed
        by :math:`\sim \epsilon \cdot` ``padding_level_shift``. A shift that is
        negligible in float64 (``1e6``) thus swamps the real eigenvalues in
        float32.

        Scale the shift with the working precision so that both the separation
        gap and the relative contamination stay :math:`\sim \sqrt{\epsilon}`:
        float64 -> ``1e6``, float32 -> ``~40``. This stays well above any
        physical MO energy in the minimal/ML basis sets this path targets while
        keeping :math:`\epsilon \cdot` ``padding_level_shift`` near the float32
        round-off floor.
        """
        eps = numpy.finfo(np.floatx).eps
        eps64 = numpy.finfo(numpy.float64).eps
        return float(1e6 * (eps64 / eps) ** 0.5)

    def _eigh(self, h, s, **kwargs):
        ao_mask = self.mol.ao_mask
        mask = np.asarray(1 - ao_mask, dtype=np.int32)
        s = s + np.diag(mask)
        h = np.where(np.outer(ao_mask, ao_mask), h, 0.)
        h = h + np.diag(mask) * self.padding_level_shift
        return eigh(h, s)

    def mo_mask(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy

        thr = .99 * self.padding_level_shift
        return np.where(np.greater(mo_energy, thr), False, True)


class SCF(_SCFPadMixin, hf.SCF):
    """Restricted SCF with padding (for batched calculations)."""

SCFPad = SCF
RHF = SCF


class UHF(_SCFPadMixin, hf.UHF):
    """Unrestricted SCF with padding (for batched calculations)."""

UHFPad = UHF
