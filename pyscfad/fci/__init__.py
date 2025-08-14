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

from functools import reduce
from pyscfad import numpy as np
from pyscfad import ao2mo
from pyscfad.fci import fci_slow
from pyscfad.fci.fci_slow import fci_ovlp

def solve_fci(mf, nroots=1):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nelec = mol.nelectron
    h1e = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
    eri = ao2mo.incore.full(mf._eri, mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    e, fcivec = fci_slow.kernel(h1e, eri, norb, nelec, mf.energy_nuc(), nroots=nroots)
    return e, fcivec
