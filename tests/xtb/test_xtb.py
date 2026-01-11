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

import pytest
import jax
from pyscfad import numpy as np
from pyscfad.gto import MoleLite as Mole
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    yield basis, param

def test_gfn1_xtb_energy_force(setup, H2O_GFN1_ref, NH3_GFN1_ref):
    basis, param = setup

    for vals in (H2O_GFN1_ref, NH3_GFN1_ref):
        numbers, coords, e0, g0, *_ = vals

        def energy(coords):
            mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=True)
            mf = GFN1XTB(mol, param=param)
            return mf.kernel()

        e1, g1 = jax.value_and_grad(energy)(coords)
        assert abs(e1 - e0) < 1e-8
        assert abs(g1 - g0).max() < 1e-8

def test_gfn1_xtb_dip_polar(setup, H2O_GFN1_ref, NH3_GFN1_ref):
    basis, param = setup

    for vals in (H2O_GFN1_ref, NH3_GFN1_ref):
        numbers, coords, _, _, mu0, alpha0 = vals

        def energy(numbers, coords, E0):
            mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=False)
            mf = GFN1XTB(mol, param=param)
            h0 = mf.get_hcore()
            mf.get_hcore = lambda *args, **kwargs: h0 + \
                np.einsum("x, xij->ij", E0, mol.intor("int1e_r", hermi=1))
            mf.kernel()
            mu = mf.dip_moment()
            return mu

        mu = energy(numbers, coords, np.zeros(3))
        assert abs(mu - mu0).max() < 1e-8
        alpha1 = jax.jacrev(energy, 2)(numbers, coords, np.zeros(3))
        assert abs(alpha1 - alpha0).max() < 1e-8
