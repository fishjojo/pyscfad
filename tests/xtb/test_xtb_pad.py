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
from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.xtb import GFN1XTB, make_param_array

@pytest.fixture
def setup():
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)
    yield basis, param

def _concat(numbers1, coords1, numbers2, coords2):
    numbers1 = np.hstack([numbers1, np.zeros(2, dtype=int)])
    numbers2 = np.hstack([numbers2, np.zeros(1, dtype=int)])
    numbers = np.asarray([numbers1, numbers2])

    coords1 = np.vstack([coords1, np.zeros(3), np.ones(3)])
    coords2 = np.vstack([coords2, np.zeros(3)])
    coords = np.asarray([coords1, coords2])
    return numbers, coords

def test_gfn1_xtb_pad_energy_force(setup, H2O_GFN1_ref, NH3_GFN1_ref):
    basis, param = setup
    numbers1, coords1, e1, g1, *_ = H2O_GFN1_ref
    numbers2, coords2, e2, g2, *_ = NH3_GFN1_ref
    numbers, coords = _concat(numbers1, coords1, numbers2, coords2)

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, trace_coords=True)
        mf = GFN1XTB(mol, param)
        e = mf.kernel()
        return e

    e, g = jax.vmap(jax.value_and_grad(energy, 1))(numbers, coords)
    e0 = np.asarray([e1, e2])
    g1 = np.vstack([g1, np.zeros((2,3))])
    g2 = np.vstack([g2, np.zeros((1,3))])
    g0 = np.asarray([g1, g2])
    assert abs(e - e0).max() < 1e-8
    assert abs(g - g0).max() < 1e-8

def test_gfn1_xtb_pad_dip_pol(setup, H2O_GFN1_ref, NH3_GFN1_ref):
    basis, param = setup
    numbers1, coords1, _, _, mu1, alpha1 = H2O_GFN1_ref
    numbers2, coords2, _, _, mu2, alpha2 = NH3_GFN1_ref
    numbers, coords = _concat(numbers1, coords1, numbers2, coords2)

    def energy(numbers, coords, E0):
        mol = MolePad(numbers, coords, basis=basis, trace_coords=False)
        mf = GFN1XTB(mol, param)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args, **kwargs: h0 + \
            np.einsum("x, xij->ij", E0, mol.intor("int1e_r", hermi=1))
        mf.kernel()
        mu = mf.dip_moment()
        return mu

    mu0 = np.asarray([mu1, mu2])
    mu = jax.vmap(energy, (0, 0, None))(numbers, coords, np.zeros(3))
    assert abs(mu - mu0).max() < 1e-8

    alpha0 = np.asarray([alpha1, alpha2])
    alpha = jax.vmap(jax.jacrev(energy, 2), (0, 0, None))(numbers, coords, np.zeros(3))
    assert abs(alpha - alpha0).max() < 1e-8
