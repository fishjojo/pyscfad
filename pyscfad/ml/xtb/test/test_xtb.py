# Copyright 2021-2025 The PySCFAD Authors
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

@pytest.fixture(scope="module")
def setup():
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)

    numbers = np.array(
        [
            [8, 1, 1, 0, 0],
            [7, 1, 1, 1, 0],
        ],
        dtype=np.int32
    )

    coords = np.array(
        [
            np.array([
                [0.00000,  0.00000,  0.00000],
                [1.43355,  0.00000, -0.95296],
                [1.43355,  0.00000,  0.95296],
                [0.00000,  0.00000,  0.00000],
                [1.00000,  0.00000,  0.00000],
            ]),
            np.array([
                [-0.80650, -1.00659,  0.02850],
                [-0.50540, -0.31299,  0.68220],
                [ 0.00620, -1.41579, -0.38500],
                [-1.32340, -0.54779, -0.69350],
                [ 0.00000,  0.00000,  0.00000],
            ]) / 0.52917721067121,
        ]
    )

    e0 = np.array([-5.72311730, -4.82989868])
    g0 = np.array(
        [
            [
                [ 3.93688346e-02, 0.,  0.,],
                [-1.96844173e-02, 0.,  1.03173514e-01,],
                [-1.96844173e-02, 0., -1.03173514e-01,],
                [ 0., 0., 0.,],
                [ 0., 0., 0.,],
            ],
            [
                [ 0.00307771,  0.00378012, -0.00250677],
                [-0.00197155, -0.00543102, -0.00679491],
                [-0.00681684,  0.00491176,  0.0032046 ],
                [ 0.00571069, -0.00326086,  0.00609709],
                [ 0., 0., 0.,],
            ],
        ]
    )

    mu0 = np.array(
        [
            [ 3.77498369e+00, 0., 0.,],
            [ 1.14759296e+00,  1.42873381e+00, -9.26322512e-01,],
        ]
    )

    polar0 = np.array(
        [
            np.diag(np.array([7.33978195, 1.07104174e-01, 4.97209375])),
            np.array(
                [
                    [ 8.40067370, -3.05980713,  1.98427637],
                    [-3.05980713,  7.05005542,  2.47070858],
                    [ 1.98427637,  2.47070858,  9.25977794],
                ]
            ),
        ]
    )
    yield numbers, coords, basis, param, e0, g0, mu0, polar0

def test_gfn1_xtb_force(setup):
    numbers, coords, basis, param, e0, g0, _, _, = setup

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, verbose=4, trace_coords=True)
        mf = GFN1XTB(mol, param)
        mf.diis = "anderson"
        mf.conv_tol = 1e-6
        mf.diis_damp = 0.5
        mf.diis_space = 6
        e = mf.kernel()
        return e

    e, g = jax.vmap(jax.value_and_grad(energy, 1))(numbers, coords)
    assert abs(e - e0).max() < 1e-6
    assert abs(g - g0).max() < 1e-6

def test_gfn1_xtb_dip_pol(setup):
    numbers, coords, basis, param, _, _, mu0, polar0, = setup
    def energy(numbers, coords, E0):
        mol = MolePad(numbers, coords, basis=basis, verbose=4, trace_coords=False)
        mf = GFN1XTB(mol, param)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args, **kwargs: h0 + \
            np.einsum("x, xij->ij", E0, mol.intor("int1e_r", hermi=1))
        mf.diis = "anderson"
        mf.conv_tol = 1e-6
        mf.diis_damp = 0.5
        mf.diis_space = 6
        e = mf.kernel()
        mu = mf.dip_moment()
        return mu

    mu = jax.vmap(energy, (0, 0, None))(numbers, coords, np.zeros(3))
    assert abs(mu - mu0).max() < 1e-6

    polar = jax.vmap(jax.jacrev(energy, 2), (0, 0, None))(numbers, coords, np.zeros(3))
    print(polar)
    print(polar0)
    assert abs(polar - polar0).max() < 1e-6
