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
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    # QM water (coordinates in Bohr)
    numbers = np.array([8, 1, 1])
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.43355,  0.00000, -0.95296],
            [1.43355,  0.00000,  0.95296],
        ]
    )
    # cubic periodic box (Bohr) and a couple of MM point charges
    a = np.eye(3) * 12.0
    mm_coords = np.array([[5.0, 0.5, 0.3], [-4.0, 1.0, -2.0]])
    mm_charges = np.array([0.4, -0.4])
    mm_radii = np.array([1.2, 1.2])
    yield basis, param, numbers, coords, a, mm_coords, mm_charges, mm_radii

def _make_energy(basis, param, numbers, a, mm_coords, mm_charges, mm_radii):
    def energy(coords):
        mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=True)
        mf = GFN1XTB(mol, param=param)
        mf = add_mm_charges(mf, mm_coords, a, mm_charges, mm_radii, unit='Bohr')
        mf.diis = None
        return mf.kernel()
    return energy

def test_gfn1_xtb_qmmm_energy_force(setup):
    basis, param, numbers, coords, a, mm_coords, mm_charges, mm_radii = setup
    energy = _make_energy(basis, param, numbers, a, mm_coords, mm_charges, mm_radii)

    e_ref = -5.703380588049711
    e, g = jax.value_and_grad(energy)(coords)
    assert abs(e - e_ref) < 1e-7
    # finite-difference check of the force along x of the O atom
    eps = 1e-4
    cp = coords.at[0, 0].add(eps)
    cm = coords.at[0, 0].add(-eps)
    g_fd = (energy(cp) - energy(cm)) / (2 * eps)
    assert abs(g[0, 0] - g_fd) < 1e-5

# Body executed under PYSCFAD_FLOATX=float32 in a subprocess (see the
# ``run_fp32`` fixture); mirrors ``_make_energy`` for the QM/MM coupling.
_FP32_QMMM_BODY = """
from pyscfad.gto import MoleLite as Mole
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges

numbers = np.asarray(IN["numbers"])
coords = np.asarray(IN["coords"])
a = np.asarray(IN["a"])
mm_coords = np.asarray(IN["mm_coords"])
mm_charges = np.asarray(IN["mm_charges"])
mm_radii = np.asarray(IN["mm_radii"])

basis = xtb_basis.get_basis_filename()
param = GFN1Param()

def energy(coords):
    mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=True)
    mf = GFN1XTB(mol, param=param)
    mf = add_mm_charges(mf, mm_coords, a, mm_charges, mm_radii, unit='Bohr')
    mf.diis = None
    return mf.kernel()

e, g = jax.value_and_grad(energy)(coords)
emit(e=float(e), g=numpy.asarray(g).tolist())
"""

def test_gfn1_xtb_qmmm_energy_force_fp32(setup, run_fp32):
    basis, param, numbers, coords, a, mm_coords, mm_charges, mm_radii = setup
    energy = _make_energy(basis, param, numbers, a, mm_coords, mm_charges, mm_radii)

    # FP64 baseline in-process; FP32 result from a float32 subprocess.
    e64, g64 = jax.value_and_grad(energy)(coords)
    res = run_fp32(
        _FP32_QMMM_BODY,
        numbers=numbers, coords=coords, a=a,
        mm_coords=mm_coords, mm_charges=mm_charges, mm_radii=mm_radii)
    e32 = res["e"]
    g32 = np.asarray(res["g"])

    assert abs(e32 - e64) < 1e-5
    assert abs(g32 - g64).max() < 1e-5
