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

import pytest
import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad.gto import MoleLite as Mole
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param
from pyscfad.pbc.gto import CellLite as Cell
from pyscfad.xtb.kxtb import GFN1KXTB

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    yield basis, param

def test_gfn1_kxtb_energy_force(setup, H2O_GFN1_ref):
    basis, param = setup
    numbers, coords, *_ = H2O_GFN1_ref

    def mol_energy(coords, diis):
        mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=True)
        mf = GFN1XTB(mol, param=param)
        mf.diis = diis
        return mf.kernel()

    def cell_energy(coords, diis):
        cell = Cell(numbers=numbers, coords=coords, a=np.eye(3)*20., rcut=22.,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param)
        mf.diis = diis
        return mf.kernel()

    for diis in ("anderson", "qbroyden"):
        e0, g0 = jax.value_and_grad(mol_energy)(coords, diis)
        e1, g1 = jax.value_and_grad(cell_energy)(coords, diis)

        assert abs(e1 - e0) < 1e-3
        assert abs(g1 - g0).max() < 1e-3

def test_gfn1_kxtb_energy_force_with_kpts_sample(setup):
    numbers = [14,14]
    coords = np.asarray([[0.0, 0.0, 0.0],
                         [1.3467560987, 1.3467560987, 1.3467560987]]) / BOHR
    a = np.asarray([[0.0, 2.6935121974, 2.6935121974],
                    [2.6935121974, 0.0, 2.6935121974],
                    [2.6935121974, 2.6935121974, 0.0]]) / BOHR
    basis, param = setup

    def cell_energy(coords):
        cell = Cell(numbers=numbers, coords=coords, a=a,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.diis = "anderson"
        return mf.kernel()

    # reference with periodic coordination numbers (cn_d3 over cell.Ls);
    # forces vanish by symmetry for the ideal diamond structure
    e0 = -3.83117442207487
    g0 = np.zeros((2, 3))

    e1, g1 = jax.value_and_grad(cell_energy)(coords)
    assert abs(e1 - e0) < 1e-6
    assert abs(g1 - g0).max() < 1e-6

def test_gfn1_kxtb_get_bands(setup):
    numbers = [14,14]
    coords = np.asarray([[0.0, 0.0, 0.0],
                         [1.3467560987, 1.3467560987, 1.3467560987]]) / BOHR
    a = np.asarray([[0.0, 2.6935121974, 2.6935121974],
                    [2.6935121974, 0.0, 2.6935121974],
                    [2.6935121974, 2.6935121974, 0.0]]) / BOHR
    basis, param = setup

    cell = Cell(numbers=numbers, coords=coords, a=a,
                basis=basis, precision=1e-6)
    mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
    mf.diis = "anderson"
    mf.conv_tol = 1e-10
    mf.kernel()

    # bands at the SCF k-points reproduce the converged mo_energy up to
    # density-convergence noise (the Fock is rebuilt from make_rdm1())
    mo_energy, mo_coeff = mf.get_bands(mf.kpts)
    assert mo_energy.shape == mf.mo_energy.shape
    assert abs(np.sort(mo_energy.real, axis=1)
               - np.sort(mf.mo_energy.real, axis=1)).max() < 1e-5

    # bands along a path: single-kpt squeeze, ordering, finite gap
    scaled_path = np.asarray([[0.0, 0.0, 0.0],
                              [0.25, 0.0, 0.25],
                              [0.5, 0.0, 0.5]])
    band_kpts = cell.get_abs_kpts(scaled_path)
    e_single, c_single = mf.get_bands(band_kpts[0])
    assert e_single.ndim == 1
    e_path, _ = mf.get_bands(band_kpts)
    assert e_path.shape == (3, e_single.shape[0])
    assert abs(np.sort(e_path[0].real) - np.sort(e_single.real)).max() < 1e-10

    bands = np.sort(e_path.real, axis=1)
    nocc = int(mf.tot_electrons) // 2 // len(mf.kpts)
    vbm = bands[:, nocc-1].max()
    cbm = bands[:, nocc].min()
    assert np.isfinite(bands).all()
    assert cbm > vbm

def test_gfn1_kxtb_lattice_gradient(setup):
    """dE/da through the lattice-shift JVP of the lattice integrals."""
    import numpy
    numbers = [14,14]
    coords = numpy.array([[0.0, 0.0, 0.0],
                          [1.3467560987, 1.3467560987, 1.3467560987]]) / BOHR
    a0 = numpy.array([[0.0, 2.6935121974, 2.6935121974],
                      [2.6935121974, 0.0, 2.6935121974],
                      [2.6935121974, 2.6935121974, 0.0]]) / BOHR
    basis, param = setup

    cell0 = Cell(numbers=numbers, coords=coords, a=a0,
                 basis=basis, precision=1e-6)
    nimgs = numpy.asarray(cell0.nimgs)
    rcut = float(cell0.rcut)

    def cell_energy(a):
        cell = Cell(numbers=numbers, coords=coords, a=a, rcut=rcut,
                    nimgs=nimgs, basis=basis, precision=1e-6)
        mf = GFN1KXTB(cell, param=param)   # Gamma point
        mf.diis = "anderson"
        mf.conv_tol = 1e-12
        return mf.kernel()

    g_ad = jax.grad(cell_energy)(np.asarray(a0))

    h = 1e-5
    for (i, j) in ((0, 1), (2, 2)):
        ap = a0.copy(); ap[i, j] += h
        am = a0.copy(); am[i, j] -= h
        fd = (cell_energy(np.asarray(ap)) - cell_energy(np.asarray(am))) / (2*h)
        assert abs(g_ad[i, j] - fd) < 1e-6 * max(abs(fd), 1.0)

# Body executed under PYSCFAD_FLOATX=float32 in a subprocess (see the
# ``run_fp32`` fixture). Runs the k-point SCF energy/force for every system in
# ``IN`` and emits the results keyed by the system name; the parent compares
# them against the FP64 references. Both reference structures have vanishing
# forces by symmetry.
_FP32_KXTB_BODY = """
from pyscfad.pbc.gto import CellLite as Cell
from pyscfad.xtb.kxtb import GFN1KXTB
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb.param import GFN1Param

basis = xtb_basis.get_basis_filename()
param = GFN1Param()

out = {}
for name, sys in IN.items():
    numbers = sys["numbers"]
    coords = np.asarray(sys["coords"])
    a = np.asarray(sys["a"])
    sigma = sys["sigma"]

    def energy(coords, numbers=numbers, a=a, sigma=sigma):
        cell = Cell(numbers=numbers, coords=coords, a=a,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.sigma = sigma
        mf.diis = "anderson"
        mf.diis_damp = .5
        mf.diis_space = 6
        mf.conv_tol = 1e-5
        return mf.kernel()

    e, g = jax.value_and_grad(energy)(coords)
    out[name] = {"e": float(e), "g": numpy.asarray(g).tolist()}

emit(**out)
"""

def test_gfn1_kxtb_energy_force_fp32(run_fp32):
    # Si diamond (insulator, no smearing) and Cu (with Fermi smearing);
    # FP64 references from test_gfn1_kxtb_energy_force_with_kpts_sample and
    # test_gfn1_kxtb_smearing.
    systems = {
        "si": {
            "numbers": [14, 14],
            "coords": (np.asarray([[0.0, 0.0, 0.0],
                                   [1.3467560987]*3]) / BOHR),
            "a": (np.asarray([[0.0, 2.6935121974, 2.6935121974],
                              [2.6935121974, 0.0, 2.6935121974],
                              [2.6935121974, 2.6935121974, 0.0]]) / BOHR),
            "sigma": None,
            "e0": -3.83117442207487,
        },
        "cu": {
            "numbers": [29, 29],
            "coords": np.asarray(
                [[0.        , 0.        , 0.        ],
                 [2.40522868, 2.40522868, 3.40150702],]),
            "a": np.asarray(
                [[4.81045737, 0.        , 0.        ],
                 [0.        , 4.81045737, 0.        ],
                 [0.        , 0.        , 6.80301405],]),
            "sigma": 0.001,
            "e0": -9.22283523848542,
        },
    }

    out = run_fp32(
        _FP32_KXTB_BODY,
        **{name: {"numbers": sys["numbers"], "coords": sys["coords"],
                  "a": sys["a"], "sigma": sys["sigma"]}
           for name, sys in systems.items()},
    )

    for name, sys in systems.items():
        e0 = sys["e0"]
        e1 = out[name]["e"]
        g1 = np.asarray(out[name]["g"])
        assert abs(e1 - e0) / abs(e0) < 1e-6
        # forces vanish by symmetry; float32 resolves them to ~1e-4
        assert abs(g1).max() < 1e-3

def test_gfn1_kxtb_smearing(setup):
    basis, param = setup
    numbers = [29, 29]
    coords = np.asarray(
        [[0.        , 0.        , 0.        ],
         [2.40522868, 2.40522868, 3.40150702],]
    )
    a = np.asarray(
        [[4.81045737, 0.        , 0.        ],
         [0.        , 4.81045737, 0.        ],
         [0.        , 0.        , 6.80301405],]
    )

    def cell_energy(coords):
        cell = Cell(numbers=numbers, coords=coords, a=a,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.sigma = 0.001
        mf.diis = "anderson"
        mf.diis_damp = .5
        mf.diis_space = 6
        return mf.kernel()

    e0 = -9.22283523848542
    g0 = np.zeros((2,3))

    e1, g1 = jax.value_and_grad(cell_energy)(coords)
    assert abs(e1 - e0) < 1e-6
    assert abs(g1 - g0).max() < 1e-6
