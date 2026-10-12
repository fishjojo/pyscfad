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

from functools import partial
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
from pyscfad.xtb.util import ke_cutoff_ewald

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    yield basis, param

@pytest.fixture
def cell_H3O_cation(setup):
    """H3O+ in each cell of a 9 Bohr cubic lattice sampled with 2 k-points.
    Yields a cell builder (cell.charge is that of the k-point supercell), the
    k-points, and the static Ewald mesh for a given alpha, so that the SCF can be jitted."""
    basis, _ = setup
    numbers = [8, 1, 1, 1]
    coords = np.array([[0., 0., 0.], [1.43355, 0., -0.95296],
                       [1.43355, 0., 0.95296], [-0.8, 0.1, 1.6]])
    a = np.eye(3) * 9.
    cell0 = Cell(numbers=numbers, coords=coords, a=a, basis=basis, rcut=12., precision=1e-6)
    nimgs = tuple(int(x) for x in cell0.nimgs)
    kpts = cell0.make_kpts([2, 1, 1])

    def make_cell():
        return Cell(numbers=numbers, coords=coords, a=a, basis=basis, rcut=12.,
                    nimgs=nimgs, precision=1e-6, charge=len(kpts))

    def ewald_mesh(alpha):
        return tuple(int(x) for x in cell0.cutoff_to_mesh(
            ke_cutoff_ewald(alpha, cell0.precision * cell0.vol)))

    yield make_cell, kpts, ewald_mesh

def test_gfn1_kxtb_energy_force(setup, H2O_GFN1_ref):
    basis, param = setup
    numbers, coords, *_ = H2O_GFN1_ref

    def mol_energy(coords, diis):
        mol = Mole(numbers=numbers, coords=coords, basis=basis)
        mf = GFN1XTB(mol, param=param)
        mf.diis = diis
        return mf.kernel()

    def cell_energy(coords, diis):
        cell = Cell(numbers=numbers, coords=coords, a=np.eye(3)*20., rcut=22.,
                    basis=basis, precision=1e-6)
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
                    basis=basis, precision=1e-6)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.diis = "anderson"
        return mf.kernel()

    # reference with periodic coordination numbers (cn_d3 over cell.Ls);
    # forces vanish by symmetry for the ideal diamond structure
    e0 = -3.62605206380876
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
                    basis=basis, precision=1e-6)
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
            "e0": -3.62605206380876,
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
            "e0": -9.066494431417272,
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
        assert abs(e1 - e0) / abs(e0) < 1e-5
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
                    basis=basis, precision=1e-6)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.sigma = 0.001
        mf.diis = "anderson"
        mf.diis_damp = .5
        mf.diis_space = 6
        return mf.kernel()

    e0 = -9.066494431417272
    g0 = np.zeros((2,3))

    e1, g1 = jax.value_and_grad(cell_energy)(coords)
    assert abs(e1 - e0) < 1e-6
    assert abs(g1 - g0).max() < 1e-6

def test_gfn1_kxtb_charged_kpts_qbroyden(setup, cell_H3O_cation):
    """Charged cell with several k-points: the charge-Broyden solver normalizes the
    per-cell shell charges to cell.charge / nkpts and agrees with Anderson mixing."""
    _, param = setup
    make_cell, kpts, ewald_mesh = cell_H3O_cation
    mesh = ewald_mesh(GFN1KXTB.ewald_alpha)

    @partial(jax.jit, static_argnums=0)
    def energy(diis):
        mf = GFN1KXTB(make_cell(), param=param, kpts=kpts)
        mf.ewald_mesh = mesh
        mf.diis = diis
        mf.conv_tol = 1e-10
        e = mf.kernel()
        return e, mf.converged

    e0, _ = energy("anderson")
    e1, conv = energy("qbroyden")
    assert conv
    assert abs(e1 - e0) < 1e-8

def test_gfn1_kxtb_charged_mo_energy_ewald_alpha(setup, cell_H3O_cation):
    """ewald_alpha only splits the lattice Coulomb sum. For a charged cell the
    neutralizing background must enter the shell potential as well as the energy;
    otherwise the MO energies shift by pi Q / (alpha^2 V) (1.5e-2 Eh between the two
    alphas here) while the energy does not."""
    _, param = setup
    make_cell, kpts, ewald_mesh = cell_H3O_cation

    def run(alpha):
        mesh = ewald_mesh(alpha)

        @jax.jit
        def kernel():
            mf = GFN1KXTB(make_cell(), param=param, kpts=kpts)
            mf.ewald_alpha = alpha
            mf.ewald_mesh = mesh
            mf.diis = "anderson"
            mf.conv_tol = 1e-10
            e = mf.kernel()
            return e, mf.mo_energy
        return kernel()

    # both erfc cutoffs (9.7 and 6.3 Bohr) are within the 12 Bohr image set
    e0, mo0 = run(0.4)
    e1, mo1 = run(0.6)
    assert abs(e1 - e0) < 1e-6
    assert abs(mo1 - mo0).max() < 1e-6

def test_gfn1_kxtb_supercell_consistency(setup):
    """A primitive cell with 2 k-points and the 2x1x1 supercell at Gamma describe the
    same crystal, so E(supercell) = 2 E(primitive). The short (6 Bohr) lattice vector
    makes the pairs of an atom with its own periodic images matter; these pairs are
    off-site, as the corresponding pairs of different atoms in the supercell. The
    charged case also checks the per-cell charge (cell.charge is that of the k-point
    supercell, so both cells carry charge 2)."""
    import numpy
    basis, param = setup
    numbers = numpy.array([8, 1, 1, 1])
    coords = numpy.array([[0., 0., 0.], [1.43355, 0., -0.95296],
                          [1.43355, 0., 0.95296], [-0.8, 0.1, 1.6]])
    a_prim = numpy.diag([6., 9., 9.])
    a_super = numpy.diag([12., 9., 9.])
    shift = numpy.array([6., 0., 0.])
    rcut, precision = 16., 1e-8

    def energy(numbers, coords, a, charge, kmesh):
        cell0 = Cell(numbers=numbers, coords=coords, a=a, basis=basis, rcut=rcut,
                     precision=precision)
        nimgs = tuple(int(x) for x in cell0.nimgs)
        mesh = tuple(int(x) for x in cell0.cutoff_to_mesh(
            ke_cutoff_ewald(GFN1KXTB.ewald_alpha, precision * cell0.vol)))
        kpts = cell0.make_kpts(kmesh)

        @jax.jit
        def e():
            cell = Cell(numbers=numbers, coords=coords, a=a, basis=basis, rcut=rcut,
                        nimgs=nimgs, precision=precision, charge=charge)
            mf = GFN1KXTB(cell, param=param, kpts=kpts)
            mf.ewald_mesh = mesh
            mf.diis = "anderson"
            mf.conv_tol = 1e-12
            return mf.kernel()
        return e()

    for natm, charge in ((3, 0), (4, 2)):   # H2O, H3O+
        n, c = numbers[:natm], coords[:natm]
        e_prim = energy(n, c, a_prim, charge, [2, 1, 1])
        e_super = energy(numpy.hstack([n, n]), numpy.vstack([c, c + shift]),
                         a_super, charge, [1, 1, 1])
        assert abs(e_super - 2 * e_prim) < 1e-6
