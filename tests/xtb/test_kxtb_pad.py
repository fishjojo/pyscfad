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

import numpy
import pytest
import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.kxtb import GFN1KXTB as GFN1KXTBRef
from pyscfad.xtb.util import ke_cutoff_ewald
from pyscfad.pbc.gto import CellLite
from pyscfad.ml.gto import make_basis_array
from pyscfad.ml.xtb import GFN1KXTB, make_param_array
from pyscfad.ml.pbc.gto import CellPad
from pyscfad.ml.pbc.gto.cell_pad import make_image_grid

MAX_NUMBER = 14
NATM = 4          # padded atoms per cell
RCUT = 15.0       # reduced lattice-sum cutoff shared by both code paths
KMESH = [2, 2, 2]

# Systems with different numbers of atoms (all insulators; Bohr):
# Si diamond primitive (2 atoms), a C diamond 1x1x2 supercell (4 atoms),
# and fcc Ne (1 atom).
A_SI = numpy.array([[0.0, 2.6935121974, 2.6935121974],
                    [2.6935121974, 0.0, 2.6935121974],
                    [2.6935121974, 2.6935121974, 0.0]]) / BOHR
COORDS_SI = numpy.array([[0.0, 0.0, 0.0], [1.3467560987] * 3]) / BOHR

_A3_C = numpy.array([1.7845, 1.7845, 0.0])
A_C2 = numpy.array([[0.0, 1.7845, 1.7845],
                    [1.7845, 0.0, 1.7845],
                    [2 * 1.7845, 2 * 1.7845, 0.0]]) / BOHR
COORDS_C2 = numpy.array([
    [0.0, 0.0, 0.0],
    [0.89225, 0.89225, 0.89225],
    [0.0, 0.0, 0.0] + _A3_C,
    [0.89225, 0.89225, 0.89225] + _A3_C,
]) / BOHR

A_NE = 2.215 * numpy.array([[0.0, 1.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0]]) / BOHR
COORDS_NE = numpy.zeros((1, 3))

SYSTEMS = (
    ([14, 14], COORDS_SI, A_SI),
    ([6, 6, 6, 6], COORDS_C2, A_C2),
    ([10], COORDS_NE, A_NE),
)


@pytest.fixture(scope="module")
def setup():
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=MAX_NUMBER)
    param = make_param_array(basis, max_number=MAX_NUMBER)

    cells = [
        CellLite(numbers=list(zs), coords=coords, a=a, basis=bfile,
                 rcut=RCUT, precision=1e-6)
        for zs, coords, a in SYSTEMS
    ]
    nimgs = numpy.max([numpy.asarray(c.nimgs) for c in cells], axis=0)
    Ts = make_image_grid(nimgs)
    ke = ke_cutoff_ewald(0.4, 1e-6 * min(float(c.vol) for c in cells))
    ewald_mesh = numpy.max(
        [numpy.asarray(c.cutoff_to_mesh(ke)) for c in cells], axis=0)
    scaled_kpts = numpy.asarray(
        cells[0].get_scaled_kpts(cells[0].make_kpts(KMESH)))
    yield bfile, basis, param, nimgs, Ts, ewald_mesh, scaled_kpts


def _abs_kpts(scaled_kpts, a):
    b = 2.0 * numpy.pi * numpy.linalg.inv(numpy.asarray(a).T)
    return scaled_kpts @ b


# fractional band path used for the get_bands parity check
SCALED_BAND_PATH = numpy.array([[0.0, 0.0, 0.0],
                                [0.25, 0.0, 0.25],
                                [0.5, 0.0, 0.5]])
N_LOWEST_BANDS = 8


def _ref(bfile, nimgs, ewald_mesh, scaled_kpts, numbers, coords, a):
    """Unbatched reference with the same static grids as the padded path."""
    cell = CellLite(numbers=numbers, coords=coords, a=a, rcut=RCUT,
                    nimgs=nimgs, basis=bfile, precision=1e-6,
                    trace_coords=True, verbose=0)
    mf = GFN1KXTBRef(cell, param=GFN1Param(),
                     kpts=_abs_kpts(scaled_kpts, a))
    mf.ewald_mesh = ewald_mesh
    mf.diis = "anderson"
    mf.conv_tol = 1e-10
    e = mf.kernel()
    nocc = int(numpy.asarray(mf.tot_electrons)) // 2 // len(scaled_kpts)
    bands = numpy.sort(numpy.asarray(mf.mo_energy), axis=1)[:, :nocc]
    path_energy, _ = mf.get_bands(_abs_kpts(SCALED_BAND_PATH, a))
    path_bands = numpy.sort(
        numpy.asarray(path_energy).real, axis=1)[:, :N_LOWEST_BANDS]
    return float(e), bands, path_bands


def test_gfn1_kxtb_pad_energy_force_bands(setup):
    bfile, basis, param, nimgs, Ts, ewald_mesh, scaled_kpts = setup
    nbas = basis.nbas
    nk = len(scaled_kpts)

    refs = [
        _ref(bfile, nimgs, ewald_mesh, scaled_kpts, list(zs), coords, a)
        for zs, coords, a in SYSTEMS
    ]

    def pad(arr, fill=0):
        arr = numpy.asarray(arr)
        out = numpy.full((NATM,) + arr.shape[1:], float(fill))
        out[: len(arr)] = arr
        return out

    # batch: 2-atom Si + 4-atom C supercell + 1-atom Ne + an empty cell
    numbers = np.asarray(
        [pad(zs).astype(int) for zs, _, _ in SYSTEMS]
        + [numpy.zeros(NATM, dtype=int)],
        dtype=np.int32,
    )
    coords = np.asarray(
        [pad(c) for _, c, _ in SYSTEMS] + [numpy.zeros((NATM, 3))]
    )
    a_batch = np.asarray([a for *_, a in SYSTEMS] + [numpy.eye(3)])
    kpts = np.asarray([_abs_kpts(scaled_kpts, a) for a in a_batch])
    n_solids = len(SYSTEMS)

    def energy(numbers, coords, a, kpts, kpts_band):
        Ls = np.asarray(Ts, dtype=np.float64) @ a
        cell = CellPad(numbers, coords, basis=basis, a=a, Ls=Ls, rcut=RCUT,
                       precision=1e-6, verbose=0, trace_coords=True)
        mf = GFN1KXTB(cell, param, kpts=kpts)
        mf.ewald_mesh = ewald_mesh
        mf.diis = "anderson"
        mf.conv_tol = 1e-10
        e = mf.kernel()

        band = mf.mo_energy.real
        mask = mf.mo_mask(band)
        nocc = mf.tot_electrons // 2
        flat = band.ravel()
        order = np.argsort(flat)
        m = mask.ravel()[order]
        pick_sorted = (np.cumsum(m) <= nocc) & m
        pick = np.zeros_like(pick_sorted).at[order].set(
            pick_sorted).reshape(band.shape)

        # bands at arbitrary k-points via the get_bands API
        path_energy, _ = mf.get_bands(kpts_band)
        path_bands = np.sort(path_energy.real, axis=1)[:, :N_LOWEST_BANDS]
        return e, (band, pick, path_bands)

    kpts_band = np.asarray([_abs_kpts(SCALED_BAND_PATH, a) for a in a_batch])
    gfn = jax.jit(jax.vmap(
        jax.value_and_grad(energy, argnums=1, has_aux=True),
        in_axes=(0, 0, 0, 0, 0)))
    (e, (band, pick, path_bands)), g = gfn(numbers, coords, a_batch, kpts,
                                           kpts_band)

    # energy and gradient parity with the unbatched references
    for i, (e_ref, _, _) in enumerate(refs):
        assert abs(e[i] - e_ref) < 1e-6
    assert abs(e[n_solids]) < 1e-12, "empty cell energy not zero"
    g = numpy.asarray(g)
    assert numpy.isfinite(g).all()
    # translation invariance: forces in every cell sum to zero
    assert numpy.abs(g[:n_solids].sum(axis=1)).max() < 1e-5
    # ideal primitive diamond / single-atom fcc: forces vanish by symmetry
    # (the C supercell's k-mesh folds asymmetrically onto the primitive
    # lattice, giving real k-sampling forces — verified identical to the
    # unbatched reference)
    assert numpy.abs(g[0]).max() < 1e-5
    assert numpy.abs(g[2]).max() < 1e-5

    # occupied valence band structure parity (eigenvalues are first order in
    # the SCF density error, hence the looser tolerance)
    band = numpy.asarray(band)
    pick = numpy.asarray(pick)
    for i, (_, ref_bands, _) in enumerate(refs):
        got = band[i][pick[i]].reshape(nk, -1)
        assert got.shape == ref_bands.shape
        assert numpy.abs(got - ref_bands).max() < 1e-5
    assert pick[n_solids].sum() == 0

    # get_bands parity along the fractional band path (each system exposes a
    # different number of real bands; compare up to the reference width)
    path_bands = numpy.asarray(path_bands)
    for i, (_, _, ref_path) in enumerate(refs):
        width = ref_path.shape[1]
        assert numpy.abs(path_bands[i][:, :width] - ref_path).max() < 1e-5
    assert numpy.isfinite(path_bands[n_solids]).all()

    # systems with different atomic numbers AND different atom counts swap
    # batch slots while reusing the same jitted executable
    perm = numpy.array([1, 2, 0, 3])
    (e_swap, _), _ = gfn(numbers[perm], coords[perm], a_batch[perm],
                         kpts[perm], kpts_band[perm])
    for slot, isys in enumerate(perm[:n_solids]):
        assert abs(e_swap[slot] - refs[isys][0]) < 1e-6
    assert gfn._cache_size() == 1
