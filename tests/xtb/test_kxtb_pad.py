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
NATM = 3          # padded atoms per cell
RCUT = 15.0       # reduced lattice-sum cutoff shared by both code paths
KMESH = [2, 2, 2]

# Si and C diamond primitive cells (Bohr)
A_SI = numpy.array([[0.0, 2.6935121974, 2.6935121974],
                    [2.6935121974, 0.0, 2.6935121974],
                    [2.6935121974, 2.6935121974, 0.0]]) / BOHR
COORDS_SI = numpy.array([[0.0, 0.0, 0.0], [1.3467560987] * 3]) / BOHR
A_C = numpy.array([[0.0, 1.7845, 1.7845],
                   [1.7845, 0.0, 1.7845],
                   [1.7845, 1.7845, 0.0]]) / BOHR
COORDS_C = numpy.array([[0.0, 0.0, 0.0], [0.89225] * 3]) / BOHR


@pytest.fixture(scope="module")
def setup():
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=MAX_NUMBER)
    param = make_param_array(basis, max_number=MAX_NUMBER)

    cells = [
        CellLite(numbers=[14, 14], coords=COORDS_SI, a=A_SI, basis=bfile,
                 rcut=RCUT, precision=1e-6),
        CellLite(numbers=[6, 6], coords=COORDS_C, a=A_C, basis=bfile,
                 rcut=RCUT, precision=1e-6),
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

    e_si, bands_si, path_si = _ref(bfile, nimgs, ewald_mesh, scaled_kpts,
                                   [14, 14], COORDS_SI, A_SI)
    e_c, bands_c, path_c = _ref(bfile, nimgs, ewald_mesh, scaled_kpts,
                                [6, 6], COORDS_C, A_C)

    def pad(arr):
        out = numpy.zeros((NATM,) + numpy.asarray(arr).shape[1:])
        out[: len(arr)] = arr
        return out

    # batch: Si + C + a fully padded (empty) cell
    numbers = np.asarray([[14, 14, 0], [6, 6, 0], [0, 0, 0]], dtype=np.int32)
    coords = np.asarray([pad(COORDS_SI), pad(COORDS_C), numpy.zeros((NATM, 3))])
    a_batch = np.asarray([A_SI, A_C, numpy.eye(3)])
    kpts = np.asarray([_abs_kpts(scaled_kpts, a) for a in a_batch])

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

    # energy and gradient parity with the unbatched reference
    assert abs(e[0] - e_si) < 1e-6
    assert abs(e[1] - e_c) < 1e-6
    assert abs(e[2]) < 1e-12
    assert numpy.isfinite(numpy.asarray(g)).all()
    # ideal diamond: forces vanish by symmetry
    assert numpy.abs(numpy.asarray(g)[:2]).max() < 1e-5

    # occupied valence band structure parity (eigenvalues are first order in
    # the SCF density error, hence the looser tolerance)
    band = numpy.asarray(band)
    pick = numpy.asarray(pick)
    for i, ref_bands in enumerate((bands_si, bands_c)):
        got = band[i][pick[i]].reshape(nk, -1)
        assert got.shape == ref_bands.shape
        assert numpy.abs(got - ref_bands).max() < 1e-5
    assert pick[2].sum() == 0

    # get_bands parity along the fractional band path
    path_bands = numpy.asarray(path_bands)
    for i, ref_path in enumerate((path_si, path_c)):
        assert numpy.abs(path_bands[i] - ref_path).max() < 1e-5
    assert numpy.isfinite(path_bands[2]).all()

    # variable atomic numbers reuse the same jitted executable
    perm = numpy.array([1, 0, 2])
    (e_swap, _), _ = gfn(numbers[perm], coords[perm], a_batch[perm],
                         kpts[perm], kpts_band[perm])
    assert abs(e_swap[0] - e_c) < 1e-6
    assert abs(e_swap[1] - e_si) < 1e-6
    assert gfn._cache_size() == 1
