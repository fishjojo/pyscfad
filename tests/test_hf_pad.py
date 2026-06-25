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

"""Tests for the batched (padded) RHF/UHF in :mod:`~pyscfad.ml.scf.hf_pad`."""
import pytest
import numpy
import jax

from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf import hf_pad

BASIS_NAME = "sto-3g"

# H2O padded to 4 atoms; NH3 fills all 4 (coordinates in Bohr)
H2O_Z = numpy.array([8, 1, 1, 0], dtype=numpy.int32)
H2O_R = numpy.array([[0., 0., 0.],
                     [0., -1.430564, 1.109132],
                     [0.,  1.430564, 1.109132],
                     [0., 0., 0.]])
NH3_Z = numpy.array([7, 1, 1, 1], dtype=numpy.int32)
NH3_R = numpy.array([[0.,  0.,      0.221],
                     [0.,  1.771,  -0.515],
                     [1.534, -0.885, -0.515],
                     [-1.534, -0.885, -0.515]])

@pytest.fixture(scope="module")
def basis():
    return make_basis_array(BASIS_NAME, 10)

def _pyscf_ref(numbers, coords, spin=0):
    import pyscf
    from pyscf.data.elements import _symbol
    atom = [(_symbol(int(z)), tuple(r))
            for z, r in zip(numbers, coords) if int(z) > 0]
    p = pyscf.M(atom=atom, basis=BASIS_NAME, unit="Bohr", spin=spin, verbose=0)
    mf = (p.UHF() if spin else p.RHF()).run()
    return mf.e_tot, mf.nuc_grad_method().kernel()

def _energy_fn(basis, cls, spin=0):
    def efn(numbers, coords):
        m = MolePad(numbers, coords, basis=basis, spin=spin, verbose=0, trace_coords=True)
        mf = cls(m)
        mf.max_cycle = 100
        mf.conv_tol = 1e-11
        return mf.kernel()
    return efn

def test_pad_rhf_matches_unpadded(basis):
    efn = _energy_fn(basis, hf_pad.RHF)
    e = float(efn(H2O_Z, np.asarray(H2O_R)))
    eref, _ = _pyscf_ref(H2O_Z, H2O_R)
    assert abs(e - eref) < 1e-9

def test_pad_rhf_padding_atom_grad_is_zero(basis):
    efn = _energy_fn(basis, hf_pad.RHF)
    g = numpy.asarray(jax.grad(efn, argnums=1)(H2O_Z, np.asarray(H2O_R)))
    # the trailing (padding) atom must not contribute any force
    assert abs(g[3]).max() < 1e-10
    # and the real-atom forces match PySCF
    _, gref = _pyscf_ref(H2O_Z, H2O_R)
    assert abs(g[:3] - gref).max() < 1e-5

def test_pad_rhf_batched(basis):
    efn = _energy_fn(basis, hf_pad.RHF)
    numbers = numpy.stack([H2O_Z, NH3_Z])
    coords = np.stack([np.asarray(H2O_R), np.asarray(NH3_R)])
    eg = jax.jit(jax.vmap(jax.value_and_grad(efn, argnums=1)))(numbers, coords)
    energies, grads = eg
    assert grads.shape == coords.shape
    for i, (z, r) in enumerate([(H2O_Z, H2O_R), (NH3_Z, NH3_R)]):
        eref, gref = _pyscf_ref(z, r)
        assert abs(float(energies[i]) - eref) < 1e-9
        nreal = int((z > 0).sum())
        assert abs(numpy.asarray(grads[i])[:nreal] - gref).max() < 1e-5

def test_pad_uhf_open_shell(basis):
    # OH doublet padded to 3 atoms
    z = numpy.array([8, 1, 0], dtype=numpy.int32)
    r = numpy.array([[0., 0., 0.], [0., 0., 1.834345], [0., 0., 0.]])
    efn = _energy_fn(basis, hf_pad.UHF, spin=1)
    e = float(efn(z, np.asarray(r)))
    eref, gref = _pyscf_ref(z, r, spin=1)
    assert abs(e - eref) < 1e-9
    g = numpy.asarray(jax.grad(efn, argnums=1)(z, np.asarray(r)))
    assert abs(g[2]).max() < 1e-10
    assert abs(g[:2] - gref).max() < 1e-5

def test_pad_uhf_batched(basis):
    z = numpy.array([8, 1, 0], dtype=numpy.int32)
    r0 = numpy.array([[0., 0., 0.], [0., 0., 1.834345], [0., 0., 0.]])
    r1 = numpy.array([[0., 0., 0.], [0., 0., 1.95], [0., 0., 0.]])
    efn = _energy_fn(basis, hf_pad.UHF, spin=1)
    numbers = numpy.stack([z, z])
    coords = np.stack([np.asarray(r0), np.asarray(r1)])
    energies = jax.jit(jax.vmap(efn))(numbers, coords)
    for i, r in enumerate([r0, r1]):
        eref, _ = _pyscf_ref(z, r, spin=1)
        assert abs(float(energies[i]) - eref) < 1e-9
