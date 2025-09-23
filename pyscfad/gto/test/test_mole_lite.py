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

from functools import partial
import pytest
import jax
import pyscf
from pyscfad import numpy as np
from pyscfad.gto.mole import Mole
from pyscfad.gto.mole_lite import Mole as MoleLite

@pytest.fixture
def atom():
    yield "h1 0 0 0; h2 0 0 2"

@pytest.fixture
def basis():
    yield {"H1" : "sto3g", "H2" : "631G**"}

@pytest.fixture
def unit():
    yield "AU"

def int1e_norm(mol, intor="int1e_ovlp"):
    ints = mol.intor(intor,
                     shls_slice=(0, 1, 2, 3),
                     hermi=0)
    return np.linalg.norm(ints)

def int1e_norm1(coords, basis, intor="int1e_ovlp"):
    mol = MoleLite(
        symbols=("h1","h2"),
        coords=coords,
        basis=basis,
        trace_coords=True,
    )
    ints = mol.intor(intor,
                     shls_slice=(0, 1, 2, 3),
                     hermi=0)
    return np.linalg.norm(ints)

def test_int1e(atom, basis, unit):
    coords = np.array([[0,0,0],[0,0,2]], dtype=float)
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    for intor in ["int1e_ovlp", "int1e_kin"]:
        fn = partial(int1e_norm, intor=intor)
        fn1 = lambda x: int1e_norm1(x, basis, intor=intor)
        assert abs(fn1(coords) - fn(mol)) < 1e-6

        gfn = jax.grad(fn)
        gfn1 = jax.jit(jax.grad(fn1))
        grad = gfn(mol).coords
        grad1 = gfn1(coords)
        assert abs(grad1 - grad).max() < 1e-6

        hfn = jax.jacfwd(jax.grad(fn))
        hfn1 = jax.jit(jax.jacfwd(jax.grad(fn1)))
        hess = hfn(mol).coords.coords
        hess1 = hfn1(coords)
        assert abs(hess1 - hess).max() < 1e-6

def test_from_to_pyscf(atom, basis, unit):
    pmol = pyscf.M(atom=atom, basis=basis, unit=unit)
    mol = MoleLite.from_pyscf(pmol)
    pmol1 = mol.to_pyscf()

    assert jax.tree.all(jax.tree.map(np.allclose, pmol._basis, pmol1._basis))
