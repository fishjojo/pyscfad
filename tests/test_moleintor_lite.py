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

"""Tests for the jittable :mod:`~pyscfad.gto.moleintor_lite` integral jvps.

The migrated ``int1e_nuc`` (rinv-at-nucleus operator term) and ``int2e`` (``s1``)
coordinate derivatives are checked against the established legacy ``pyscfad.gto``
analytic gradients and PySCF, plus jit/finite-difference consistency.
"""
import pytest
import numpy
import jax
from pyscfad import numpy as np
from pyscfad import gto
from pyscfad.gto import MoleLite

BASIS = "sto3g"


@pytest.fixture
def water():
    symbols = ("O", "H", "H")
    coords = numpy.array([[0.0, 0.0, 0.0],
                          [0.0, -1.43, 1.1],
                          [0.0, 1.43, 1.1]])
    return symbols, coords


def _lite(coords, symbols):
    return MoleLite(symbols=symbols, coords=coords, basis=BASIS,
                    trace_coords=True, verbose=0)


def _legacy(coords, symbols):
    atom = list(zip(symbols, numpy.asarray(coords).tolist()))
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.unit = "Bohr"
    mol.verbose = 0
    mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)
    return mol


def _fd_grad(f, coords, eps=1e-5):
    g = numpy.zeros_like(coords)
    for ia in range(coords.shape[0]):
        for x in range(3):
            cp = coords.copy(); cp[ia, x] += eps
            cm = coords.copy(); cm[ia, x] -= eps
            g[ia, x] = (float(f(cp)) - float(f(cm))) / (2 * eps)
    return g


@pytest.mark.parametrize("intor", ["int1e_ovlp", "int1e_kin", "int1e_nuc"])
def test_int1e_forward(water, intor):
    symbols, coords = water
    mol = _lite(coords, symbols)
    ref = mol.to_pyscf().intor_symmetric(intor)
    val = numpy.asarray(mol.intor_symmetric(intor))
    assert abs(val - ref).max() < 1e-10


def test_int2e_forward(water):
    symbols, coords = water
    mol = _lite(coords, symbols)
    ref = mol.to_pyscf().intor("int2e")
    val = numpy.asarray(mol.intor("int2e"))
    assert abs(val - ref).max() < 1e-10


@pytest.mark.parametrize("intor", ["int1e_nuc"])
@pytest.mark.parametrize("hermi", [0, 1])
def test_int1e_nuc_grad(water, intor, hermi):
    symbols, coords = water

    def f_lite(c):
        return np.sum(_lite(c, symbols).intor(intor, hermi=hermi) ** 2)

    def f_leg(mol):
        return np.sum(mol.intor(intor, hermi=hermi) ** 2)

    g_lite = numpy.asarray(jax.grad(f_lite)(coords))
    g_leg = numpy.asarray(jax.grad(f_leg)(_legacy(coords, symbols)).coords)
    g_jit = numpy.asarray(jax.jit(jax.grad(f_lite))(coords))

    # analytic lite vs analytic legacy, and jit vs eager
    assert abs(g_lite - g_leg).max() < 1e-8
    assert abs(g_jit - g_lite).max() < 1e-10
    # finite-difference sanity
    assert abs(g_lite - _fd_grad(f_lite, coords)).max() < 1e-6


def test_int2e_grad(water):
    symbols, coords = water

    def f_lite(c):
        return np.sum(_lite(c, symbols).intor("int2e") ** 2)

    def f_leg(mol):
        return np.sum(mol.intor("int2e", aosym="s1") ** 2)

    g_lite = numpy.asarray(jax.grad(f_lite)(coords))
    g_leg = numpy.asarray(jax.grad(f_leg)(_legacy(coords, symbols)).coords)
    g_jit = numpy.asarray(jax.jit(jax.grad(f_lite))(coords))

    assert abs(g_lite - g_leg).max() < 1e-8
    assert abs(g_jit - g_lite).max() < 1e-10
    assert abs(g_lite - _fd_grad(f_lite, coords)).max() < 1e-6


def test_higher_order_not_implemented(water):
    symbols, coords = water

    def f_nuc(c):
        return np.sum(_lite(c, symbols).intor("int1e_nuc", hermi=1) ** 2)

    def f_eri(c):
        return np.sum(_lite(c, symbols).intor("int2e") ** 2)

    with pytest.raises(NotImplementedError):
        jax.jacfwd(jax.grad(f_nuc))(coords)
    with pytest.raises(NotImplementedError):
        jax.jacfwd(jax.grad(f_eri))(coords)
