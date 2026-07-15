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

"""Basis-set parameter derivatives of the PBC lattice / k-point integrals."""
import numpy
import pytest
import jax

from pyscf.data.nist import BOHR

from pyscfad import numpy as np
from pyscfad.pbc.gto import CellLite
from pyscfad.pbc.gto import _latintor
from pyscfad.pbc.gto import _pbcintor_lite
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

A_SI = numpy.array([[0.0, 2.6935, 2.6935],
                    [2.6935, 0.0, 2.6935],
                    [2.6935, 2.6935, 0.0]]) / BOHR
COORDS_SI = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
RCUT = 8.0


@pytest.fixture(scope="module")
def cell():
    return CellLite(numbers=[14, 14], coords=COORDS_SI, a=A_SI,
                    basis="gth-szv", rcut=RCUT, precision=1e-6, verbose=0)


def _basis_slots(cell):
    _, _, exp_of = setup_exp(cell)
    _, _, cs_of = setup_ctr_coeff(cell)
    return numpy.concatenate([exp_of, cs_of])


def four_point_fd(f, env, slot, disp=1e-4):
    env = numpy.asarray(env, dtype=float)

    def at(d):
        env1 = env.copy()
        env1[slot] += d
        return numpy.asarray(f(env1))

    return (8.0 * (at(disp) - at(-disp))
            - (at(2 * disp) - at(-2 * disp))) / (12.0 * disp)


@pytest.mark.parametrize("hermi", [0, 1])
def test_lattice_intor_basis_jvp_fd(cell, hermi):
    Ls = numpy.asarray(cell.Ls, dtype=float).reshape(-1, 3)
    Ls_mask = numpy.ones(len(Ls), dtype=numpy.int32)
    atm, bas = cell._atm, cell._bas
    env = numpy.asarray(cell._env)

    def f(env_):
        return _latintor._lattice_intor(
            "int1e_ovlp_sph", Ls, Ls_mask, atm, bas, env_,
            hermi=hermi, trace_coords=True, trace_basis=True,
        )

    jac_fwd = numpy.asarray(jax.jacfwd(f)(np.asarray(env)))
    jac_rev = numpy.asarray(jax.jacrev(f)(np.asarray(env)))
    assert abs(jac_fwd - jac_rev).max() < 1e-12

    for slot in _basis_slots(cell)[::3]:
        fd = four_point_fd(f, env, slot)
        assert abs(jac_fwd[..., slot] - fd).max() < 1e-8


@pytest.mark.parametrize("hermi", [0, 1])
def test_pbc_intor_basis_jvp_fd(cell, hermi):
    atm, bas = cell._atm, cell._bas
    env = numpy.asarray(cell._env)
    a = numpy.asarray(cell.a, dtype=float)
    kpts = numpy.asarray(cell.make_kpts([2, 1, 1]), dtype=float)

    def f(env_):
        return _pbcintor_lite._pbc_intor(
            "int1e_ovlp_sph", a, kpts, RCUT, atm, bas, env_,
            hermi=hermi, trace_coords=True, trace_basis=True,
        )

    jac_fwd = numpy.asarray(jax.jacfwd(f)(np.asarray(env)))
    for slot in _basis_slots(cell)[::3]:
        fd = four_point_fd(f, env, slot)
        assert abs(jac_fwd[..., slot] - fd).max() < 1e-8

    # reverse mode through a real scalar loss
    def loss(env_):
        s = f(env_)
        return np.sum(np.abs(s) ** 2)

    g_rev = numpy.asarray(jax.grad(loss)(np.asarray(env)))
    g_fwd = numpy.asarray(jax.jacfwd(loss)(np.asarray(env)))
    assert abs(g_rev - g_fwd).max() < 1e-10


def test_cell_lite_basis_grad(cell):
    """kxtb-shaped end-to-end check: gradient of a lattice-overlap loss
    w.r.t. the raw basis parameters of a CellLite."""
    Ls = numpy.asarray(cell.Ls, dtype=float).reshape(-1, 3)
    basis = cell.basis
    numbers = [14, 14]

    def loss(basis_):
        cell_ = CellLite(numbers=numbers, coords=COORDS_SI, a=A_SI,
                         basis=basis_, rcut=RCUT, precision=1e-6,
                         verbose=0, trace_basis=True)
        s1e_lat = cell_.lattice_intor("int1e_ovlp", hermi=1, Ls=Ls)
        return np.sum(s1e_lat ** 2)

    grad = jax.grad(loss)(basis)
    leaves, treedef = jax.tree.flatten(basis)
    grad_leaves = jax.tree.leaves(grad)

    # 2-point FD on a few basis-parameter leaves
    checked = 0
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        for flat in range(0, leaf.size, max(1, leaf.size // 2)):
            if checked >= 6:
                break
            checked += 1
            disp = 1e-4 * max(1.0, abs(leaf.flat[flat]))

            def at(d):
                leaf1 = leaf.copy()
                leaf1.flat[flat] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return float(loss(jax.tree.unflatten(treedef, leaves1)))

            fd = (at(disp) - at(-disp)) / (2 * disp)
            got = numpy.asarray(grad_leaves[i]).flat[flat]
            assert abs(got - fd) < 1e-6 * max(1.0, abs(fd))
