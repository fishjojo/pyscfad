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

"""Basis-set parameter (exponent / contraction-coefficient) derivatives
of one-electron integrals on the array-level (lite) path.
"""
import dataclasses

import numpy
import pytest
import jax
import pyscf

from pyscfad import numpy as np
from pyscfad import gto
from pyscfad.gto import MoleLite, moleintor_lite
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

# Small basis covering the tricky cases: a general contraction (nctr=2),
# multiple primitives per shell, an l=2 shell, and (via the two H atoms)
# shells sharing the same env slots.
BASIS = {
    "O": [
        [0, [13.07, 0.15, 0.06], [2.38, 0.53, 0.35], [0.64, 0.44, 0.7]],
        [1, [5.03, 0.5], [1.16, 0.6]],
        [2, [0.8, 1.0]],
    ],
    "H": [
        [0, [3.42, 0.15], [0.62, 0.55]],
        [1, [0.9, 1.0]],
    ],
}

ATOM = "O 0 0 0.2; H 0.1 0.7 -0.5; H -0.1 -0.76 -0.4"


@pytest.fixture(scope="module")
def mol_ref():
    """Reference pyscf molecule providing concrete atm/bas/env arrays."""
    return pyscf.M(atom=ATOM, basis=BASIS, verbose=0)


def four_point_fd(f, env, slot, disp=1e-4):
    """4-point finite difference of ``f(env)`` w.r.t. ``env[slot]``."""
    env = numpy.asarray(env, dtype=float)

    def at(d):
        env1 = env.copy()
        env1[slot] += d
        return numpy.asarray(f(env1))

    return (8.0 * (at(disp) - at(-disp))
            - (at(2 * disp) - at(-2 * disp))) / (12.0 * disp)


def _basis_slots(mol_ref):
    """env indices of the (unique) exponents and contraction coefficients."""
    _, _, exp_of = setup_exp(mol_ref)
    _, _, cs_of = setup_ctr_coeff(mol_ref)
    return numpy.concatenate([exp_of, cs_of])


@pytest.mark.parametrize(
    "intor,comp,hermi",
    [
        ("int1e_ovlp_sph", None, 1),
        ("int1e_kin_sph", None, 0),
        ("int1e_r_sph", 3, 1),
        ("int1e_rinv_sph", None, 1),
        ("int1e_ovlp_dr10_sph", 3, 0),
        ("int1e_ovlp_cart", None, 1),
    ],
)
def test_int1e_basis_jvp_fd(mol_ref, intor, comp, hermi):
    atm, bas = mol_ref._atm, mol_ref._bas
    env = numpy.asarray(mol_ref._env)

    def f(env_):
        return moleintor_lite.getints(
            intor, atm, bas, env_, comp=comp, hermi=hermi,
            trace_coords=True, trace_basis=True,
        )

    jac_fwd = numpy.asarray(jax.jacfwd(f)(np.asarray(env)))
    jac_rev = numpy.asarray(jax.jacrev(f)(np.asarray(env)))
    assert abs(jac_fwd - jac_rev).max() < 1e-12

    slots = _basis_slots(mol_ref)
    for slot in slots[::3]:
        fd = four_point_fd(f, env, slot)
        assert abs(jac_fwd[..., slot] - fd).max() < 1e-8


def test_basis_jvp_vs_legacy(mol_ref):
    mol = gto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.build(trace_coords=False, trace_ctr_coeff=True, trace_exp=True)

    atm, bas = mol_ref._atm, mol_ref._bas
    env = np.asarray(mol_ref._env)
    _, _, exp_of = setup_exp(mol_ref)
    _, _, cs_of = setup_ctr_coeff(mol_ref)

    for intor, hermi in [("int1e_ovlp", 1), ("int1e_kin", 0),
                         ("int1e_ovlp_dr10", 0)]:
        jac_legacy = jax.jacfwd(mol.__class__.intor)(mol, intor)

        def f(env_):
            return moleintor_lite.getints(
                mol_ref._add_suffix(intor), atm, bas, env_, hermi=hermi,
                trace_coords=False, trace_basis=True,
            )

        jac = numpy.asarray(jax.jacfwd(f)(env))
        assert abs(jac[..., exp_of] - numpy.asarray(jac_legacy.exp)).max() < 1e-10
        assert abs(jac[..., cs_of] - numpy.asarray(jac_legacy.ctr_coeff)).max() < 1e-10


@pytest.fixture(scope="module")
def lite_args(mol_ref):
    symbols = tuple(mol_ref.atom_symbol(i) for i in range(mol_ref.natm))
    coords = np.asarray(mol_ref.atom_coords())
    basis = MoleLite(symbols=symbols, coords=coords, basis=BASIS).basis
    return symbols, coords, basis


def _fd_tree_grad(loss, basis, n_checks=6, disp=1e-4):
    """2-point FD gradient of ``loss(basis)`` at a few tree leaves,
    returned as a list of (leaf_idx, flat_idx, value)."""
    leaves, treedef = jax.tree.flatten(basis)
    out = []
    count = 0
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        for flat in range(0, leaf.size, max(1, leaf.size // 2)):
            if count >= n_checks:
                return out
            count += 1

            def at(d):
                leaf1 = leaf.copy()
                leaf1.flat[flat] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return float(loss(jax.tree.unflatten(treedef, leaves1)))

            out.append((i, flat, (at(disp) - at(-disp)) / (2 * disp)))
    return out


@pytest.mark.parametrize("cart", [False, True])
def test_mole_lite_basis_grad(lite_args, cart):
    symbols, coords, basis = lite_args

    def loss(basis):
        mol = MoleLite(symbols=symbols, coords=coords, basis=basis,
                       cart=cart, trace_basis=True)
        return np.linalg.norm(mol.intor("int1e_kin", hermi=1))

    grad = jax.grad(loss)(basis)
    grad_fwd = jax.jacfwd(loss)(basis)
    for g_rev, g_fwd in zip(jax.tree.leaves(grad), jax.tree.leaves(grad_fwd)):
        assert abs(numpy.asarray(g_rev) - numpy.asarray(g_fwd)).max() < 1e-12

    grad_leaves = jax.tree.leaves(grad)
    for i, flat, fd in _fd_tree_grad(loss, basis):
        assert abs(numpy.asarray(grad_leaves[i]).flat[flat] - fd) < 1e-7

    # jit consistency
    grad_jit = jax.jit(jax.grad(loss))(basis)
    for g, gj in zip(jax.tree.leaves(grad), jax.tree.leaves(grad_jit)):
        assert abs(numpy.asarray(g) - numpy.asarray(gj)).max() < 1e-12


def test_mole_lite_mixed_coords_basis(lite_args):
    """d/d(basis) of the coordinate-gradient norm (coords inner, basis outer),
    mirroring the legacy ``test_chain_deriv`` oracle."""
    symbols, coords, basis = lite_args

    def gnorm(basis):
        def inner(coords_):
            mol = MoleLite(symbols=symbols, coords=coords_, basis=basis,
                           trace_coords=True, trace_basis=True)
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=1))
        g = jax.grad(inner)(coords)
        return np.linalg.norm(g)

    grad = jax.grad(gnorm)(basis)
    grad_leaves = jax.tree.leaves(grad)
    for i, flat, fd in _fd_tree_grad(gnorm, basis):
        assert abs(numpy.asarray(grad_leaves[i]).flat[flat] - fd) < 1e-6


def test_mole_pad_basis_grad():
    from pyscfad.ml.gto import MolePad, make_basis_array

    b = make_basis_array("sto-3g", max_number=8)
    numbers = numpy.array([8, 1, 1, 0], dtype=numpy.int32)
    coords = numpy.array([[0.0, 0.0, 0.4],
                          [0.2, 1.4, -1.0],
                          [-0.2, -1.45, -0.9],
                          [0.0, 0.0, 0.0]])

    def loss(data):
        basis = dataclasses.replace(b, data=data)
        mol = MolePad(numbers, coords, basis=basis, trace_basis=True)
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s * s)

    grad = numpy.asarray(jax.grad(loss)(b.data))
    assert numpy.isfinite(grad).all()

    grad_fwd = numpy.asarray(jax.jacfwd(loss)(b.data))
    assert abs(grad - grad_fwd).max() < 1e-12

    grad_jit = numpy.asarray(jax.jit(jax.grad(loss))(b.data))
    assert abs(grad - grad_jit).max() < 1e-12

    # FD at a handful of real (unmasked) parameter slots
    mask = numpy.asarray(b.mask_data)
    data = numpy.asarray(b.data)
    real_slots = numpy.flatnonzero(mask.ravel())
    for flat in real_slots[:: max(1, len(real_slots) // 8)]:
        disp = 1e-4 * max(1.0, abs(data.flat[flat]))

        def at(d):
            data1 = data.copy()
            data1.flat[flat] += d
            return float(loss(np.asarray(data1)))

        fd = (at(disp) - at(-disp)) / (2 * disp)
        assert abs(grad.flat[flat] - fd) < 1e-6 * max(1.0, abs(fd))

    # padded slots: exponent gradients vanish (their coefficients are zero);
    # padded coefficient gradients are the true (tiny) derivatives
    pad = ~mask
    exp_col = numpy.zeros_like(mask)
    exp_col[..., 0] = True
    assert abs(grad[pad & exp_col]).max() < 1e-12
    assert abs(grad[pad & ~exp_col]).max() < 1e-6


def test_mole_pad_basis_grad_traced_numbers():
    """Basis-parameter gradients with traced atomic numbers (ML training):
    numbers as a jit argument and vmap over a batch of systems."""
    from pyscfad.ml.gto import MolePad, make_basis_array

    b = make_basis_array("sto-3g", max_number=8)
    numbers_b = numpy.array([[8, 1, 1, 0], [7, 1, 1, 1]], dtype=numpy.int32)
    coords_b = numpy.array([
        [[0.0, 0.0, 0.4], [0.2, 1.4, -1.0], [-0.2, -1.45, -0.9], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.9, 0.4], [1.7, -0.8, 0.4], [-1.6, -0.9, 0.5]],
    ])

    def loss(data, numbers, coords):
        basis = dataclasses.replace(b, data=data)
        mol = MolePad(numbers, coords, basis=basis, trace_basis=True)
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s * s)

    # eager references (per system)
    g_ref = [numpy.asarray(jax.grad(loss)(b.data, numbers_b[i], coords_b[i]))
             for i in range(2)]

    # numbers as a traced jit argument
    g_jit = jax.jit(jax.grad(loss))
    for i in range(2):
        assert abs(numpy.asarray(g_jit(b.data, numbers_b[i], coords_b[i]))
                   - g_ref[i]).max() < 1e-12

    # vmap over the batch of systems
    g_vmap = numpy.asarray(
        jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))(
            b.data, numbers_b, coords_b))
    for i in range(2):
        assert abs(g_vmap[i] - g_ref[i]).max() < 1e-12
