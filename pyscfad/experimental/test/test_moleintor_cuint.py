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
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.gto import MoleLite
from pyscfad.experimental.moleintor_cuint import (
    _cuint,
    cuint_create_plan,
    cuint_merge_plans,
)

if not _cuint:
    pytest.skip("cuint is needed for the tests", allow_module_level=True)

INTORS = ["int1e_ovlp", "int1e_r", "int1e_rr"]

@pytest.fixture
def OH():
    numbers = [1,8]
    coords = np.array(
        [
            [-5.8768512,  5.12198377, -1.26494537],
            [0.74030004,  3.22499019, -2.63607016],
        ]
    ) / BOHR
    basis = "ccpvdz"
    spin = 1

    mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis)
    plan = cuint_create_plan(mol)

    return {"numbers": numbers,
            "coords": coords,
            "basis": basis,
            "spin": spin,
            "plan": plan}

@pytest.fixture
def mol_batch():
    basis = make_basis_array("sto-3g", 8)
    numbers = np.array([[8, 1, 1, 0],
                        [7, 1, 1, 1]], dtype=np.int32)
    coords = np.array([np.array([[0.00000,  0.00000,  0.00000],
                                 [1.43355,  0.00000, -0.95296],
                                 [1.43355,  0.00000,  0.95296],
                                 [0.00000,  0.00000,  0.00000]]),
                       np.array([[-0.80650, -1.00659,  0.02850],
                                 [-0.50540, -0.31299,  0.68220],
                                 [ 0.00620, -1.41579, -0.38500],
                                 [-1.32340, -0.54779, -0.69350]])/BOHR])

    plans = []
    for number, coord in zip(numbers, coords):
        mol = MolePad(number, coord, basis=basis, verbose=0)
        plan = cuint_create_plan(mol)
        plans.append(plan)
    plans, in_axes = cuint_merge_plans(plans)

    return {"numbers": numbers,
            "coords": coords,
            "basis": basis,
            "plans": plans,
            "in_axes": in_axes}

def test_cuint_batched(mol_batch):
    numbers = mol_batch["numbers"]
    coords = mol_batch["coords"]
    basis = mol_batch["basis"]
    plans = mol_batch["plans"]
    in_axes = mol_batch["in_axes"]

    def intor_pad(numbers, coords, plan, intor):
        mol = MolePad(numbers, coords, basis=basis, cuint_plan=plan)
        return mol.intor(intor, hermi=1)

    intor_pad_jitted = jax.jit(jax.vmap(intor_pad, (0, 0, in_axes, None)), static_argnames=["intor"])
    jac = jax.jacrev(intor_pad, 1)
    intor_grad_pad_jitted = jax.jit(jax.vmap(jac, (0, 0, in_axes, None)), static_argnames=["intor"])

    def intor_ref_pad(numbers, coords, intor):
        mol = MolePad(numbers, coords, basis=basis)
        return mol.intor(intor, hermi=1)

    intor_ref_pad_jitted = jax.jit(jax.vmap(intor_ref_pad, (0,0,None)), static_argnames=["intor"])
    jac1 = jax.jacrev(intor_ref_pad, 1)
    intor_grad_ref_pad_jitted = jax.jit(jax.vmap(jac1, (0,0,None)), static_argnames=["intor"])

    for intor in INTORS:
        s = intor_pad_jitted(numbers, coords, plans, intor)
        s_deriv = intor_grad_pad_jitted(numbers, coords, plans, intor)

        s_ref = intor_ref_pad_jitted(numbers, coords, intor)
        s_ref_deriv = intor_grad_ref_pad_jitted(numbers, coords, intor)

        assert abs(s - s_ref).max() < 1e-9
        assert abs(s_deriv - s_ref_deriv).max() < 1e-9

def test_cuint(OH):
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    basis = OH["basis"]
    plan = OH["plan"]

    def intor_lite(coords, plan, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        return mol.intor(intor, hermi=1)

    intor_lite_jitted = jax.jit(intor_lite, static_argnames=["intor"])
    intor_grad_lite_jitted = jax.jit(jax.jacrev(intor_lite), static_argnames=["intor"])

    def intor_lite_ref(coords, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis)
        return mol.intor(intor, hermi=1)

    intor_lite_ref_jitted = jax.jit(intor_lite_ref, static_argnames=["intor"])
    intor_grad_lite_ref_jitted = jax.jit(jax.jacrev(intor_lite_ref), static_argnames=["intor"])

    for intor in INTORS:
        s1e = intor_lite_jitted(coords, plan, intor)
        s1e_deriv = intor_grad_lite_jitted(coords, plan, intor)

        s1e_ref = intor_lite_ref_jitted(coords, intor)
        s1e_deriv_ref = intor_grad_lite_ref_jitted(coords, intor)

        assert abs(s1e - s1e_ref).max() < 1e-9
        assert abs(s1e_deriv - s1e_deriv_ref).max() < 1e-9

def test_rc_deriv(OH):
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    basis = OH["basis"]
    plan = OH["plan"]

    def fn(coords, origin, plan, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    fn_jitted = jax.jit(fn, static_argnames=["intor"])
    fn_grad_jitted = jax.jit(jax.jacrev(fn, 1), static_argnames=["intor"])

    def fn_ref(coords, origin, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    fn_ref_jitted = jax.jit(fn_ref, static_argnames=["intor"])
    fn_grad_ref_jitted = jax.jit(jax.jacrev(fn_ref, 1), static_argnames=["intor"])

    origin = np.array([0.1,0.2,0.3])

    for intor in ["int1e_r", "int1e_rr"]:
        s1e = fn_jitted(coords, origin, plan, intor)
        s1e_deriv = fn_grad_jitted(coords, origin, plan, intor)

        s1e_ref = fn_ref_jitted(coords, origin, intor)
        s1e_deriv_ref = fn_grad_ref_jitted(coords, origin, intor)

        assert abs(s1e - s1e_ref).max() < 1e-9
        assert abs(s1e_deriv - s1e_deriv_ref).max() < 1e-9

def test_cuint_basis_deriv(OH):
    """Basis-set parameter derivatives: cuint vs the lite CPU path."""
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    mol0 = MoleLite(numbers=numbers, coords=coords, spin=spin, basis="ccpvdz")
    basis0 = mol0.basis

    def loss(basis, plan):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        s1e = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s1e ** 2)

    g_gpu = jax.grad(loss)(basis0, plan)
    g_cpu = jax.grad(loss)(basis0, None)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_cpu)):
        assert abs(a - b).max() < 1e-9

    # forward mode and jit consistency
    g_fwd = jax.jacfwd(loss)(basis0, plan)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_fwd)):
        assert abs(a - b).max() < 1e-12
    g_jit = jax.jit(jax.grad(loss))(basis0, plan)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_jit)):
        assert abs(a - b).max() < 1e-12


def test_cuint_basis_deriv_fd(OH):
    """The cuint basis gradient against finite differences, checked
    separately for the exponent (column 0 of a shell block) and the
    contraction-coefficient columns.

    The other basis-derivative tests compare against the CPU path, which
    reaches the exponent term a different way (l+2 Cartesian promotion,
    versus the solid-harmonic identity used here); this pins the cuint
    result on its own.
    """
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis

    def loss(basis):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        s1e = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s1e ** 2)

    grad_leaves = jax.tree.leaves(jax.grad(loss)(basis0))
    leaves, treedef = jax.tree.flatten(basis0)

    n_exp = n_cs = 0
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        nprim, ncol = leaf.shape
        # one exponent and one contraction coefficient per shell block
        for idx in ((0, 0), (nprim - 1, ncol - 1)):
            disp = 1e-4 * max(1.0, abs(leaf[idx]))

            def at(d):
                leaf1 = leaf.copy()
                leaf1[idx] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return float(loss(jax.tree.unflatten(treedef, leaves1)))

            fd = (at(disp) - at(-disp)) / (2 * disp)
            got = numpy.asarray(grad_leaves[i])[idx]
            assert abs(got - fd) < 1e-6 * max(1.0, abs(fd))
            n_exp += idx[1] == 0
            n_cs += idx[1] > 0
    assert n_exp and n_cs


def test_cuint_basis_deriv_ipovlp(OH):
    """Basis-parameter gradients of the coordinate-gradient overlap
    (``int1e_ipovlp``): cuint vs the CPU lite path.

    This is the integral the geometry-gradient tangent is built from, so its
    basis derivative is what makes mixed coordinate/basis derivatives work.
    """
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis

    def loss(basis, plan):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        # both backends return the full (antisymmetric) matrix; cuint only
        # implements hermi=1, the CPU path builds it with hermi=0
        s1e = mol.intor("int1e_ipovlp", hermi=1 if plan is not None else 0)
        return np.sum(s1e ** 2)

    g_gpu = jax.grad(loss)(basis0, plan)
    g_cpu = jax.grad(loss)(basis0, None)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_cpu)):
        assert abs(a - b).max() < 1e-9

    g_jit = jax.jit(jax.grad(loss))(basis0, plan)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_jit)):
        assert abs(a - b).max() < 1e-12


def _mixed_loss(numbers, spin):
    """sum(S**2) as a function of (coords, basis), traced for both."""
    def loss(coords, basis, plan):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        return np.sum(mol.intor("int1e_ovlp", hermi=1) ** 2)
    return loss


def test_cuint_mixed_coord_basis_deriv(OH):
    """Mixed coordinate/basis second derivative (the basis-parameter
    gradient of the geometry gradient): cuint vs the CPU lite path, and
    against finite differences of the geometry gradient.
    """
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis
    loss = _mixed_loss(numbers, spin)

    # differentiate the geometry gradient w.r.t. the basis parameters; the
    # coordinate derivative has to be the inner one (the cross integrals of
    # the basis tangent are evaluated on the primal geometry)
    mixed = jax.jacfwd(jax.grad(loss, argnums=0), argnums=1)
    g_gpu = mixed(coords, basis0, plan)
    g_cpu = mixed(coords, basis0, None)
    for a, b in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_cpu)):
        assert abs(a - b).max() < 1e-9

    # finite differences of the geometry gradient w.r.t. one exponent and one
    # contraction coefficient per shell block
    grad_coords = jax.grad(loss, argnums=0)
    leaves, treedef = jax.tree.flatten(basis0)
    got_leaves = jax.tree.leaves(g_gpu)
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        for idx in ((0, 0), (leaf.shape[0] - 1, leaf.shape[1] - 1)):
            disp = 1e-5 * max(1.0, abs(leaf[idx]))

            def at(d):
                leaf1 = leaf.copy()
                leaf1[idx] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return numpy.asarray(
                    grad_coords(coords, jax.tree.unflatten(treedef, leaves1),
                                plan))

            fd = (at(disp) - at(-disp)) / (2 * disp)
            got = numpy.asarray(got_leaves[i])[..., idx[0], idx[1]]
            assert abs(got - fd).max() < 1e-6 * max(1.0, abs(fd).max())


def test_cuint_second_coord_deriv_not_implemented(OH):
    """cuint implements one coordinate derivative only, so a geometry
    Hessian must fail loudly rather than quietly return something.

    This replaces the old ``max_coord_deriv`` budget. That kwarg existed
    because the basis tangent used to route its cross integrals through a
    nested ``getints`` that kept tracing coordinates and so asked for
    ``int1e_ovlp_dr20_sph``. The tangent now evaluates the cross integrals
    directly (``gen_overlap_cross`` on a stopped ``env``), so a mixed
    coordinate/basis derivative never reaches the second geometry
    derivative and needs no budget -- see
    :func:`test_cuint_mixed_coord_basis_deriv`, which takes exactly that
    derivative and checks it against finite differences.
    """
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis
    loss = _mixed_loss(numbers, spin)

    with pytest.raises(NotImplementedError):
        jax.jacfwd(jax.grad(loss, argnums=0), argnums=0)(coords, basis0, plan)


def test_cuint_mixed_coord_basis_deriv_batched(mol_batch):
    """Batched (padded, traced atomic numbers) mixed coordinate/basis
    second derivatives through the cuint backend vs the CPU pad path."""
    import dataclasses

    numbers = mol_batch["numbers"]
    coords = mol_batch["coords"]
    basis = mol_batch["basis"]
    plans = mol_batch["plans"]
    in_axes = mol_batch["in_axes"]

    # differentiate w.r.t. the basis parameters only (the BasisArray also
    # carries boolean mask leaves, which forward mode has no tangents for)
    def loss(data, numbers, coords, plan):
        mol = MolePad(numbers, coords,
                      basis=dataclasses.replace(basis, data=data),
                      cuint_plan=plan)
        return np.sum(mol.intor("int1e_ovlp", hermi=1) ** 2)

    mixed = jax.jacfwd(jax.grad(loss, argnums=2), argnums=0)
    g_gpu = jax.jit(jax.vmap(mixed, in_axes=(None, 0, 0, in_axes)))(
        basis.data, numbers, coords, plans)

    for i in range(len(numbers)):
        g_cpu = numpy.asarray(mixed(basis.data, numbers[i], coords[i], None))
        g = numpy.asarray(g_gpu[i])
        assert abs(g - g_cpu).max() < 1e-9
        # padding entries are frozen in make_bas_env
        mask = numpy.broadcast_to(numpy.asarray(basis.mask_data), g.shape)
        assert not g[~mask].any()


def test_cuint_cross_block_chunking(OH):
    """The cross integrals are evaluated a chunk of fake functions at a time
    so that the square block the kernels write is no bigger than the
    (fake x real) one that is kept. That split is pure bookkeeping: the
    assembled block must not depend on the chunk width.
    """
    from pyscfad.experimental.moleintor_cuint import (
        BasisCrossPlan, _cross_blocks, _plan_bas_concrete,
    )

    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan_cu = OH["plan"]

    mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis="ccpvdz")
    atm = numpy.asarray(mol._atm, dtype=numpy.int32)
    bas = numpy.asarray(mol._bas)
    env = numpy.asarray(mol._env)
    bas_conc = _plan_bas_concrete(plan_cu, bas)
    envc = np.concatenate([np.asarray(env), np.ones(1)])

    whole = BasisCrossPlan(bas_conc, env.shape[-1], budget=10**6)
    split = BasisCrossPlan(bas_conc, env.shape[-1], budget=5)
    assert len(whole.chunks) == 1
    assert len(split.chunks) > 1
    assert sum(c.n_fn for c in split.chunks) == split.nao_fake

    for n_deriv in (0, 1):
        for lap in (False, True):
            a = _cross_blocks(atm, envc, whole, whole.make_rows(bas),
                              n_deriv, lap)
            b = _cross_blocks(atm, envc, split, split.make_rows(bas),
                              n_deriv, lap)
            a = numpy.asarray(a)
            b = numpy.asarray(b)
            assert a.shape == (3 ** n_deriv, whole.nao_fake, whole.nao)
            # only the atomicAdd order over the pair lists differs
            assert abs(a - b).max() < 1e-12 * max(1.0, abs(a).max())


def test_cuint_basis_deriv_gradient_old_plugin(OH, monkeypatch):
    """A plugin whose kernels stop at total derivative order 2 cannot do the
    exponent term of the gradient integral; cuint would silently return
    zeros, so the Python side has to refuse.
    """
    from pyscfad.experimental import moleintor_cuint

    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis
    monkeypatch.setattr(moleintor_cuint, "cuint_max_deriv", lambda: 2)

    def loss(basis):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        return np.sum(mol.intor("int1e_ipovlp", hermi=1) ** 2)

    with pytest.raises(NotImplementedError, match="derivative order 3"):
        jax.grad(loss)(basis0)

    # the plain overlap only needs order 2 and still works
    def loss_ovlp(basis):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        return np.sum(mol.intor("int1e_ovlp", hermi=1) ** 2)

    assert jax.tree.leaves(jax.grad(loss_ovlp)(basis0))


@pytest.mark.parametrize("intor", ["int1e_r", "int1e_rr"])
def test_cuint_basis_deriv_unsupported(OH, intor):
    """Only the overlap has a basis-parameter derivative on cuint; the
    dipole/quadrupole exponent terms need their own kernels.
    """
    numbers = OH["numbers"]
    spin = OH["spin"]
    coords = OH["coords"]
    plan = OH["plan"]

    basis0 = MoleLite(numbers=numbers, coords=coords, spin=spin,
                      basis="ccpvdz").basis

    def loss(basis):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis,
                       cuint_plan=plan)
        return np.sum(mol.intor(intor, hermi=1) ** 2)

    with pytest.raises(NotImplementedError):
        jax.grad(loss)(basis0)


def test_cuint_basis_deriv_batched(mol_batch):
    """Batched basis-parameter gradients (traced atomic numbers) through
    the cuint backend vs the CPU pad path."""
    numbers = mol_batch["numbers"]
    coords = mol_batch["coords"]
    basis = mol_batch["basis"]
    plans = mol_batch["plans"]
    in_axes = mol_batch["in_axes"]

    def loss(basis, numbers, coords, plan):
        mol = MolePad(numbers, coords, basis=basis, cuint_plan=plan)
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s ** 2)

    # the BasisArray also carries boolean mask leaves, whose tangents come
    # back as float0; the parameters live in 'data'
    grad = jax.grad(loss, allow_int=True)
    g_gpu = jax.jit(jax.vmap(grad, in_axes=(None, 0, 0, in_axes)))(
        basis, numbers, coords, plans)

    for i in range(len(numbers)):
        g_cpu = grad(basis, numbers[i], coords[i], None)
        assert abs(g_gpu.data[i] - g_cpu.data).max() < 1e-9


def test_rc_deriv_batched(mol_batch):
    numbers = mol_batch["numbers"]
    coords = mol_batch["coords"]
    basis = mol_batch["basis"]
    plans = mol_batch["plans"]
    in_axes = mol_batch["in_axes"]

    def intor_pad(numbers, coords, origin, plan, intor):
        mol = MolePad(numbers, coords, basis=basis, cuint_plan=plan)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    intor_pad_jitted = jax.jit(jax.vmap(intor_pad, (0, 0, None, in_axes, None)), static_argnames=["intor"])
    jac = jax.jacrev(intor_pad, 2)
    intor_grad_pad_jitted = jax.jit(jax.vmap(jac, (0, 0, None, in_axes, None)), static_argnames=["intor"])

    def intor_ref_pad(numbers, coords, origin, intor):
        mol = MolePad(numbers, coords, basis=basis)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    intor_ref_pad_jitted = jax.jit(jax.vmap(intor_ref_pad, (0,0,None,None)), static_argnames=["intor"])
    jac1 = jax.jacrev(intor_ref_pad, 2)
    intor_grad_ref_pad_jitted = jax.jit(jax.vmap(jac1, (0,0,None,None)), static_argnames=["intor"])

    origin = np.array([0.1,0.2,0.3])

    for intor in ["int1e_r", "int1e_rr"]:
        s1e = intor_pad_jitted(numbers, coords, origin, plans, intor)
        s1e_deriv = intor_grad_pad_jitted(numbers, coords, origin, plans, intor)

        s1e_ref = intor_ref_pad_jitted(numbers, coords, origin, intor)
        s1e_deriv_ref = intor_grad_ref_pad_jitted(numbers, coords, origin, intor)

        assert abs(s1e - s1e_ref).max() < 1e-9
        assert abs(s1e_deriv - s1e_deriv_ref).max() < 1e-9
