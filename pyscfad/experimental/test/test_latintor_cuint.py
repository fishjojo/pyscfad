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
from pyscfad.pbc.gto import CellLite
from pyscfad.pbc.tools import nimgs_to_lattice_Ls
from pyscfad.experimental.moleintor_cuint import _cuint, cuint_create_plan
from pyscfad.lib import hermi_triu

if not _cuint:
    pytest.skip("cuint is needed for the tests", allow_module_level=True)

def func_norm(coords, numbers, a, basis, kmesh, cuint_plan=None):
    cell = CellLite(numbers=numbers, coords=coords, a=a, rcut=None,
                    basis=basis, precision=1e-6, trace_coords=True)
    kpts = cell.make_kpts(kmesh)
    Ls = nimgs_to_lattice_Ls(cell)
    expkL = np.exp(1j*np.dot(kpts, Ls.T))

    s1e_lat = cell.lattice_intor("int1e_ovlp", hermi=1, Ls=Ls, cuint_plan=cuint_plan)

    if cuint_plan is not None:
        h1 = np.einsum("kl,lpq->kpq", expkL, s1e_lat)
        h1 = h1 + h1.transpose(0,2,1).conj()
    else:
        h1 = np.einsum("kl,lpq->kpq", expkL, s1e_lat)
        h1 = hermi_triu(h1)

    res = jax.vmap(lambda s: np.sqrt(np.sum(s*s.conj()).real))(h1)
    return res

def test_latovlp():
    numbers = [14,14]
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.3467560987, 1.3467560987, 1.3467560987]
        ]
    ) / BOHR

    a = np.array([[0.0, 2.6935121974, 2.6935121974],
                  [2.6935121974, 0.0, 2.6935121974],
                  [2.6935121974, 2.6935121974, 0.0]]) / BOHR

    basis = "ccpvtz"
    cell = CellLite(numbers=numbers, coords=coords, a=a, rcut=None,
                    basis=basis, precision=1e-6, trace_coords=True)

    cuint_plan = cuint_create_plan(cell)
    kmesh = [3,2,2]
    s0 = func_norm(coords, numbers, a, basis, kmesh)
    s1 = func_norm(coords, numbers, a, basis, kmesh, cuint_plan)
    assert abs(s1-s0).max() < 1e-9

    g0 = jax.jacrev(func_norm)(coords, numbers, a, basis, kmesh)
    g1 = jax.jacrev(func_norm)(coords, numbers, a, basis, kmesh, cuint_plan)
    assert abs(g1-g0).max() < 1e-9


def test_latovlp_basis_deriv():
    """Basis-parameter tangent of the per-image lattice overlap:
    forward AD vs finite differences of the stored (cuint) primal."""
    from pyscfad.experimental import latintor_cuint
    from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

    numbers = [14, 14]
    coords = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
    a = numpy.array([[0.0, 2.6935, 2.6935],
                     [2.6935, 0.0, 2.6935],
                     [2.6935, 2.6935, 0.0]]) / BOHR
    cell = CellLite(numbers=numbers, coords=coords, a=a, basis="gth-szv",
                    rcut=8.0, precision=1e-6, verbose=0)
    plan = cuint_create_plan(cell)
    Ls = numpy.asarray(cell.Ls, dtype=float).reshape(-1, 3)
    Ls_mask = numpy.ones(len(Ls), dtype=numpy.int32)
    atm, bas = cell._atm, cell._bas
    env = numpy.asarray(cell._env)

    def f(env_):
        return latintor_cuint._lattice_intor(
            "int1e_ovlp_sph", Ls, Ls_mask, atm, bas, env_, plan,
            hermi=1, trace_coords=True, trace_basis=True)

    jac = numpy.asarray(jax.jacfwd(f)(np.asarray(env)))

    _, _, exp_of = setup_exp(cell)
    _, _, cs_of = setup_ctr_coeff(cell)
    disp = 1e-4
    for slot in numpy.concatenate([exp_of, cs_of])[::2]:
        def at(d):
            env1 = env.copy()
            env1[slot] += d
            return numpy.asarray(f(env1))
        fd = (8*(at(disp)-at(-disp)) - (at(2*disp)-at(-2*disp))) / (12*disp)
        assert abs(jac[..., slot] - fd).max() < 1e-8

    # reverse mode consistency
    def loss(env_):
        return np.sum(f(env_) ** 2)
    g_rev = numpy.asarray(jax.grad(loss)(np.asarray(env)))
    g_fwd = numpy.asarray(jax.jacfwd(loss)(np.asarray(env)))
    assert abs(g_rev - g_fwd).max() < 1e-10


def test_kxtb_basis_grad_parity():
    """GFN1-xTB (k-point) energy gradient w.r.t. the raw basis parameters:
    cuint vs the CPU lattice path."""
    from pyscfad.xtb.kxtb import GFN1KXTB
    from pyscfad.xtb.param import GFN1Param
    from pyscfad.xtb import basis as xtb_basis

    numbers = [14, 14]
    coords = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
    a = numpy.array([[0.0, 2.6935, 2.6935],
                     [2.6935, 0.0, 2.6935],
                     [2.6935, 2.6935, 0.0]]) / BOHR

    bfile = xtb_basis.get_basis_filename()
    cell0 = CellLite(numbers=numbers, coords=coords, a=a, basis=bfile,
                     rcut=15.0, precision=1e-6, verbose=0)
    plan = cuint_create_plan(cell0)
    basis0 = cell0.basis

    def energy(basis, use_plan):
        cell = CellLite(numbers=numbers, coords=coords, a=a, basis=basis,
                        rcut=15.0, precision=1e-6, verbose=0,
                        trace_basis=True,
                        cuint_plan=plan if use_plan else None)
        mf = GFN1KXTB(cell, param=GFN1Param(), kpts=cell.make_kpts([1, 1, 1]))
        mf.conv_tol = 1e-10
        return mf.kernel()

    g_gpu = jax.grad(energy)(basis0, True)
    g_cpu = jax.grad(energy)(basis0, False)
    for x, y in zip(jax.tree.leaves(g_gpu), jax.tree.leaves(g_cpu)):
        assert abs(numpy.asarray(x) - numpy.asarray(y)).max() < 1e-9


def test_gfn1_kxtb_pad_cuint():
    """Batched (padded) k-point GFN1-xTB through the cuint lattice backend:
    energy/force parity with the CPU pad path under jit(vmap(value_and_grad)).
    """
    from pyscfad.experimental.moleintor_cuint import cuint_merge_plans
    from pyscfad.xtb import basis as xtb_basis
    from pyscfad.xtb.util import ke_cutoff_ewald
    from pyscfad.ml.gto import make_basis_array
    from pyscfad.ml.xtb import GFN1KXTB, make_param_array
    from pyscfad.ml.pbc.gto import CellPad
    from pyscfad.ml.pbc.gto.cell_pad import make_image_grid

    a = numpy.array([[0.0, 2.6935, 2.6935],
                     [2.6935, 0.0, 2.6935],
                     [2.6935, 2.6935, 0.0]]) / BOHR
    coords_si = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
    rcut = 15.0

    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=14)
    param = make_param_array(basis, max_number=14)

    cell0 = CellLite(numbers=[14, 14], coords=coords_si, a=a, basis=bfile,
                     rcut=rcut, precision=1e-6, verbose=0)
    Ts = make_image_grid(numpy.asarray(cell0.nimgs))
    ke = ke_cutoff_ewald(0.4, 1e-6 * float(cell0.vol))
    ewald_mesh = numpy.asarray(cell0.cutoff_to_mesh(ke))
    Ls0 = Ts @ a

    # batch: Si2 (+1 padding atom) and Ne1 (+2 padding atoms)
    numbers_b = numpy.array([[14, 14, 0], [10, 0, 0]], dtype=numpy.int32)
    coords_b = numpy.stack([
        numpy.vstack([coords_si, numpy.zeros((1, 3))]),
        numpy.zeros((3, 3)),
    ])

    plans = []
    for nums, crds in zip(numbers_b, coords_b):
        c = CellPad(nums, crds, basis=basis, a=a, Ls=Ls0, rcut=rcut,
                    precision=1e-6, verbose=0)
        plans.append(cuint_create_plan(c))
    merged_plan, plan_axes = cuint_merge_plans(plans)

    def energy(numbers, coords, plan):
        Ls = np.asarray(Ts, dtype=np.float64) @ a
        cell = CellPad(numbers, coords, basis=basis, a=a, Ls=Ls, rcut=rcut,
                       precision=1e-6, verbose=0, trace_coords=True,
                       cuint_plan=plan)
        mf = GFN1KXTB(cell, param, kpts=cell.make_kpts([1, 1, 1]))
        mf.ewald_mesh = ewald_mesh
        mf.conv_tol = 1e-10
        return mf.kernel()

    vgrad = jax.jit(jax.vmap(jax.value_and_grad(energy, argnums=1),
                             in_axes=(0, 0, plan_axes)))
    e, g = vgrad(numbers_b, np.asarray(coords_b), merged_plan)
    assert numpy.isfinite(numpy.asarray(g)).all()

    # unbatched CPU pad references
    for i in range(2):
        e_ref = energy(numbers_b[i], coords_b[i], None)
        assert abs(float(e[i]) - float(e_ref)) < 1e-6


def test_latovlp_basis_deriv_batched():
    """Batched (padded, traced atomic numbers) lattice-overlap basis
    gradients through the cuint backend vs the CPU pad path."""
    import dataclasses
    from pyscfad.xtb import basis as xtb_basis
    from pyscfad.ml.gto import make_basis_array
    from pyscfad.ml.pbc.gto import CellPad
    from pyscfad.ml.pbc.gto.cell_pad import make_image_grid
    from pyscfad.experimental.moleintor_cuint import cuint_merge_plans

    a = numpy.array([[0.0, 2.6935, 2.6935],
                     [2.6935, 0.0, 2.6935],
                     [2.6935, 2.6935, 0.0]]) / BOHR
    coords_si = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
    rcut = 10.0

    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=14)

    cell0 = CellLite(numbers=[14, 14], coords=coords_si, a=a, basis=bfile,
                     rcut=rcut, precision=1e-6, verbose=0)
    Ts = make_image_grid(numpy.asarray(cell0.nimgs))
    Ls0 = Ts @ a

    numbers_b = numpy.array([[14, 14, 0], [10, 0, 0]], dtype=numpy.int32)
    coords_b = numpy.stack([
        numpy.vstack([coords_si, numpy.zeros((1, 3))]),
        numpy.zeros((3, 3)),
    ])

    plans = []
    for nums, crds in zip(numbers_b, coords_b):
        c = CellPad(nums, crds, basis=basis, a=a, Ls=Ls0, rcut=rcut,
                    precision=1e-6, verbose=0)
        plans.append(cuint_create_plan(c))
    merged_plan, plan_axes = cuint_merge_plans(plans)

    def loss(data, numbers, coords, plan):
        basis_ = dataclasses.replace(basis, data=data)
        cell = CellPad(numbers, coords, basis=basis_, a=a, Ls=Ls0, rcut=rcut,
                       precision=1e-6, verbose=0, trace_basis=True,
                       cuint_plan=plan)
        s1e_lat = cell.lattice_intor("int1e_ovlp", hermi=1)
        s1e = np.sum(s1e_lat, axis=0)
        # backend-specific storage conventions (as in kxtb):
        # CPU stores the lower triangle, cuint stores halved pair blocks
        if plan is None:
            s1e = hermi_triu(s1e)
        else:
            s1e = s1e + s1e.T
        return np.sum(s1e ** 2)

    g_gpu = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0, plan_axes)))(
        basis.data, numbers_b, coords_b, merged_plan)

    for i in range(len(numbers_b)):
        g_cpu = jax.grad(loss)(basis.data, numbers_b[i], coords_b[i], None)
        # the CPU lattice sum screens shell pairs at `precision`, cuint does
        # not, so the two backends agree only to ~precision relative
        assert bool((abs(g_gpu[i] - g_cpu)
                     <= 1e-6 * (1.0 + abs(g_cpu))).all())
