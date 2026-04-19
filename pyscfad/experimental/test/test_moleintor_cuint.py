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
        mol = MolePad(numbers, coords, basis=basis, verbose=0,
                      trace_coords=True, cuint_plan=plan)
        return mol.intor(intor, hermi=1)

    intor_pad_jitted = jax.jit(jax.vmap(intor_pad, (0, 0, in_axes, None)), static_argnames=["intor"])
    jac = jax.jacrev(intor_pad, 1)
    intor_grad_pad_jitted = jax.jit(jax.vmap(jac, (0, 0, in_axes, None)), static_argnames=["intor"])

    def intor_ref_pad(numbers, coords, intor):
        mol = MolePad(numbers, coords, basis=basis, verbose=0, trace_coords=True)
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
                       trace_coords=True, cuint_plan=plan)
        return mol.intor(intor, hermi=1)

    intor_lite_jitted = jax.jit(intor_lite, static_argnames=["intor"])
    intor_grad_lite_jitted = jax.jit(jax.jacrev(intor_lite), static_argnames=["intor"])

    def intor_lite_ref(coords, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis, trace_coords=True)
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
                       trace_coords=True, cuint_plan=plan)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    fn_jitted = jax.jit(fn, static_argnames=["intor"])
    fn_grad_jitted = jax.jit(jax.jacrev(fn, 1), static_argnames=["intor"])

    def fn_ref(coords, origin, intor):
        mol = MoleLite(numbers=numbers, coords=coords, spin=spin, basis=basis, trace_coords=True)
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

def test_rc_deriv_batched(mol_batch):
    numbers = mol_batch["numbers"]
    coords = mol_batch["coords"]
    basis = mol_batch["basis"]
    plans = mol_batch["plans"]
    in_axes = mol_batch["in_axes"]

    def intor_pad(numbers, coords, origin, plan, intor):
        mol = MolePad(numbers, coords, basis=basis, verbose=0,
                      trace_coords=True, cuint_plan=plan)
        with mol.with_common_origin(origin):
            s1e = mol.intor(intor, hermi=1)
        return s1e

    intor_pad_jitted = jax.jit(jax.vmap(intor_pad, (0, 0, None, in_axes, None)), static_argnames=["intor"])
    jac = jax.jacrev(intor_pad, 2)
    intor_grad_pad_jitted = jax.jit(jax.vmap(jac, (0, 0, None, in_axes, None)), static_argnames=["intor"])

    def intor_ref_pad(numbers, coords, origin, intor):
        mol = MolePad(numbers, coords, basis=basis, verbose=0, trace_coords=True)
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
