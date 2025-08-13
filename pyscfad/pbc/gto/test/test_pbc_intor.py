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

import pytest
import numpy
import jax
from jax import numpy as jnp
from pyscfad.pbc import gto

TEST_SET = ["int1e_ovlp", "int1e_kin"]

@pytest.fixture
def get_cell():
    cell = gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.
                Si 1.3467560987,  1.3467560987,  1.3467560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build(trace_exp=False, trace_ctr_coeff=False)
    return cell

def func_norm(cell, intor, kpts=None):
    s1 = cell.pbc_intor(intor, hermi=1, kpts=kpts)
    if getattr(s1, "ndim", None) == 2:
        s1 = [s1,]
    res = [jnp.sqrt(jnp.sum(s*s.conj()).real) for s in s1]
    return res

def int1e_deriv1_r0(cell, intor, kpts=None):
    intor_ip = intor.replace("int1e_", "int1e_ip")
    s1 = cell.pbc_intor(intor_ip, kpts=kpts)
    if getattr(s1, "ndim", None) == 2:
        s1 = [s1,]

    aoslices = cell.aoslice_by_atom()
    nao = cell.nao
    natm = cell.natm
    def get_grad(s1_k):
        grad = numpy.zeros((natm,3,nao,nao), dtype=s1_k.dtype)
        for ia in range(natm):
            p0, p1 = aoslices [ia,2:]
            grad[ia,:,p0:p1] += -s1_k[:,p0:p1]
        grad += grad.transpose(0,1,3,2).conj()
        return grad.transpose(2,3,0,1)

    grad = [get_grad(s) for s in s1]
    return grad


def test_int1e_r0(get_cell):
    cell = get_cell
    kpts = cell.make_kpts([2,2,2])

    for intor in TEST_SET:
        jac_fwd = jax.jacfwd(func_norm)(cell, intor, kpts)
        jac_bwd = jax.jacrev(func_norm)(cell, intor, kpts)

        s1 = cell.pbc_intor(intor, kpts=kpts)
        norm = func_norm(cell, intor, kpts)
        g0 = int1e_deriv1_r0(cell, intor, kpts)
        for i in range(len(kpts)):
            grad = jnp.einsum("ijnx,ij->nx", g0[i], s1[i].conj())
            grad += grad.conj()
            grad = (grad * 0.5 / norm[i]).real
            assert abs(grad - jac_fwd[i].coords).max() < 1e-9
            assert abs(grad - jac_bwd[i].coords).max() < 1e-9
