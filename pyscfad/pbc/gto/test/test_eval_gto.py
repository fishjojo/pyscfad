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
from pyscf.pbc.gto import Cell
from pyscf.pbc.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad.pbc import gto

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
    cell.mesh = [5,5,5]
    cell.build(trace_coords=True)
    return cell

def eval_gto_deriv1_r0(cell, intor, coords, kpts=None, kpt=None):
    intor_ip = intor.replace("GTOval", "GTOval_ip")
    ao1 = pyscf_eval_gto(cell, intor_ip, coords, kpts=kpts, kpt=kpt)
    single_kpt = False
    if isinstance(ao1, numpy.ndarray):
        ao1 = [ao1,]
        single_kpt = True
    nkpts = len(ao1)
    natm = cell.natm
    nao = cell.nao
    ng = len(coords)
    ao_loc = None
    aoslices = cell.aoslice_by_atom(ao_loc)
    grad = []
    for k in range(nkpts):
        grad_k = numpy.zeros([natm,3,ng,nao], dtype=ao1[k].dtype)
        for ia in range(natm):
            p0, p1 = aoslices [ia, 2:]
            grad_k[ia,...,p0:p1] += -ao1[k][...,p0:p1]
        grad.append(grad_k.transpose(2,3,0,1))
    if single_kpt:
        grad = grad[0]
    return grad

def test_eval_gto(get_cell):
    cell = get_cell
    kpts = cell.make_kpts([2,1,1])
    grids = cell.get_uniform_grids()
    ao = cell.pbc_eval_gto("GTOval", grids)
    ao_ref = pyscf_eval_gto(cell.view(Cell), "GTOval", grids)
    assert abs(ao - ao_ref).max() < 1e-10

    g0 = eval_gto_deriv1_r0(cell.view(Cell), "GTOval", grids)
    jac_fwd = jax.jacfwd(cell.__class__.pbc_eval_gto)(cell, "GTOval", grids)
    jac_bwd = jax.jacrev(cell.__class__.pbc_eval_gto)(cell, "GTOval", grids)
    assert abs(jac_fwd.coords - g0).max() < 1e-10
    assert abs(jac_bwd.coords - g0).max() < 1e-10

    ao = cell.pbc_eval_gto("GTOval", grids, kpts=kpts)
    ao_ref = pyscf_eval_gto(cell.view(Cell), "GTOval", grids, kpts=kpts)
    for i in range(len(kpts)):
        assert abs(ao[i] - ao_ref[i]).max() < 1e-10

    g0 = eval_gto_deriv1_r0(cell.view(Cell), "GTOval", grids, kpts=kpts)
    jac_fwd = jax.jacfwd(cell.__class__.pbc_eval_gto)(cell, "GTOval", grids, kpts=kpts)
    for i in range(len(kpts)):
        assert abs(jac_fwd[i].coords - g0[i]).max() < 1e-10

    # TODO reverse mode autodiff
    pass

    for i in range(4):
        intor = "PBCGTOval_sph_deriv" + str(i)
        ao = cell.pbc_eval_gto(intor, grids)
        ao_ref = pyscf_eval_gto(cell.view(Cell), intor, grids)
        assert abs(ao - ao_ref).max() < 1e-10

        # TODO test gradient
        pass
