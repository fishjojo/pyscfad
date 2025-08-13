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
from pyscf.pbc import gto as pyscf_gto
from pyscf.pbc import scf as pyscf_scf
from pyscf.pbc import grad as pyscf_grad
from pyscfad.pbc import gto, scf

BOHR = 0.52917721092

basis = 'gth-szv'
pseudo = 'gth-pade'

a = 5.431020511
lattice = [[0., a/2, a/2],
          [a/2, 0., a/2],
          [a/2, a/2, 0.]]
mesh = [21,]*3
disp = 0.01
atom = [['Si', [0., 0., 0.]],
        ['Si', [a/4+disp, a/4+disp, a/4+disp]]]

atom_p = [['Si', [0., 0., 0.]],
          ['Si', [a/4+disp, a/4+disp, a/4+disp+0.001]]]

atom_m = [['Si', [0., 0., 0.]],
          ['Si', [a/4+disp, a/4+disp, a/4+disp-0.001]]]

@pytest.fixture
def get_cell():
    cell = gto.Cell()
    cell.atom = atom
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.mesh = mesh
    cell.build()
    return cell

@pytest.fixture
def get_cellp_ref():
    cell = pyscf_gto.Cell()
    cell.atom = atom_p
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.mesh = mesh
    cell.build()
    return cell

@pytest.fixture
def get_cellm_ref():
    cell = pyscf_gto.Cell()
    cell.atom = atom_m
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.mesh = mesh
    cell.build()
    return cell

def test_get_hcore(get_cell):
    cell = get_cell
    def get_hcore(cell):
        mf = scf.RHF(cell)
        h1 = mf.get_hcore()
        return h1
    h1 = get_hcore(cell)

    kpts = numpy.zeros([1,3])
    cell_ref = cell.view(pyscf_gto.Cell)
    mf_ref = pyscf_scf.KRHF(cell_ref, kpts=kpts)
    h1_ref = mf_ref.get_hcore()
    assert abs(h1-h1_ref[0]).max() < 1e-8

    g_fwd = jax.jacfwd(get_hcore)(cell).coords
    g_bwd = jax.jacrev(get_hcore)(cell).coords

    mf_grad = pyscf_grad.krhf.Gradients(mf_ref)
    hcore_deriv = mf_grad.hcore_generator(cell_ref, kpts)
    for ia in range(cell_ref.natm):
        g0 = hcore_deriv(ia)[:,0].transpose(1,2,0)
        assert abs(g_fwd[...,ia,:] - g0).max() < 1e-8
        assert abs(g_bwd[...,ia,:] - g0).max() < 1e-8

def test_get_veff(get_cell, get_cellp_ref, get_cellm_ref):
    cell = get_cell
    def get_veff(cell, dm0=None):
        mf = scf.RHF(cell, exxdiv=None)
        return mf.get_veff(dm=dm0)

    nao = cell.nao
    dm0 = numpy.random.rand(nao,nao)
    dm0 = (dm0 + dm0.T) / 2.

    g_fwd = jax.jacfwd(get_veff)(cell, dm0).coords
    g_bwd = jax.jacrev(get_veff)(cell, dm0).coords

    cell_p = get_cellp_ref
    mf_p = pyscf_scf.RHF(cell_p, exxdiv=None)
    vjk_p = mf_p.get_veff(dm=dm0)

    cell_m = get_cellm_ref
    mf_m = pyscf_scf.RHF(cell_m, exxdiv=None)
    vjk_m = mf_m.get_veff(dm=dm0)
    g0z = (vjk_p - vjk_m) / (0.002 / BOHR)
    assert abs(g_fwd[...,1,2] - g0z).max() < 1e-6
    assert abs(g_bwd[...,1,2] - g0z).max() < 1e-6

def test_rhf(get_cell):
    cell = get_cell
    def hf_energy(cell):
        mf = scf.RHF(cell, exxdiv=None)
        ehf = mf.kernel()
        return ehf
    e_tot, jac_bwd = jax.value_and_grad(hf_energy)(cell)

    cell_ref = cell.view(pyscf_gto.Cell)
    mf_ref = pyscf_scf.KRHF(cell_ref, kpts=numpy.zeros([1,3]), exxdiv=None)
    e_tot_ref = mf_ref.kernel()
    mf_grad = pyscf_grad.krhf.Gradients(mf_ref)
    g0 = mf_grad.kernel()
    assert abs(e_tot - e_tot_ref) < 1e-8
    assert abs(jac_bwd.coords - g0).max() < 1e-8
