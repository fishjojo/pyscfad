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

from packaging.version import Version
import pytest
import numpy
import jax
from jax import numpy as jnp
import pyscf
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
    cell.build()
    return cell

@pytest.fixture
def get_cell_ref():
    cell = pyscf_gto.Cell()
    cell.atom = atom
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.build()
    return cell

@pytest.fixture
def get_cellp_ref():
    cell = pyscf_gto.Cell()
    cell.atom = atom_p
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.build()
    return cell

@pytest.fixture
def get_cellm_ref():
    cell = pyscf_gto.Cell()
    cell.atom = atom_m
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.build()
    return cell

@pytest.mark.skipif(
    Version(pyscf.__version__) >= Version("2.12.0"),
    reason="Gradient of the nonlocal part of the pp is excluded since pyscf 2.12.0."
)
def test_get_hcore(get_cell, get_cell_ref):
    cell = get_cell
    kpts = cell.make_kpts([2,1,1])
    def get_hcore(cell, kpts):
        mf = scf.KRHF(cell, kpts=kpts)
        h1 = mf.get_hcore()
        return h1
    h1 = get_hcore(cell, kpts)

    cell_ref = get_cell_ref
    mf_ref = pyscf_scf.KRHF(cell_ref, kpts=kpts)
    h1_ref = mf_ref.get_hcore()
    assert abs(h1-h1_ref).max() < 1e-8

    g_fwd = jax.jacfwd(get_hcore)(cell, kpts).coords
    #g_bwd = jax.jacrev(get_hcore)(cell, kpts).coords

    mf_grad = pyscf_grad.krhf.Gradients(mf_ref)
    hcore_deriv = mf_grad.hcore_generator(cell_ref, kpts)
    for ia in range(cell_ref.natm):
        g0 = hcore_deriv(ia).transpose(1,2,3,0)
        assert abs(g_fwd[...,ia,:] - g0).max() < 1e-8
        #assert abs(g_bwd[...,ia,:] - g0).max() < 1e-8

def test_get_veff(get_cell, get_cellp_ref, get_cellm_ref):
    cell = get_cell
    kpts = cell.make_kpts([2,1,1])
    def get_veff(cell, dm_kpts, kpts):
        mf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
        veff = mf.get_veff(dm_kpts=dm_kpts, kpts=kpts)
        return veff

    nao = cell.nao
    nk = len(kpts)
    dm0 = numpy.random.rand(nk,nao,nao)
    for i in range(nk):
        dm0[i] = (dm0[i] + dm0[i].T.conj()) / 2.

    g_fwd = jax.jacfwd(get_veff)(cell, dm0, kpts).coords
    #g_bwd = jax.jacrev(get_veff)(cell, dm0, kpts).coords

    cell_p = get_cellp_ref
    mf_p = pyscf_scf.KRHF(cell_p, kpts=kpts, exxdiv=None)
    vjk_p = mf_p.get_veff(dm_kpts=dm0)

    cell_m = get_cellm_ref
    mf_m = pyscf_scf.KRHF(cell_m, kpts=kpts, exxdiv=None)
    vjk_m = mf_m.get_veff(dm_kpts=dm0)
    g0z = (vjk_p - vjk_m) / (0.002 / BOHR)
    assert abs(g_fwd[...,1,2] - g0z).max() < 1e-6
    #assert abs(g_bwd[...,1,2] - g0z).max() < 1e-6

def test_krhf(get_cell, get_cell_ref):
    cell = get_cell
    kpts = cell.make_kpts([2,1,1])
    def hf_energy(cell, kpts):
        mf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
        e_tot = mf.kernel()
        return e_tot
    #jac_fwd = mf.energy_grad(mode='fwd')
    e_tot, jac_bwd = jax.value_and_grad(hf_energy)(cell, kpts)

    cell_ref = get_cell_ref
    mf_ref = pyscf_scf.KRHF(cell_ref, kpts=kpts, exxdiv=None)
    e_tot_ref = mf_ref.kernel()
    mf_grad = pyscf_grad.krhf.Gradients(mf_ref)
    g0 = mf_grad.kernel()

    assert abs(e_tot - e_tot_ref) < 1e-7
    #assert abs(jac_fwd.coords - g0).max() < 1e-7
    assert abs(jac_bwd.coords - g0).max() < 1e-7

def test_get_veff_get_occ_traced(cell_H2, monkeypatch):
    # pyscf's KSCF.get_veff attaches the Coulomb energy to the returned
    # potential with numpy.einsum, and its get_occ requires numpy arrays of
    # MO energies. Neither works on traced arrays, so pyscfad must build veff
    # itself and detach the MO energies. See issue #161.
    cell = cell_H2
    kpts = numpy.zeros((1,3))
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None)

    def _not_traceable(*args, **kwargs):
        raise AssertionError('pyscf KSCF.get_veff must not be called')
    monkeypatch.setattr(pyscf_scf.khf.KSCF, 'get_veff', _not_traceable)

    nao = cell.nao
    dm0 = jnp.asarray(numpy.eye(nao)[None] * (cell.nelectron / nao))
    veff, tangent = jax.jvp(lambda dm: mf.get_veff(dm_kpts=dm), (dm0,), (dm0,))
    vj, vk = mf.get_jk(dm_kpts=dm0)
    assert abs(veff - (vj - vk * .5)).max() < 1e-10
    # veff is linear in the density matrix
    assert abs(tangent - veff).max() < 1e-10

    mo_energy = jnp.asarray(numpy.arange(nkpts*nao, dtype=float).reshape(nkpts,nao))
    mo_occ = numpy.asarray(mf.get_occ(mo_energy))
    occ0 = numpy.zeros(nkpts*nao)
    occ0[:cell.tot_electrons(nkpts)//2] = 2
    assert abs(mo_occ.ravel() - occ0).max() < 1e-10
