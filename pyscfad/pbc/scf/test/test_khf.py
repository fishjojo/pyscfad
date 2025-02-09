import pytest
import numpy
import jax
from jax import numpy as jnp
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
    cell.build(trace_exp=False, trace_ctr_coeff=False)
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
    assert abs(h1-h1_ref).max() < 1e-10

    g_fwd = jax.jacfwd(get_hcore)(cell, kpts).coords
    #g_bwd = jax.jacrev(get_hcore)(cell, kpts).coords

    mf_grad = pyscf_grad.krhf.Gradients(mf_ref)
    hcore_deriv = mf_grad.hcore_generator(cell_ref, kpts)
    for ia in range(cell_ref.natm):
        g0 = hcore_deriv(ia).transpose(1,2,3,0)
        assert abs(g_fwd[...,ia,:] - g0).max() < 1e-10
        #assert abs(g_bwd[...,ia,:] - g0).max() < 1e-10

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

def test_e_tot(get_cell, get_cell_ref):
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

    assert abs(e_tot - e_tot_ref) < 1e-10
    #assert abs(jac_fwd.coords - g0).max() < 1e-8
    assert abs(jac_bwd.coords - g0).max() < 1e-8
