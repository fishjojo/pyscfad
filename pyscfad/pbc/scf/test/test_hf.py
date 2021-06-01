import pytest
import numpy
import jax
from pyscf.pbc import gto as pyscf_gto
from pyscf.pbc import scf as pyscf_scf
from pyscfad.lib import numpy as jnp
from pyscfad.pbc import gto, scf

BOHR = 0.52917721092

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
    cell.build(trace_coords=True)
    return cell

@pytest.fixture
def get_cell_ref():
    cell = pyscf_gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.
                Si 1.3467560987,  1.3467560987,  1.3467560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

@pytest.fixture
def get_cell_ref_p():
    cell = pyscf_gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.
                Si 1.3467560987,  1.3467560987,  1.3468560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

@pytest.fixture
def get_cell_ref_m():
    cell = pyscf_gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.
                Si 1.3467560987,  1.3467560987,  1.3466560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell


def test_get_hcore(get_cell, get_cell_ref, get_cell_ref_p, get_cell_ref_m):
    cell = get_cell
    mf = scf.RHF(cell)
    h1 = mf.get_hcore()

    cell_ref = get_cell_ref
    mf_ref = pyscf_scf.RHF(cell_ref)
    h1_ref = mf_ref.get_hcore()
    assert abs(h1-h1_ref).max() < 1e-10

    cell_ref_p = get_cell_ref_p
    mf_ref_p = pyscf_scf.RHF(cell_ref_p)
    h1_ref_p = mf_ref_p.get_hcore()

    cell_ref_m = get_cell_ref_m
    mf_ref_m = pyscf_scf.RHF(cell_ref_m)
    h1_ref_m = mf_ref_m.get_hcore()

    g_z = (h1_ref_p - h1_ref_m) / (0.0002/BOHR)
    jac_fwd = jax.jacfwd(mf.__class__.get_hcore)(mf)
    assert abs(jac_fwd.cell.coords[...,1,2] - g_z).max() < 1e-6
    jac_bwd = jax.jacrev(mf.__class__.get_hcore)(mf)
    assert abs(jac_bwd.cell.coords[...,1,2] - g_z).max() < 1e-6
