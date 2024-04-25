import pytest
import numpy
import jax
from pyscf.pbc import gto as pyscf_gto
from pyscf.pbc.df import fft as pyscf_fft
from pyscfad.pbc import gto
from pyscfad.pbc.df import fft

BOHR = 0.52917721092

@pytest.fixture
def get_cell():
    cell = gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.001
                Si 1.3467560987,  1.3467560987,  1.3467560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build(trace_exp=False, trace_ctr_coeff=False)
    return cell

@pytest.fixture
def get_cell_ref():
    cell = pyscf_gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.001
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
    cell.atom = '''Si 0.,  0.,  0.001
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
    cell.atom = '''Si 0.,  0.,  0.001
                Si 1.3467560987,  1.3467560987,  1.3466560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell


def test_get_pp(get_cell, get_cell_ref, get_cell_ref_p, get_cell_ref_m):
    cell = get_cell
    kpts = cell.make_kpts([3,1,1])

    def get_pp(cell, kpts=None):
        mydf = fft.FFTDF(cell, kpts=kpts)
        vpp = mydf.get_pp(kpts=kpts)
        return vpp
    vpp = get_pp(cell, kpts)

    cell_ref = get_cell_ref
    mydf_ref = pyscf_fft.FFTDF(cell_ref, kpts=kpts)
    vpp_ref = mydf_ref.get_pp(kpts=kpts)
    assert abs(vpp-vpp_ref).max() < 1e-10

    cell_ref_p = get_cell_ref_p
    mydf_ref_p = pyscf_fft.FFTDF(cell_ref_p, kpts=kpts)
    vpp_ref_p = mydf_ref_p.get_pp(kpts=kpts)

    cell_ref_m = get_cell_ref_m
    mydf_ref_m = pyscf_fft.FFTDF(cell_ref_m, kpts=kpts)
    vpp_ref_m = mydf_ref_m.get_pp(kpts=kpts)

    g_z = (vpp_ref_p - vpp_ref_m) / (0.0002/BOHR)
    jac_fwd = jax.jacfwd(get_pp)(cell, kpts=kpts)
    assert abs(jac_fwd.coords[...,1,2] - g_z).max() < 1e-6
    # FIXME jacrev requires real-valued outputs
    #jac_bwd = jax.jacrev(get_pp)(cell, kpts=kpts)
    #assert abs(jac_bwd.cell.coords[...,1,2] - g_z).max() < 1e-6
