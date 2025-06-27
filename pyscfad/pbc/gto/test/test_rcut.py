import numpy
import pytest
from pyscfad.pbc import gto

@pytest.fixture
def get_cell():
    cell = gto.Cell()
    cell.atom = """
        C 0.875 0.875 0.875
        C 0.25 0.25 0.25
    """
    cell.a = """
        4.136576868, 0.000000000, 2.388253772
        1.378858962, 3.900002074, 2.388253772
        0.000000000, 0.000000000, 4.776507525
    """
    cell.unit = "B"
    cell.precision = 1e-15
    cell.basis = "gth-tzv2p"
    cell.pseudo = "gth-lda"
    cell.mesh = [15]*3
    cell.verbose = 0
    cell.fractional = True
    cell.build()
    return cell

def test_rcut(get_cell):
    cell = get_cell
    kpts = cell.make_kpts([2,2,2])
    s0 = numpy.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
    t0 = numpy.asarray(cell.pbc_intor("int1e_kin", hermi=1, kpts=kpts))
    for i in range(1, 10):
        prec = 1e-13 * 10**i
        cell.rcut = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
        s1 = numpy.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
        t1 = numpy.asarray(cell.pbc_intor("int1e_kin", hermi=1, kpts=kpts))
        #print(prec, cell.rcut, "error = ", abs(s1-s0).max(), abs(t1-t0).max())
        assert abs(s1-s0).max() < prec
        assert abs(t1-t0).max() < prec
