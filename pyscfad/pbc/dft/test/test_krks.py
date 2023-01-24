import pytest
import numpy
from pyscfad.pbc import dft

def test_nuc_grad(get_Si2):
    cell = get_Si2
    kpts = cell.make_kpts([2,1,1])

    mf = dft.KRKS(cell, kpts=kpts, exxdiv=None)
    mf.xc = 'pbe'
    mf.kernel()
    jac = mf.energy_grad()

    g0 = numpy.asarray([[-0.0151487891,  0.0023280773,  0.0023280773],
                        [ 0.0151486150, -0.0023404710, -0.0023404710]])
    assert abs(jac.coords - g0).max() < 1e-5
