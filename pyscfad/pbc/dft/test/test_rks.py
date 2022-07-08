import pytest
import numpy
from pyscfad.pbc import dft

def test_nuc_grad(get_Si2):
    cell = get_Si2

    mf = dft.RKS(cell, exxdiv=None)
    mf.xc = 'pbe'
    mf.kernel()
    jac = mf.energy_grad()

    g0 = numpy.asarray([[-0.0071742877, -0.0071742877, -0.0071742877],
                        [ 0.0071739026,  0.0071739026,  0.0071739026]])
    assert abs(jac.coords - g0).max() < 1e-6
