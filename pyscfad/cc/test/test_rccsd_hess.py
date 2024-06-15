import pytest
import numpy
import jax
from pyscf import lib
from pyscfad import gto, scf, cc
from pyscfad import config

config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

def test_nuc_hessian(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    with jax.disable_jit():
        h1 = jax.jacrev(jax.jacrev(energy))(mol).coords.coords
    h0 = numpy.array(
        [[[[ 4.20246014e-02, 0, 0],
           [-4.20246014e-02, 0, 0]],
          [[0,  4.20246014e-02, 0],
           [0, -4.20246014e-02, 0]],
          [[0, 0,  1.53246547e-01],
           [0, 0, -1.53246547e-01]]],
         [[[-4.20246015e-02, 0, 0],
           [ 4.20246015e-02, 0, 0]],
          [[0, -4.20246017e-02, 0],
           [0,  4.20246017e-02, 0]],
          [[0, 0, -1.53241777e-01],
           [0, 0,  1.53241780e-01]]]]
    )
    assert(abs(h1-h0).max() < 5e-5)
    #f = lib.fp(h1)
    #assert(abs(f - -0.18529155578401263) < 1e-6)
