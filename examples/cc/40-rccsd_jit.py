"""RCCSD nuclear and basis gradients, fully jittable.

Unlike 00-simple.py, the whole calculation -- the SCF, the integral
transformation and the amplitude equations -- is traced in one go, and the
basis gradient is with respect to raw parameters, before basis function
normalization.
Implicit differentiation is used always.
"""
import jax
import numpy
from pyscfad.gto import MoleLite
from pyscfad.scf.hf_lite import SCFLite
from pyscfad.cc.rccsd import RCCSDLite

# coordinates in a.u.
coords = numpy.array([[0., 0., 0.,], [0., 0., 1.4,]])

# basis in nwchem format, where
# the first column stores exponents and the
# following columns store contraction coefficients.
basis = {
    'H': {
        0: [
            numpy.array(
                [[13.01  ,  0.019685],
                 [ 1.962 ,  0.137977],
                 [ 0.4446,  0.478148]]
            ), # 1s
            numpy.array([[0.122, 1.]]),  # 2s
        ],
        1: [numpy.array([[0.727, 1.]])], # 2p
    },
}

def energy(coords, basis):
    mol = MoleLite(["H", "H"], coords, basis=basis, verbose=4)
    mf = SCFLite(mol)
    mf.init_guess = "hcore"
    mf.diis = "anderson"
    mf.kernel()

    mycc = RCCSDLite(mf)
    mycc.kernel()
    return mycc.e_tot

gfn = jax.grad(energy, (0, 1))
g = jax.jit(gfn)(coords, basis)
print("Nuclear gradient:\n", g[0])
print("Basis gradient:\n", g[1])
