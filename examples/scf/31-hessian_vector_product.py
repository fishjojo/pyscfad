"""
Example of computing the nuclear Hessian vector product
"""
import jax
from pyscfad import gto, scf, cc
from pyscfad import config

config.update("pyscfad_scf_implicit_diff", True)

mol = gto.Mole()
mol.atom = "H 0. 0. 0.; F 0. 0. 1.1"
mol.basis = 'ccpvdz'
mol.verbose = 4
mol.incore_anyway = True
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    ehf = mf.kernel()
    return ehf

# function for computing the Jacobian
jac = lambda x, *args: jax.jacrev(energy)(x)
# function for computing the Hessian vector product
hessp = lambda x, p, *args: jax.vjp(jac, x)[1](p)[0]

g = jac(mol)
print("Nuclear gradient:")
print(g.coords)

# prepare a Mole object which has its coords
# attribute set as the nuclear gradient
mol1 = mol.copy()
mol1.coords = g.coords
hvp = hessp(mol, mol1)
print("Nuclear Hessian gradient product:")
print(hvp.coords)
