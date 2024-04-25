import jax
from jax import numpy as jnp
from pyscfad import gto, scf, tdscf

mol = gto.Mole()
mol.atom = 'H 0 0 0; F 0 0 1.09'
mol.basis = '6-31G*'
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    mytd = tdscf.rhf.CIS(mf)
    mytd.nstates = 1
    e = mytd.kernel()[0]
    return e[0]+e_hf

grad = jax.grad(energy)(mol)

mf = scf.RHF(mol)
mf.kernel()
mytd = tdscf.rhf.CIS(mf)
mytd.nstates = 1
e, xy = mytd.kernel()

# Using Hellman-Feynman formalism.
# The amplitude is closed over, so there is no tracing through the Davidson iteration.
def hellman(mol):
    mf = scf.RHF(mol)
    e_hf = mf.kernel()
    mytd = tdscf.rhf.CIS(mf)
    mytd.nstates = 1

    vind, _ = mytd.gen_vind(mytd._scf)
    ax = vind(xy[0][0])
    e00 = 2.*jnp.dot(xy[0][0].ravel(), ax.ravel())
    return e00 + e_hf

e_tot, grad1 = jax.value_and_grad(hellman)(mol)
print(f"e_tot = {e_tot}")
print(grad.coords)
print(grad1.coords)
