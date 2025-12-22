"""RHF nuclear hessian
"""
import jax
from pyscfad import gto, scf

mol = gto.Mole()
mol.atom = [
    ["O", 0.000,  0.000, 0.000],
    ["H", 0.000, -0.757, 0.587],
    ["H", 0.000,  0.757, 0.587],
]
mol.basis = "631g*"
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

def rhf_energy(mol):
    mf = scf.RHF(mol)
    ehf = mf.kernel()
    return ehf

jac = jax.hessian(rhf_energy)(mol)
print(f"Nuclear hessian:\n{jac.coords.coords}")
