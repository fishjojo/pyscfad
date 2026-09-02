"""Batched RHF nuclear and basis gradients, fully jit-able.

The batched counterpart of 40-rhf_jit.py. A batch mixes molecules of
different composition, so it uses the padded classes: MolePad pads every
element to the same shell structure and every molecule to the same number
of atoms, which keeps every array shape static under jax.vmap. The atomic
number 0 marks a padding atom; it carries no electrons and no force.
"""
import dataclasses
import jax
from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf.hf_pad import SCFPad

# one padded basis array covering every element up to oxygen (Z=8)
# basis set parameters are stored in basis.data
basis = make_basis_array("sto3g", max_number=8)

# water and ammonia, padded to four atoms each
numbers = np.array([[8, 1, 1, 0],
                    [7, 1, 1, 1]], dtype=np.int32)
# coordinates in a.u.
coords = np.array([[[ 0.00000,  0.00000,  0.00000],
                    [ 1.43355,  0.00000, -0.95296],
                    [ 1.43355,  0.00000,  0.95296],
                    [ 0.00000,  0.00000,  0.00000]],
                   [[-1.52370, -1.90220,  0.05386],
                    [-0.95500, -0.59150,  1.28920],
                    [ 0.01172, -2.67480, -0.72760],
                    [-2.50090, -1.03550, -1.31090]]])

def energy(numbers, coords, basis):
    mol = MolePad(numbers, coords, basis=basis, verbose=4)
    mf = SCFPad(mol)
    mf.init_guess = "hcore"
    return mf.kernel()

# the batch shares one basis (in_axes None), so its gradient comes back per
# molecule; sum over the batch to update the shared parameters
efn = jax.value_and_grad(energy, (1, 2), allow_int=True)
e, (g_coords, g_basis) = jax.jit(jax.vmap(efn, (0, 0, None)))(numbers, coords, basis)

print("Energy:\n", e)
print("Nuclear gradient:\n", g_coords)
# g_basis is indexed by atomic number, then by shell, primitive and
# (exponent, contraction coefficients); the elements absent from the batch
# and the padding slots of the present ones are zero
print("Basis gradient of H:\n", g_basis.data[:,1])
print("Basis gradient of N:\n", g_basis.data[:,7])
print("Basis gradient of O:\n", g_basis.data[:,8])
