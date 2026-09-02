"""Batched RCCSD nuclear and basis gradients, fully jittable.

The batched counterpart of 40-rccsd_jit.py, and the correlated counterpart
of ../scf/41-rhf_batched.py. A batch mixes molecules of different
composition, so it uses the padded classes: MolePad pads every element to
the same shell structure and every molecule to the same number of atoms,
which keeps every array shape static under jax.vmap. The atomic number 0
marks a padding atom; it carries no electrons and no force.

The number of occupied orbitals has to be static as well, and cannot be
read off a traced atomic number, so it is passed as the maximum over the
batch. A molecule with fewer electrons than that then has padding MOs
filling its extra occupied slots. Those enter no integral, so its energy is
unchanged: the water below is solved with nocc=7 against its true 5 and
still reproduces the unpadded result to machine precision.
"""
import jax
from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf.hf_pad import SCFPad
from pyscfad.cc.rccsd import RCCSDLite

# one padded basis array covering every element up to oxygen (Z=8), which
# must cover every atomic number in the batch
# basis set parameters are stored in basis.data
basis = make_basis_array("631g", max_number=8)

# water (10 electrons) and hydrogen cyanide (14), padded to four atoms each
numbers = np.array([[8, 1, 1, 0],
                    [1, 6, 7, 0]], dtype=np.int32)
# coordinates in a.u.
coords = np.array([[[ 0.00000,  0.00000,  0.00000],
                    [ 1.43355,  0.00000, -0.95296],
                    [ 1.43355,  0.00000,  0.95296],
                    [ 0.00000,  0.00000,  0.00000]],
                   [[ 0.00000,  0.00000, -2.01000],
                    [ 0.00000,  0.00000,  0.00000],
                    [ 0.00000,  0.00000,  2.18000],
                    [ 0.00000,  0.00000,  0.00000]]])

# largest number of doubly occupied orbitals in the batch (here HCN's)
nocc = 7

def energy(numbers, coords, basis):
    mol = MolePad(numbers, coords, basis=basis, verbose=0)
    mf = SCFPad(mol)
    mf.init_guess = "hcore"
    mf.kernel()
    # the amplitudes are extrapolated with DIIS and the amplitude equations
    # are differentiated implicitly, through jax.lax.custom_root, rather than
    # by unrolling the iterations
    mycc = RCCSDLite(mf, nocc=nocc)
    mycc.kernel()
    return mycc.e_tot

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
print("Basis gradient of C:\n", g_basis.data[:,6])
print("Basis gradient of N:\n", g_basis.data[:,7])
print("Basis gradient of O:\n", g_basis.data[:,8])
