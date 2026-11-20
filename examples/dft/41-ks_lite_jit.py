"""Kohn-Sham nuclear and basis gradients with the lightweight
classes, fully jit-able, for closed and open shells, and with
the SP2 density matrix purification solver.

The integration grids move with the atoms, so the nuclear
gradient includes the grid response.
"""
import jax
import numpy
from pyscfad.gto import MoleLite
from pyscfad.dft.rks_lite import RKSLite
from pyscfad.dft.uks_lite import UKSLite

# coordinates in a.u.
coords = numpy.array([[0., 0., 0.23], [0., 1.43, -0.92], [0., -1.43, -0.92]])

def energy(coords, basis, cls=RKSLite, charge=0, spin=0, use_sp2=False):
    mol = MoleLite(["O", "H", "H"], coords, basis=basis,
                   charge=charge, spin=spin, verbose=0)
    mf = cls(mol, xc="pbe,pbe")
    # Only the hcore guess works inside jit.
    mf.init_guess = "hcore"
    if use_sp2:
        mf.use_sp2 = True
        mf.diis = "anderson"
    return mf.kernel()

basis = MoleLite(["O", "H", "H"], coords, basis="sto3g").basis
gfn = jax.jit(jax.grad(energy, (0, 1)), static_argnames=("cls", "charge", "spin", "use_sp2"))

g = gfn(coords, basis)
print("RKS nuclear gradient:\n", g[0])
print("RKS basis gradient of H:\n", g[1]["H"])

g = gfn(coords, basis, cls=UKSLite, charge=1, spin=1, use_sp2=True)
print("UKS (SP2) nuclear gradient of H2O+:\n", g[0])
