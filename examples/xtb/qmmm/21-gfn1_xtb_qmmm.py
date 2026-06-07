"""GFN1-XTB QMMM

Chorismate Mutase

# FP32 set up:
export PYSCFAD_FLOATX="float32"

# Reference energy computed in FP64:
-121.43359301944021

# Reference nuclear gradients computed in FP64:
gqm.npy: QM atom nuclear gradients
gmm.npy: MM atom nuclear gradients
"""
import jax
import numpy

from ase.io import read
from pyscf.data.nist import BOHR

from pyscfad import numpy as np
from pyscfad.xtb import basis as xtb_basis
from pyscfad.gto import MoleLite
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.qmmm_pbc import itrf
from pyscfad.scf.addons import _fermi_entropy
from pyscfad.experimental.moleintor_cuint import (
    _cuint,
    cuint_create_plan,
)

basis = xtb_basis.get_basis_filename()
param = GFN1Param()

atoms = read("qm.xyz")
numbers = numpy.array(atoms.get_atomic_numbers(), dtype=np.int32)
coords = np.array(atoms.positions) / BOHR

qm_indexes = np.array("1411 5668 5669 5670 5682 5685 5686 1414 1415 1417 1418 1420 1421 1423 1426 1427 1429 1430 5664 5665 5672 5674 5677 5679 5681 5683 1413 1416 1419 1424 5663 5666 5667 5671 5673 5675 5676 5678 5680 5684 1422 1425 1428 91 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 1230 1232 1233 1234 1235 1236 1237 1238 1239 1240".split(), dtype=np.int32) - 1
mm_charges = np.array(numpy.loadtxt("./partial_charges.dat")).at[qm_indexes].set(0.)
mm_coords = np.array(read("geom.xyz").positions) / BOHR
a = np.diag(np.array([79.00600, 79.68200, 79.03000])) / BOHR
mm_radii = np.ones_like(mm_charges)

key = jax.random.key(23333)
randparam = jax.random.uniform(key=key, shape=(2, numbers.shape[0]))

use_cuint = True
if _cuint and use_cuint:
    mol = MoleLite(numbers, coords, basis=basis, charge=-1, verbose=0)
    plan = cuint_create_plan(mol)
else:
    # fall back to CPU intor
    plan = None

def energy(coords, mm_coords, randparam, mm_charges, mm_radii, a):
    mol = MoleLite(numbers=numbers, coords=coords, basis=basis, charge=-1,
                   cuint_plan=plan, trace_coords=True, verbose=4)
    mf = GFN1XTB(mol, param)
    mf = itrf.add_mm_charges(
        mf, mm_coords, a, mm_charges, mm_radii,
        mm_ew_eta=0.1, mm_ew_rcut=60., mm_ew_mesh=[30, 30, 30], qm_ew_mesh=[25,25,25], unit="BOHR",
    )
    mf.param.dipgam = randparam[0].astype(np.floatx)
    mf.param.quadgam = randparam[1].astype(np.floatx)
    mf.diis = "qbroyden"
    mf.conv_tol = 1e-5
    mf.diis_damp = 0.6
    mf.sigma = 0.0009494077738033897
    e = mf.kernel()
    return e - mf.sigma * _fermi_entropy(mf.mo_occ)

gfn = jax.jit(jax.value_and_grad(energy, (0, 1)))
e, (gqm, gmm) = gfn(coords, mm_coords, randparam, mm_charges, mm_radii, a)
print("energy = ", e)
g = np.vstack([gqm, gmm])
print(np.sum(g, axis=0))

gqm_ref = numpy.load("gqm.npy")
gmm_ref = numpy.load("gmm.npy")

print(abs(gqm-gqm_ref).max())
print(abs(gmm-gmm_ref).max())
