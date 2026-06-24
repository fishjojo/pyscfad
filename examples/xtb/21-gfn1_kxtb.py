"""GFN1-KXTB
"""
import numpy
import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad.pbc.gto import CellLite
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.kxtb import GFN1KXTB
from pyscfad.xtb import util
from pyscfad.experimental.moleintor_cuint import _cuint, cuint_create_plan

# Si
numbers = [14,14]
coords = np.array(
    [
        [0.00000,  0.00000,  0.00000],
        [1.3467560987, 1.3467560987, 1.3467560987]
    ]
) / BOHR

a = np.array([[0.0, 2.6935121974, 2.6935121974],
              [2.6935121974, 0.0, 2.6935121974],
              [2.6935121974, 2.6935121974, 0.0]]) / BOHR

basis = xtb_basis.get_basis_filename()
param = GFN1Param()

# pre-compute static variables for jitted calculation
# not needed in non-jitted cacluation
cell = CellLite(numbers=numbers, coords=coords, a=a, basis=basis, precision=1e-6)
kpts = cell.make_kpts([3,3,3])

nimgs = numpy.asarray(cell.nimgs)
print(f"lattice sum mesh: {nimgs}")

ewald_eta = 0.4
ke_cutoff = util.ke_cutoff_ewald(ewald_eta, cell.precision * cell.vol)
ewald_mesh = numpy.asarray(cell.cutoff_to_mesh(ke_cutoff))
print(f"ewald sum mesh: {ewald_mesh}")

if _cuint:
    print("use cuint backend for integrals")
    cuint_plan = cuint_create_plan(cell)
else:
    cuint_plan = None

def xtb_energy(coords, param, kpts, cuint_plan=None):
    cell = CellLite(numbers=numbers, coords=coords, a=a, nimgs=nimgs,
                    basis=basis, precision=1e-6, trace_coords=True,
                    cuint_plan=cuint_plan, verbose=4)
    mf = GFN1KXTB(cell, param=param, kpts=kpts)
    mf.ewald_eta = ewald_eta
    mf.ewald_mesh = ewald_mesh
    mf.diis = "qbroyden"
    e = mf.kernel()
    return e

gfn = jax.jit(jax.value_and_grad(xtb_energy, (0,1)))
e, g = gfn(coords, param, kpts, cuint_plan)
print("Nuclear Gradient:\n", g[0])
print("Parameter Gradient:")
print(f"Element['Si']{g[1].element['Si']}")
print("...")
