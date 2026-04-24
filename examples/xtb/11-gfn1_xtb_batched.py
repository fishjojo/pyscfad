"""Batched GFN1-XTB calculation.
"""
import jax
from pyscfad import numpy as np
from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.xtb import GFN1XTB, make_param_array
from pyscfad.experimental.moleintor_cuint import (
    _cuint,
    cuint_create_plan,
    cuint_merge_plans,
)

bfile = xtb_basis.get_basis_filename()
basis = make_basis_array(bfile, max_number=8)
param = make_param_array(basis, max_number=8)

numbers = np.array([[8, 1, 1, 0],
                    [7, 1, 1, 1]], dtype=np.int32)
coords = np.array([np.array([[0.00000,  0.00000,  0.00000],
                             [1.43355,  0.00000, -0.95296],
                             [1.43355,  0.00000,  0.95296],
                             [0.00000,  0.00000,  0.00000]]),
                   np.array([[-0.80650, -1.00659,  0.02850],
                             [-0.50540, -0.31299,  0.68220],
                             [ 0.00620, -1.41579, -0.38500],
                             [-1.32340, -0.54779, -0.69350]])/0.529])

def energy(numbers, coords, plan=None):
    mol = MolePad(numbers, coords, basis=basis, verbose=0,
                  trace_coords=True, cuint_plan=plan)
    mf = GFN1XTB(mol, param)
    e = mf.kernel()
    mu = mf.dip_moment()
    return e, mu

use_cuint = True
if _cuint and use_cuint:
    # create a batch of cuint plans
    plans = []
    for number, coord in zip(numbers, coords):
        mol = MolePad(number, coord, basis=basis, verbose=0)
        plan = cuint_create_plan(mol)
        plans.append(plan)
    plans, in_axes = cuint_merge_plans(plans)
else:
    # fall back to CPU intor
    plans = None
    in_axes = None

gfn = jax.value_and_grad(energy, 1, has_aux=True)
(e, mu), g = jax.jit(jax.vmap(gfn, (0, 0, in_axes)))(numbers, coords, plans)
print(f"energy:\n{e}")
print(f"force:\n{-g}")
print(f"dipole:\n{mu}")
