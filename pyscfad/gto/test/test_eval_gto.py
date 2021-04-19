import jax
import pytest
import pyscf
from pyscf.dft import gen_grid
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad import gto

BOHR = 0.52917721092
bas = '631g'

def test_eval_gto_nuc():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = bas
    mol.build(trace_coords=True)

    grids = gen_grid.Grids(mol)
    grids.build(with_non0tab=True)
    coords = grids.coords

    molp = pyscf.gto.Mole()
    molp.atom = 'H 0 0 0; H 0 0 0.740005'  # in Angstrom
    molp.basis = bas
    molp.build()

    molm = pyscf.gto.Mole()
    molm.atom = 'H 0 0 0; H 0 0 0.739995'  # in Angstrom
    molm.basis = bas
    molm.build()

    eval_names = ["GTOval_sph", "GTOval_sph_deriv1",
                  "GTOval_sph_deriv2", "GTOval_sph_deriv3",]
    tol = [1e-6, 1e-6, 1e-6, 3e-6]

    for i, eval_name in enumerate(eval_names):
        ao0 = pyscf_eval_gto(mol, eval_name, coords)
        ao = mol.eval_gto(eval_name, coords)
        assert abs(ao-ao0).max() < 1e-10

        aop = molp.eval_gto(eval_name, coords)
        aom = molm.eval_gto(eval_name, coords)
        g_fd = (aop-aom) / (1e-5 / BOHR)
        jac = jax.jacfwd(mol.__class__.eval_gto)(mol, eval_name, coords)
        assert abs(jac.coords[...,1,2] - g_fd).max() < tol[i]
