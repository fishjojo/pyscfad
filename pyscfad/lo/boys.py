from functools import partial
import numpy
import scipy
from jaxopt import linear_solve
from pyscf import numpy as np
from pyscf.lo.boys import Boys
from pyscfad import config
from pyscfad.lib import stop_grad
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import extract_rotation, pack_uniq_var
from pyscfad.scipy.sparse.linalg import gmres

def dipole_integral(mol, mo_coeff):
    charge_center = numpy.einsum('z,zx->x', mol.atom_charges(),
                                 stop_grad(mol.atom_coords()))
    with mol.with_common_origin(charge_center):
        r = mol.intor('int1e_r')
        dip = np.einsum('ui,xuv,vj->xij', mo_coeff.conj(), r, mo_coeff)
    return dip

def opt_cond(x, mol, mo_coeff):
    u = extract_rotation(x)
    mo_coeff = np.dot(mo_coeff, u)
    dip = dipole_integral(mol, mo_coeff)
    g0 = np.einsum('xii,xip->pi', dip, dip)
    g = -pack_uniq_var(g0-g0.conj().T) * 2
    return g

def _boys(x, mol, mo_coeff):
    loc = Boys(mol, mo_coeff=mo_coeff)
    u, sorted_idx = loc.kernel(mo_coeff=mo_coeff, return_u=True)
    mat = scipy.linalg.logm(u)
    x = pack_uniq_var(mat)
    return x, sorted_idx

def boys(mol, mo_coeff):
    if mo_coeff.shape[-1] == 1:
        return mo_coeff

    if config.moleintor_opt:
        solver = partial(gmres, tol=1e-5)
    else:
        solver = partial(linear_solve.solve_gmres, tol=1e-5,
                         solve_method='incremental')
    _boys_iter = make_implicit_diff(_boys, implicit_diff=True,
                                    fixed_point=False,
                                    optimality_cond=opt_cond,
                                    solver=solver, has_aux=True)

    x, sorted_idx = _boys_iter(None, mol, mo_coeff)
    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])
