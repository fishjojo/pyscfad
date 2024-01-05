import numpy
import scipy
import jax
from pyscf import numpy as np
from pyscf.lib import logger
from pyscf.lo.boys import Boys
from pyscfad.lib import stop_grad
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import (
    extract_rotation,
    pack_uniq_var,
)
from pyscfad.tools.linear_solver import gen_gmres

def dipole_integral(mol, mo_coeff):
    charge_center = numpy.einsum('z,zx->x', mol.atom_charges(),
                                 stop_grad(mol.atom_coords()))
    with mol.with_common_origin(charge_center):
        r = mol.intor('int1e_r')
        dip = np.einsum('ui,xuv,vj->xij', mo_coeff.conj(), r, mo_coeff)
    return dip

def cost_function(x, mol, mo_coeff):
    u = extract_rotation(x)
    mo_coeff = np.dot(mo_coeff, u)
    dip = dipole_integral(mol, mo_coeff)
    r2 = mol.intor_symmetric('int1e_r2')
    r2 = np.einsum('pi,pi->', mo_coeff, np.dot(r2, mo_coeff))
    val = r2 - np.einsum('xii,xii->', dip, dip)
    return val

def _opt_cond(x, mol, mo_coeff):
    g = jax.grad(cost_function, 0)(x, mol, mo_coeff)
    return g

def _boys(x, mol, mo_coeff, *,
          init_guess=None,
          conv_tol=None, conv_tol_grad=None, max_cycle=None):
    mo_coeff = numpy.array(mo_coeff, dtype=mo_coeff.dtype, order='C')
    loc = Boys(mol, mo_coeff=mo_coeff)
    if init_guess is not None:
        loc.init_guess = init_guess
    if conv_tol is not None:
        loc.conv_tol = conv_tol
    if conv_tol_grad is not None:
        loc.conv_tol_grad = conv_tol_grad
    if max_cycle is not None:
        loc.max_cycle = max_cycle

    if init_guess is None:
        u, sorted_idx = loc.kernel(mo_coeff=mo_coeff, return_u=True)
    else:
        u, sorted_idx = loc.kernel(mo_coeff=None, return_u=True)
    h_diag = loc.gen_g_hop(u)[2]
    if numpy.any(h_diag < 0):
        logger.warn(loc, 'Saddle point reached in orbital localization.')
    if numpy.linalg.det(u) < 0:
        u[:,0] *= -1
    mat = scipy.linalg.logm(u)
    x = pack_uniq_var(mat)
    if numpy.any(abs(x.imag) > 1e-6):
        raise RuntimeError('Complex solutions are not supported for '
                           'differentiating the Boys localiztion.')
    else:
        x = x.real
    return x, sorted_idx

def boys(mol, mo_coeff, *,
         init_guess=None,
         conv_tol=None, conv_tol_grad=None, max_cycle=None,
         symmetry=False, gmres_options=None):
    '''
    Boys localization. See also `pyscf.lo.boys.Boys`.

    Arguments:
        symmetry : bool
            For certain symmetric molecules, orbital localizations
            may have degenerate solutions. The orbital hessians are
            singular in such cases. Setting `symmetry=True` will
            solve the linear equations in the range space when computing
            the derivatives. Default valule is `False`.
        gmres_options : dict
            Options of the GMRES solver for computing the derivatives.
            See `pyscfad.tools.linear_solver.gen_gmres` for more information.
    '''
    if mo_coeff.shape[-1] == 1:
        return mo_coeff
    if gmres_options is None:
        gmres_options = {}

    solver = gen_gmres(safe=symmetry, **gmres_options)
    _boys_iter = make_implicit_diff(_boys,
                                    implicit_diff=True,
                                    fixed_point=False,
                                    optimality_cond=_opt_cond,
                                    solver=solver,
                                    has_aux=True)

    x, sorted_idx = _boys_iter(None, mol, mo_coeff,
                               init_guess=init_guess,
                               conv_tol=conv_tol,
                               conv_tol_grad=conv_tol_grad,
                               max_cycle=max_cycle)

    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])
