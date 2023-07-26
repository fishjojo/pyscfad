import numpy
import scipy
from pyscf import numpy as np
from pyscf.lo.boys import Boys
from pyscfad import config
from pyscfad.lib import stop_grad
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import extract_rotation, pack_uniq_var
from pyscfad.tools.linear_solver import precond_by_hdiag, gen_gmres

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

    h_diag = np.einsum('xii,xpp->pi', dip, dip) * 2
    h_diag-= g0.diagonal() + g0.diagonal().reshape(-1,1)
    h_diag+= np.einsum('xip,xip->pi', dip, dip) * 2
    h_diag+= np.einsum('xip,xpi->pi', dip, dip) * 2
    h_diag = -pack_uniq_var(h_diag) * 2
    return g, h_diag

def _boys(x, mol, mo_coeff, *,
          init_guess=None, conv_tol=None, conv_tol_grad=None, max_cycle=None):
    # make a numpy array copy of mo_coeff to pass it into pyscf
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
    mat = scipy.linalg.logm(u)
    x = pack_uniq_var(mat)
    if numpy.any(abs(x.imag) > 1e-9):
        raise RuntimeError('Complex solutions are not supported for '
                           'differentiating the Boys localiztion.')
    else:
        x = x.real
    return x, sorted_idx

def boys(mol, mo_coeff, *,
         init_guess=None, conv_tol=None, conv_tol_grad=None, max_cycle=None):
    if mo_coeff.shape[-1] == 1:
        return mo_coeff

    gen_precond = None
    if config.moleintor_opt:
        gen_precond = precond_by_hdiag
    _boys_iter = make_implicit_diff(_boys, implicit_diff=True,
                                    fixed_point=False,
                                    optimality_cond=opt_cond,
                                    solver=gen_gmres(), has_aux=True,
                                    optimality_fun_has_aux=True,
                                    gen_precond=gen_precond)

    x, sorted_idx = _boys_iter(None, mol, mo_coeff,
                               init_guess=init_guess,
                               conv_tol=conv_tol,
                               conv_tol_grad=conv_tol_grad,
                               max_cycle=max_cycle)
    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])
