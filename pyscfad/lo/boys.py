import numpy
import scipy
import jax
from pyscf.lib import logger
from pyscf.lo import boys as pyscf_boys
from pyscfad import numpy as np
from pyscfad.ops import stop_grad
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import (
    extract_rotation,
    pack_uniq_var,
)
from pyscfad.tools.linear_solver import gen_gmres

# modified from pyscf v2.3
def kernel(localizer, mo_coeff=None, callback=None, verbose=None,
           return_u=False):
    from pyscf.tools import mo_mapping
    from pyscf.soscf import ciah
    if mo_coeff is not None:
        localizer.mo_coeff = numpy.asarray(mo_coeff, order='C')
    if localizer.mo_coeff.shape[1] <= 1:
        return localizer.mo_coeff

    if localizer.verbose >= logger.WARN:
        localizer.check_sanity()
    localizer.dump_flags()

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(localizer, verbose=verbose)

    if localizer.conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(localizer.conv_tol*.1)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = localizer.conv_tol_grad

    if mo_coeff is None:
        if getattr(localizer, 'mol', None) and localizer.mol.natm == 0:
            # For customized Hamiltonian
            u0 = localizer.get_init_guess('random')
        else:
            u0 = localizer.get_init_guess(localizer.init_guess)
    else:
        u0 = localizer.get_init_guess(None)

    rotaiter = ciah.rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=log)
    u, g_orb, stat = next(rotaiter)
    cput1 = log.timer('initializing CIAH', *cput0)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = 0
    for imacro in range(localizer.max_cycle):
        norm_gorb = numpy.linalg.norm(g_orb)
        u0 = numpy.dot(u0, u)
        e = localizer.cost_function(u0)
        e_last, de = e, e-e_last

        log.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        cput1 = log.timer(f'cycle= {imacro+1}', *cput1)

        if (norm_gorb < conv_tol_grad and abs(de) < localizer.conv_tol
                and stat.tot_hop < localizer.ah_max_cycle):
            conv = True

        if callable(callback):
            callback(locals())

        if conv:
            break

        u, g_orb, stat = rotaiter.send(u0)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    sorted_idx = mo_mapping.mo_1to1map(u0)
    if return_u:
        return u0, sorted_idx
    localizer.mo_coeff = numpy.dot(localizer.mo_coeff, u0[:,sorted_idx])
    return localizer.mo_coeff


class Boys(pyscf_boys.Boys):
    kernel = kernel


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
