from functools import reduce, partial
import numpy
import scipy
from pyscf import numpy as np
from pyscf.lo.pipek import PM
from pyscfad import config
from pyscfad.lib import vmap
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import extract_rotation, pack_uniq_var
from pyscfad.tools.linear_solver import precond_by_hdiag, gen_gmres
from pyscfad.lo import orth

def atomic_pops(mol, mo_coeff, method='mulliken'):
    method = method.lower().replace('_', '-')
    nmo = mo_coeff.shape[1]
    proj = None

    s = mol.intor_symmetric('int1e_ovlp')

    if method == 'becke':
        raise NotImplementedError

    elif method == 'mulliken':
        def fn(aoslice, mos, idx):
            p0, p1 = aoslice[:]
            mask = (idx >= p0) & (idx < p1)
            mos1 = np.where(mask, mos, np.array(0, dtype=mos.dtype))
            csc = reduce(np.dot, (mos1.conj().T, s, mos))
            return (csc + csc.conj().T) * .5

        aoslices = mol.aoslice_by_atom()[:,2:]
        idx = np.arange(mol.nao)
        proj = vmap(fn, in_axes=(0, None, None))(aoslices, mo_coeff, idx[:,None])

    elif method in ('lowdin', 'meta-lowdin'):
        raise NotImplementedError

    elif method in ('iao', 'ibo'):
        from pyscfad.lo import iao
        # NOTE mo_coeff must be the occupied
        orb_occ = mo_coeff

        iao_coeff = iao.iao(mol, orb_occ)
        iao_coeff = orth.vec_lowdin(iao_coeff, s)
        csc = reduce(np.dot, (mo_coeff.conj().T, s, iao_coeff))

        def fn(aoslice, csc, idx):
            p0, p1 = aoslice[:]
            mask = (idx >= p0) & (idx < p1)
            csc1 = np.where(mask, csc, np.array(0, dtype=csc.dtype))
            return np.dot(csc1, csc1.conj().T)

        iao_mol = iao.reference_mol(mol)
        aoslices = iao_mol.aoslice_by_atom()[:,2:]
        idx = np.arange(iao_mol.nao)
        proj = vmap(fn, in_axes=(0, None, None))(aoslices, csc, idx[None,:])

    else:
        raise KeyError

    return proj


def opt_cond(x, mol, mo_coeff, pop_method='mulliken', exponent=2):
    u = extract_rotation(x)
    mo_coeff = np.dot(mo_coeff, u)
    pop = atomic_pops(mol, mo_coeff, pop_method)
    if exponent == 2:
        g0 = np.einsum('xii,xip->pi', pop, pop)
        g = -pack_uniq_var(g0-g0.conj().T) * 2
    else:
        pop3 = np.einsum('xii->xi', pop)**3
        g0 = np.einsum('xi,xip->pi', pop3, pop)
        g = -pack_uniq_var(g0-g0.conj().T) * 4

    h_diag = np.einsum('xii,xpp->pi', pop, pop) * 2
    g_diag = g0.diagonal()
    h_diag-= g_diag + g_diag.reshape(-1,1)
    h_diag+= np.einsum('xip,xip->pi', pop, pop) * 2
    h_diag+= np.einsum('xip,xpi->pi', pop, pop) * 2
    h_diag = -pack_uniq_var(h_diag) * 2
    return g, h_diag

def _pm(x, mol, mo_coeff, *,
        pop_method=None, exponent=None, init_guess=None,
        conv_tol=None, conv_tol_grad=None, max_cycle=None):
    mo_coeff = numpy.array(mo_coeff, dtype=mo_coeff.dtype, order='C')
    loc = PM(mol, mo_coeff=mo_coeff)
    if pop_method is not None:
        loc.pop_method = pop_method
    if exponent is not None:
        loc.exponent = exponent
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

def pm(mol, mo_coeff, *,
       pop_method='mulliken', exponent=2, init_guess=None,
       conv_tol=None, conv_tol_grad=None, max_cycle=None):
    if mo_coeff.shape[-1] == 1:
        return mo_coeff

    gen_precond = None
    if config.moleintor_opt:
        gen_precond = precond_by_hdiag
    _pm_iter = make_implicit_diff(_pm, implicit_diff=True,
                                  fixed_point=False,
                                  optimality_cond=partial(opt_cond,
                                                          pop_method=pop_method,
                                                          exponent=exponent),
                                  solver=gen_gmres(), has_aux=True,
                                  optimality_fun_has_aux=True,
                                  gen_precond=gen_precond)

    x, sorted_idx = _pm_iter(None, mol, mo_coeff,
                             pop_method=pop_method,
                             exponent=exponent,
                             init_guess=init_guess,
                             conv_tol=conv_tol,
                             conv_tol_grad=conv_tol_grad,
                             max_cycle=max_cycle)
    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])

pipekmezey = pm
