from functools import reduce, partial
import numpy
import scipy
import jax
from pyscf.lib import logger
from pyscf.lo import pipek as pyscf_pipek
from pyscfad import numpy as np
from pyscfad.lib import vmap
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import (
    extract_rotation,
    pack_uniq_var,
)
from pyscfad.tools.linear_solver import gen_gmres
from pyscfad.lo import orth, boys

def atomic_pops(mol, mo_coeff, method='mulliken', s=None):
    method = method.lower().replace('_', '-')
    nmo = mo_coeff.shape[1]
    proj = None

    if s is None:
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

class PipekMezey(pyscf_pipek.PipekMezey):
    def atomic_pops(self, mol, mo_coeff, method=None, s=None):
        if method is None:
            method = self.pop_method
        return numpy.asarray(atomic_pops(mol, mo_coeff, method, s=s))

    kernel = boys.kernel

PM = Pipek = PipekMezey

def cost_function(x, mol, mo_coeff, pop_method='mulliken', exponent=2):
    u = extract_rotation(x)
    mo_coeff = np.dot(mo_coeff, u)
    pop = atomic_pops(mol, mo_coeff, pop_method)
    if exponent == 2:
        return -np.einsum('xii,xii->', pop, pop)
    else:
        pop2 = np.einsum('xii->xi', pop)**2
        return -np.einsum('xi,xi', pop2, pop2)

def _opt_cond(x, mol, mo_coeff, pop_method='mulliken', exponent=2):
    g = jax.grad(cost_function, 0)(x, mol, mo_coeff, pop_method, exponent)
    return g

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

def pm(mol, mo_coeff, *,
       pop_method='mulliken', exponent=2, init_guess=None,
       conv_tol=None, conv_tol_grad=None, max_cycle=None,
       symmetry=False, gmres_options=None):
    if mo_coeff.shape[-1] == 1:
        return mo_coeff
    if gmres_options is None:
        gmres_options = {}

    solver = gen_gmres(safe=symmetry, **gmres_options)
    optcond = partial(_opt_cond,
                      pop_method=pop_method,
                      exponent=exponent)
    _pm_iter = make_implicit_diff(_pm,
                                  implicit_diff=True,
                                  fixed_point=False,
                                  optimality_cond=optcond,
                                  solver=solver,
                                  has_aux=True)

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

def jacobi_sweep(mlo):
    isstable, mo1 = mlo.stability_jacobi()
    if not isstable:
        mo = mlo.kernel(mo1)
        mlo = jacobi_sweep(mlo)
    return mlo

