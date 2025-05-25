'''
This example demonstrates a method for optimizing an auxiliary basis set against
the entire two-electron integral tensor of a molecule. The exponents of the
auxiliary basis orbitals, or the alpha and beta parameters for even-tempered
Gaussian functions, are optimized to minimize the overall difference between the
density-fitting approximate integrals and the two-electron integrals in a
least-squares sense.

For more theoretical background, please refer to:
    Enhancing the Productivity of Quantum Chemistry Simulations in PySCF, arXiv:xxxx.xxxx
This example provides a complete implementation of the code discussed in that paper.
'''

import numpy as np
from scipy.optimize import least_squares
import jax
from jax import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
from pyscfad import gto
from pyscf.df.incore import aux_e2
from pyscfad.gto.mole import setup_exp
from pyscfad.df.incore import int3c_cross

def setup_auxbasis(mol, auxbasis):
    ref = mol.to_pyscf().intor('int2e', aosym='s4')
    tril_idx = jnp.tril_indices(mol.nao)

    auxmol = gto.Mole()
    auxmol.atom = mol.atom
    auxmol.basis = auxbasis
    auxmol.build(trace_ctr_coeff=False)
    x0 = auxmol.exp
    _, _, env_ptr, unflatten_exp = setup_exp(auxmol, return_unravel_fn=True)

    def f_residual_for_jacobian(x):
        auxmol.exp = jnp.array(x)
        int3c2e = int3c_cross(mol, auxmol, aosym='s1')[tril_idx]
        int2c2e = auxmol.intor('int2c2e')
        return (int3c2e.dot(jnp.linalg.solve(int2c2e, int3c2e.T)) - ref).ravel()

    # The jacobian is typically a very tall matrix
    _jac = jax.jacfwd(f_residual_for_jacobian)
    def jac(x):
        # Before calling jac, update the exponents for the current x.
        # This assignment cannot be performed within f_residual_for_jacobian.
        # The side-effect operation will cause TracerArrayConversionError
        auxmol._env[env_ptr] = x
        return _jac(x)

    def f_residual(x):
        _auxmol = auxmol.to_pyscf()
        _auxmol._env[env_ptr] = x
        int3c2e = aux_e2(mol.to_pyscf(), _auxmol, aosym='s2')
        int2c2e = _auxmol.intor('int2c2e')
        return (int3c2e.dot(np.linalg.solve(int2c2e, int3c2e.T)) - ref).ravel()

    return f_residual, jac, x0, unflatten_exp

def setup_etbs(mol, etbs):
    ref = mol.to_pyscf().intor('int2e', aosym='s4')
    tril_idx = jnp.tril_indices(mol.nao)

    auxbasis = {}
    for element, etb in etbs.items():
        auxbasis[element] = []
        for l, n, alpha, beta in etb:
            exps = alpha * beta**np.arange(n)
            auxbasis[element].extend([[l, [e, 1.]] for e in exps])
    auxmol = gto.Mole()
    auxmol.atom = mol.atom
    auxmol.basis = auxbasis
    auxmol.build(trace_ctr_coeff=False)
    x0, _, env_ptr, unflatten_exp = setup_exp(auxmol, return_unravel_fn=True)

    # The order of elements in rebuilt basis are consist to that in auxmol.exp
    sorted_elements = list(unflatten_exp(x0).keys())
    x0 = [(alpha, beta) for elem in sorted_elements for l, n, alpha, beta in etbs[elem]]
    ngroup = len(x0)
    x0 = np.array(x0).ravel()

    def rebuild_etbs(inp):
        inp = inp.reshape(ngroup, 2)
        new_etbs = {}
        k = 0
        for element in sorted_elements:
            element_etbs = []
            for l, n, _, _ in etbs[element]:
                alpha, beta = inp[k]
                element_etbs.append((l, n, alpha, beta))
                k += 1
            new_etbs[element] = element_etbs
        return new_etbs

    def to_exp(inp):
        inp = inp.reshape(ngroup, 2)
        exps = []
        k = 0
        for element in sorted_elements:
            for l, n, _, _ in etbs[element]:
                alpha, beta = inp[k]
                exps.append(alpha * beta**jnp.arange(n))
                k += 1
        return jnp.hstack(exps)

    def f_residual_for_jacobian(x):
        auxmol.exp = to_exp(x)
        int3c2e = int3c_cross(mol, auxmol, aosym='s1')[tril_idx]
        int2c2e = auxmol.intor('int2c2e')
        return (int3c2e.dot(jnp.linalg.solve(int2c2e, int3c2e.T)) - ref).ravel()

    _jac = jax.jacfwd(f_residual_for_jacobian)
    def jac(x):
        auxmol._env[env_ptr] = to_exp(x)
        return _jac(x)

    def f_residual(x):
        _auxmol = auxmol.to_pyscf()
        _auxmol._env[env_ptr] = to_exp(x)
        int3c2e = aux_e2(mol.to_pyscf(), _auxmol, aosym='s2')
        int2c2e = _auxmol.intor('int2c2e')
        return (int3c2e.dot(np.linalg.solve(int2c2e, int3c2e.T)) - ref).ravel()

    return f_residual, jac, x0, rebuild_etbs

if __name__ == '__main__':
    from pyscf.gto import charge
    from pyscf.df.addons import autoaux, aug_etb, _aug_etb_element

    mol = gto.Mole()
    mol.atom = 'C 0. 0. 0.0; O 0. 0. 1.1'
    mol.basis = 'cc-pvdz'
    mol.build(trace_ctr_coeff=False, trace_exp=False)

    # Optimize auxbasis exponents
    #auxbasis = {
    #    'C': [[0, [9., 1.]],
    #          [0, [3., 1.]],
    #          [0, [1., 1.]],
    #          [0, [.3, 1.]],
    #          [1, [3., 1.]],
    #          [1, [1., 1.]],
    #          [2, [1., 1.]]],
    #    'O': [[0, [9., 1.]],
    #          [0, [3., 1.]],
    #          [0, [1., 1.]],
    #          [0, [.3, 1.]],
    #          [1, [3., 1.]],
    #          [1, [1., 1.]],
    #          [2, [1., 1.]]]
    #}
    #auxbasis = autoaux(mol)
    auxbasis = aug_etb(mol)
    f_residual, jac, x0, unravel_exp = setup_auxbasis(mol, auxbasis)
    result = least_squares(f_residual, x0, jac=jac, gtol=1e-6, verbose=2)
    print(unravel_exp(result.x))

    # Optimize alpha, beta in even-tempered Gaussian
    # etb = (l, n, alpha, beta)
    #etbs = {
    #    'C': [[0, 6, .7, 2.], [1, 4, 1.2, 2.], [2, 2, 1.2, 2.]],
    #    'O': [[0, 6, .7, 2.], [1, 4, 1.2, 2.], [2, 2, 1.2, 2.]]
    #}
    uniq_atoms = {a[0] for a in mol._atom}
    etbs = {
        symb: _aug_etb_element(charge(symb), mol._basis[symb], 2.0)
        for symb in uniq_atoms
    }
    f_residual, jac, x0, rebuild_etbs = setup_etbs(mol, etbs)
    result = least_squares(f_residual, x0, jac=jac, gtol=1e-6, verbose=2)
    print(rebuild_etbs(result.x))
