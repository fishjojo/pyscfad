import jax
from pyscfad import gto
from pyscfad.df import incore
from pyscf.df.incore import aux_e2
from pyscf.gto import PTR_EXP, PTR_COEFF

def test_int3c_aux_exp():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.8'
    mol.basis = 'sto3g'
    mol.build(trace_ctr_coeff=False, trace_exp=False)

    def func(auxmol):
        return incore.int3c_cross(mol, auxmol)

    auxmol = gto.Mole()
    auxmol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.8'
    auxmol.basis = {'H': [[0, [1., 1.]],
                         [1, [2., 1., .5], [3., .1, .4]]]}
    auxmol.build(trace_ctr_coeff=False)

    jac = jax.jacobian(func)(auxmol)
    dat = jac.exp

    mol = mol.to_pyscf()
    auxmol = auxmol.to_pyscf()

    ptrs = [auxmol._bas[0,PTR_EXP], auxmol._bas[1,PTR_EXP]+0,
            auxmol._bas[1,PTR_EXP]+1]
    for i, ptr in enumerate(ptrs):
        auxmol._env[ptr] += .001
        ints1 = aux_e2(mol, auxmol, aosym='s1')

        auxmol._env[ptr] -= .002
        ints2 = aux_e2(mol, auxmol, aosym='s1')
        auxmol._env[ptr] += .001
        ref = (ints1 - ints2) / 0.002
        assert abs(ref - dat[:,:,:,i]).max() < 1e-5

def test_int3c_aux_ctr_coeff():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.8'
    mol.basis = 'sto3g'
    mol.build(trace_ctr_coeff=False, trace_exp=False)

    def func(auxmol):
        return incore.int3c_cross(mol, auxmol)

    auxmol = gto.Mole()
    auxmol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.8'
    auxmol.basis = {'H': [[0, [1., 1.]],
                         [1, [2., 1., .5], [3., .1, .4]]]}
    auxmol.build(trace_exp=False)
    jac = jax.jacobian(func)(auxmol)
    dat = jac.ctr_coeff

    mol = mol.to_pyscf()
    auxmol = auxmol.to_pyscf()

    ptrs = [auxmol._bas[0,PTR_COEFF],
            auxmol._bas[1,PTR_COEFF]+0,
            auxmol._bas[1,PTR_COEFF]+1,
            auxmol._bas[1,PTR_COEFF]+2,
            auxmol._bas[1,PTR_COEFF]+3]

    for i, ptr in enumerate(ptrs):
        auxmol._env[ptr] += .01
        ints1 = aux_e2(mol, auxmol, aosym='s1')

        auxmol._env[ptr] -= .02
        ints2 = aux_e2(mol, auxmol, aosym='s1')
        auxmol._env[ptr] += .01
        ref = (ints1 - ints2) / 0.02
        assert abs(ref - dat[:,:,:,i]).max() < 1e-12
