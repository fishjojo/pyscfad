import pytest
import ctypes
import numpy
import pyscf
from pyscf import lib
from pyscf.gto.moleintor import make_loc#, make_cintopt
from pyscfadlib import libcgto_vjp as libcgto

@pytest.fixture
def mol_h2():
    mol = pyscf.M(
        atom = 'H 0., 0., 0.; H 0., 0., 0.74',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

def test_int2c_r0_vjp(mol_h2):
    mol = mol_h2

    intor = 'int1e_ovlp_dr10_sph'
    drv = getattr(libcgto, 'GTOint2c_r0_vjp')
    fn = getattr(libcgto, intor)

    nao = mol.nao
    natm = mol.natm
    nbas = mol.nbas
    comp = 1
    ndim = 3
    hermi = 1
    shls_slice = (0, nbas, 0, nbas)
    ao_loc = make_loc(mol._bas, intor)
    ao_loc =numpy.asarray(ao_loc, order='C', dtype=numpy.int32)
    atm = numpy.asarray(mol._atm, order='C', dtype=numpy.int32)
    bas = numpy.asarray(mol._bas, order='C', dtype=numpy.int32)
    env = numpy.asarray(mol._env, order='C', dtype=numpy.double)
    cintopt = lib.c_null_ptr()#make_cintopt(atm, bas, env, intor)

    vjp = numpy.zeros((natm, ndim), order='C', dtype=numpy.double)
    numpy.random.seed(12345)
    ybar = numpy.random.rand(comp, nao, nao)
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

    drv(fn,
        vjp.ctypes.data_as(ctypes.c_void_p),
        ybar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(ndim), ctypes.c_int(hermi),
        (ctypes.c_int*4)(*(shls_slice[:4])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    finger = numpy.linalg.norm(numpy.cos(vjp))
    assert abs(finger - 2.446220919742891) < 1e-6

