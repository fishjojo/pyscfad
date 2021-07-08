import pytest
import numpy
import jax
import pyscf
from pyscfad import gto

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'H 0. 0. 0.; F 0. , 0. , 0.91',
        basis = '631g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. , 0. , 0.91'
    mol.basis = '631g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def hess_analyt(mol):
    nao = mol.nao
    natm = mol.natm
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    aa = mol.intor("int2e_ipip1").reshape(3,3,nao,nao,nao,nao).transpose(2,3,4,5,0,1)
    ab = mol.intor("int2e_ipvip1").reshape(3,3,nao,nao,nao,nao).transpose(2,3,4,5,0,1)
    ac = mol.intor("int2e_ip1ip2").reshape(3,3,nao,nao,nao,nao).transpose(2,3,4,5,0,1)

    h = numpy.zeros((nao,nao,nao,nao,natm,3,natm,3))
    for ia in atmlst:
        p0, p1 = aoslices[ia,2:]
        h[p0:p1,...,ia,:,ia,:] += aa[p0:p1]
        h[:,p0:p1,...,ia,:,ia,:] += aa[p0:p1].transpose(1,0,2,3,4,5)
        h[...,p0:p1,:,ia,:,ia,:] += aa[p0:p1].transpose(2,3,0,1,4,5)
        h[...,p0:p1,ia,:,ia,:] += aa[p0:p1].transpose(2,3,1,0,4,5)
        for ja in atmlst:
            q0, q1 = aoslices[ja,2:]
            h[p0:p1,q0:q1,...,ia,:,ja,:] += ab[p0:p1,q0:q1]
            h[q0:q1,p0:p1,...,ia,:,ja,:] += ab[q0:q1,p0:p1].transpose(0,1,2,3,5,4)
            h[...,p0:p1,q0:q1,ia,:,ja,:] += ab[p0:p1,q0:q1].transpose(2,3,0,1,4,5)
            h[...,q0:q1,p0:p1,ia,:,ja,:] += ab[q0:q1,p0:p1].transpose(2,3,0,1,5,4)
            h[p0:p1,:,q0:q1,:,ia,:,ja,:] += ac[p0:p1,:,q0:q1]
            h[q0:q1,:,p0:p1,:,ia,:,ja,:] += ac[q0:q1,:,p0:p1].transpose(0,1,2,3,5,4)
            h[:,p0:p1,:,q0:q1,ia,:,ja,:] += ac[p0:p1,:,q0:q1].transpose(1,0,3,2,4,5)
            h[:,q0:q1,:,p0:p1,ia,:,ja,:] += ac[q0:q1,:,p0:p1].transpose(1,0,3,2,5,4)
            h[p0:p1,...,q0:q1,ia,:,ja,:] += ac[p0:p1,:,q0:q1].transpose(0,1,3,2,4,5)
            h[q0:q1,...,p0:p1,ia,:,ja,:] += ac[q0:q1,:,p0:p1].transpose(0,1,3,2,5,4)
            h[:,p0:p1,q0:q1,:,ia,:,ja,:] += ac[p0:p1,:,q0:q1].transpose(1,0,2,3,4,5)
            h[:,q0:q1,p0:p1,:,ia,:,ja,:] += ac[q0:q1,:,p0:p1].transpose(1,0,2,3,5,4)
    return h


# pylint: disable=redefined-outer-name
def test_int2e(get_mol0, get_mol):
    mol0 = get_mol0
    h0 = hess_analyt(mol0)

    mol1 = get_mol
    h1 = jax.jacfwd(jax.jacfwd(mol1.__class__.intor))(mol1, "int2e").coords.coords
    assert abs(h1-h0).max() < 1e-6
