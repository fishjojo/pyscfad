import pytest
import numpy as np
import jax
import pyscf
from pyscfad import gto
from pyscfad.lib import numpy as jnp

TOL_NUC2 = 5e-9

TEST_SET = ["int1e_ovlp", "int1e_kin", "int1e_nuc",
            "int1e_rinv",]
TEST_SET_NUC = ["int1e_nuc"]

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'ccpvdz',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'ccpvdz'
    mol.verbose=0
    mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)
    return mol


def hess_analyt(mol, intor):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    s0 = mol.intor(intor)
    nao = s0.shape[-1]
    h = np.zeros((nao,nao,mol.natm,3,mol.natm,3))
    ip2 = intor.replace("int1e_","int1e_ipip")
    ipip = intor.replace("_sph","").replace("_cart","").replace("int1e_","int1e_ip")+"ip"
    if "_sph" in intor:
        ipip = ipip + "_sph"
    elif "_cart" in intor:
        ipip = ipip + "_cart"
    s2 = mol.intor(ip2).reshape(3,3,nao,nao).transpose(2,3,0,1)
    s12 = mol.intor(ipip).reshape(3,3,nao,nao).transpose(2,3,0,1)
    for ia in atmlst:
        p0, p1 = aoslices[ia,2:]
        h[p0:p1,:,ia,:,ia] += s2[p0:p1]
        h[:,p0:p1,ia,:,ia] += s2[p0:p1].transpose(1,0,2,3)
        for ja in atmlst:
            q0, q1 = aoslices[ja,2:]
            h[p0:p1,q0:q1,ia,:,ja,:] += s12[p0:p1,q0:q1]
            h[q0:q1,p0:p1,ia,:,ja,:] += s12[q0:q1,p0:p1].transpose(0,1,3,2)
    return h

def _test_int1e_deriv2_nuc(intor, mol0, mol1, funanal, args, tol=TOL_NUC2):
    hess0 = funanal(*args)
    hess_fwd = jax.jacfwd(jax.jacfwd(mol1.__class__.intor))(mol1, intor)
    hess_rev = jax.jacrev(jax.jacrev(mol1.__class__.intor))(mol1, intor)
    assert abs(hess_fwd.coords.coords - hess0).max() < tol
    assert abs(hess_rev.coords.coords - hess0).max() < tol

def test_int1e_deriv2(get_mol0, get_mol):
    mol0 = get_mol0
    mol1 = get_mol
    for intor in set(TEST_SET) - set(TEST_SET_NUC):
        _test_int1e_deriv2_nuc(intor, mol0, mol1, hess_analyt, (mol0, intor))
