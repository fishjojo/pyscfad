import pytest
import numpy
import jax
from jax import numpy as jnp
from pyscf.pbc import grad as pyscf_grad
from pyscfad.pbc import gto

@pytest.fixture
def get_cell():
    cell = gto.Cell()
    cell.atom = '''Si 0.,  0.,  0.001
                Si 1.3467560987,  1.3467560987,  1.3467560987'''
    cell.a = '''0.            2.6935121974    2.6935121974
             2.6935121974  0.              2.6935121974
             2.6935121974  2.6935121974    0.    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [5,5,5]
    cell.build()
    return cell

def test_SI(get_cell):
    cell = get_cell
    Gv = cell.get_Gv()
    SI = cell.get_SI()
    natm = cell.natm
    ng = Gv.shape[0]
    g0 = numpy.zeros((natm,ng,natm,3), dtype=numpy.complex128)
    for i in range(natm):
        g0[i,:,i] += -1j * numpy.einsum("gx,g->gx", Gv, SI[i])
    jac_fwd = jax.jacfwd(cell.__class__.get_SI)(cell)
    assert abs(jac_fwd.coords - g0).max() < 1e-10

    # NOTE vjp for functions f:R->C will lose the imaginary part,
    #      and reverse-mode autodiff will fail in such cases. For
    #      functions f:R->R or f:C->C, both jvp and vjp will work.
    _, func_vjp = jax.vjp(cell.__class__.get_SI, cell)
    ct = jnp.eye((natm*ng), dtype=jnp.complex128).reshape(natm*ng,natm,ng)
    jac_bwd = jax.vmap(func_vjp)(ct)[0].coords.reshape(natm,ng,natm,3)
    assert abs(jac_bwd - g0.real).max() < 1e-10

    def fun(cell):
        out = []
        SI = cell.get_SI()
        for i in range(natm):
            out.append((SI[i] * SI[i].conj()).real.sum())
        return out
    jac_fwd = jax.jacfwd(fun)(cell)
    jac_bwd = jax.jacrev(fun)(cell)

    norm = fun(cell)
    for i in range(natm):
        grad = jnp.einsum("gnx,g->nx", g0[i], SI[i].conj())
        grad += grad.conj()
        grad = (grad * 0.5 / norm[i]).real
        assert abs(grad - jac_fwd[i].coords).max() < 1e-10
        assert abs(grad - jac_bwd[i].coords).max() < 1e-10


def test_ewald(get_cell):
    cell = get_cell
    def fun(cell):
        return cell.energy_nuc()
    jac_fwd = jax.jacfwd(fun)(cell)
    jac_bwd = jax.jacrev(fun)(cell)
    g0 = pyscf_grad.krhf.grad_nuc(cell, None)
    assert abs(g0 - jac_fwd.coords).max() < 1e-10
    assert abs(g0 - jac_bwd.coords).max() < 1e-10
