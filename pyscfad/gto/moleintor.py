import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.config import config
config.update("jax_enable_x64", True)

@custom_jvp
def int1e_ovlp(mol):
    return mol.mol.intor('int1e_ovlp')

@int1e_ovlp.defjvp
def int1e_ovlp_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_ovlp(mol)

    mol_t, = tangents
    coords = mol_t.coords
    atmlst = range(mol.mol.natm)
    aoslices = mol.mol.aoslice_by_atom()
    nao = mol.mol.nao
    tangent_out = jnp.zeros((nao,nao))
    s1 = -mol.mol.intor('int1e_ipovlp', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = jnp.einsum('xij,x->ij',s1[:,p0:p1],coords[k])
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[p0:p1], tmp)
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,p0:p1], tmp.T)
    return primal_out, tangent_out
