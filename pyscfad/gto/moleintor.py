import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.config import config
config.update("jax_enable_x64", True)

@custom_jvp
def int1e_ovlp(mol):
    return mol.mol.intor("int1e_ovlp")

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

@custom_jvp
def int1e_kin(mol):
    return mol.mol.intor("int1e_kin")

@int1e_kin.defjvp
def int1e_kin_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_kin(mol)

    mol_t, = tangents
    coords = mol_t.coords
    atmlst = range(mol.mol.natm)
    aoslices = mol.mol.aoslice_by_atom()
    nao = mol.mol.nao
    tangent_out = jnp.zeros((nao,nao))
    s1 = -mol.mol.intor('int1e_ipkin', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = jnp.einsum('xij,x->ij',s1[:,p0:p1],coords[k])
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[p0:p1], tmp)
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,p0:p1], tmp.T)
    return primal_out, tangent_out

@custom_jvp
def int1e_nuc(mol):
    return mol.mol.intor("int1e_nuc")

@int1e_nuc.defjvp
def int1e_nuc_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_nuc(mol)

    mol_t, = tangents
    coords = mol_t.coords
    atmlst = range(mol.mol.natm)
    aoslices = mol.mol.aoslice_by_atom()
    nao = mol.mol.nao
    tangent_out = jnp.zeros((nao,nao))

    h1 = -mol.mol.intor('int1e_ipnuc', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        with mol.mol.with_rinv_at_nucleus(ia):
            vrinv = mol.mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.mol.atom_charge(ia)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        tmp = vrinv + vrinv.transpose(0,2,1)
        tmp1 = jnp.einsum('xij,x->ij',tmp, coords[k])
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,:], tmp1)
    return primal_out, tangent_out

@custom_jvp
def int2e(mol):
    return mol.mol.intor("int2e")

@int2e.defjvp
def int2e_jvp(primals, tangents):
    mol, = primals
    primal_out = int2e(mol)

    mol_t, = tangents
    coords = mol_t.coords
    atmlst = range(mol.mol.natm)
    aoslices = mol.mol.aoslice_by_atom()
    nao = mol.mol.nao
    tangent_out = jnp.zeros([nao,]*4)

    eri1 = -mol.mol.intor("int2e_ip1", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = jnp.einsum("xijkl,x->ijkl", eri1[:,p0:p1], coords[k])
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[p0:p1], tmp)
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,p0:p1], tmp.transpose((1,0,2,3)))
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,:,p0:p1], tmp.transpose((2,3,0,1)))
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,:,:,p0:p1], tmp.transpose((2,3,1,0)))
    return primal_out, tangent_out
